"""
app/pages/2_Alertes.py
-----------------------
Page des alertes — stress hydrique, azoté, ravageurs.
Inférence par cellules 100m×100m, agrégation sévérité parcelle = max cellules.
Onglets par type de stress.
"""

import sys
import hashlib
from pathlib import Path
from datetime import date, timedelta

import folium
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from streamlit_folium import st_folium
from sentinelhub import BBox, CRS

from src.ingestion.sentinel2 import get_sh_config, search_available_scenes, download_scene
from src.ingestion.meteo import fetch_historical_weather
from src.indices.vegetation import load_bands
from src.models.anomaly_detection import infer_parcel_by_cells

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Alertes - Parcelle Watch", page_icon="🚨", layout="wide")

ROOT       = Path(__file__).parent.parent.parent
DATA_PROC  = ROOT / "data" / "processed"
DATA_RAW   = ROOT / "data" / "raw"
MODELS_DIR = DATA_PROC / "models"
SCENES_DIR = DATA_RAW / "interface_test"
BBOX_LIMIT = 4

STRESS_TYPES = {
    "Hydrique 💧" : "HYDRIQUE",
    "Azoté 🌿"    : "AZOTE",
    "Ravageurs 🐛": "RAVAGEURS",
}

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("tif_path", None), ("date_acq", None), ("meteo_df", None),
    ("bands_cache", None), ("transform_cache", None),
    # Résultats par stress_type → dict {stress_type: df_results}
    ("results_by_stress", {}),
    ("intra_pid", None),
    ("intra_closed", False),
    ("intra_stress", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if "today"      not in st.session_state:
    st.session_state["today"] = date.today()
if "map_center" not in st.session_state:
    st.session_state["map_center"] = [48.69, 2.62]
if "parcelles"  not in st.session_state:
    st.session_state["parcelles"] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def sev_color(sev):
    return {"critical": "#d9534f", "warning": "#f0ad4e",
            "normal": "#5cb85c"}.get(sev, "#aaaaaa")


def global_bbox(parcelles):
    margin = 0.002
    return (
        min(p["bbox"][0] for p in parcelles.values()) - margin,
        min(p["bbox"][1] for p in parcelles.values()) - margin,
        max(p["bbox"][2] for p in parcelles.values()) + margin,
        max(p["bbox"][3] for p in parcelles.values()) + margin,
    )


def bbox_hash(minx, miny, maxx, maxy) -> str:
    key = f"{minx:.5f}_{miny:.5f}_{maxx:.5f}_{maxy:.5f}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


def scene_path(bbox_h: str, acq_date: date) -> Path:
    SCENES_DIR.mkdir(parents=True, exist_ok=True)
    return SCENES_DIR / f"sentinel2_{bbox_h}_{acq_date.strftime('%Y%m%d')}.tif"


def compute_meteo_features(meteo_df, ref_date) -> dict:
    mi  = meteo_df.set_index("date")
    ref = pd.Timestamp(ref_date)

    def agg(days_start, days_end, col, fn="sum"):
        w = mi.loc[ref - pd.Timedelta(days=days_end):
                   ref - pd.Timedelta(days=days_start), col]
        return float(getattr(w, fn)()) if len(w) else np.nan

    precip_7d  = agg(1, 7,  "precipitation_sum",          "sum")
    precip_14d = agg(1, 14, "precipitation_sum",          "sum")
    precip_30d = agg(1, 30, "precipitation_sum",          "sum")
    et0_7d     = agg(1, 7,  "et0_fao_evapotranspiration", "sum")
    tmax_7d    = agg(1, 7,  "temperature_2m_max",         "mean")
    tmin_7d    = agg(1, 7,  "temperature_2m_min",         "mean")

    tmax_vals = mi.loc[ref - pd.Timedelta(days=30):
                       ref - pd.Timedelta(days=1), "temperature_2m_max"]
    temp_sum_30d = float(tmax_vals.clip(lower=0).sum()) if len(tmax_vals) else np.nan

    precip_15_30 = agg(15, 30, "precipitation_sum", "sum")
    precip_delta = (precip_14d - precip_15_30
                    if not (np.isnan(precip_14d) or np.isnan(precip_15_30))
                    else np.nan)
    amp_7d = (tmax_7d - tmin_7d
              if not (np.isnan(tmax_7d) or np.isnan(tmin_7d)) else np.nan)
    hum_proxy = (precip_7d / (et0_7d + 0.01)
                 if not (np.isnan(precip_7d) or np.isnan(et0_7d)) else np.nan)

    return {
        "precip_7d"        : precip_7d,
        "precip_14d"       : precip_14d,
        "precip_30d"       : precip_30d,
        "tmax_7d"          : tmax_7d,
        "tmin_7d"          : tmin_7d,
        "et0_7d"           : et0_7d,
        "deficit_7d"       : (precip_7d - et0_7d
                               if not (np.isnan(precip_7d) or np.isnan(et0_7d))
                               else np.nan),
        "temp_sum_30d"     : temp_sum_30d,
        "precip_delta"     : precip_delta,
        "tmax_tmin_amp_7d" : amp_7d,
        "humidity_proxy_7d": hum_proxy,
    }


def run_inference_stress(parcelles_session: dict, bands: np.ndarray,
                          transform, meteo_feats: dict,
                          date_acq, stress_type: str) -> pd.DataFrame:
    """
    Lance l'inférence par cellules sur toutes les parcelles
    pour un type de stress donné.
    Retourne un DataFrame une ligne par parcelle.
    """
    rows = []
    progress = st.progress(0, text=f"Analyse {stress_type}...")
    total = len(parcelles_session)

    for i, (pid, pmeta) in enumerate(parcelles_session.items()):
        progress.progress((i + 1) / total,
                           text=f"{stress_type} — parcelle {i+1}/{total}")
        result = infer_parcel_by_cells(
            geom        = pmeta["geometry"],
            code_cultu  = pmeta["code_cultu"],
            bands       = bands,
            transform   = transform,
            meteo_feats = meteo_feats,
            acq_date    = date_acq,
            stress_type = stress_type,
            models_dir  = MODELS_DIR,
        )
        rows.append({
            "parcelle_id" : pid,
            "code_cultu"  : pmeta["code_cultu"],
            "surf_parc"   : pmeta["surf_parc"],
            "severity"    : result.get("severity", "unknown"),
            "worst_score" : result.get("worst_score", np.nan),
            "n_cells"     : result.get("n_cells", 0),
            "n_critical"  : result.get("n_critical", 0),
            "n_warning"   : result.get("n_warning", 0),
            "n_normal"    : result.get("n_normal", 0),
            "pct_critical": result.get("pct_critical", 0),
            "pct_warning" : result.get("pct_warning", 0),
            "model_source": result.get("model_source", "unknown"),
            "model_name"  : result.get("model_name", "general"),
            "error"       : result.get("error"),
            "_cells"      : result.get("cells", []),  # pour intra-parc
        })
    progress.empty()
    return pd.DataFrame(rows)


# ── Rendu carte + tableau pour un stress_type ─────────────────────────────────

def render_stress_tab(stress_type: str, df_res: pd.DataFrame,
                       parcelles_session: dict, date_acq):
    """Affiche carte + filtres + tableau + intra-parcellaire pour un stress."""

    # ── Filtres ───────────────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
    with col_f1:
        cultures = sorted(df_res["code_cultu"].dropna().unique().tolist())
        sel_cult = st.multiselect("Cultures", options=cultures, default=cultures,
                                   key=f"filt_cult_{stress_type}")
    with col_f2:
        sev_filt = st.selectbox(
            "Sévérité", ["Toutes", "Attention et critique", "Critique uniquement"],
            key=f"filt_sev_{stress_type}",
        )
    with col_f3:
        sort_by = st.selectbox(
            "Trier par", ["Score (pire en premier)", "Surface", "Culture"],
            key=f"filt_sort_{stress_type}",
        )

    df = df_res.copy()
    if sel_cult:
        df = df[df["code_cultu"].isin(sel_cult)]
    if sev_filt == "Attention et critique":
        df = df[df["severity"].isin(["warning", "critical"])]
    elif sev_filt == "Critique uniquement":
        df = df[df["severity"] == "critical"]

    sort_map = {"Score (pire en premier)": ("worst_score", True),
                "Surface": ("surf_parc", False), "Culture": ("code_cultu", True)}
    scol, sasc = sort_map[sort_by]
    df = df.sort_values(scol, ascending=sasc)

    # ── Métriques ─────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    n_tot = len(df)
    m1.metric("Parcelles", n_tot)
    m2.metric("Critique",  int((df["severity"] == "critical").sum()))
    m3.metric("Attention", int((df["severity"] == "warning").sum()))
    m4.metric("Normal",    int((df["severity"] == "normal").sum()))

    st.divider()
    col_map, col_tbl = st.columns([3, 2])
    filtered_ids = set(df["parcelle_id"].astype(str).tolist())
    res_by_id    = df_res.set_index("parcelle_id").to_dict("index")

    with col_map:
        date_str = date_acq.strftime("%d %B %Y") if date_acq else "N/A"
        m = folium.Map(location=st.session_state["map_center"], zoom_start=14)
        m.get_root().html.add_child(folium.Element(
            f"<div style='position:fixed;top:10px;left:50%;transform:translateX(-50%);"
            f"background:rgba(0,0,0,0.75);color:white;padding:8px 14px;"
            f"border-radius:6px;font-size:13px;z-index:9999;font-family:monospace;'>"
            f"Parcelle Watch — {stress_type} — {date_str}</div>"
        ))
        m.get_root().html.add_child(folium.Element(
            "<div style='position:fixed;bottom:20px;right:20px;"
            "background:rgba(0,0,0,0.8);color:white;padding:10px 14px;"
            "border-radius:6px;font-size:11px;z-index:9999;font-family:monospace;'>"
            "<b>Sévérité</b><br>"
            "<span style='color:#d9534f'>■</span> Critique<br>"
            "<span style='color:#f0ad4e'>■</span> Attention<br>"
            "<span style='color:#5cb85c'>■</span> Normal<br>"
            "<span style='color:#cccccc'>■</span> Filtré</div>"
        ))

        for pid, pmeta in parcelles_session.items():
            r         = res_by_id.get(pid, {})
            in_filter = pid in filtered_ids
            sev       = r.get("severity", "unknown") if in_filter else "filtered"
            color     = sev_color(sev) if in_filter else "#cccccc"
            opacity   = 0.65 if in_filter else 0.2

            popup_html = (
                f"<div style='font-family:monospace;font-size:12px;min-width:180px'>"
                f"<b>{pid[:12]}</b><br>"
                f"Culture : <b>{pmeta['code_cultu']}</b><br>"
                f"Surface : {pmeta['surf_parc']:.1f} ha<br>"
                f"<hr style='margin:3px 0'>"
                f"Sévérité : <b style='color:{color}'>{sev.upper()}</b><br>"
                f"Cellules critiques : {r.get('n_critical',0)} "
                f"({r.get('pct_critical',0):.0f}% surface)<br>"
                f"Modèle : {r.get('model_source','?')}"
                f"</div>"
            )
            folium.GeoJson(
                pmeta["geometry"].__geo_interface__,
                style_function=lambda _, c=color, o=opacity: {
                    "fillColor": c, "color": "white",
                    "weight": 1.5, "fillOpacity": o,
                },
                tooltip=folium.Tooltip(f"{pmeta['code_cultu']} — {sev.upper()}"),
                popup=folium.Popup(popup_html, max_width=220),
            ).add_to(m)

        st_folium(m, width=None, height=480, key=f"map_{stress_type}")

    with col_tbl:
        st.subheader("Tableau")
        df_disp = df[[
            "parcelle_id", "code_cultu", "surf_parc",
            "severity", "worst_score", "n_cells",
            "pct_critical", "pct_warning",
        ]].rename(columns={
            "parcelle_id" : "Parcelle",
            "code_cultu"  : "Culture",
            "surf_parc"   : "Ha",
            "severity"    : "Sévérité",
            "worst_score" : "Score",
            "n_cells"     : "Cellules",
            "pct_critical": "% Crit.",
            "pct_warning" : "% Warn.",
        }).copy()
        df_disp["Parcelle"] = df_disp["Parcelle"].str[:12]
        df_disp["Score"]    = df_disp["Score"].round(3)
        df_disp["% Crit."] = df_disp["% Crit."].round(1)
        df_disp["% Warn."] = df_disp["% Warn."].round(1)

        if df_disp.empty:
            st.info("Aucune parcelle pour ces filtres.")
        else:
            event = st.dataframe(
                df_disp, use_container_width=True, height=380,
                hide_index=True, on_select="rerun",
                selection_mode="single-row",
                key=f"tbl_{stress_type}",
            )
            sel_rows = (event.selection.get("rows", [])
                        if event and event.selection else [])

            if sel_rows and not st.session_state.get("intra_closed", False):
                full_pid = df.iloc[sel_rows[0]]["parcelle_id"]
                if (full_pid != st.session_state["intra_pid"]
                        or st.session_state["intra_stress"] != stress_type):
                    st.session_state["intra_pid"]    = full_pid
                    st.session_state["intra_stress"] = stress_type
                    st.rerun()
            if st.session_state.get("intra_closed", False):
                st.session_state["intra_closed"] = False

            if st.session_state["intra_pid"] is not None:
                st.caption(
                    f"↓ Analyse : `{str(st.session_state['intra_pid'])[:12]}`"
                )

    # ── Intra-parcellaire ─────────────────────────────────────────────────────
    intra_pid = st.session_state.get("intra_pid")
    # N'afficher que si l'intra appartient à ce stress_type
    if (intra_pid is not None
            and st.session_state.get("intra_stress") == stress_type):
        _render_intra(intra_pid, df_res, parcelles_session, stress_type)


def _render_intra(pid: str, df_res: pd.DataFrame,
                   parcelles_session: dict, stress_type: str):
    """Affiche l'analyse intra-parcellaire sous le tableau."""
    st.divider()

    row = df_res[df_res["parcelle_id"] == pid]
    if row.empty:
        st.warning("Résultats introuvables pour cette parcelle.")
        return

    cells      = row.iloc[0]["_cells"]
    pmeta      = parcelles_session[pid]
    nom        = pmeta.get("nom", pid[:12])
    surf       = pmeta["surf_parc"]
    n_cells    = len(cells)

    col_title, col_close = st.columns([5, 1])
    with col_title:
        st.subheader(
            f"Intra-parcellaire — {nom} ({pmeta['code_cultu']}) "
            f"— {stress_type} — {n_cells} cellules"
        )
    with col_close:
        if st.button("✕ Fermer", key=f"close_intra_{stress_type}"):
            st.session_state["intra_pid"]    = None
            st.session_state["intra_stress"] = None
            st.session_state["intra_closed"] = True
            st.rerun()

    if not cells:
        st.warning("Aucune cellule valide pour cette parcelle.")
        return

    col_carte, col_stats = st.columns([2, 1])

    with col_carte:
        geom     = pmeta["geometry"]
        centroid = geom.centroid
        m_intra  = folium.Map(location=[centroid.y, centroid.x],
                               zoom_start=16, tiles="Esri WorldImagery")
        folium.GeoJson(
            geom.__geo_interface__,
            style_function=lambda _: {
                "fillColor": "transparent", "color": "white",
                "weight": 2, "interactive": False,
            },
        ).add_to(m_intra)

        for cell in cells:
            color = sev_color(cell["severity"])
            folium.GeoJson(
                cell["geometry"].__geo_interface__,
                style_function=lambda _, c=color: {
                    "fillColor": c, "color": "white",
                    "weight": 1, "fillOpacity": 0.65, "interactive": False,
                },
                tooltip=folium.Tooltip(
                    f"{cell['cell_id']} | Score:{cell['score']:.3f} "
                    f"| {cell['severity'].upper()} | {cell['area_ha']:.2f} ha"
                ),
            ).add_to(m_intra)

        st_folium(m_intra, width=None, height=380,
                   key=f"map_intra_{pid}_{stress_type}")

    with col_stats:
        st.markdown("**Répartition des cellules**")
        df_cells = pd.DataFrame([{
            "Cellule"  : c["cell_id"],
            "Score"    : round(c["score"], 3),
            "Sévérité" : c["severity"],
            "Surface ha": c["area_ha"],
        } for c in cells]).sort_values("Score")

        st.dataframe(df_cells, use_container_width=True,
                      hide_index=True, height=250)

        total_area   = sum(c["area_ha"] for c in cells)
        stress_area  = sum(c["area_ha"] for c in cells
                           if c["severity"] in ("warning", "critical"))
        pct_stress   = stress_area / total_area * 100 if total_area else 0

        st.divider()
        st.metric("Surface totale",     f"{total_area:.1f} ha")
        st.metric("Surface en stress",  f"{stress_area:.1f} ha ({pct_stress:.0f}%)")

        # Économie d'eau (stress hydrique uniquement)
        if stress_type == "HYDRIQUE":
            apport_mm    = 30
            eau_uniforme = total_area  * apport_mm * 10
            eau_ciblee   = stress_area * apport_mm * 10
            economie     = eau_uniforme - eau_ciblee
            pct_eco      = economie / eau_uniforme * 100 if eau_uniforme > 0 else 0
            st.divider()
            st.markdown("**Économie d'eau estimée**")
            st.caption(f"Hypothèse : {apport_mm} mm/ha")
            st.metric("Irrigation uniforme", f"{eau_uniforme:.0f} m³")
            st.metric("Irrigation ciblée",   f"{eau_ciblee:.0f} m³")
            st.metric("Économie potentielle", f"{economie:.0f} m³",
                       delta=f"{pct_eco:.0f}%",
                       delta_color="normal" if economie >= 0 else "inverse")


# ════════════════════════════════════════════════════════════════════════════
# PAGE PRINCIPALE
# ════════════════════════════════════════════════════════════════════════════
st.title("Alertes")

parcelles_session = st.session_state.get("parcelles", {})
if not parcelles_session:
    st.warning("Aucune parcelle sélectionnée — revenez à 'Mes Parcelles'.")
    st.stop()

# ── Sidebar : nettoyage TIF ──────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Gestion des données")
    tif_files = list(SCENES_DIR.glob("sentinel2_*.tif")) if SCENES_DIR.exists() else []
    total_mb  = sum(f.stat().st_size for f in tif_files) / 1_048_576
    if tif_files:
        st.caption(f"{len(tif_files)} image(s) — {total_mb:.1f} Mo")
        if st.button("🗑️ Supprimer toutes les images", use_container_width=True):
            for f in tif_files:
                f.unlink(missing_ok=True)
            st.toast(f"{len(tif_files)} image(s) supprimée(s)")
            st.rerun()
    else:
        st.caption("Aucune image en cache")

# ── Acquisition satellite + météo (commune à tous les stress) ────────────────
minx, miny, maxx, maxy = global_bbox(parcelles_session)
sh_bbox = BBox(bbox=[minx, miny, maxx, maxy], crs=CRS.WGS84)
bbox_h  = bbox_hash(minx, miny, maxx, maxy)

############ AJOUT
st.markdown(f"Bbox : `{minx:.4f},{miny:.4f} → {maxx:.4f},{maxy:.4f}` (hash: `{bbox_h}`)")
############# / AJOUT

if (minx - maxx) ** 2 > BBOX_LIMIT or (miny - maxy) ** 2 > BBOX_LIMIT:
    st.warning("Parcelles trop éloignées pour une seule image satellite.")
    st.stop()

with st.expander("📡 Acquisition satellite & météo", expanded=False):
    config  = get_sh_config()
    end_d   = st.session_state["today"]
    start_d = end_d - timedelta(days=30)

    with st.spinner("Recherche scènes..."):
        scenes = search_available_scenes(sh_bbox, start_d, end_d,
                                         max_cloud_coverage=0.30, config=config)
    if not scenes:
        st.warning("Aucune scène disponible.")
        st.stop()

    latest   = scenes[-1]
    date_acq = latest["date"]
    st.session_state["date_acq"] = date_acq
    st.markdown(f"Scène : **{date_acq}** — nuages : {latest['cloud_coverage']:.1f}%")

    tif_path = scene_path(bbox_h, date_acq)
    if tif_path.exists():
        st.info(f"Cache : `{tif_path.name}`")
    else:
        with st.spinner("Téléchargement..."):
            tif_path = download_scene(
                bbox=sh_bbox, acquisition_date=date_acq,
                output_dir=SCENES_DIR, config=config,
            )
            expected = scene_path(bbox_h, date_acq)
            if tif_path != expected:
                tif_path.rename(expected)
                tif_path = expected
    st.session_state["tif_path"] = tif_path

    lat_c = (miny + maxy) / 2
    lon_c = (minx + maxx) / 2
    with st.spinner("Météo..."):
        meteo_df = fetch_historical_weather(
            lat_c, lon_c,
            date_acq - timedelta(days=30), date_acq,
        )
    meteo_df["date"] = pd.to_datetime(meteo_df["date"])
    st.session_state["meteo_df"] = meteo_df
    st.success(f"✅ Prêt — {date_acq}")

# Vérification que l'acquisition est faite
if st.session_state["tif_path"] is None:
    st.info("Ouvrez 'Acquisition satellite & météo' ci-dessus pour lancer l'analyse.")
    st.stop()

# Chargement bandes (une seule fois, mis en cache session)
if st.session_state["bands_cache"] is None:
    bands, meta = load_bands(st.session_state["tif_path"])
    st.session_state["bands_cache"]     = bands
    st.session_state["transform_cache"] = meta["transform"]

bands     = st.session_state["bands_cache"]
transform = st.session_state["transform_cache"]
meteo_feats = compute_meteo_features(
    st.session_state["meteo_df"],
    st.session_state["date_acq"],
)
date_acq = st.session_state["date_acq"]

# ── Onglets par type de stress ────────────────────────────────────────────────
tab_hydrique, tab_azote, tab_ravageurs = st.tabs(list(STRESS_TYPES.keys()))

for tab, (tab_label, stress_type) in zip(
    [tab_hydrique, tab_azote, tab_ravageurs], STRESS_TYPES.items()
):
    with tab:
        results_cache = st.session_state["results_by_stress"]

        # Bouton pour lancer/relancer l'analyse de ce stress
        col_btn, col_info = st.columns([2, 4])
        with col_btn:
            run_label = ("🔄 Relancer" if stress_type in results_cache
                         else "▶ Lancer l'analyse")
            do_run = st.button(run_label, key=f"run_{stress_type}",
                                type="primary")
        with col_info:
            if stress_type in results_cache:
                st.caption(f"Résultats disponibles — {date_acq}")

        if do_run:
            df_res = run_inference_stress(
                parcelles_session, bands, transform,
                meteo_feats, date_acq, stress_type,
            )
            results_cache[stress_type] = df_res
            st.session_state["results_by_stress"] = results_cache
            # Réinitialiser l'intra si on relance
            if st.session_state.get("intra_stress") == stress_type:
                st.session_state["intra_pid"]    = None
                st.session_state["intra_stress"] = None
            st.rerun()

        if stress_type in results_cache:
            render_stress_tab(
                stress_type,
                results_cache[stress_type],
                parcelles_session,
                date_acq,
            )
        else:
            st.info(f"Cliquez sur '▶ Lancer l'analyse' pour analyser le stress {tab_label}.")
