"""
app/pages/2_Alertes.py
-----------------------
Page des alertes de stress hydrique.
- Téléchargement scène Sentinel-2 + météo
- Feature engineering + inférence Isolation Forest
- Carte des résultats avec filtres (culture, sévérité)
- Analyse intra-parcellaire déclenchée par sélection dans le tableau
"""

import sys
from pathlib import Path
from datetime import date, timedelta

import folium
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import joblib
import streamlit as st
from shapely.geometry import Point, box as sbox, mapping
from streamlit_folium import st_folium
from sentinelhub import BBox, CRS

from src.ingestion.sentinel2 import get_sh_config, search_available_scenes, download_scene
from src.ingestion.meteo import fetch_historical_weather
from src.indices.vegetation import compute_ndvi, compute_ndwi, load_bands

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Alertes - Parcelle Watch", page_icon="🚨", layout="wide")

ROOT       = Path(__file__).parent.parent.parent
DATA_PROC  = ROOT / "data" / "processed"
DATA_RAW   = ROOT / "data" / "raw"
MODELS_DIR = DATA_PROC / "models"
DOCS_DIR   = ROOT / "docs"
BBOX_LIMIT = 4

CODE_CULTU_LABELS = {
    "BTH": "Ble tendre hiver",  "MIS": "Mais",          "CZH": "Colza hiver",
    "ORH": "Orge hiver",        "ORP": "Orge printemps","JAC": "Jachere",
    "PPH": "Prairie permanente","BTA": "Ble tendre autre",
    "BTN": "Ble tendre printemps","BOR": "Ble orge",
    "FVL": "Feverole",          "LEC": "Lentille",      "BFS": "Betterave",
}

FEATURE_COLS = [
    "ndwi_mean", "ndwi_p10", "ndwi_std", "ndvi_mean",
    "ndwi_deviation", "ndvi_deviation",
    "ndwi_delta", "ndvi_delta",
    "day_of_year",
    "precip_14d", "tmax_7d", "deficit_7d",
]

# ── Session state ────────────────────────────────────────────────────────────
for key, default in [
    ("tif_path", None), ("date_acq", None), ("meteo_df", None),
    ("df_features", None), ("df_results", None),
    ("intra_pid", None),          # parcelle sélectionnée pour intra-parcellaire
]:
    if key not in st.session_state:
        st.session_state[key] = default

if "today" not in st.session_state:
    st.session_state["today"] = date.today()
if "map_center" not in st.session_state:
    st.session_state["map_center"] = [48.69, 2.62]
if "parcelles" not in st.session_state:
    st.session_state["parcelles"] = {}


# ── Helpers ──────────────────────────────────────────────────────────────────
def sev_color(sev):
    return {"critical": "#d9534f", "warning": "#f0ad4e", "normal": "#5cb85c"}.get(sev, "#aaaaaa")


def global_bbox(parcelles):
    margin = 0.002
    return (
        min(p["bbox"][0] for p in parcelles.values()) - margin,
        min(p["bbox"][1] for p in parcelles.values()) - margin,
        max(p["bbox"][2] for p in parcelles.values()) + margin,
        max(p["bbox"][3] for p in parcelles.values()) + margin,
    )


def zonal_stats_parcel(geom, index_arr, transform, min_pixels=2):
    mask   = rasterio.features.geometry_mask(
        [mapping(geom)], out_shape=index_arr.shape,
        transform=transform, invert=True,
    )
    pixels = index_arr[mask]
    pixels = pixels[~np.isnan(pixels)]
    pixels = pixels[(pixels >= -1.0) & (pixels <= 1.0)]
    if len(pixels) < min_pixels:
        return None
    return {
        "mean": float(np.mean(pixels)), "std":  float(np.std(pixels)),
        "p10":  float(np.percentile(pixels, 10)),
        "p90":  float(np.percentile(pixels, 90)),
        "n":    len(pixels),
    }


def compute_meteo_features(meteo_df, ref_date):
    mi  = meteo_df.set_index("date")
    ref = pd.Timestamp(ref_date)
    def agg(days, col, fn="sum"):
        w = mi.loc[ref - pd.Timedelta(days=days): ref - pd.Timedelta(days=1), col]
        return float(getattr(w, fn)()) if len(w) else np.nan
    return {
        "precip_7d"  : agg(7,  "precipitation_sum",          "sum"),
        "precip_14d" : agg(14, "precipitation_sum",          "sum"),
        "tmax_7d"    : agg(7,  "temperature_2m_max",         "mean"),
        "et0_7d"     : agg(7,  "et0_fao_evapotranspiration", "sum"),
        "deficit_7d" : agg(7,  "precipitation_sum",          "sum")
                     - agg(7,  "et0_fao_evapotranspiration", "sum"),
    }


def get_model(code_cultu, stress_type="HYDRIQUE"):
    index_path = MODELS_DIR / "models_index.csv"
    if not index_path.exists():
        raise FileNotFoundError("models_index.csv manquant.")
    index = pd.read_csv(index_path)
    index = index[index["stress_type"] == stress_type]
    for _, row in index.iterrows():
        if code_cultu in str(row["cultures_codes"]).split(","):
            data = joblib.load(MODELS_DIR / row["filename"])
            data.update({"model_name": row["model_name"], "source": "specific"})
            return data
    general = index[index["cultures_codes"] == "ALL"]
    if not general.empty:
        data = joblib.load(MODELS_DIR / general.iloc[0]["filename"])
        data["source"] = "fallback"
        return data
    raise FileNotFoundError(f"Aucun modèle {stress_type} pour {code_cultu}")


def score_row(row, feature_cols):
    try:
        model_data = get_model(row["code_cultu"])
    except FileNotFoundError as e:
        st.warning(f"⚠️  {row['parcelle_id']} : {e}")
        return np.nan, False, "unknown", "N/A"
    model  = model_data["model"]
    scaler = model_data["scaler"]
    feats  = model_data.get("features", feature_cols)
    X      = row[feats].values.reshape(1, -1)
    X      = np.nan_to_num(X, nan=0.0)
    X_sc   = scaler.transform(X)
    score  = float(model.decision_function(X_sc)[0])
    is_anom = bool(model.predict(X_sc)[0] == -1)
    sev = "critical" if score < -0.12 else "warning" if score < -0.04 else "normal"
    return score, is_anom, sev, model_data.get("model_name", "general")


def build_grid(geom, n):
    """Grille NxN intersectée avec la géométrie réelle de la parcelle."""
    minx, miny, maxx, maxy = geom.bounds
    dx, dy = (maxx - minx) / n, (maxy - miny) / n
    cells  = []
    for r in range(n):
        for c in range(n):
            cell_box  = sbox(minx+c*dx, miny+r*dy, minx+(c+1)*dx, miny+(r+1)*dy)
            cell_geom = geom.intersection(cell_box)
            if cell_geom.is_empty:
                continue
            area_ha = cell_geom.area * (111_000 ** 2) / 10_000
            cells.append({
                "row": r, "col": c, "cell_id": f"R{r}C{c}",
                "geometry": cell_geom, "area_ha": round(area_ha, 2),
            })
    return cells


def run_intra_parcellaire(pid, parcelles_session, tif_path, ndwi_arr, transform):
    """
    Calcule et affiche l'analyse intra-parcellaire pour une parcelle.
    Retourne (fig_heatmap, df_cells, eau_stats) ou lève une exception.
    """
    meta   = parcelles_session[pid]
    geom   = meta["geometry"]
    surf   = meta["surf_parc"]
    grid_n = 5 if surf >= 15 else 3 if surf >= 5 else 2

    cells  = build_grid(geom, grid_n)
    matrix = np.full((grid_n, grid_n), np.nan)
    areas  = np.zeros((grid_n, grid_n))

    for cell in cells:
        stats = zonal_stats_parcel(cell["geometry"], ndwi_arr, transform)
        if stats:
            matrix[grid_n - 1 - cell["row"], cell["col"]] = stats["mean"]
            areas[grid_n  - 1 - cell["row"], cell["col"]] = cell["area_ha"]

    # Heatmap matplotlib
    fig, ax = plt.subplots(figsize=(5, 5))
    norm    = mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=-0.15, vmax=0.2)
    im      = ax.imshow(matrix, cmap=plt.cm.RdYlBu, norm=norm, aspect="equal")
    plt.colorbar(im, ax=ax, label="NDWI", fraction=0.04)
    for r in range(grid_n):
        for c in range(grid_n):
            v = matrix[r, c]
            if not np.isnan(v):
                icon = "🔴" if v < -0.3 else "🟠" if v < -0.15 else "🟢"
                ax.text(c, r, f"{icon}\n{v:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if v < -0.2 else "black",
                        fontweight="bold")
    nom = meta.get("nom", pid)
    ax.set_title(
        f"{nom} ({meta['code_cultu']})\n"
        f"NDWI intra-parcellaire — grille {grid_n}×{grid_n}",
        fontsize=10,
    )
    ax.set_xticks(range(grid_n))
    ax.set_yticks(range(grid_n))
    plt.tight_layout()

    # Carte Folium intra
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
        r, c = cell["row"], cell["col"]
        val  = matrix[grid_n - 1 - r, c]
        if np.isnan(val):
            continue
        color = "#d9534f" if val < -0.3 else "#f0ad4e" if val < -0.15 else \
                "#ffe066" if val < 0 else "#5cb85c"
        label = "Critique" if val < -0.3 else "Modéré" if val < -0.15 else \
                "Léger" if val < 0 else "Normal"
        folium.GeoJson(
            cell["geometry"].__geo_interface__,
            style_function=lambda _, c2=color: {
                "fillColor": c2, "color": "white",
                "weight": 1, "fillOpacity": 0.65, "interactive": False,
            },
            tooltip=folium.Tooltip(
                f"{cell['cell_id']} | NDWI:{val:.3f} | {label} | {cell['area_ha']:.2f} ha"
            ),
            popup=folium.Popup(
                f"<b>{cell['cell_id']}</b><br>NDWI : {val:.3f}<br>"
                f"{'Irrigation recommandée' if val < -0.15 else 'OK'}",
                max_width=160,
            ),
        ).add_to(m_intra)

    # Économie d'eau
    stress_ha    = sum(
        cell["area_ha"] for cell in cells
        if not np.isnan(matrix[grid_n - 1 - cell["row"], cell["col"]])
        and matrix[grid_n - 1 - cell["row"], cell["col"]] < -0.15
    )
    total_ha     = sum(c["area_ha"] for c in cells)
    apport_mm    = 30
    eau_uniforme = total_ha  * apport_mm * 10
    eau_ciblee   = stress_ha * apport_mm * 10
    economie     = eau_uniforme - eau_ciblee
    pct          = economie / eau_uniforme * 100 if eau_uniforme > 0 else 0

    eau_stats = {
        "stress_ha": stress_ha, "total_ha": total_ha,
        "eau_uniforme": eau_uniforme, "eau_ciblee": eau_ciblee,
        "economie": economie, "pct": pct, "apport_mm": apport_mm,
    }

    # DataFrame cellules pour affichage
    df_cells = pd.DataFrame([
        {
            "Cellule"  : cell["cell_id"],
            "NDWI"     : round(matrix[grid_n - 1 - cell["row"], cell["col"]], 3)
                         if not np.isnan(matrix[grid_n - 1 - cell["row"], cell["col"]])
                         else None,
            "Surface ha": cell["area_ha"],
            "Statut"   : (
                "Critique" if not np.isnan(matrix[grid_n-1-cell["row"], cell["col"]])
                              and matrix[grid_n-1-cell["row"], cell["col"]] < -0.3
                else "Modéré"  if not np.isnan(matrix[grid_n-1-cell["row"], cell["col"]])
                              and matrix[grid_n-1-cell["row"], cell["col"]] < -0.15
                else "Normal"
            ),
        }
        for cell in cells
    ])

    return fig, m_intra, df_cells, eau_stats


# ════════════════════════════════════════════════════════════════════════════
# PAGE PRINCIPALE
# ════════════════════════════════════════════════════════════════════════════
st.title("Alertes stress hydrique")

parcelles_session = st.session_state.get("parcelles", {})

if not parcelles_session:
    st.warning("Aucune parcelle sélectionnée — revenez à la page 'Mes Parcelles'.")
    st.stop()

# ── Étape 1 : Scène satellite ────────────────────────────────────────────────
minx, miny, maxx, maxy = global_bbox(parcelles_session)
sh_bbox = BBox(bbox=[minx, miny, maxx, maxy], crs=CRS.WGS84)
st.markdown(f"Bbox : `{minx:.4f},{miny:.4f} → {maxx:.4f},{maxy:.4f}`")

if (minx - maxx) ** 2 > BBOX_LIMIT or (miny - maxy) ** 2 > BBOX_LIMIT:
    st.warning("Les parcelles sélectionnées sont trop éloignées pour une seule image satellite.")
    st.stop()

config  = get_sh_config()
end_d   = st.session_state["today"]
start_d = end_d - timedelta(days=30)

with st.spinner(f"Recherche scènes du {start_d} au {end_d}..."):
    scenes = search_available_scenes(sh_bbox, start_d, end_d,
                                     max_cloud_coverage=0.30, config=config)

if not scenes:
    st.warning("⚠️  Aucune scène disponible — couverture nuageuse trop importante.")
    st.stop()

latest = scenes[-1]
st.session_state["date_acq"] = latest["date"]
st.markdown(f"Scène retenue : **{latest['date']}** (nuages : {latest['cloud_coverage']:.1f}%)")

out_dir  = DATA_RAW / "interface_test"
out_dir.mkdir(parents=True, exist_ok=True)

with st.spinner("Téléchargement de la scène..."):
    tif_path = download_scene(
        bbox=sh_bbox, acquisition_date=st.session_state["date_acq"],
        output_dir=out_dir, config=config,
    )
st.session_state["tif_path"] = tif_path
st.success(f"TIF téléchargé : `{tif_path.name}`")

#################
# Il faut trouver le moyen d'intégrer la zone de vue dans le nom de l'image satellite parce que si on rajoute dans la même journée d'autres parcelles qui ne sont pas dans la zone l'image n'est pas retéléchargées et la nouvelle parcelle n'apparait pas. 
# Je pense qu'il faudrait aussi un bouton pour effacer les images téléchargées précedemment, parce que l'agriculteur il va pas se charger son ordi avec des millions de vues satellite. 
#################

# ── Étape 2 : Météo ──────────────────────────────────────────────────────────
date_acq    = st.session_state["date_acq"]
lat_c       = (miny + maxy) / 2
lon_c       = (minx + maxx) / 2
meteo_start = date_acq - timedelta(days=14)

with st.spinner("Récupération météo..."):
    meteo_df = fetch_historical_weather(lat_c, lon_c, meteo_start, date_acq)
meteo_df["date"] = pd.to_datetime(meteo_df["date"])
st.session_state["meteo_df"] = meteo_df
st.success(f"Météo : {len(meteo_df)} jours ({meteo_start} → {date_acq})")

# ── Étape 3 : Features + inférence ──────────────────────────────────────────
bands, meta  = load_bands(tif_path)
ndvi_arr     = compute_ndvi(bands)
ndwi_arr     = compute_ndwi(bands)
transform    = meta["transform"]
meteo_feat   = compute_meteo_features(meteo_df, date_acq)

rows = []
for pid, pmeta in parcelles_session.items():
    geom   = pmeta["geometry"]
    s_ndvi = zonal_stats_parcel(geom, ndvi_arr, transform)
    s_ndwi = zonal_stats_parcel(geom, ndwi_arr, transform)
    if s_ndvi is None or s_ndwi is None:
        st.warning(f"⚠️  Parcelle {pid[:12]} : trop peu de pixels valides")
        continue
    row = {
        "parcelle_id": pid,
        "source"     : pmeta["source"],
        "code_cultu" : pmeta["code_cultu"],
        "surf_parc"  : pmeta["surf_parc"],
        "date"       : date_acq,
        "ndvi_mean"  : s_ndvi["mean"], "ndvi_std": s_ndvi["std"],
        "ndvi_p10"   : s_ndvi["p10"],  "ndvi_p90": s_ndvi["p90"],
        "ndwi_mean"  : s_ndwi["mean"], "ndwi_std": s_ndwi["std"],
        "ndwi_p10"   : s_ndwi["p10"],  "ndwi_p90": s_ndwi["p90"],
        "n_pixels"   : s_ndvi["n"],
        "day_of_year": date_acq.timetuple().tm_yday,
        "ndwi_delta" : np.nan, "ndvi_delta"    : np.nan,
        "ndwi_deviation": 0.0, "ndvi_deviation": 0.0,
    }
    row.update(meteo_feat)
    rows.append(row)

if not rows:
    st.error("Aucune parcelle n'a pu être analysée.")
    st.stop()

df_feat = pd.DataFrame(rows)
st.session_state["df_features"] = df_feat

# Inférence
results = []
for _, row in df_feat.iterrows():
    score, is_anom, sev, model_name = score_row(row, FEATURE_COLS)
    results.append({
        "parcelle_id"  : row["parcelle_id"],
        "code_cultu"   : row["code_cultu"],
        "surf_parc"    : row["surf_parc"],
        "ndwi_mean"    : row["ndwi_mean"],
        "ndvi_mean"    : row["ndvi_mean"],
        "deficit_7d"   : row["deficit_7d"],
        "anomaly_score": score,
        "is_anomaly"   : is_anom,
        "severity"     : sev,
        "model_used"   : model_name,
    })

df_res = pd.DataFrame(results)
st.session_state["df_results"] = df_res
st.success("✅ Inférence terminée")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# SECTION RÉSULTATS
# ════════════════════════════════════════════════════════════════════════════
st.subheader("Résultats")

# ── Filtres ──────────────────────────────────────────────────────────────────
col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
with col_f1:
    cultures_dispo = sorted(df_res["code_cultu"].dropna().unique().tolist())
    sel_cultures   = st.multiselect(
        "Cultures", options=cultures_dispo, default=cultures_dispo,
        key="filter_cultures",
    )
with col_f2:
    sev_filter = st.selectbox(
        "Sévérité", ["Toutes", "Attention et critique", "Critique uniquement"],
        key="filter_sev",
    )
with col_f3:
    # Tri
    sort_by = st.selectbox(
        "Trier par", ["Score (pire en premier)", "Surface", "Culture"],
        key="filter_sort",
    )

# Application des filtres
df_filtered = df_res.copy()
if sel_cultures:
    df_filtered = df_filtered[df_filtered["code_cultu"].isin(sel_cultures)]
if sev_filter == "Attention et critique":
    df_filtered = df_filtered[df_filtered["severity"].isin(["warning", "critical"])]
elif sev_filter == "Critique uniquement":
    df_filtered = df_filtered[df_filtered["severity"] == "critical"]

if sort_by == "Score (pire en premier)":
    df_filtered = df_filtered.sort_values("anomaly_score")
elif sort_by == "Surface":
    df_filtered = df_filtered.sort_values("surf_parc", ascending=False)
elif sort_by == "Culture":
    df_filtered = df_filtered.sort_values("code_cultu")

# ── Métriques ─────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
n_tot  = len(df_filtered)
n_anom = int(df_filtered["is_anomaly"].sum()) if n_tot else 0
m1.metric("Parcelles", n_tot)
m2.metric("Anomalies", f"{n_anom} ({n_anom/n_tot*100:.0f}%)" if n_tot else "0")
m3.metric("Attention",  int((df_filtered["severity"] == "warning").sum()))
m4.metric("Critique",   int((df_filtered["severity"] == "critical").sum()))

st.divider()

# ── Carte + Tableau ───────────────────────────────────────────────────────────
col_map, col_tbl = st.columns([3, 2])

# Ensemble des IDs filtrés (pour n'afficher que les parcelles visibles)
filtered_ids = set(df_filtered["parcelle_id"].astype(str).tolist())

with col_map:
    date_str = date_acq.strftime("%d %B %Y")
    m_folium = folium.Map(
        location=st.session_state["map_center"],
        zoom_start=14,
    )

    title_html = (
        f"<div style='position:fixed;top:10px;left:50%;transform:translateX(-50%);"
        f"background:rgba(0,0,0,0.75);color:white;padding:8px 14px;"
        f"border-radius:6px;font-size:13px;z-index:9999;font-family:monospace;'>"
        f"Parcelle Watch — Stress hydrique — {date_str}</div>"
    )
    legend_html = (
        "<div style='position:fixed;bottom:20px;right:20px;"
        "background:rgba(0,0,0,0.8);color:white;padding:10px 14px;"
        "border-radius:6px;font-size:11px;z-index:9999;font-family:monospace;'>"
        "<b>Stress hydrique</b><br>"
        "<span style='color:#d9534f'>■</span> Critique<br>"
        "<span style='color:#f0ad4e'>■</span> Attention<br>"
        "<span style='color:#5cb85c'>■</span> Normal<br>"
        "<span style='color:#cccccc'>■</span> Filtré</div>"
    )
    m_folium.get_root().html.add_child(folium.Element(title_html))
    m_folium.get_root().html.add_child(folium.Element(legend_html))

    res_by_id = df_res.set_index("parcelle_id").to_dict("index")

    for pid, pmeta in parcelles_session.items():
        r      = res_by_id.get(pid, {})
        in_filter = pid in filtered_ids
        sev    = r.get("severity", "unknown") if in_filter else "filtered"
        color  = sev_color(sev) if in_filter else "#cccccc"
        ndwi_s = f"{r.get('ndwi_mean', 0):.3f}" if r else "N/A"
        score_s = f"{r.get('anomaly_score', 0):.3f}" if r else "N/A"
        opacity = 0.65 if in_filter else 0.2

        popup_html = (
            f"<div style='font-family:monospace;font-size:12px;min-width:170px'>"
            f"<b>{pid[:12]}</b><br>"
            f"Culture : <b>{pmeta['code_cultu']}</b><br>"
            f"Surface : {pmeta['surf_parc']:.1f} ha<br>"
            f"<hr style='margin:3px 0'>"
            f"NDWI : {ndwi_s}<br>"
            f"Score : {score_s}<br>"
            f"Sévérité : <b style='color:{color}'>{sev.upper()}</b>"
            f"{'<br><i>(filtré)</i>' if not in_filter else ''}"
            f"</div>"
        )
        folium.GeoJson(
            pmeta["geometry"].__geo_interface__,
            style_function=lambda _, c=color, o=opacity: {
                "fillColor": c, "color": "white",
                "weight": 1.5, "fillOpacity": o,
            },
            tooltip=folium.Tooltip(f"{pmeta['code_cultu']} — {sev.upper()}"),
            popup=folium.Popup(popup_html, max_width=200),
        ).add_to(m_folium)

    st_folium(m_folium, width=None, height=500, key="map_res")

with col_tbl:
    st.subheader("Tableau des alertes")

    # Tableau cliquable — sélection pour intra-parcellaire
    df_display = df_filtered[[
        "parcelle_id", "code_cultu", "surf_parc",
        "ndwi_mean", "anomaly_score", "severity",
    ]].rename(columns={
        "parcelle_id"  : "Parcelle",
        "code_cultu"   : "Culture",
        "surf_parc"    : "Ha",
        "ndwi_mean"    : "NDWI",
        "anomaly_score": "Score",
        "severity"     : "Sévérité",
    }).copy()
    for c in ["NDWI", "Score"]:
        if c in df_display.columns:
            df_display[c] = df_display[c].round(3)

    # Raccourcir l'ID pour l'affichage
    df_display["Parcelle"] = df_display["Parcelle"].str[:12]

    if df_display.empty:
        st.info("Aucune parcelle pour ces filtres.")
    else:
        # st.dataframe avec selection
        event = st.dataframe(
            df_display,
            use_container_width=True,
            height=420,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="tbl_alertes",
        )

        selected_rows = event.selection.get("rows", []) if event and event.selection else []
        if selected_rows:
            row_idx = selected_rows[0]
            # Récupérer le pid complet depuis df_filtered (même ordre)
            full_pid = df_filtered.iloc[row_idx]["parcelle_id"]
            if full_pid != st.session_state["intra_pid"]:
                st.session_state["intra_pid"] = full_pid

        if st.session_state["intra_pid"] is not None:
            st.caption(
                f"↓ Analyse intra-parcellaire : `{str(st.session_state['intra_pid'])[:12]}`"
            )

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# SECTION INTRA-PARCELLAIRE
# ════════════════════════════════════════════════════════════════════════════
intra_pid = st.session_state.get("intra_pid")

if intra_pid is None:
    st.info("👆 Cliquez sur une ligne du tableau pour afficher l'analyse intra-parcellaire.")
else:
    pmeta = parcelles_session.get(intra_pid)
    if pmeta is None:
        st.warning("Parcelle introuvable en session.")
    else:
        nom  = pmeta.get("nom", intra_pid[:12])
        surf = pmeta["surf_parc"]
        grid_n = 5 if surf >= 15 else 3 if surf >= 5 else 2

        st.subheader(f"Analyse intra-parcellaire — {nom} ({pmeta['code_cultu']}) — grille {grid_n}×{grid_n}")

        col_heat, col_carte_intra = st.columns([1, 2])

        with st.spinner("Calcul de la grille intra-parcellaire..."):
            try:
                fig, m_intra, df_cells, eau = run_intra_parcellaire(
                    intra_pid, parcelles_session, tif_path, ndwi_arr, transform
                )

                with col_heat:
                    st.pyplot(fig)
                    plt.close(fig)

                with col_carte_intra:
                    st_folium(m_intra, width=None, height=400,
                               key=f"map_intra_{intra_pid}")

                # Tableau cellules + économie d'eau
                col_cells, col_eau = st.columns([2, 1])

                with col_cells:
                    st.markdown("**Détail par cellule**")
                    st.dataframe(df_cells, use_container_width=True,
                                  hide_index=True, height=200)

                with col_eau:
                    st.markdown("**Économie d'eau estimée**")
                    st.markdown(
                        f"Hypothèse : **{eau['apport_mm']} mm/ha**"
                    )
                    st.metric(
                        "Surface en stress (NDWI < -0.15)",
                        f"{eau['stress_ha']:.1f} ha / {eau['total_ha']:.1f} ha",
                    )
                    st.metric("Irrigation uniforme",  f"{eau['eau_uniforme']:.0f} m³")
                    st.metric("Irrigation ciblée",    f"{eau['eau_ciblee']:.0f} m³")
                    delta_col = "normal" if eau["economie"] >= 0 else "inverse"
                    st.metric(
                        "Économie potentielle",
                        f"{eau['economie']:.0f} m³",
                        delta=f"{eau['pct']:.0f}%",
                        delta_color=delta_col,
                    )

            except Exception as e:
                st.error(f"Erreur analyse intra-parcellaire : {e}")

        if st.button("Fermer l'analyse intra-parcellaire"):
            st.session_state["intra_pid"] = None
            st.rerun()
##############
# Pour l'analyse intraparcellaire, on n'analyse que la moitié basse de la parcelle, bizarre. 
# Et pour fermer l'analyse intraparcellaire, on est obligé de déselectionner la ligne dans le tableau d'abord, c'est dommage. 
##############
