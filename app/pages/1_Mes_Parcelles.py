import sys
import json
import uuid
import math
from pathlib import Path

import folium
from folium.plugins import Draw
import geopandas as gpd
import requests
import streamlit as st
from shapely.geometry import Point, Polygon, shape, mapping
from shapely import wkt
from streamlit_folium import st_folium

from app.components.culture_selector import culture_selector
 

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(
    page_title="Mes Parcelles - Parcelle Watch",
    page_icon="🗺️",
    layout="wide",
)

ROOT            = Path(__file__).parent.parent.parent
PARCELLES_FILE  = ROOT / "data" / "user" / "parcelles.json"
CULTURES_CSV = ROOT / "data" / "external" / "CULTURE.csv"

DEFAULT_LAT  = 48.69
DEFAULT_LON  = 2.62
DEFAULT_ZOOM = 13

CODE_CULTU_LABELS = {
    "BTH": "Ble tendre hiver",   "BTN": "Ble tendre printemps",
    "BTP": "Ble tendre pollen",  "BTA": "Ble tendre autre",
    "ORH": "Orge hiver",         "ORP": "Orge printemps",
    "MIS": "Mais grain",         "CZH": "Colza hiver",
    "JAC": "Jachere",            "PPH": "Prairie permanente",
    "BOR": "Ble orge",           "FVL": "Feverole",
    "LEC": "Lentille",           "BFS": "Betterave",
    "SNE": "Surface non exploitee",
}

# ── Persistance JSON ─────────────────────────────────────────────────────────

def _geom_to_wkt(geom):
    """Shapely geometry → WKT string pour sérialisation JSON."""
    return geom.wkt if geom is not None else None

def _wkt_to_geom(wkt_str):
    """WKT string → Shapely geometry."""
    try:
        return wkt.loads(wkt_str) if wkt_str else None
    except Exception:
        return None

def save_parcelles(parcelles: dict):
    """Sérialise les parcelles dans data/user/parcelles.json."""
    PARCELLES_FILE.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for pid, meta in parcelles.items():
        entry = {k: v for k, v in meta.items() if k != "geometry"}
        # bbox : tuple → list pour JSON
        if "bbox" in entry and entry["bbox"] is not None:
            entry["bbox"] = list(entry["bbox"])
        entry["geometry_wkt"] = _geom_to_wkt(meta.get("geometry"))
        serializable[pid] = entry
    with open(PARCELLES_FILE, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)

def load_parcelles() -> dict:
    """Charge les parcelles depuis data/user/parcelles.json (si existant)."""
    if not PARCELLES_FILE.exists():
        return {}
    try:
        with open(PARCELLES_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        parcelles = {}
        for pid, entry in raw.items():
            geom = _wkt_to_geom(entry.pop("geometry_wkt", None))
            if "bbox" in entry and entry["bbox"] is not None:
                entry["bbox"] = tuple(entry["bbox"])
            entry["geometry"] = geom
            parcelles[pid] = entry
        return parcelles
    except Exception as e:
        st.warning(f"Impossible de charger les parcelles sauvegardées : {e}")
        return {}

def clear_parcelles_file():
    """Supprime le fichier de persistance."""
    if PARCELLES_FILE.exists():
        PARCELLES_FILE.unlink()

# ── Session state (chargement initial depuis fichier) ────────────────────────

if "parcelles_loaded" not in st.session_state:
    # Premier chargement : on tente de lire le fichier
    st.session_state["parcelles"]       = load_parcelles()
    st.session_state["parcelles_loaded"] = True

if "map_center" not in st.session_state:
    st.session_state["map_center"] = [DEFAULT_LAT, DEFAULT_LON]
if "map_zoom" not in st.session_state:
    st.session_state["map_zoom"] = DEFAULT_ZOOM
if "editing_pid" not in st.session_state:
    st.session_state["editing_pid"] = None
# Clé de la carte dessin — incrémentée pour forcer le re-rendu
if "draw_map_version" not in st.session_state:
    st.session_state["draw_map_version"] = 0


# ── Helpers géo ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def geocode_address(query):
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": "ParcelleWatch/1.0"},
            timeout=5,
        )
        results = resp.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception:
        pass
    return None


def parse_gps(text):
    try:
        parts = text.replace(";", ",").split(",")
        if len(parts) == 2:
            lat, lon = float(parts[0].strip()), float(parts[1].strip())
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
    except Exception:
        pass
    return None


def check_rpg_availability(rpg_year, today):
    current_year  = today.year
    current_month = today.month
    if rpg_year < 2021:
        return False, "RPG disponible uniquement depuis 2021 via WFS IGN"
    if rpg_year > current_year:
        return False, f"RPG {rpg_year} pas encore disponible"
    if rpg_year == current_year and current_month < 6:
        return True, f"RPG {rpg_year} partiellement disponible (publication vers juin {rpg_year+1})"
    return True, f"RPG {rpg_year} disponible"


def load_rpg_around_center(center, zoom, rpg_year=2024):
    delta    = max(0.02, 0.5 / (2 ** (zoom - 10)))
    lat_min, lat_max = center[0] - delta, center[0] + delta
    lon_min, lon_max = center[1] - delta * 1.5, center[1] + delta * 1.5
    wfs_url = (
        f"https://data.geopf.fr/wfs/ows?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
        f"&TYPENAMES=RPG.{rpg_year}:parcelles_graphiques"
        f"&BBOX={lon_min},{lat_min},{lon_max},{lat_max},EPSG:4326"
        f"&OUTPUTFORMAT=application/json"
    )
    try:
        gdf = gpd.read_file(wfs_url)
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        gdf = gdf[gdf["code_cultu"] != "SNE"].copy()
        gdf["id_parcel"] = gdf["id_parcel"].astype(str)
        return gdf, (lon_min, lat_min, lon_max, lat_max)
    except Exception as e:
        st.warning(f"Erreur chargement RPG : {e}")
        return gpd.GeoDataFrame(), None


def find_clicked_parcelle(click_lat, click_lon, gdf):
    if gdf.empty:
        return None
    point = Point(click_lon, click_lat)
    try:
        candidates_idx = list(gdf.sindex.intersection(point.bounds))
        for _, row in gdf.iloc[candidates_idx].iterrows():
            if row.geometry is not None:
                try:
                    if row.geometry.contains(point):
                        return row
                except Exception:
                    continue
    except Exception:
        pass
    return None


def parse_gps_polygon(text):
    points = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        result = parse_gps(line)
        if result is None:
            return None, f"Ligne invalide : '{line}' — format attendu : lat, lon"
        lat, lon = result
        points.append((lon, lat))
    if len(points) < 3:
        return None, "Il faut au moins 3 points pour définir une parcelle."
    if points[0] != points[-1]:
        points.append(points[0])
    try:
        poly = Polygon(points)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly, None
    except Exception as e:
        return None, str(e)


def surface_from_polygon(poly):
    centroid_lat = poly.centroid.y
    lat_m = 111_000
    lon_m = 111_000 * math.cos(math.radians(centroid_lat))
    area_m2 = 0.0
    coords = list(poly.exterior.coords)
    n = len(coords)
    for i in range(n - 1):
        x1 = coords[i][0]   * lon_m;  y1 = coords[i][1]   * lat_m
        x2 = coords[i+1][0] * lon_m;  y2 = coords[i+1][1] * lat_m
        area_m2 += (x1 * y2 - x2 * y1)
    return abs(area_m2) / 2 / 10_000


def build_map(center, zoom, gdf=None, selected_ids=None,
              rpg_enabled=False, draw_enabled=False):
    if selected_ids is None:
        selected_ids = set()
    m = folium.Map(location=center, zoom_start=zoom, tiles="Esri WorldImagery")

    if rpg_enabled and gdf is not None and not gdf.empty:
        for _, row in gdf.iterrows():
            pid    = str(row["id_parcel"])
            sel    = pid in selected_ids
            color  = "#3d9bf0" if sel else "#ffffff"
            opac   = 0.65     if sel else 0.15
            weight = 2.5      if sel else 1.0
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda _, c=color, o=opac, w=weight: {
                    "fillColor": c, "color": c, "weight": w, "fillOpacity": o,
                },
                tooltip=folium.Tooltip(
                    f"{row['code_cultu']} | {row.get('surf_parc', 0):.1f} ha | id:{pid}"
                ),
            ).add_to(m)

    if draw_enabled:
        Draw(
            export=False,
            draw_options={
                "polygon":      {"allowIntersection": False},
                "rectangle":    True,
                "circle":       False,
                "marker":       False,
                "polyline":     False,
                "circlemarker": False,
            },
            edit_options={"edit": True, "remove": True},
        ).add_to(m)

    # Parcelles dessin/GPS déjà enregistrées
    for pid, meta in st.session_state["parcelles"].items():
        if meta.get("source") in ("dessin", "gps"):
            geom = meta.get("geometry")
            if geom is not None:
                folium.GeoJson(
                    geom.__geo_interface__,
                    style_function=lambda _: {
                        "fillColor": "#f0ad4e", "color": "#f0ad4e",
                        "weight": 2.5, "fillOpacity": 0.50,
                    },
                    tooltip=folium.Tooltip(
                        f"[Perso] {meta.get('nom', '?')} — {meta.get('surf_parc', 0):.1f} ha"
                    ),
                ).add_to(m)
    return m


# ════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ════════════════════════════════════════════════════════════════════════════
st.title("Mes Parcelles")
st.caption("Sélectionnez, dessinez ou saisissez vos parcelles à surveiller.")

tab_clic, tab_dessin, tab_gps = st.tabs([
    "Clic sur RPG", "Dessiner sur la carte", "Coordonnées GPS"
])

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Localisation")
    with st.form("search_form"):
        query     = st.text_input("Adresse ou commune",
                                   placeholder="Ex: Melun, Seine-et-Marne")
        gps_input = st.text_input("Ou coordonnées GPS", placeholder="48.682, 2.581")
        submitted = st.form_submit_button("Centrer la carte", type="primary")

    if submitted:
        coords = None
        if gps_input.strip():
            coords = parse_gps(gps_input)
            if coords is None:
                st.error("Format GPS invalide. Utiliser : lat, lon")
        elif query.strip():
            with st.spinner("Géocodage..."):
                coords = geocode_address(query)
            if coords is None:
                st.error(f"Adresse introuvable : {query}")
        if coords is not None:
            st.session_state["map_center"] = list(coords)
            st.session_state["map_zoom"]   = 14
            st.rerun()

    st.divider()
    n_sel = len(st.session_state["parcelles"])
    st.subheader(f"Mes parcelles ({n_sel})")

    if not st.session_state["parcelles"]:
        st.info("Aucune parcelle.\nUtilisez la carte ou les onglets.")
    else:
        to_delete   = []
        editing_pid = st.session_state["editing_pid"]

        for pid, meta in st.session_state["parcelles"].items():
            source  = meta.get("source", "rpg")
            culture = meta.get("code_cultu", "?")
            label   = CODE_CULTU_LABELS.get(culture, culture)
            if source in ("dessin", "gps"):
                label = meta.get("nom", "Parcelle perso")
            surface = meta.get("surf_parc", 0)
            badge   = "🟦" if source == "rpg" else "🟧"

            if editing_pid != pid:
                col_i, col_e, col_d = st.columns([3, 1, 1])
                with col_i:
                    st.markdown(f"{badge} **{label}**  \n`{pid[:12]}` — {surface:.1f} ha")
                with col_e:
                    if st.button("✏️", key=f"edit_{pid}", help="Modifier la culture"):
                        st.session_state["editing_pid"] = pid
                        st.rerun()
                with col_d:
                    if st.button("✕", key=f"del_{pid}", help="Retirer"):
                        to_delete.append(pid)
            else:
                current_culture = meta.get("code_cultu", "BTH")
                current_idx = list(CODE_CULTU_LABELS.keys()).index(current_culture) \
                              if current_culture in CODE_CULTU_LABELS else 0
                st.markdown(f"{badge} **{label}**  \n`{pid[:12]}` — {surface:.1f} ha")
                new_culture = culture_selector(
                                key=f"edit_{pid}",           # clé unique par parcelle
                                csv_path=CULTURES_CSV,
                                default=meta.get("code_cultu", "BTH"),   
                                label="Culture",
                            )

                col_ok, col_cancel = st.columns(2)
                with col_ok:
                    if st.button("✓ OK", key=f"confirm_{pid}", type="primary"):
                        st.session_state["parcelles"][pid]["code_cultu"] = new_culture
                        st.session_state["editing_pid"] = None
                        save_parcelles(st.session_state["parcelles"])
                        st.toast(f"Culture mise à jour : {new_culture}")
                        st.rerun()
                with col_cancel:
                    if st.button("Annuler", key=f"cancel_{pid}"):
                        st.session_state["editing_pid"] = None
                        st.rerun()
            st.divider()

        for pid in to_delete:
            del st.session_state["parcelles"][pid]
        if to_delete:
            st.session_state["editing_pid"] = None
            save_parcelles(st.session_state["parcelles"])
            st.rerun()

        total_ha = sum(m.get("surf_parc", 0)
                       for m in st.session_state["parcelles"].values())
        st.metric("Surface totale", f"{total_ha:.1f} ha")
        st.caption("🟦 RPG  🟧 Personnalisée")
        st.divider()
        if st.button("Lancer l'analyse satellite",
                     type="primary", use_container_width=True):
            # Sauvegarde avant de quitter la page
            save_parcelles(st.session_state["parcelles"])
            st.switch_page("pages/2_Alertes.py")

    # ── Bouton reset parcelles sauvegardées ───────────────────────────────
    if PARCELLES_FILE.exists():
        st.divider()
        if st.button("🗑️ Effacer les parcelles sauvegardées",
                     use_container_width=True, help="Supprime le fichier de sauvegarde"):
            clear_parcelles_file()
            st.session_state["parcelles"] = {}
            st.session_state["editing_pid"] = None
            st.toast("Sauvegarde effacée")
            st.rerun()

selected_ids = set(st.session_state["parcelles"].keys())


# ════════════════════════════════════════════════════════════════════════════
# ONGLET 1 — Clic RPG
# ════════════════════════════════════════════════════════════════════════════
with tab_clic:
    st.markdown("Naviguez sur la carte et cliquez sur une parcelle pour la sélectionner.")

    rpg_year = st.number_input("Année de référence RPG", value=2024,
                                min_value=2021, max_value=2030, key="rpg_year")
    import datetime
    today = st.session_state.get("today", datetime.date.today())
    ok, msg = check_rpg_availability(int(rpg_year), today)

    if ok:
        gdf_rpg, bbox_extent = load_rpg_around_center(
            st.session_state["map_center"],
            st.session_state["map_zoom"],
            rpg_year=int(rpg_year),
        )
        st.markdown(f"RPG {int(rpg_year)} : {msg} — {len(gdf_rpg)} parcelles chargées")
    else:
        st.warning(f"⚠️  {msg}")
        gdf_rpg     = gpd.GeoDataFrame()
        bbox_extent = None

    if gdf_rpg.empty:
        st.warning("Aucune parcelle RPG chargée. Centrez la carte sur une zone agricole.")
    else:
        map_key_clic = f"map_clic_{len(selected_ids)}"
        m_rpg = build_map(
            center=st.session_state["map_center"],
            zoom=st.session_state["map_zoom"],
            gdf=gdf_rpg,
            selected_ids=selected_ids,
            rpg_enabled=True,
        )
        map_data = st_folium(
            m_rpg,
            width=None, height=520,
            returned_objects=["last_object_clicked", "zoom", "center"],
            key=map_key_clic,
        )

        col_a, col_b = st.columns(2)
        col_a.metric("Parcelles RPG visibles", len(gdf_rpg))
        col_b.metric("Sélectionnées", len(selected_ids))

        if map_data:
            if map_data.get("center"):
                st.session_state["map_center"] = [
                    map_data["center"]["lat"], map_data["center"]["lng"],
                ]
            if map_data.get("zoom"):
                st.session_state["map_zoom"] = map_data["zoom"]

            if map_data.get("last_object_clicked") and not gdf_rpg.empty:
                click_lat = map_data["last_object_clicked"]["lat"]
                click_lon = map_data["last_object_clicked"]["lng"]
                parcelle_row = find_clicked_parcelle(click_lat, click_lon, gdf_rpg)

                if parcelle_row is not None:
                    pid = str(parcelle_row.get("id_parcel", ""))
                    if pid in st.session_state["parcelles"]:
                        del st.session_state["parcelles"][pid]
                        st.toast(f"Parcelle {pid[:12]} retirée")
                    else:
                        culture = parcelle_row.get("code_cultu", "?")
                        surf    = parcelle_row.get("surf_parc", 0)
                        geom    = parcelle_row.get("geometry")
                        min_lon, min_lat, max_lon, max_lat = geom.bounds
                        st.session_state["parcelles"][pid] = {
                            "source"    : "rpg",
                            "code_cultu": culture,
                            "surf_parc" : surf,
                            "bbox"      : (min_lon, min_lat, max_lon, max_lat),
                            "geometry"  : geom,
                            "lat"       : click_lat,
                            "lon"       : click_lon,
                        }
                        st.toast(f"{CODE_CULTU_LABELS.get(culture, culture)} ({surf:.1f} ha) ajoutée")
                    save_parcelles(st.session_state["parcelles"])
                    st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# ONGLET 2 — Dessin
# ════════════════════════════════════════════════════════════════════════════
with tab_dessin:
    st.markdown(
        "Utilisez les outils de dessin pour tracer votre parcelle directement "
        "sur la carte, puis donnez-lui un nom."
    )

    col_carte, col_form = st.columns([3, 1])

    with col_carte:
        map_key_draw = f"map_draw_{st.session_state['draw_map_version']}_{len(selected_ids)}"
        m_draw = build_map(
            center=st.session_state["map_center"],
            zoom=st.session_state["map_zoom"],
            draw_enabled=True,
        )
        draw_data = st_folium(
            m_draw,
            width=None, height=480,
            returned_objects=["last_active_drawing", "center", "zoom"],
            key=map_key_draw,
        )

    with col_form:
        st.markdown("**Nommer la parcelle**")
        nom_dessin     = st.text_input("Nom", placeholder="Ex: Champ du nord",
                                        key="nom_dessin")
        culture_dessin = culture_selector(
                            key="dessin",
                            csv_path=CULTURES_CSV,
                            default="BTH",
                            label="Culture",
                        )

        if draw_data and draw_data.get("center"):
            st.session_state["map_center"] = [
                draw_data["center"]["lat"], draw_data["center"]["lng"],
            ]
        if draw_data and draw_data.get("zoom"):
            st.session_state["map_zoom"] = draw_data["zoom"]

        drawing     = None
        surf_dessin = 0.0
        if draw_data and draw_data.get("last_active_drawing"):
            drawing = draw_data["last_active_drawing"]
            geom    = drawing.get("geometry", {})
            st.success(f"Forme : {geom.get('type', '?')}")
            try:
                poly = shape(geom)
                if poly.geom_type == "MultiPolygon":
                    poly = list(poly.geoms)[0]
                surf_dessin = surface_from_polygon(poly)
                st.metric("Surface", f"{surf_dessin:.2f} ha")
            except Exception:
                st.warning("Surface non calculable.")

        st.divider()
        can_add = drawing is not None and nom_dessin.strip()
        if st.button("Ajouter", type="primary",
                      disabled=not can_add, use_container_width=True):
            pid = f"dessin_{uuid.uuid4().hex[:8]}"
            try:
                poly     = shape(drawing["geometry"])
                centroid = poly.centroid
                min_lon, min_lat, max_lon, max_lat = poly.bounds
                st.session_state["parcelles"][pid] = {
                    "source"    : "dessin",
                    "nom"       : nom_dessin.strip(),
                    "code_cultu": culture_dessin,
                    "surf_parc" : surf_dessin,
                    "bbox"      : (min_lon, min_lat, max_lon, max_lat),
                    "geometry"  : poly,
                    "lat"       : centroid.y,
                    "lon"       : centroid.x,
                }
                save_parcelles(st.session_state["parcelles"])
                st.toast(f"'{nom_dessin}' ajoutée ({surf_dessin:.2f} ha)")
                # Incrémenter la version force le re-rendu de la carte au rerun
                st.session_state["draw_map_version"] += 1
                st.rerun()
            except Exception as e:
                st.error(f"Erreur : {e}")

        if not can_add:
            st.caption("Dessinez une parcelle puis nommez-la.")


# ════════════════════════════════════════════════════════════════════════════
# ONGLET 3 — GPS
# ════════════════════════════════════════════════════════════════════════════
with tab_gps:
    st.markdown(
        "Saisissez les coordonnées GPS des sommets de votre parcelle, "
        "une paire `latitude, longitude` par ligne."
    )

    col_form_gps, col_apercu = st.columns([1, 2])

    with col_form_gps:
        nom_gps     = st.text_input("Nom de la parcelle",
                                     placeholder="Ex: Vigne sud", key="nom_gps")
        culture_gps = culture_selector(
                                key="gps",
                                csv_path=CULTURES_CSV,
                                default="BTH",
                                label="Culture",
                            )

        coords_text = st.text_area(
            "Points GPS (un par ligne)",
            placeholder="48.683, 2.571\n48.685, 2.575\n48.682, 2.578\n48.680, 2.573",
            height=180,
            key="coords_gps",
        )

        poly_gps = None
        surf_gps = 0.0
        if coords_text.strip():
            poly_gps, err = parse_gps_polygon(coords_text)
            if err:
                st.error(f"Erreur : {err}")
            else:
                surf_gps = surface_from_polygon(poly_gps)
                n_pts    = len(list(poly_gps.exterior.coords)) - 1
                st.success(f"Polygone valide — {n_pts} points")
                st.metric("Surface estimée", f"{surf_gps:.2f} ha")

        can_add_gps = poly_gps is not None and nom_gps.strip()
        if st.button("Ajouter cette parcelle", type="primary",
                      disabled=not can_add_gps, use_container_width=True,
                      key="btn_add_gps"):
            pid      = f"gps_{uuid.uuid4().hex[:8]}"
            centroid = poly_gps.centroid
            min_lon, min_lat, max_lon, max_lat = poly_gps.bounds
            st.session_state["parcelles"][pid] = {
                "source"    : "gps",
                "nom"       : nom_gps.strip(),
                "code_cultu": culture_gps,
                "surf_parc" : surf_gps,
                "bbox"      : (min_lon, min_lat, max_lon, max_lat),
                "geometry"  : poly_gps,
                "lat"       : centroid.y,
                "lon"       : centroid.x,
            }
            st.session_state["map_center"] = [centroid.y, centroid.x]
            st.session_state["map_zoom"]   = 15
            save_parcelles(st.session_state["parcelles"])
            st.toast(f"'{nom_gps}' ajoutée ({surf_gps:.2f} ha)")
            st.rerun()

        with st.expander("Comment obtenir des coordonnées GPS ?"):
            st.markdown(
                "**Google Maps** : clic droit sur votre parcelle → coordonnées en haut du menu.\n\n"
                "**Géoportail** : clic droit → 'Créer un point'.\n\n"
                "**Smartphone** : épinglez un point, les détails affichent les coordonnées."
            )

    with col_apercu:
        st.markdown("**Aperçu de la parcelle**")
        lat_ap = st.session_state["map_center"][0]
        lon_ap = st.session_state["map_center"][1]
        if poly_gps is not None:
            centroid = poly_gps.centroid
            lat_ap, lon_ap = centroid.y, centroid.x

        m_gps = folium.Map(location=[lat_ap, lon_ap], zoom_start=15,
                            tiles="Esri WorldImagery")

        if poly_gps is not None:
            folium.GeoJson(
                poly_gps.__geo_interface__,
                style_function=lambda _: {
                    "fillColor": "#f0ad4e", "color": "#f0ad4e",
                    "weight": 2.5, "fillOpacity": 0.50,
                },
                tooltip=folium.Tooltip(f"{nom_gps or 'Parcelle'} — {surf_gps:.2f} ha"),
            ).add_to(m_gps)
            for i, (lon_p, lat_p) in enumerate(list(poly_gps.exterior.coords)[:-1]):
                folium.CircleMarker(
                    location=[lat_p, lon_p], radius=5,
                    color="#f0ad4e", fill=True,
                    fill_color="#ffffff", fill_opacity=0.9,
                    tooltip=f"Point {i+1} : {lat_p:.5f}, {lon_p:.5f}",
                ).add_to(m_gps)

        for pid, meta in st.session_state["parcelles"].items():
            if meta.get("source") == "gps":
                geom = meta.get("geometry")
                if geom is not None:
                    folium.GeoJson(
                        geom.__geo_interface__,
                        style_function=lambda _: {
                            "fillColor": "#f0ad4e", "color": "#f0ad4e",
                            "weight": 1.5, "fillOpacity": 0.30,
                        },
                        tooltip=folium.Tooltip(meta.get("nom", "?")),
                    ).add_to(m_gps)

        st_folium(m_gps, width=None, height=400, key="map_gps_preview")
