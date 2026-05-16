"""
app/pages/1_Mes_Parcelles.py
----------------------------
Page de selection et gestion des parcelles surveillees.

Trois modes de selection :
  1. Clic sur une parcelle RPG existante
  2. Dessin libre sur la carte (polygone ou rectangle via folium Draw)
  3. Saisie manuelle de coordonnees GPS

Optimisations :
  - sindex (R-tree) pour la recherche spatiale au clic
  - cache TTL=300s sur le WFS RPG
  - map_key dynamique pour forcer le re-rendu apres selection
  - "interactive": False sur les GeoJson pour eviter le vol de clic

Session state :
  st.session_state["parcelles"] : dict {id -> metadata}
"""

import sys
import uuid
import math
from pathlib import Path

import folium
from folium.plugins import Draw
import geopandas as gpd
import requests
import streamlit as st
from shapely.geometry import Point, Polygon, shape
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(
    page_title="Mes Parcelles - Parcelle Watch",
    page_icon="🗺️",
    layout="wide",
)

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

if "parcelles"  not in st.session_state:
    st.session_state["parcelles"]  = {} #liste de dict
if "map_center" not in st.session_state:
    st.session_state["map_center"] = [DEFAULT_LAT, DEFAULT_LON]
if "map_zoom"   not in st.session_state:
    st.session_state["map_zoom"]   = DEFAULT_ZOOM


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
    """
    Vérifie si on a des données satellite pour une année RPG donnée.
    Le RPG N est disponible en données ouvertes environ en juin N+1.
    Les données satellite sont disponibles dès l'année N.
    """
    current_year = today.year
    current_month = today.month

    if rpg_year < 2021:
        return False, 'RPG disponible uniquement depuis 2021 via WFS IGN'
    if rpg_year > current_year:
        return False, f'RPG {rpg_year} pas encore disponible'
    if rpg_year == current_year and current_month < 6:
        return True, f'RPG {rpg_year} partiellement disponible (publication vers juin {rpg_year+1})'
    return True, f'RPG {rpg_year} disponible — coordonnées parcelles RPG {rpg_year} accessibles'


def load_rpg_around_center(center, zoom, rpg_year=2024):
    """
    Charge les parcelles RPG via WFS IGN autour d'un point central.   
    #center = list[latitude, longitude], soit st.session_state["map_center"]
    #zoom   = int, soit st.session_state["map_zoom"]
    """
    delta  = max(0.02, 0.5 / (2 ** (zoom - 10)))
    lat_min, lat_max = center[0] - delta, center[0] + delta
    lon_min, lon_max = center[1] - delta * 1.5, center[1] + delta * 1.5

    wfs_url = (
        f'https://data.geopf.fr/wfs/ows?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature'
        f'&TYPENAMES=RPG.{rpg_year}:parcelles_graphiques'
        f'&BBOX={lon_min},{lat_min},{lon_max},{lat_max},EPSG:4326'
        f'&OUTPUTFORMAT=application/json'
    )
    try:
        gdf = gpd.read_file(wfs_url)
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        gdf = gdf[gdf['code_cultu'] != 'SNE'].copy()
        gdf['id_parcel'] = gdf['id_parcel'].astype(str)
        return gdf, (lon_min, lat_min, lon_max, lat_max)
    except Exception as e:
        st.warning(f'Erreur chargement RPG : {e}')
        return gpd.GeoDataFrame(), None


def find_clicked_parcelle(click_lat, click_lon, gdf):
    """Recherche spatiale rapide via sindex (R-tree)."""
    if gdf.empty:
        return None
    point = Point(click_lon, click_lat)
    try:
        candidates_idx = list(gdf.sindex.intersection(point.bounds))
        candidates     = gdf.iloc[candidates_idx]
        for _, row in candidates.iterrows():
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
    """
    Parse une liste de points GPS (une paire lat,lon par ligne).
    Retourne (Polygon, None) ou (None, message_erreur).
    """
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
    """Estime la surface en ha depuis un polygone WGS84."""
    centroid_lat = poly.centroid.y
    lat_m = 111_000
    lon_m = 111_000 * math.cos(math.radians(centroid_lat))
    area_m2 = 0.0
    coords = list(poly.exterior.coords)
    n = len(coords)
    for i in range(n - 1):
        x1 = coords[i][0]   * lon_m
        y1 = coords[i][1]   * lat_m
        x2 = coords[i+1][0] * lon_m
        y2 = coords[i+1][1] * lat_m
        area_m2 += (x1 * y2 - x2 * y1)
    return abs(area_m2) / 2 / 10_000


def build_map(center, zoom, gdf, selected_ids=None, rpg_enabled=False, draw_enabled=False):
    """Carte Folium des parcelles RPG, parcelles sélectionnées en bleu.
    dessin ou construction des parcelles avec coordonnées GPS"""
    if selected_ids is None:
        selected_ids = set()

    m = folium.Map(location=center, zoom_start=zoom, tiles='Esri WorldImagery')

    if rpg_enabled:
        for _, row in gdf.iterrows():
            pid    = str(row['id_parcel'])
            sel    = pid in selected_ids
            color  = '#3d9bf0' if sel else '#ffffff'
            opac   = 0.65 if sel else 0.15
            weight = 2.5  if sel else 1.0
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda _, c=color, o=opac, w=weight: {
                    'fillColor': c, 'color': c,
                    'weight': w, 'fillOpacity': o,
                    'interactive': True,
                },
                tooltip=folium.Tooltip(
                    f"{row['code_cultu']} | {row.get('surf_parc',0):.1f} ha | id:{pid}"
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

    # Afficher les parcelles personnalisées (dessin/GPS) déja en session
    for pid, meta in st.session_state["parcelles"].items():
        if meta.get("source") in ("dessin", "gps"):
            geojson = meta.get("geometry_geojson")
            if geojson:
                folium.GeoJson(
                    geojson,
                    style_function=lambda _: {
                        "fillColor": "#f0ad4e", "color": "#f0ad4e",
                        "weight": 2.5, "fillOpacity": 0.50,
                        "interactive": False,
                    },
                    tooltip=folium.Tooltip(
                        f"[Perso] {meta.get('nom','?')} - {meta.get('surf_parc',0):.1f} ha"
                    ),
                ).add_to(m)

    #instr = (
    #    "<div style='position:fixed;bottom:20px;left:20px;"
    #    "background:rgba(0,0,0,0.75);color:white;padding:10px 14px;"
    #    "border-radius:6px;font-size:12px;z-index:9999;font-family:monospace;"
    #    "max-width:240px;line-height:1.6'>"
    #    "<b>Selectionner une parcelle</b><br>"
    #    "Cliquez sur une zone coloree<br>pour la selectionner."
    #    "</div>"
    #)
    #m.get_root().html.add_child(folium.Element(instr))
    return m

# ── Layout ─────────────────────────────────────────────────────────────────────
st.title("Mes Parcelles")
st.caption("Selectionnez, dessinez ou saisissez vos parcelles a surveiller.")

tab_clic, tab_dessin, tab_gps = st.tabs([
    "Clic sur RPG", "Dessiner sur la carte", "Coordonnées GPS"
])

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Localisation")
    with st.form("search_form"):
        query     = st.text_input("Adresse ou commune",
                                   placeholder="Ex: Melun, Seine-et-Marne")
        gps_input = st.text_input("Ou coordonnees GPS", placeholder="48.682, 2.581")
        submitted = st.form_submit_button("Centrer la carte", type="primary")

    if submitted:
        coords = None
        if gps_input.strip():
            coords = parse_gps(gps_input)
            if coords is None:
                st.error("Format GPS invalide. Utiliser : lat, lon")
        elif query.strip():
            with st.spinner("Geocodage..."):
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
        to_delete = []
        for pid, meta in st.session_state["parcelles"].items():
            source  = meta.get("source", "rpg")
            culture = meta.get("code_cultu", "?")
            label   = CODE_CULTU_LABELS.get(culture, culture)
            if source in ("dessin", "gps"):
                label = meta.get("nom", "Parcelle perso")
            surface = meta.get("surf_parc", 0)
            badge   = "🟦" if source == "rpg" else "🟧"
            col_i, col_d = st.columns([4, 1])
            with col_i:
                st.markdown(f"{badge} **{label}**  \n`{pid[:12]}` - {surface:.1f} ha")
            with col_d:
                if st.button("x", key=f"del_{pid}", help="Retirer"):
                    to_delete.append(pid)
        for pid in to_delete:
            del st.session_state["parcelles"][pid]
        if to_delete:
            st.rerun()

        st.divider()
        total_ha = sum(m.get("surf_parc", 0)
                       for m in st.session_state["parcelles"].values())
        st.metric("Surface totale", f"{total_ha:.1f} ha")
        st.caption("🟦 RPG  🟧 Personnalisee")
        st.divider()
        if st.button("Lancer l'analyse satellite",
                     type="primary", use_container_width=True):
            st.switch_page("pages/2_Alertes.py")

# ── Calcul bbox ────────────────────────────────────────────────────────────────
#center = st.session_state["map_center"]
#zoom   = st.session_state["map_zoom"]
#delta  = max(0.02, 0.5 / (2 ** (zoom - 10)))
#lat_min, lat_max = center[0] - delta, center[0] + delta
#lon_min, lon_max = center[1] - delta * 1.5, center[1] + delta * 1.5

#with st.spinner("Chargement des parcelles RPG..."):
    #gdf = load_rpg_bbox(lat_min, lon_min, lat_max, lon_max)
    #gdf, bbox_extent = load_rpg_around_center(center, zoom, rpg_year=2024)

selected_ids = set(st.session_state["parcelles"].keys())



# ═════════════════════════════════════════════════════════════════════════════
# ONGLET 1 — Clic RPG
# ═════════════════════════════════════════════════════════════════════════════
with tab_clic:
    st.markdown("Naviguez sur la carte et cliquez sur une parcelle pour la sélectionner.")
    map_key_clic = f"map_clic_{len(selected_ids)}"

    rpg_year = st.number_input("Année de référence RPG", value=2024, key="rpg_year", width=300)
    if rpg_year:
        ok, msg = check_rpg_availability(rpg_year, st.session_state["today"])
        #Creation of st.session_state["rpg_year"] with arg: key

    if ok:
        gdf_rpg, bbox_extent = load_rpg_around_center(
            st.session_state['map_center'], st.session_state['map_zoom'],
            rpg_year=st.session_state['rpg_year'])
        st.markdown(f'RPG {rpg_year} : {msg} ; {len(gdf_rpg)} parcelles RPG chargées')
    else:
        st.warning('⚠️  Données RPG non disponibles pour cette année')
        gdf_rpg = gpd.GeoDataFrame()
        bbox_extent = None

    if gdf_rpg.empty:
        map_data = False
        st.warning("Aucune parcelle RPG chargée. Centrez la carte sur une zone agricole.")

    if not gdf_rpg.empty:
        m_rpg = build_map(
            center=st.session_state["map_center"], 
            zoom=st.session_state["map_zoom"],
            gdf=gdf_rpg,
            selected_ids=selected_ids,
            rpg_enabled=True)

        map_data = st_folium(
            m_rpg, 
            width=None,
            height=520,
            returned_objects=["last_object_clicked","zoom", "center"],
            key=map_key_clic,
            )

    
    #m_clic = build_map(center, zoom, gdf_rpg, selected_ids, draw_enabled=False)

    #map_data = st_folium(
        #m_clic,
        #width=None,
        #height=520,
        #returned_objects=["last_object_clicked", "center", "zoom"],
        #key=map_key_clic,
    #)

    col_a, col_b = st.columns(2)
    col_a.metric("Parcelles RPG visibles", len(gdf_rpg))
    col_b.metric("Sélectionnees", len(selected_ids))

    if map_data:
        if map_data.get("center"):
            st.session_state["map_center"] = [
                map_data["center"]["lat"],
                map_data["center"]["lng"],
            ]
        if map_data.get("zoom"):
            st.session_state["map_zoom"] = map_data["zoom"]

        if map_data.get("last_object_clicked") and not gdf_rpg.empty:
            click_lat = map_data["last_object_clicked"]["lat"]
            click_lon = map_data["last_object_clicked"]["lng"]
            parcelle_row = find_clicked_parcelle(click_lat, click_lon, gdf_rpg) #return dict

            if parcelle_row is not None:
                pid = str(parcelle_row.get("id_parcel", ""))
                if pid in st.session_state["parcelles"]:
                    del st.session_state["parcelles"][pid]
                    st.toast(f"Parcelle {pid} retiree", icon="X")
                else:
                    culture = parcelle_row.get("code_cultu", "?")
                    label   = CODE_CULTU_LABELS.get(culture, culture)
                    surf    = parcelle_row.get("surf_parc", 0)
                    st.session_state["parcelles"][pid] = {
                        "source"           : "rpg",
                        "code_cultu"       : culture,
                        "surf_parc"        : surf,
                        "geometry_geojson" : parcelle_row.geometry.__geo_interface__,
                        "lat"              : click_lat,
                        "lon"              : click_lon,
                    }
                    st.toast(f"{label} ({surf:.1f} ha) ajoutee")#, icon="✅")
                st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# ONGLET 2 — Dessin
# ═════════════════════════════════════════════════════════════════════════════
with tab_dessin:
    st.markdown(
        "Utilisez les outils de dessin pour tracer votre parcelle directement "
        "sur la carte, puis donnez-lui un nom."
    )

    col_carte, col_form = st.columns([3, 1])

    with col_carte:
        map_key_draw = f"map_draw_{len(selected_ids)}"
        m_draw = build_map(
            center=st.session_state["map_center"], 
            zoom=st.session_state["map_zoom"],
            gdf=None, 
            selected_ids=selected_ids, 
            draw_enabled=True)
        draw_data = st_folium(
            m_draw,
            width=None,
            height=480,
            returned_objects=["last_active_drawing", "center", "zoom"],
            key=map_key_draw,
        )

    with col_form:
        st.markdown("**Nommer la parcelle**")
        nom_dessin = st.text_input("Nom", placeholder="Ex: Champ du nord",
                                    key="nom_dessin")
        culture_dessin = st.selectbox(
            "Culture",
            options=list(CODE_CULTU_LABELS.keys()),
            format_func=lambda k: f"{k} - {CODE_CULTU_LABELS[k]}",
            key="culture_dessin",
        )

        if draw_data and draw_data.get("center"):
            st.session_state["map_center"] = [
                draw_data["center"]["lat"], draw_data["center"]["lng"]
            ]
        if draw_data and draw_data.get("zoom"):
            st.session_state["map_zoom"] = draw_data["zoom"]

        drawing = None
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
                st.session_state["parcelles"][pid] = {
                    "source"           : "dessin",
                    "nom"              : nom_dessin.strip(),
                    "code_cultu"       : culture_dessin,
                    "surf_parc"        : surf_dessin,
                    "geometry_geojson" : drawing["geometry"],
                    "lat"              : centroid.y,
                    "lon"              : centroid.x,
                }
                st.toast(f"'{nom_dessin}' ajoutee ({surf_dessin:.2f} ha)")#, icon="✅")
                #st.rerun()
            except Exception as e:
                st.error(f"Erreur : {e}")

        if not can_add:
            st.caption("Dessinez une parcelle puis nommez-la.")


# ═════════════════════════════════════════════════════════════════════════════
# ONGLET 3 — GPS
# ═════════════════════════════════════════════════════════════════════════════
with tab_gps:
    st.markdown(
        "Saisissez les coordonnées GPS des sommets de votre parcelle, "
        "une paire `latitude, longitude` par ligne."
    )

    col_form_gps, col_apercu = st.columns([1, 2])

    with col_form_gps:
        nom_gps = st.text_input("Nom de la parcelle",
                                 placeholder="Ex: Vigne sud", key="nom_gps")
        culture_gps = st.selectbox(
            "Culture",
            options=list(CODE_CULTU_LABELS.keys()),
            format_func=lambda k: f"{k} - {CODE_CULTU_LABELS[k]}",
            key="culture_gps",
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
                st.success(f"Polygone valide - {n_pts} points")
                st.metric("Surface estimee", f"{surf_gps:.2f} ha")

        can_add_gps = poly_gps is not None and nom_gps.strip()
        if st.button("Ajouter cette parcelle", type="primary",
                      disabled=not can_add_gps, use_container_width=True,
                      key="btn_add_gps"):
            pid      = f"gps_{uuid.uuid4().hex[:8]}"
            centroid = poly_gps.centroid
            st.session_state["parcelles"][pid] = {
                "source"           : "gps",
                "nom"              : nom_gps.strip(),
                "code_cultu"       : culture_gps,
                "surf_parc"        : surf_gps,
                "geometry_geojson" : poly_gps.__geo_interface__,
                "lat"              : centroid.y,
                "lon"              : centroid.x,
            }
            st.session_state["map_center"] = [centroid.y, centroid.x]
            st.session_state["map_zoom"]   = 15
            st.toast(f"'{nom_gps}' ajoutee ({surf_gps:.2f} ha)")#, icon="✅")
            st.rerun()

        with st.expander("Comment obtenir des coordonnees GPS ?"):
            st.markdown(
                "**Google Maps** : clic droit sur votre parcelle. "
                "Les coordonnees apparaissent en haut du menu contextuel.\n\n"
                "**Geoportail** : clic droit -> 'Creer un point'.\n\n"
                "**Smartphone** : epinglez un point sur votre app GPS, "
                "les details affichent les coordonnees."
            )

    with col_apercu:
        st.markdown("**Apercu de la parcelle**")
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
                    "interactive": True,
                },
                tooltip=folium.Tooltip(f"{nom_gps or 'Parcelle'} - {surf_gps:.2f} ha"),
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
                geojson = meta.get("geometry_geojson")
                if geojson:
                    folium.GeoJson(
                        geojson,
                        style_function=lambda _: {
                            "fillColor": "#f0ad4e", "color": "#f0ad4e",
                            "weight": 1.5, "fillOpacity": 0.30,
                            "interactive": True,
                        },
                        tooltip=folium.Tooltip(meta.get("nom", "?")),
                    ).add_to(m_gps)

        st_folium(m_gps, width=None, height=400, key="map_gps_preview")
