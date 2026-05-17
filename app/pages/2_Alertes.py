"""
app/pages/2_Alertes.py
-----------------------
Page des alertes de stress hydrique.
Effectue les calculs nécessaires à l'emploi du modèle.
Charge le modèle et affiche les résultats. 
"""

import sys
from pathlib import Path
from datetime import date, timedelta

import folium
import geopandas as gpd
import rasterio
import rasterio.features
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from sentinelhub import BBox, CRS
from shapely.geometry import Point, Polygon, box, shape, mapping

from src.ingestion.sentinel2 import get_sh_config, search_available_scenes, download_scene
from src.ingestion.meteo import fetch_historical_weather
from src.indices.vegetation import compute_ndvi, compute_ndwi, load_bands

#On retire un .parent pour tester si ça suffit pour faire redescendre 
#la création du fichier data dans la hierarchie.
#sys.path.insert(0, str(Path(__file__).parent.parent))#.parent)) 
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Alertes - Parcelle Watch", page_icon="🚨", layout="wide")

ROOT          = Path(__file__).parent.parent.parent
DATA_PROC     = ROOT / "data" / "processed"
DATA_RAW      = ROOT / "data" / "raw"
#DATA_RAW   = Path('../data/raw')
#DATA_PROC  = Path('../data/processed')
MODELS_DIR = DATA_PROC / 'models'
BBOX_LIMIT    = 800
#ANOMALIES_CSV = DATA_PROC / "anomalies_brie.csv"
#BBOX_COORDS   = (2.551592, 48.668972, 2.611508, 48.695606)
#WFS_BASE      = (
#    "https://data.geopf.fr/wfs/ows?"
#    "SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
#    "&TYPENAMES=RPG.2023:parcelles_graphiques"
#    "&OUTPUTFORMAT=application/json"
#)

CODE_CULTU_LABELS = {
    "BTH": "Ble tendre hiver", "MIS": "Mais", "CZH": "Colza hiver",
    "ORH": "Orge hiver", "ORP": "Orge printemps", "JAC": "Jachere",
    "PPH": "Prairie permanente", "BTA": "Ble tendre autre",
    "BTN": "Ble tendre printemps", "BOR": "Ble orge",
    "FVL": "Feverole", "LEC": "Lentille", "BFS": "Betterave",
}


#@st.cache_data
#def load_anomalies(path):
    #df = pd.read_csv(path, parse_dates=["date"])
    #return df


#@st.cache_data
#def load_rpg():
    #bbox_str = ",".join(map(str, BBOX_COORDS))
    #url = f"{WFS_BASE}&BBOX={bbox_str},EPSG:4326"
    #gdf = gpd.read_file(url)
    #return gdf[gdf["code_cultu"] != "SNE"].copy()

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



def sev_color(sev):
    if sev == "critical": return "#d9534f"
    if sev == "warning":  return "#f0ad4e"
    if sev == "normal":   return "#5cb85c"
    return "#aaaaaa"



#if not ANOMALIES_CSV.exists():
    #st.error("Donnees manquantes. Lancer les notebooks 02, 03 et 04 d'abord.")
    #st.stop()

#df   = load_anomalies(ANOMALIES_CSV)

#__Recherche de l'image satellite la plus récente disponible (nuages < 30%)
def global_bbox(parcelles):
    """Calcule la bbox englobante de toutes les parcelles."""
    all_minx = [parcelles[k]['bbox'][0] for k in parcelles.keys()]
    all_miny = [parcelles[k]['bbox'][1] for k in parcelles.keys()]
    all_maxx = [parcelles[k]['bbox'][2] for k in parcelles.keys()]
    all_maxy = [parcelles[k]['bbox'][3] for k in parcelles.keys()]
    margin = 0.002   # ~200m de marge
    return (
        min(all_minx) - margin,
        min(all_miny) - margin,
        max(all_maxx) + margin,
        max(all_maxy) + margin,
    )

def zonal_stats_parcel(geom, index_arr, transform, min_pixels=2):
    """Statistiques d'un indice sur une géométrie. Retourne None si trop peu de pixels."""
    mask = rasterio.features.geometry_mask(
        [mapping(geom)], out_shape=index_arr.shape,
        transform=transform, invert=True,
    )
    pixels = index_arr[mask]
    pixels = pixels[~np.isnan(pixels)]
    pixels = pixels[(pixels >= -1.0) & (pixels <= 1.0)]
    if len(pixels) < min_pixels:
        return None
    return {
        'mean': float(np.mean(pixels)),
        'std' : float(np.std(pixels)),
        'p10' : float(np.percentile(pixels, 10)),
        'p90' : float(np.percentile(pixels, 90)),
        'n'   : len(pixels),
    }


def compute_meteo_features(meteo_df, ref_date):
    """
    Calcule les features météo agrégées sur 7j et 14j
    pour une date de référence.
    """
    mi = meteo_df.set_index('date')
    ref = pd.Timestamp(ref_date)

    def agg(days, col, fn='sum'):
        w = mi.loc[ref - pd.Timedelta(days=days) : ref - pd.Timedelta(days=1), col]
        return float(getattr(w, fn)()) if len(w) else np.nan

    return {
        'precip_7d'  : agg(7,  'precipitation_sum',          'sum'),
        'precip_14d' : agg(14, 'precipitation_sum',          'sum'),
        'tmax_7d'    : agg(7,  'temperature_2m_max',         'mean'),
        'et0_7d'     : agg(7,  'et0_fao_evapotranspiration', 'sum'),
        'deficit_7d' : agg(7,  'precipitation_sum',          'sum')
                     - agg(7,  'et0_fao_evapotranspiration', 'sum'),
    }


st.title("Alertes stress hydrique")

if not st.session_state['parcelles']:
    st.warning("Aucune parcelle sélectionnée, revenir à la page 'Mes Parcelles'.")

# Bbox globale
minx, miny, maxx, maxy = global_bbox(st.session_state['parcelles'])
sh_bbox = BBox(bbox=[minx, miny, maxx, maxy], crs=CRS.WGS84)
st.markdown(f'Bbox globale : {minx:.4f},{miny:.4f} -> {maxx:.4f},{maxy:.4f}')

if np.square(minx-maxx) > BBOX_LIMIT or np.square(miny-maxy) > BBOX_LIMIT:
    st.warning("Les parcelles sélectionnées sont trop éloignées les unes des autres pour récupérer une image satellite couvrant la zone.")
    st.stop()
else:
    
# Est-ce qu'on a besoin de gdf_rpg ?? 
# oui, pour l'affichage après ? ou non, on a déjà les coordonnées et le type de culture pour les parcelles sélectionnées
#gdf  = load_rpg_around_center(center, zoom, rpg_year=2024)
#gdf, bbox_extent = load_rpg_around_center(
            #st.session_state['map_center'], st.session_state['map_zoom'],
            #rpg_year=st.session_state['rpg_year'])
#gdf["parcelle_id"] = gdf["id_parcel"].astype(str)

# Recherche de la scène la plus récente (30 derniers jours, nuages < 30%)
    config = get_sh_config()
    end_d   = st.session_state['today']
    start_d = end_d - timedelta(days=30)

with st.spinner(f'Recherche scènes du {start_d} au {end_d}...'):
    scenes = search_available_scenes(sh_bbox, start_d, end_d,
                                  max_cloud_coverage=0.30, config=config)

if not scenes:
    st.warning('⚠️  Aucune scène disponible — élargir la fenêtre temporelle')
    st.session_state['date_acq'] = None
else:
    # Scène la plus récente en premier
    latest = scenes[-1]
    st.session_state['date_acq'] = latest['date']
    st.markdown(f"Scène la plus récente : {latest['date']} (nuages : {latest['cloud_coverage']:.1f}%)")
    st.markdown(f'Scènes disponibles : {[s["date"].isoformat() for s in scenes[-5:]]}')

    # Téléchargement de la scène
    out_dir = DATA_RAW / 'interface_test'
    out_dir.mkdir(parents=True, exist_ok=True)

    st.markdown(f"Téléchargement scène {st.session_state['date_acq']}...")
    tif_path = download_scene(
        bbox=sh_bbox,
        acquisition_date=st.session_state['date_acq'],
        output_dir=out_dir,
        config=config,
    )
    st.session_state['tif_path'] = tif_path
    st.markdown(f'TIF téléchargé : {tif_path}')

    # Météo sur les 14 jours précédant la prise de vue
    date_acq = st.session_state['date_acq']
    lat_c = (miny + maxy) / 2
    lon_c = (minx + maxx) / 2

    meteo_start = date_acq - timedelta(days=14)
    meteo_df    = fetch_historical_weather(lat_c, lon_c, meteo_start, date_acq)
    meteo_df['date'] = pd.to_datetime(meteo_df['date'])
    st.session_state['meteo_df'] = meteo_df

    st.markdown(f'Météo : {len(meteo_df)} jours ({meteo_start} -> {date_acq})')
    st.markdown(f"\n✅ ÉTAPE 3 OK")
    st.markdown(f"   Date acquisition : {st.session_state['date_acq']}")
    st.markdown(f"   TIF path         : {st.session_state['tif_path'].name}")



# Parcelles sélectionnees en session (depuis page Mes Parcelles)
parcelles_session = st.session_state.get("parcelles", {})

##########
# Actuellement on va bien jusque là !!! :)
# Sauf que ça a crée un fichier data à coté du fichier parent, oups :/

##########

#Elaboration du Dataframe de features, une ligne par parcelle sélectionnée
if st.session_state['tif_path'] is None:
    st.warning('Pas de TIF disponible — Erreur de chargement de la prise de vue satellite.')
else:
    # Charger le raster
    bands, meta = load_bands(st.session_state['tif_path'])
    ndvi_arr    = compute_ndvi(bands)
    ndwi_arr    = compute_ndwi(bands)
    transform   = meta['transform']

    # Features météo (une seule date d'acquisition)
    meteo_feat = compute_meteo_features(st.session_state['meteo_df'], st.session_state['date_acq'])

    # Pour chaque parcelle
    rows = []
    for pid in st.session_state['parcelles'].keys():
        #geom = st.session_state['parcelles'][pid]['geometry_geojson']
        geom = st.session_state['parcelles'][pid]['geometry']
        s_ndvi = zonal_stats_parcel(geom, ndvi_arr, transform)
        st.markdown(s_ndvi)
        s_ndwi = zonal_stats_parcel(geom, ndwi_arr, transform)
        st.markdown(s_ndwi)

        if s_ndvi is None or s_ndwi is None:
            st.warning(f"⚠️  Parcelle {pid} : trop peu de pixels valides")
            continue

        row = {
            'parcelle_id'  : pid,
            'source'       : st.session_state['parcelles'][pid]['source'],
            'code_cultu'   : st.session_state['parcelles'][pid]['code_cultu'],
            'surf_parc'    : st.session_state['parcelles'][pid]['surf_parc'],
            'date'         : st.session_state['date_acq'],
            'ndvi_mean'    : s_ndvi['mean'],
            'ndvi_std'     : s_ndvi['std'],
            'ndvi_p10'     : s_ndvi['p10'],
            'ndvi_p90'     : s_ndvi['p90'],
            'ndwi_mean'    : s_ndwi['mean'],
            'ndwi_std'     : s_ndwi['std'],
            'ndwi_p10'     : s_ndwi['p10'],
            'ndwi_p90'     : s_ndwi['p90'],
            'n_pixels'     : s_ndvi['n'],
            'day_of_year'  : st.session_state['date_acq'].timetuple().tm_yday,
        }
        row.update(meteo_feat)

        # Features temporelles : pas d'historique ici (une seule date)
        # On met NaN — le modèle doit gérer l'absence de delta
        row['ndwi_delta']     = np.nan
        row['ndvi_delta']     = np.nan
        row['ndwi_deviation'] = np.nan   # calculé ci-dessous
        row['ndvi_deviation'] = np.nan
        rows.append(row)

    df_feat = pd.DataFrame(rows)

    # Déviation par rapport à la médiane de la même culture dans le dataset
    # Charger les références depuis le dataset d'entraînement
    # (utilise la moyenne des modèles comme proxy si pas d'historique)
    # Note : en production charger les refs depuis le CSV d'entraînement
    df_feat['ndwi_deviation'] = 0.0   # neutre si pas de référence
    df_feat['ndvi_deviation'] = 0.0

    st.session_state['df_features'] = df_feat
    st.markdown('\n✅ ÉTAPE 4 OK')
    st.markdown(df_feat[['parcelle_id','code_cultu','ndvi_mean','ndwi_mean',
                   'deficit_7d','n_pixels']].to_string())




###############
## On arrive jusqu'ici ok !!
# En vrai on va même jusqu'à l'affichage de toutes les colonnes de filtrage
#################

FEATURE_COLS = [
    'ndwi_mean','ndwi_p10','ndwi_std','ndvi_mean',
    'ndwi_deviation','ndvi_deviation',
    'ndwi_delta','ndvi_delta',
    'day_of_year',
    'precip_14d','tmax_7d','deficit_7d',
]


def get_model(code_cultu, stress_type='HYDRIQUE'):
    """Charge le modèle le plus adapté (spécifique > général)."""
    index_path = MODELS_DIR / 'models_index.csv'
    st.markdown(f'index_path: {index_path}')
    if not index_path.exists():
        raise FileNotFoundError('models_index.csv manquant. Télécharger les modèles.')

    index = pd.read_csv(index_path)
    index = index[index['stress_type'] == stress_type]

    for _, row in index.iterrows():
        codes = str(row['cultures_codes']).split(',')
        if code_cultu in codes:
            data = joblib.load(Path(row['path']))
            data['model_name'] = row['model_name']
            data['source']     = 'specific'
            return data

    # Fallback
    general = index[index['cultures_codes'] == 'ALL']
    if not general.empty:
        data = joblib.load(Path(general.iloc[0]['path']))
        data['source'] = 'fallback'
        return data

    raise FileNotFoundError(f'Aucun modèle {stress_type} pour {code_cultu}')


def score_row(row, feature_cols):
    """Score une ligne de features avec le modèle approprié."""
    try:
        model_data = get_model(row['code_cultu'])
    except FileNotFoundError as e:
        st.warning(f"⚠️  {row['parcelle_id']} : {e}")
        return np.nan, False, 'unknown', 'N/A'

    model  = model_data['model']
    scaler = model_data['scaler']
    feats  = model_data.get('features', feature_cols)

    X = row[feats].values.reshape(1, -1)
    # Remplacer les NaN par 0 (neutre) pour les features temporelles manquantes
    X = np.nan_to_num(X, nan=0.0)
    X_sc = scaler.transform(X)

    score  = float(model.decision_function(X_sc)[0])
    is_anom = bool(model.predict(X_sc)[0] == -1)

    if score < -0.12:  sev = 'critical'
    elif score < -0.04: sev = 'warning'
    else:               sev = 'normal'

    return score, is_anom, sev, model_data.get('model_name','general')


if st.session_state['df_features'] is None:
    st.warning('df_features absent — Erreur dans le calcul des features.')
else:
    df = st.session_state['df_features'].copy()
    results = []
    for _, row in df.iterrows():
        score, is_anom, sev, model_name = score_row(row, FEATURE_COLS)
        results.append({
            'parcelle_id'   : row['parcelle_id'],
            'code_cultu'    : row['code_cultu'],
            'surf_parc'     : row['surf_parc'],
            'ndwi_mean'     : row['ndwi_mean'],
            'ndvi_mean'     : row['ndvi_mean'],
            'deficit_7d'    : row['deficit_7d'],
            'anomaly_score' : score,
            'is_anomaly'    : is_anom,
            'severity'      : sev,
            'model_used'    : model_name,
        })

    df_res = pd.DataFrame(results)
    st.session_state['df_results'] = df_res

    st.markdown('\n✅ ÉTAPE 5 OK — Résultats d\'inférence :')
    st.markdown(df_res[['parcelle_id','code_cultu','ndwi_mean',
                  'anomaly_score','severity','model_used']].to_string())


#########
# Alors oui, on arrive jusque là, mais ils ne trouvent pas les modèles, on a cette erreur : 
# ⚠️ 11378520 : [Errno 2] No such file or directory: '../data/processed/models/isolation_forest_HYDRIQUE_MIS.joblib'
#la suite au prochain épisode.
##########



# Selecteurs
dates_dispo = sorted(st.session_state['df_features']["date"].unique())
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    selected_date = st.selectbox(
        "Date", options=dates_dispo, index=len(dates_dispo)-1,
        format_func=lambda d: pd.Timestamp(d).strftime("%d %B %Y"),
    )
with col2:
    cultures = sorted(st.session_state['df_features']["code_cultu"].dropna().unique())
    sel_cultures = st.multiselect("Cultures", options=cultures, default=cultures)
with col3:
    sev_filter = st.selectbox("Severite", ["Toutes", "Warning+", "Critique"])

df_feat = st.session_state['df_features']
# Filtre
df_date = df_feat[df_feat["date"] == selected_date].copy()
if sel_cultures:
    df_date = df_date[df_date["code_cultu"].isin(sel_cultures)]
if sev_filter == "Warning+":
    df_date = df_date[df_date["severity"].isin(["warning", "critical"])]
elif sev_filter == "Critique":
    df_date = df_date[df_date["severity"] == "critical"]

# Metriques
m1, m2, m3, m4 = st.columns(4)
n_tot  = len(df_date)
n_anom = int(df_date["is_anomaly"].sum()) if "is_anomaly" in df_date.columns else 0
m1.metric("Parcelles", n_tot)
m2.metric("Anomalies", f"{n_anom} ({n_anom/n_tot*100:.0f}%)" if n_tot else "0")
m3.metric("Attention", int((df_date["severity"] == "warning").sum()))
m4.metric("Critique",  int((df_date["severity"] == "critical").sum()))

st.divider()

col_map, col_tbl = st.columns([3, 2])

with col_map:
    df_date["parcelle_id"] = df_date["parcelle_id"].astype(str)
    gdf_s = gdf.merge(
        df_date[["parcelle_id", "anomaly_score", "is_anomaly",
                 "severity", "ndwi_mean", "ndvi_mean"]],
        on="parcelle_id", how="left",
    )

    lat_c = (BBOX_COORDS[1] + BBOX_COORDS[3]) / 2
    lon_c = (BBOX_COORDS[0] + BBOX_COORDS[2]) / 2
    m_folium = folium.Map(location=[lat_c, lon_c], zoom_start=14, tiles="Esri WorldImagery")

    # Surligner les parcelles de la session
    session_ids = set(parcelles_session.keys())

    for _, row in gdf_s.iterrows():
        sev  = str(row.get("severity", ""))
        pid  = str(row.get("id_parcel", ""))
        color = sev_color(sev)
        ndwi_s = f"{row['ndwi_mean']:.3f}" if pd.notna(row.get("ndwi_mean")) else "N/A"
        sev_s  = sev if sev else "N/A"
        in_session = pid in session_ids

        popup_html = (
            f"<div style='font-family:monospace;font-size:12px'>"
            f"<b>Parcelle {pid}</b><br>"
            f"Culture : {row.get('code_cultu','?')}<br>"
            f"NDWI : {ndwi_s}<br>"
            f"Severite : <b style='color:{color}'>{sev_s.upper()}</b>"
            f"{'<br><b>⭐ Votre parcelle</b>' if in_session else ''}"
            f"</div>"
        )

        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _, c=color, sel=in_session: {
                "fillColor": c,
                "color":     "#ffff00" if sel else "white",
                "weight":    2.5 if sel else 0.8,
                "fillOpacity": 0.65,
            },
            tooltip=f"{row.get('code_cultu','?')} - {sev_s.upper()}",
            popup=folium.Popup(popup_html, max_width=200),
        ).add_to(m_folium)

    st_folium(m_folium, width=None, height=450)

with col_tbl:
    st.subheader("Tableau des alertes")
    df_alerts = df_date[df_date.get("is_anomaly", False) == True] if "is_anomaly" in df_date.columns else pd.DataFrame()
    if hasattr(df_date, "columns") and "is_anomaly" in df_date.columns:
        df_alerts = df_date[df_date["is_anomaly"] == True].sort_values("anomaly_score")

    if df_alerts.empty:
        st.success("Aucune anomalie pour ces filtres.")
    else:
        disp = df_alerts[
            ["parcelle_id", "code_cultu", "surf_parc", "ndwi_mean", "anomaly_score", "severity"]
        ].rename(columns={
            "parcelle_id": "Parcelle", "code_cultu": "Culture",
            "surf_parc": "Ha", "ndwi_mean": "NDWI",
            "anomaly_score": "Score", "severity": "Severite",
        }).copy()
        for c in ["NDWI", "Score"]:
            if c in disp.columns:
                disp[c] = disp[c].round(3)
        st.dataframe(disp, use_container_width=True, height=420, hide_index=True)
