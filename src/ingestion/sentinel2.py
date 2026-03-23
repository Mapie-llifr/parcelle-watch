"""
src/ingestion/sentinel2.py
--------------------------
Téléchargement et gestion des images Sentinel-2 via l'API Copernicus Data Space.

Flux :
    1. Authentification OAuth2 (CLIENT_ID / CLIENT_SECRET)
    2. Recherche des scènes disponibles via Catalog API (filtre nuages)
    3. Téléchargement des bandes utiles en FLOAT32
    4. Sauvegarde locale en GeoTIFF géoréférencé

Bandes Sentinel-2 stockées dans le GeoTIFF (ordre fixe) :
    bande 1 — B04  Rouge (665nm)        → NDVI, EVI
    bande 2 — B05  Red Edge (705nm)     → NDRE
    bande 3 — B08  PIR large (842nm)    → NDVI, NDRE, EVI
    bande 4 — B8A  PIR étroit (865nm)   → NDRE
    bande 5 — B11  SWIR (1610nm)        → NDWI
    bande 6 — B03  Vert (560nm)         → NDWI, EVI

Dépendances : sentinelhub, rasterio, numpy, pandas, python-dotenv, loguru, tqdm
"""

import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from dotenv import load_dotenv
from loguru import logger
from rasterio.crs import CRS as RasterioCRS
from rasterio.transform import from_bounds
from sentinelhub import (
    BBox,
    CRS,
    DataCollection,
    SentinelHubCatalog,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
)
from shapely.geometry import shape
from tqdm import tqdm

load_dotenv()

DEFAULT_RESOLUTION = 10

# Evalscript commun — 6 bandes FLOAT32, ordre documenté dans le docstring
EVALSCRIPT_6BANDS = """
//VERSION=3
function setup() {
    return {
        input: ["B03", "B04", "B05", "B08", "B8A", "B11"],
        output: { bands: 6, sampleType: "FLOAT32" }
    };
}
function evaluatePixel(sample) {
    return [
        sample.B04,   // bande 1 — Rouge      → NDVI, EVI
        sample.B05,   // bande 2 — Red Edge   → NDRE
        sample.B08,   // bande 3 — PIR large  → NDVI, NDRE, EVI
        sample.B8A,   // bande 4 — PIR étroit → NDRE
        sample.B11,   // bande 5 — SWIR       → NDWI
        sample.B03    // bande 6 — Vert       → NDWI, EVI
    ];
}
"""


# ── Configuration ──────────────────────────────────────────────────────────────

def get_sh_config() -> SHConfig:
    """Charge la configuration Sentinel Hub depuis les variables d'environnement."""
    config = SHConfig()
    config.sh_client_id = os.getenv("COPERNICUS_CLIENT_ID")
    config.sh_client_secret = os.getenv("COPERNICUS_CLIENT_SECRET")
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    config.sh_token_url = (
        "https://identity.dataspace.copernicus.eu"
        "/auth/realms/CDSE/protocol/openid-connect/token"
    )
    if not config.sh_client_id or not config.sh_client_secret:
        raise EnvironmentError(
            "Variables COPERNICUS_CLIENT_ID et COPERNICUS_CLIENT_SECRET manquantes. "
            "Copier .env.example en .env et remplir les credentials."
        )
    return config


def build_bbox_from_geojson(geojson_geometry: dict) -> BBox:
    """
    Construit un BBox Sentinel Hub depuis une géométrie GeoJSON (ex: RPG).

    Args:
        geojson_geometry: dict GeoJSON (type Polygon ou MultiPolygon)

    Returns:
        BBox en WGS84
    """
    geom = shape(geojson_geometry)
    minx, miny, maxx, maxy = geom.bounds
    return BBox(bbox=[minx, miny, maxx, maxy], crs=CRS.WGS84)


# ── Recherche de scènes ────────────────────────────────────────────────────────

def search_available_scenes(
    bbox: BBox,
    start_date: date,
    end_date: date,
    max_cloud_coverage: float = 0.3,
    config: SHConfig | None = None,
) -> list[dict]:
    """
    Recherche les scènes Sentinel-2 L2A disponibles via le Catalog API Copernicus.

    Args:
        bbox: Emprise géographique
        start_date: Date de début
        end_date: Date de fin
        max_cloud_coverage: Seuil de couverture nuageuse (0.0 à 1.0)
        config: Configuration SentinelHub (chargée depuis .env si None)

    Returns:
        Liste de dicts {date, cloud_coverage, id} triés chronologiquement,
        dédoublonnés par date.
    """
    if config is None:
        config = get_sh_config()

    catalog = SentinelHubCatalog(config=config)

    search_iterator = catalog.search(
        DataCollection.SENTINEL2_L2A.define_from(
            "s2l2a", service_url=config.sh_base_url
        ),
        bbox=bbox,
        time=(
            start_date.strftime("%Y-%m-%dT00:00:00Z"),
            end_date.strftime("%Y-%m-%dT23:59:59Z"),
        ),
        filter=f"eo:cloud_cover < {int(max_cloud_coverage * 100)}",
        fields={
            "include": [
                "id",
                "properties.datetime",
                "properties.eo:cloud_cover",
            ]
        },
    )

    scenes = []
    for item in search_iterator:
        dt_str = item["properties"]["datetime"][:10]  # "YYYY-MM-DD"
        scenes.append({
            "date": date.fromisoformat(dt_str),
            "cloud_coverage": item["properties"].get("eo:cloud_cover", 0),
            "id": item["id"],
        })

    # Dédoublonner par date (plusieurs granules peuvent couvrir la même bbox)
    seen: set[date] = set()
    unique_scenes = []
    for s in sorted(scenes, key=lambda x: x["date"]):
        if s["date"] not in seen:
            seen.add(s["date"])
            unique_scenes.append(s)

    logger.info(
        f"{len(unique_scenes)} scènes disponibles "
        f"({start_date} → {end_date}, nuages < {max_cloud_coverage*100:.0f}%)"
    )
    return unique_scenes


# ── Utilitaire de sauvegarde ───────────────────────────────────────────────────

def _save_tiff(image: np.ndarray, bbox: BBox, output_path: Path) -> None:
    """
    Sauvegarde un array (H, W, C) en GeoTIFF géoréférencé WGS84.
    Normalise les valeurs en réflectance [0, 1] via division par 10000.
    """
    height, width, n_bands = image.shape
    transform = from_bounds(
        bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y, width, height
    )
    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=height, width=width,
        count=n_bands,
        dtype=np.float32,
        crs=RasterioCRS.from_epsg(4326),
        transform=transform,
        nodata=np.nan,
    ) as dst:
        for i in range(n_bands):
            band = image[:, :, i].astype(np.float32)
            band = np.where(band > 0, band / 10000.0, np.nan)
            dst.write(band, i + 1)


# ── Téléchargement d'une scène unique ─────────────────────────────────────────

def download_scene(
    bbox: BBox,
    acquisition_date: date,
    output_dir: Path,
    resolution: int = DEFAULT_RESOLUTION,
    config: SHConfig | None = None,
) -> Path:
    """
    Télécharge les 6 bandes d'intérêt pour une date et une zone données.

    Args:
        bbox: Emprise géographique
        acquisition_date: Date d'acquisition souhaitée
        output_dir: Répertoire de sortie
        resolution: Résolution en mètres (10 par défaut)
        config: Configuration SentinelHub (chargée depuis .env si None)

    Returns:
        Chemin vers le fichier GeoTIFF sauvegardé
    """
    if config is None:
        config = get_sh_config()

    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = acquisition_date.strftime("%Y%m%d")
    output_path = output_dir / f"sentinel2_{date_str}.tif"

    if output_path.exists():
        logger.info(f"Scène déjà téléchargée : {output_path}")
        return output_path

    logger.info(f"Téléchargement scène {date_str}...")

    time_interval = (
        acquisition_date.strftime("%Y-%m-%d"),
        (acquisition_date + timedelta(days=1)).strftime("%Y-%m-%d"),
    )

    request = SentinelHubRequest(
        evalscript=EVALSCRIPT_6BANDS,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C.define_from(
                    "s2l1c", service_url=config.sh_base_url
                ),
                time_interval=time_interval,
            )
        ],
        responses=[{"identifier": "default", "format": {"type": "image/tiff"}}],
        bbox=bbox,
        size=bbox_to_dimensions(bbox, resolution=resolution),
        config=config,
    )

    data = request.get_data()
    if not data or data[0] is None:
        raise ValueError(
            f"Aucune donnée pour le {date_str}. "
            "Vérifier la couverture nuageuse ou la disponibilité."
        )

    _save_tiff(data[0], bbox, output_path)
    h, w, _ = data[0].shape
    logger.success(f"Sauvegardé : {output_path.name} ({w}x{h}px)")
    return output_path


# ── Téléchargement batch — série temporelle ────────────────────────────────────

def download_time_series_batch(
    bbox: BBox,
    start_date: date,
    end_date: date,
    output_dir: Path,
    max_cloud_coverage: float = 0.3,
    resolution: int = DEFAULT_RESOLUTION,
    config: SHConfig | None = None,
) -> list[Path]:
    """
    Télécharge toutes les scènes disponibles sur une période — approche batch
    optimisée pour construire une série temporelle.

    Stratégie :
        1. Catalog API → liste des dates disponibles (filtre nuages inclus)
        2. Une requête SentinelHub par date, avec cache disque automatique
        3. Reprise possible : les scènes déjà présentes sont skippées
        4. Barre de progression tqdm

    Exemple d'utilisation :
        paths = download_time_series_batch(
            bbox=BBox([2.55, 48.67, 2.61, 48.70], crs=CRS.WGS84),
            start_date=date(2023, 4, 1),
            end_date=date(2023, 10, 31),
            output_dir=Path("data/raw/brie_2023"),
            max_cloud_coverage=0.3,
        )

    Args:
        bbox: Emprise géographique
        start_date: Début de période (ex: date(2023, 4, 1))
        end_date: Fin de période   (ex: date(2023, 10, 31))
        output_dir: Répertoire de sortie (créé si inexistant)
        max_cloud_coverage: Seuil nuages 0.0→1.0 (défaut 0.30 = 30%)
        resolution: Résolution en mètres (défaut 10)
        config: Configuration SentinelHub (chargée depuis .env si None)

    Returns:
        Liste des Path des GeoTIFF téléchargés ou déjà présents sur disque
    """
    if config is None:
        config = get_sh_config()

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Recherche des scènes disponibles ───────────────────────────────────
    scenes = search_available_scenes(
        bbox, start_date, end_date, max_cloud_coverage, config
    )

    if not scenes:
        logger.warning(
            "Aucune scène disponible. "
            "Essayer d'augmenter max_cloud_coverage ou d'élargir la période."
        )
        return []

    size = bbox_to_dimensions(bbox, resolution=resolution)
    downloaded: list[Path] = []
    skipped = 0
    failed = 0

    # ── 2. Téléchargement avec cache et progression ───────────────────────────
    for scene in tqdm(scenes, desc="Téléchargement Sentinel-2", unit="scène"):
        scene_date = scene["date"]
        date_str = scene_date.strftime("%Y%m%d")
        output_path = output_dir / f"sentinel2_{date_str}.tif"

        # Cache disque : ne pas re-télécharger ce qui existe déjà
        if output_path.exists():
            logger.debug(f"Cache hit : {output_path.name}")
            downloaded.append(output_path)
            skipped += 1
            continue

        time_interval = (
            scene_date.strftime("%Y-%m-%d"),
            (scene_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        )

        try:
            request = SentinelHubRequest(
                evalscript=EVALSCRIPT_6BANDS,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L1C.define_from(
                            "s2l1c", service_url=config.sh_base_url
                        ),
                        time_interval=time_interval,
                    )
                ],
                responses=[
                    {"identifier": "default", "format": {"type": "image/tiff"}}
                ],
                bbox=bbox,
                size=size,
                config=config,
            )

            data = request.get_data()

            if not data or data[0] is None:
                logger.warning(f"Pas de données pour {date_str}")
                failed += 1
                continue

            _save_tiff(data[0], bbox, output_path)
            downloaded.append(output_path)

        except Exception as e:
            logger.warning(f"Échec {date_str} : {e}")
            failed += 1

    # ── 3. Résumé ─────────────────────────────────────────────────────────────
    logger.success(
        f"Batch terminé — {len(downloaded)} scènes disponibles "
        f"({skipped} depuis cache, {failed} échecs) → {output_dir}"
    )
    return downloaded


# ── Index des scènes locales ───────────────────────────────────────────────────

def build_scene_index(output_dir: Path) -> pd.DataFrame:
    """
    Construit un index DataFrame des GeoTIFF présents dans output_dir.

    Utile pour savoir quelles dates sont disponibles avant de lancer
    le calcul des indices ou l'entraînement du modèle.

    Returns:
        DataFrame [date, path, filename] trié chronologiquement
    """
    tifs = sorted(output_dir.glob("sentinel2_*.tif"))
    records = []
    for tif in tifs:
        date_str = tif.stem.replace("sentinel2_", "")
        try:
            d = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
            records.append({"date": d, "path": tif, "filename": tif.name})
        except ValueError:
            logger.warning(f"Nom de fichier non reconnu : {tif.name}")

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)
    logger.info(f"{len(df)} scènes indexées dans {output_dir}")
    return df
