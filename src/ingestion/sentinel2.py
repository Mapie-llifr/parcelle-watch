"""
src/ingestion/sentinel2.py
--------------------------
Téléchargement et gestion des images Sentinel-2 via l'API Copernicus Data Space.

Flux :
    1. Authentification OAuth2 (CLIENT_ID / CLIENT_SECRET)
    2. Recherche des scènes disponibles sur une zone (bbox ou polygone) et une période
    3. Filtrage par couverture nuageuse
    4. Téléchargement des bandes utiles uniquement (B04, B08, B05, B11...)
    5. Sauvegarde locale en GeoTIFF

Bandes Sentinel-2 utilisées :
    B02 — Bleu (490nm)
    B03 — Vert (560nm)
    B04 — Rouge (665nm)       ← NDVI
    B05 — Red Edge (705nm)    ← NDRE
    B08 — PIR large (842nm)   ← NDVI
    B8A — PIR étroit (865nm)  ← NDRE
    B11 — SWIR (1610nm)       ← NDWI

Dépendances : sentinelhub, rasterio, shapely, python-dotenv
"""

import io
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
from dotenv import load_dotenv
from loguru import logger
from rasterio.crs import CRS as RasterioCRS
from rasterio.transform import from_bounds
from sentinelhub import (
    BBox,
    CRS,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
)
from shapely.geometry import shape

load_dotenv()

# Bandes à télécharger (suffisantes pour NDVI, NDRE, NDWI)
BANDS_OF_INTEREST = ["B04", "B05", "B08", "B8A", "B11", "B03"]

# Résolution cible en mètres
DEFAULT_RESOLUTION = 10


def get_sh_config() -> SHConfig:
    """Charge la configuration Sentinel Hub depuis les variables d'environnement."""
    config = SHConfig()
    config.sh_client_id = os.getenv("COPERNICUS_CLIENT_ID")
    config.sh_client_secret = os.getenv("COPERNICUS_CLIENT_SECRET")
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

    if not config.sh_client_id or not config.sh_client_secret:
        raise EnvironmentError(
            "Variables COPERNICUS_CLIENT_ID et COPERNICUS_CLIENT_SECRET manquantes. "
            "Copier .env.example en .env et remplir les credentials."
        )
    return config


def build_bbox_from_geojson(geojson_geometry: dict) -> BBox:
    """
    Construit un BBox Sentinel Hub depuis une géométrie GeoJSON (ex: issue du RPG).

    Args:
        geojson_geometry: dict GeoJSON (type Polygon ou MultiPolygon)

    Returns:
        BBox en WGS84
    """
    geom = shape(geojson_geometry)
    minx, miny, maxx, maxy = geom.bounds
    return BBox(bbox=[minx, miny, maxx, maxy], crs=CRS.WGS84)


def search_available_scenes(
    bbox: BBox,
    start_date: date,
    end_date: date,
    max_cloud_coverage: float = 0.3,
) -> list[dict]:
    """
    Recherche les scènes Sentinel-2 L2A disponibles sur la zone et la période.

    Args:
        bbox: Emprise géographique
        start_date: Date de début
        end_date: Date de fin
        max_cloud_coverage: Seuil de couverture nuageuse (0.0 à 1.0)

    Returns:
        Liste de métadonnées de scènes disponibles, triées par date
    """
    # TODO: implémenter via sentinelhub.catalog ou OGC WFS
    # Voir : https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Catalog.html
    raise NotImplementedError("À implémenter dans le notebook 01")


def download_scene(
    bbox: BBox,
    acquisition_date: date,
    output_dir: Path,
    resolution: int = DEFAULT_RESOLUTION,
    config: SHConfig | None = None,
) -> Path:
    """
    Télécharge les bandes d'intérêt pour une scène et une zone données.

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

    logger.info(f"Téléchargement scène {date_str} sur bbox {bbox}...")

    # Evalscript : 6 bandes en FLOAT32 pour calculer NDVI, NDRE, NDWI, EVI
    # Ordre des bandes : B04, B05, B08, B8A, B11, B03
    # (cohérent avec load_bands() dans src/indices/vegetation.py)
    evalscript = """
    //VERSION=3
    function setup() {
        return {
            input: ["B02", "B03", "B04", "B05", "B08", "B8A", "B11"],
            output: { bands: 6, sampleType: "FLOAT32" }
        };
    }
    function evaluatePixel(sample) {
        return [
            sample.B04,   // bande 0 — Rouge      → NDVI, EVI
            sample.B05,   // bande 1 — Red Edge   → NDRE
            sample.B08,   // bande 2 — PIR large  → NDVI, NDRE, EVI
            sample.B8A,   // bande 3 — PIR étroit → NDRE
            sample.B11,   // bande 4 — SWIR       → NDWI
            sample.B03    // bande 5 — Vert       → NDWI, EVI
        ];
    }
    """

    # Fenêtre temporelle : jour J uniquement (± 1 jour pour tolérance)
    time_interval = (
        acquisition_date.strftime("%Y-%m-%d"),
        (acquisition_date + timedelta(days=1)).strftime("%Y-%m-%d"),
    )

    size = bbox_to_dimensions(bbox, resolution=resolution)

    request = SentinelHubRequest(
        evalscript=evalscript,
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
        size=size,
        config=config,
    )

    data = request.get_data()

    if not data or data[0] is None:
        raise ValueError(
            f"Aucune donnée retournée pour la date {date_str}. "
            "Vérifier la couverture nuageuse ou la disponibilité de la scène."
        )

    image = data[0]  # shape : (height, width, 6), dtype float32

    # --- Sauvegarde en GeoTIFF géoréférencé ---
    # La transform relie pixels et coordonnées géographiques (WGS84)
    height, width, n_bands = image.shape
    minx, miny, maxx, maxy = bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=n_bands,
        dtype=np.float32,
        crs=RasterioCRS.from_epsg(4326),  # WGS84
        transform=transform,
        nodata=np.nan,
    ) as dst:
        for i in range(n_bands):
            band = image[:, :, i].astype(np.float32)
            # Sentinel-2 L1C : réflectance TOA encodée en uint16 → diviser par 10000
            band = np.where(band > 0, band / 10000.0, np.nan)
            dst.write(band, i + 1)  # rasterio : bandes indexées à partir de 1

    logger.success(f"Scène sauvegardée : {output_path} ({width}×{height}px, {n_bands} bandes)")
    return output_path


def download_time_series(
    bbox: BBox,
    start_date: date,
    end_date: date,
    output_dir: Path,
    max_cloud_coverage: float = 0.3,
    resolution: int = DEFAULT_RESOLUTION,
) -> list[Path]:
    """
    Télécharge toutes les scènes disponibles sur une période (série temporelle).

    Args:
        bbox: Emprise géographique
        start_date: Date de début
        end_date: Date de fin
        output_dir: Répertoire de sortie
        max_cloud_coverage: Seuil nuages
        resolution: Résolution en mètres

    Returns:
        Liste des chemins vers les GeoTIFF téléchargés
    """
    config = get_sh_config()
    scenes = search_available_scenes(bbox, start_date, end_date, max_cloud_coverage)
    logger.info(f"{len(scenes)} scènes trouvées ({max_cloud_coverage*100:.0f}% nuages max)")

    downloaded = []
    for scene in scenes:
        try:
            path = download_scene(bbox, scene["date"], output_dir, resolution, config)
            downloaded.append(path)
        except Exception as e:
            logger.warning(f"Échec téléchargement {scene['date']} : {e}")

    logger.success(f"{len(downloaded)}/{len(scenes)} scènes téléchargées dans {output_dir}")
    return downloaded
