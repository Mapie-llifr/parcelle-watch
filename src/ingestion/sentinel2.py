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

import os
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from sentinelhub import (
    BBox,
    BBoxSplitter,
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

    # TODO: construire l'evalscript et la requête SentinelHubRequest
    # Voir notebook 01 pour le prototype
    raise NotImplementedError("À implémenter dans le notebook 01")


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
