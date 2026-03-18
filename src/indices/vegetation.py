"""
src/indices/vegetation.py
--------------------------
Calcul des indices de végétation et stress à partir des bandes Sentinel-2.

Indices implémentés :
    NDVI  — Normalized Difference Vegetation Index
            = (B08 - B04) / (B08 + B04)
            Interprétation : < 0.2 sol nu | 0.2-0.5 végétation faible | > 0.5 végétation dense

    NDRE  — Normalized Difference Red Edge Index
            = (B08 - B05) / (B08 + B05)
            Plus sensible que NDVI au stress azoté, meilleur sur cultures denses

    NDWI  — Normalized Difference Water Index
            = (B03 - B11) / (B03 + B11)
            Détecte le stress hydrique : valeurs basses = stress

    EVI   — Enhanced Vegetation Index
            = 2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1)
            Moins saturé que NDVI en végétation dense

Dépendances : numpy, rasterio, xarray
"""

from pathlib import Path

import numpy as np
import rasterio
from loguru import logger


def load_bands(tif_path: Path) -> dict[str, np.ndarray]:
    """
    Charge les bandes d'un GeoTIFF Sentinel-2 en dictionnaire numpy.

    Convention d'ordre des bandes dans le fichier (défini dans sentinel2.py) :
        0: B04 (Rouge)
        1: B05 (Red Edge)
        2: B08 (PIR)
        3: B8A (PIR étroit)
        4: B11 (SWIR)
        5: B03 (Vert)

    Returns:
        Dict {'B04': array, 'B05': array, ...} avec valeurs float [0, 1]
    """
    band_names = ["B04", "B05", "B08", "B8A", "B11", "B03"]

    with rasterio.open(tif_path) as src:
        bands = {}
        for i, name in enumerate(band_names, start=1):
            data = src.read(i).astype(np.float32)
            # Normalisation : Sentinel-2 L2A → réflectance [0, 1]
            data = np.where(data > 0, data / 10000.0, np.nan)
            bands[name] = data
        meta = src.meta

    return bands, meta


def compute_ndvi(bands: dict[str, np.ndarray]) -> np.ndarray:
    """
    NDVI = (B08 - B04) / (B08 + B04)
    Indice général de vigueur végétative.
    """
    nir = bands["B08"]
    red = bands["B04"]
    denominator = nir + red
    ndvi = np.where(denominator != 0, (nir - red) / denominator, np.nan)
    return np.clip(ndvi, -1, 1)


def compute_ndre(bands: dict[str, np.ndarray]) -> np.ndarray:
    """
    NDRE = (B08 - B05) / (B08 + B05)
    Stress azoté et chlorophyllien — plus précis que NDVI sur cultures denses.
    """
    nir = bands["B08"]
    re = bands["B05"]
    denominator = nir + re
    ndre = np.where(denominator != 0, (nir - re) / denominator, np.nan)
    return np.clip(ndre, -1, 1)


def compute_ndwi(bands: dict[str, np.ndarray]) -> np.ndarray:
    """
    NDWI = (B03 - B11) / (B03 + B11)
    Stress hydrique — valeur basse = sol/végétation sec(e).
    """
    green = bands["B03"]
    swir = bands["B11"]
    denominator = green + swir
    ndwi = np.where(denominator != 0, (green - swir) / denominator, np.nan)
    return np.clip(ndwi, -1, 1)


def compute_evi(bands: dict[str, np.ndarray]) -> np.ndarray:
    """
    EVI = 2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1)
    Moins sujet à la saturation que NDVI en fortes densités végétales.
    Note: nécessite B02 (bleu) — retourne NaN si absent.
    """
    if "B02" not in bands:
        logger.warning("Bande B02 absente — EVI non calculable")
        return np.full_like(bands["B08"], np.nan)

    nir = bands["B08"]
    red = bands["B04"]
    blue = bands["B02"]
    denominator = nir + 6 * red - 7.5 * blue + 1
    evi = np.where(denominator != 0, 2.5 * (nir - red) / denominator, np.nan)
    return np.clip(evi, -1, 1)


def compute_all_indices(tif_path: Path) -> dict[str, np.ndarray]:
    """
    Point d'entrée principal : charge les bandes et calcule tous les indices.

    Args:
        tif_path: Chemin vers le GeoTIFF Sentinel-2

    Returns:
        Dict {'NDVI': array, 'NDRE': array, 'NDWI': array, 'EVI': array}
        + 'meta' : métadonnées rasterio pour la sauvegarde
    """
    logger.info(f"Calcul des indices sur {tif_path.name}...")
    bands, meta = load_bands(tif_path)

    indices = {
        "NDVI": compute_ndvi(bands),
        "NDRE": compute_ndre(bands),
        "NDWI": compute_ndwi(bands),
        "EVI": compute_evi(bands),
        "_meta": meta,
    }

    for name, arr in indices.items():
        if name.startswith("_"):
            continue
        valid = arr[~np.isnan(arr)]
        logger.debug(f"  {name}: min={valid.min():.3f} | mean={valid.mean():.3f} | max={valid.max():.3f}")

    return indices


def save_index(
    index_array: np.ndarray,
    meta: dict,
    output_path: Path,
) -> None:
    """Sauvegarde un indice en GeoTIFF single-band."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta.update({"count": 1, "dtype": "float32", "nodata": np.nan})

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(index_array.astype(np.float32), 1)
    logger.info(f"Indice sauvegardé : {output_path}")
