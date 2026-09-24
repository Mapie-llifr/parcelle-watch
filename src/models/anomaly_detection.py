"""
src/models/anomaly_detection.py  — section inférence par cellules
------------------------------------------------------------------
Nouvelle logique d'inférence :
  1. Découper la parcelle en cellules 100m×100m (build_parcel_grid_fixed)
  2. Calculer les indices + features météo sur chaque cellule
  3. Scorer chaque cellule avec le modèle approprié
  4. Agréger : sévérité parcelle = max(sévérité cellules)

Sévérité :  critical  > warning  > normal
Règle      : une cellule critique → parcelle critique
             aucune critique, une warning → parcelle warning
             toutes normales → parcelle normale
"""

import math
import numpy as np
import pandas as pd
import joblib
import rasterio.features
from pathlib import Path
from shapely.geometry import Polygon, box, mapping

# ── Constantes ───────────────────────────────────────────────────────────────
CELL_SIZE_M     = 100
LAT_REF         = 47.0
MIN_PIXELS_CELL = 20

SEV_ORDER = {"critical": 2, "warning": 1, "normal": 0, "unknown": -1}

FEATURE_COLS = {
    "HYDRIQUE": [
        "ndwi_mean", "ndwi_p10", "ndwi_std", "ndvi_mean",
        "ndwi_deviation", "ndvi_deviation",
        "ndwi_delta", "ndvi_delta",
        "day_of_year",
        "precip_14d", "tmax_7d", "deficit_7d",
    ],
    "AZOTE": [
        "ndre_mean", "ndre_std", "ndre_p10", "ndre_deviation",
        "ndre_delta", "ndre_ndvi_ratio", "evi_mean", "evi_deviation",
        "ndvi_mean", "day_of_year",
        "precip_30d", "precip_delta", "temp_sum_30d", "tmin_7d",
    ],
    "RAVAGEURS": [
        "ndre_std", "ndvi_delta", "ndre_ndvi_ratio", "evi_delta",
        "ndwi_mean", "day_of_year",
        "tmin_7d", "tmax_tmin_amp_7d", "humidity_proxy_7d",
    ],
}


# ── Grille cellules taille fixe ───────────────────────────────────────────────

def _deg_per_meter(lat_ref: float = LAT_REF):
    lat_rad = math.radians(lat_ref)
    return 1 / 111_000, 1 / (111_000 * math.cos(lat_rad))


def build_parcel_grid_fixed(geom: Polygon,
                             cell_size_m: float = CELL_SIZE_M,
                             lat_ref: float = LAT_REF) -> list[dict]:
    """
    Découpe une parcelle en cellules de taille fixe (cell_size_m × cell_size_m).
    Applique buffer(0) si la géométrie est invalide.
    Retourne une liste de dicts {row, col, cell_id, geometry, area_ha}.
    """
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.is_empty or not geom.is_valid:
        return []

    deg_lat, deg_lon = _deg_per_meter(lat_ref)
    cell_deg_lat = cell_size_m * deg_lat
    cell_deg_lon = cell_size_m * deg_lon

    minx, miny, maxx, maxy = geom.bounds
    n_cols = math.ceil((maxx - minx) / cell_deg_lon)
    n_rows = math.ceil((maxy - miny) / cell_deg_lat)

    cells = []
    for row in range(n_rows):
        for col in range(n_cols):
            x0 = minx + col * cell_deg_lon
            y0 = miny + row * cell_deg_lat
            cell_box  = box(x0, y0, x0 + cell_deg_lon, y0 + cell_deg_lat)
            cell_geom = geom.intersection(cell_box)
            if cell_geom.is_empty:
                continue
            area_ha = cell_geom.area * (111_000 ** 2) / 10_000
            cells.append({
                "row"     : row,
                "col"     : col,
                "cell_id" : f"R{row}C{col}",
                "geometry": cell_geom,
                "area_ha" : round(area_ha, 2),
            })
    return cells


# ── Indices par cellule ───────────────────────────────────────────────────────

def _zonal_stats_cell(geom, arr: np.ndarray, transform,
                      min_pixels: int = MIN_PIXELS_CELL) -> dict | None:
    try:
        mask   = rasterio.features.geometry_mask(
            [mapping(geom)], out_shape=arr.shape,
            transform=transform, invert=True,
        )
        pixels = arr[mask]
        pixels = pixels[~np.isnan(pixels)]
        pixels = pixels[(pixels >= -1.0) & (pixels <= 1.0)]
        if len(pixels) < min_pixels:
            return None
        return {
            "mean": float(np.mean(pixels)),
            "std" : float(np.std(pixels)),
            "p10" : float(np.percentile(pixels, 10)),
            "p90" : float(np.percentile(pixels, 90)),
            "n"   : len(pixels),
        }
    except Exception:
        return None


def compute_cell_indices(cell_geom, bands: np.ndarray,
                          transform) -> dict:
    """
    Calcule NDVI, NDWI, NDRE, EVI sur une cellule.
    Retourne un dict avec mean/std/p10 pour chaque indice disponible.
    Valeurs manquantes → NaN (géré par nan_to_num dans le scorer).

    Convention bandes (evalscript 7 bandes) :
        0=B04, 1=B03, 2=B02, 3=B05, 4=B08, 5=B8A, 6=B11
    """
    def _ndvi(b): return _ratio(b[4], b[0])
    def _ndwi(b): return _ratio(b[1], b[6])
    def _ndre(b): return _ratio(b[4], b[3])
    def _evi(b):
        denom = b[4] + 6*b[0] - 7.5*b[2] + 1
        denom[np.abs(denom) < 1e-10] = np.nan
        return np.clip(2.5 * (b[4] - b[0]) / denom, -1, 1)

    def _ratio(a, b):
        a, b = a.astype(float), b.astype(float)
        denom = a + b
        denom[denom == 0] = np.nan
        return (a - b) / denom

    result = {}
    for name, fn in [("ndvi", _ndvi), ("ndwi", _ndwi),
                     ("ndre", _ndre), ("evi",  _evi)]:
        try:
            arr   = fn(bands)
            stats = _zonal_stats_cell(cell_geom, arr, transform)
            if stats:
                result[f"{name}_mean"] = stats["mean"]
                result[f"{name}_std"]  = stats["std"]
                result[f"{name}_p10"]  = stats["p10"]
                result[f"{name}_p90"]  = stats["p90"]
                result[f"n_pixels_{name}"] = stats["n"]
            else:
                for s in ("mean", "std", "p10", "p90"):
                    result[f"{name}_{s}"] = np.nan
        except Exception:
            for s in ("mean", "std", "p10", "p90"):
                result[f"{name}_{s}"] = np.nan

    # Ratio NDRE/NDVI
    ndvi = result.get("ndvi_mean", np.nan)
    ndre = result.get("ndre_mean", np.nan)
    result["ndre_ndvi_ratio"] = (ndre / ndvi
                                  if (not np.isnan(ndvi) and ndvi != 0)
                                  else np.nan)
    return result


# ── Chargement modèle ─────────────────────────────────────────────────────────

def get_model(code_cultu: str, stress_type: str,
              models_dir: Path) -> dict:
    """
    Charge le modèle le plus adapté (spécifique culture > général).
    Lève FileNotFoundError si aucun modèle disponible.
    """
    index_path = models_dir / "models_index.csv"
    if not index_path.exists():
        raise FileNotFoundError("models_index.csv manquant.")

    index = pd.read_csv(index_path)
    index = index[index["stress_type"] == stress_type]

    for _, row in index.iterrows():
        if code_cultu in str(row["cultures_codes"]).split(","):
            data = joblib.load(models_dir / row["filename"])
            data.update({"model_name": row["model_name"], "source": "specific"})
            return data

    general = index[index["cultures_codes"] == "ALL"]
    if not general.empty:
        data = joblib.load(models_dir / general.iloc[0]["filename"])
        data["source"] = "fallback"
        return data

    raise FileNotFoundError(f"Aucun modèle {stress_type} pour {code_cultu}")


# ── Scorer une cellule ────────────────────────────────────────────────────────

def _sev_from_score(score: float, stress_type: str) -> str:
    """
    Seuils de sévérité.
    Conservés identiques au stress hydrique pour l'instant —
    à recalibrer après évaluation de chaque modèle.
    """
    if score < -0.12:
        return "critical"
    if score < -0.04:
        return "warning"
    return "normal"


def score_cell(cell_features: dict, model_data: dict,
               stress_type: str) -> tuple[float, str]:
    """
    Score une cellule avec un modèle chargé.
    Retourne (anomaly_score, severity).
    """
    model  = model_data["model"]
    scaler = model_data["scaler"]
    feats  = model_data.get("features", FEATURE_COLS[stress_type])

    X = np.array([cell_features.get(f, np.nan) for f in feats]).reshape(1, -1)
    X = np.nan_to_num(X, nan=0.0)
    X_sc  = scaler.transform(X)
    score = float(model.decision_function(X_sc)[0])
    sev   = _sev_from_score(score, stress_type)
    return score, sev


# ── Inférence parcelle complète par cellules ──────────────────────────────────

def infer_parcel_by_cells(
    geom        : Polygon,
    code_cultu  : str,
    bands       : np.ndarray,
    transform,
    meteo_feats : dict,
    acq_date,
    stress_type : str,
    models_dir  : Path,
    # Features temporelles optionnelles (NaN si non disponibles)
    prev_indices: dict | None = None,
    seasonal_ref: dict | None = None,
) -> dict:
    """
    Découpe la parcelle en cellules 100m, score chaque cellule,
    agrège en sévérité parcelle = max(sévérité cellules).

    Args:
        geom         : géométrie Shapely de la parcelle (EPSG:4326)
        code_cultu   : code culture RPG (ex: "BTH")
        bands        : array numpy (7, H, W) chargé depuis le TIF
        transform    : transform rasterio
        meteo_feats  : dict des features météo pré-calculées pour cette date
        acq_date     : date d'acquisition (date ou Timestamp)
        stress_type  : "HYDRIQUE" | "AZOTE" | "RAVAGEURS"
        models_dir   : Path vers le dossier des modèles
        prev_indices : dict {ndvi_mean, ndwi_mean, ...} de la date précédente
                       pour calculer les deltas (None si première acquisition)
        seasonal_ref : dict {ndvi_mean_ref, ndwi_mean_ref, ...} médiane saisonnière
                       (None si non disponible → deviation = 0.0)

    Returns:
        {
          "severity"      : str   — sévérité agrégée parcelle
          "worst_score"   : float — score de la cellule la plus anormale
          "n_cells"       : int   — nombre de cellules analysées
          "n_critical"    : int
          "n_warning"     : int
          "n_normal"      : int
          "pct_critical"  : float — % surface en critique
          "pct_warning"   : float
          "cells"         : list[dict] — détail par cellule (pour intra-parc)
          "model_source"  : str   — "specific" ou "fallback"
        }
    """
    # Chargement modèle (une fois pour toute la parcelle)
    try:
        model_data = get_model(code_cultu, stress_type, models_dir)
    except FileNotFoundError as e:
        return {"severity": "unknown", "error": str(e),
                "n_cells": 0, "cells": []}

    # Grille cellules
    cells = build_parcel_grid_fixed(geom)
    if not cells:
        return {"severity": "unknown", "error": "Géométrie invalide",
                "n_cells": 0, "cells": []}

    day_of_year = (pd.Timestamp(acq_date).timetuple().tm_yday
                   if acq_date else 0)

    cell_results = []
    total_area   = sum(c["area_ha"] for c in cells)

    for cell in cells:
        # Indices satellite sur la cellule
        idx = compute_cell_indices(cell["geometry"], bands, transform)
        if np.isnan(idx.get("ndvi_mean", np.nan)):
            continue  # cellule sans pixels valides

        # Features temporelles
        # Delta : valeur actuelle - valeur date précédente
        # Deviation : valeur actuelle - médiane saisonnière culture
        for indice in ("ndvi", "ndwi", "ndre", "evi"):
            mean_key = f"{indice}_mean"
            # Delta
            if prev_indices and mean_key in prev_indices:
                idx[f"{indice}_delta"] = (idx.get(mean_key, np.nan)
                                          - prev_indices[mean_key])
            else:
                idx[f"{indice}_delta"] = np.nan
            # Deviation
            ref_key = f"{mean_key}_ref"
            if seasonal_ref and ref_key in seasonal_ref:
                idx[f"{indice}_deviation"] = (idx.get(mean_key, np.nan)
                                              - seasonal_ref[ref_key])
            else:
                idx[f"{indice}_deviation"] = 0.0  # neutre si pas de référence

        # Assemblage features complètes
        features = {
            **idx,
            "day_of_year": day_of_year,
            **meteo_feats,
        }

        score, sev = score_cell(features, model_data, stress_type)

        cell_results.append({
            "cell_id"  : cell["cell_id"],
            "row"      : cell["row"],
            "col"      : cell["col"],
            "geometry" : cell["geometry"],
            "area_ha"  : cell["area_ha"],
            "score"    : score,
            "severity" : sev,
            **{k: v for k, v in idx.items()
               if k.endswith("_mean") or k.endswith("_std")},
        })

    if not cell_results:
        return {"severity": "unknown", "error": "Aucune cellule valide",
                "n_cells": 0, "cells": []}

    # ── Agrégation : sévérité parcelle = max(sévérité cellules) ─────────────
    def sev_rank(s): return SEV_ORDER.get(s, -1)

    worst_cell  = max(cell_results, key=lambda c: sev_rank(c["severity"]))
    parcel_sev  = worst_cell["severity"]
    worst_score = worst_cell["score"]

    n_critical = sum(1 for c in cell_results if c["severity"] == "critical")
    n_warning  = sum(1 for c in cell_results if c["severity"] == "warning")
    n_normal   = sum(1 for c in cell_results if c["severity"] == "normal")

    area_critical = sum(c["area_ha"] for c in cell_results
                        if c["severity"] == "critical")
    area_warning  = sum(c["area_ha"] for c in cell_results
                        if c["severity"] == "warning")

    return {
        "severity"    : parcel_sev,
        "worst_score" : worst_score,
        "n_cells"     : len(cell_results),
        "n_critical"  : n_critical,
        "n_warning"   : n_warning,
        "n_normal"    : n_normal,
        "pct_critical": area_critical / total_area * 100 if total_area else 0,
        "pct_warning" : area_warning  / total_area * 100 if total_area else 0,
        "cells"       : cell_results,
        "model_source": model_data.get("source", "unknown"),
        "model_name"  : model_data.get("model_name", "general"),
    }
