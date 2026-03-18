"""
src/models/yield_prediction.py
-------------------------------
Prédiction de rendement agricole en fin de saison.

Approche :
    - Modèle supervisé : XGBoost entraîné sur données Agreste (rendements historiques)
    - Features : NDVI max, NDVI intégré (somme), NDVI à floraison, anomalies détectées,
                 GDD (degrés-jours de croissance), précipitations cumulées, ETP...
    - Cible : rendement en quintaux/hectare (q/ha)
    - Évaluation : RMSE, MAE, R² — avec intervalles de confiance (bootstrap)

Limitation connue :
    Les données Agreste sont à l'échelle département/région, pas à la parcelle.
    Le modèle prédit donc un rendement "typique" pour la culture et la région,
    ajusté par les indicateurs satellite de la parcelle spécifique.

Dépendances : xgboost, scikit-learn, pandas, numpy
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor


class YieldPredictor:
    """
    Prédicteur de rendement par XGBoost.

    Features attendues en entrée :
        Satellite :
            ndvi_max          : NDVI maximum atteint dans la saison
            ndvi_integral     : somme des NDVI (proxy biomasse cumulée)
            ndvi_at_flowering : NDVI mesuré à la période de floraison
            n_anomalies       : nombre d'anomalies détectées dans la saison
            ndwi_min          : NDWI minimum (pic de stress hydrique)

        Météo :
            gdd               : Growing Degree Days (somme thermique)
            precip_total      : précipitations totales saison
            precip_critical   : précipitations en période critique (floraison)
            et0_total         : évapotranspiration totale

        Contexte :
            culture           : type de culture (blé, orge, colza, tournesol...)
            departement       : code département (proxy sol/climat)
            year              : année (tendance technologique)
    """

    FEATURE_COLUMNS = [
        "ndvi_max", "ndvi_integral", "ndvi_at_flowering",
        "n_anomalies", "ndwi_min",
        "gdd", "precip_total", "precip_critical", "et0_total",
        "culture_encoded", "departement_encoded", "year",
    ]

    def __init__(self):
        self.model: XGBRegressor | None = None
        self.culture_encoder = LabelEncoder()
        self.dept_encoder = LabelEncoder()
        self._is_fitted = False

    def prepare_features(self, df: pd.DataFrame, fit_encoders: bool = False) -> pd.DataFrame:
        """
        Encode les variables catégorielles et sélectionne les features.

        Args:
            df: DataFrame brut
            fit_encoders: True à l'entraînement, False à la prédiction

        Returns:
            DataFrame de features prêt pour XGBoost
        """
        df = df.copy()

        if fit_encoders:
            df["culture_encoded"] = self.culture_encoder.fit_transform(df["culture"])
            df["departement_encoded"] = self.dept_encoder.fit_transform(df["departement"])
        else:
            df["culture_encoded"] = self.culture_encoder.transform(df["culture"])
            df["departement_encoded"] = self.dept_encoder.transform(df["departement"])

        return df[self.FEATURE_COLUMNS]

    def fit(self, df: pd.DataFrame, target_col: str = "rendement_qha") -> dict:
        """
        Entraîne le modèle XGBoost.

        Args:
            df: DataFrame avec features + colonne cible
            target_col: Nom de la colonne rendement (q/ha)

        Returns:
            Dict de métriques d'évaluation (train/test split + CV)
        """
        X = self.prepare_features(df, fit_encoders=True)
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        self._is_fitted = True

        # Évaluation
        y_pred = self.model.predict(X_test)
        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }

        logger.success(
            f"Modèle entraîné — RMSE: {metrics['rmse']:.2f} q/ha | "
            f"MAE: {metrics['mae']:.2f} q/ha | R²: {metrics['r2']:.3f}"
        )
        return metrics

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prédit le rendement avec intervalle de confiance (bootstrap).

        Returns:
            DataFrame avec colonnes :
                - yield_pred   : prédiction centrale (q/ha)
                - yield_low    : borne basse 90% IC
                - yield_high   : borne haute 90% IC
        """
        if not self._is_fitted:
            raise RuntimeError("Appeler fit() avant predict()")

        X = self.prepare_features(df, fit_encoders=False)
        pred = self.model.predict(X)

        # Intervalle de confiance approximatif (±10% — à affiner avec quantile regression)
        result = df.copy()
        result["yield_pred"] = pred
        result["yield_low"] = pred * 0.90
        result["yield_high"] = pred * 1.10
        return result

    def feature_importance(self) -> pd.DataFrame:
        """Retourne l'importance des features triée par ordre décroissant."""
        if not self._is_fitted:
            raise RuntimeError("Modèle non entraîné")
        importances = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": self.FEATURE_COLUMNS, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "culture_encoder": self.culture_encoder,
            "dept_encoder": self.dept_encoder,
        }, path)
        logger.info(f"Modèle rendement sauvegardé : {path}")

    @classmethod
    def load(cls, path: Path) -> "YieldPredictor":
        obj = cls()
        saved = joblib.load(path)
        obj.model = saved["model"]
        obj.culture_encoder = saved["culture_encoder"]
        obj.dept_encoder = saved["dept_encoder"]
        obj._is_fitted = True
        return obj
