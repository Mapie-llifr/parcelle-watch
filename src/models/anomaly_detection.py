"""
src/models/anomaly_detection.py
--------------------------------
Détection d'anomalies sur séries temporelles d'indices de végétation.

Approche :
    - Modèle non supervisé : pas besoin de labels terrain
    - Features : valeurs NDVI/NDRE/NDWI + jour de l'année + météo
    - Algorithme : Isolation Forest (scikit-learn)
    - Granularité : par pixel ou par zone homogène de la parcelle

Pipeline :
    1. Construire la série temporelle des indices sur la parcelle
    2. Calculer les features (valeur courante, écart à la moyenne historique,
       tendance sur 2 semaines, variables météo associées)
    3. Entraîner Isolation Forest sur les données d'une ou plusieurs saisons
    4. Scorer les nouvelles acquisitions → score d'anomalie par pixel
    5. Agréger par zone et lever des alertes si seuil dépassé

Dépendances : scikit-learn, numpy, pandas
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class VegetationAnomalyDetector:
    """
    Détecteur d'anomalies végétatives basé sur Isolation Forest.

    Attributes:
        contamination: Proportion estimée d'anomalies (0.05 à 0.15 recommandé)
        model: Instance IsolationForest entraînée
        scaler: StandardScaler pour normalisation des features
    """

    FEATURE_COLUMNS = [
        "ndvi",
        "ndre",
        "ndwi",
        "ndvi_deviation",       # écart à la moyenne historique même période
        "ndvi_trend_14d",       # pente sur les 14 derniers jours
        "day_of_year",          # saisonnalité
        "temperature_2m_max",
        "precipitation_sum",
        "et0_fao_evapotranspiration",
    ]

    def __init__(self, contamination: float = 0.08, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model: IsolationForest | None = None
        self.scaler: StandardScaler | None = None

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Construit les features à partir d'un DataFrame de série temporelle.

        Args:
            df: DataFrame avec colonnes [date, ndvi, ndre, ndwi, + météo]

        Returns:
            DataFrame de features prêt pour l'entraînement ou la prédiction
        """
        df = df.copy().sort_values("date")
        df["date"] = pd.to_datetime(df["date"])
        df["day_of_year"] = df["date"].dt.dayofyear

        # Écart à la moyenne historique (même semaine de l'année)
        df["week"] = df["date"].dt.isocalendar().week
        weekly_mean = df.groupby("week")["ndvi"].transform("mean")
        df["ndvi_deviation"] = df["ndvi"] - weekly_mean

        # Tendance NDVI sur 14 jours (pente de régression linéaire glissante)
        df["ndvi_trend_14d"] = (
            df["ndvi"]
            .rolling(window=4, min_periods=2)  # ~14j avec revisite 5j
            .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
        )

        # Sélection et nettoyage
        features = df[self.FEATURE_COLUMNS].dropna()
        logger.info(f"Features construites : {len(features)} observations, "
                    f"{len(self.FEATURE_COLUMNS)} variables")
        return features

    def fit(self, df: pd.DataFrame) -> "VegetationAnomalyDetector":
        """
        Entraîne le modèle sur les données historiques.

        Args:
            df: DataFrame de série temporelle (plusieurs saisons recommandées)

        Returns:
            self (pour chaînage)
        """
        features = self.build_features(df)

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(features)

        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=200,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X)

        logger.success(f"Modèle entraîné sur {len(features)} observations "
                       f"(contamination={self.contamination})")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score de nouvelles observations.

        Args:
            df: DataFrame de série temporelle à scorer

        Returns:
            DataFrame original enrichi de :
                - anomaly_score : score brut (plus négatif = plus anormal)
                - is_anomaly    : booléen (True si anomalie détectée)
                - severity      : 'normal' | 'warning' | 'critical'
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Le modèle doit être entraîné avant predict(). Appeler fit() d'abord.")

        features = self.build_features(df)
        X = self.scaler.transform(features)

        scores = self.model.decision_function(X)    # plus négatif = plus anormal
        predictions = self.model.predict(X)         # -1 = anomalie, 1 = normal

        result = df.loc[features.index].copy()
        result["anomaly_score"] = scores
        result["is_anomaly"] = predictions == -1
        result["severity"] = pd.cut(
            scores,
            bins=[-np.inf, -0.15, -0.05, np.inf],
            labels=["critical", "warning", "normal"],
        )

        n_anomalies = result["is_anomaly"].sum()
        logger.info(f"Prédiction : {n_anomalies}/{len(result)} anomalies détectées")
        return result

    def save(self, path: Path) -> None:
        """Sauvegarde le modèle entraîné (joblib)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)
        logger.info(f"Modèle sauvegardé : {path}")

    @classmethod
    def load(cls, path: Path) -> "VegetationAnomalyDetector":
        """Charge un modèle sauvegardé."""
        obj = cls()
        saved = joblib.load(path)
        obj.model = saved["model"]
        obj.scaler = saved["scaler"]
        logger.info(f"Modèle chargé : {path}")
        return obj
