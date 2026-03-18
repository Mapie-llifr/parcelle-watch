"""
src/ingestion/meteo.py
----------------------
Téléchargement des données météorologiques via l'API Open-Meteo.

Open-Meteo est gratuit, sans clé API, avec un historique depuis 1940.
Documentation : https://open-meteo.com/en/docs/historical-weather-api

Variables récupérées (utiles pour la prédiction de rendement) :
    - temperature_2m_max / min    : températures quotidiennes
    - precipitation_sum           : cumul de précipitations
    - et0_fao_evapotranspiration  : évapotranspiration de référence (ETP)
    - shortwave_radiation_sum     : rayonnement solaire
    - windspeed_10m_max           : vitesse du vent

Dépendances : openmeteo-requests, pandas, requests-cache, retry-requests
"""

from datetime import date
from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from loguru import logger
from retry_requests import retry

# Variables météo à récupérer
DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "shortwave_radiation_sum",
    "windspeed_10m_max",
]


def get_openmeteo_client() -> openmeteo_requests.Client:
    """Crée un client Open-Meteo avec cache et retry automatique."""
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def fetch_historical_weather(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Télécharge les données météo historiques pour un point géographique.

    Args:
        latitude: Latitude du point (centre de la parcelle)
        longitude: Longitude du point
        start_date: Date de début
        end_date: Date de fin

    Returns:
        DataFrame avec une ligne par jour et une colonne par variable météo
    """
    client = get_openmeteo_client()

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": DAILY_VARIABLES,
        "timezone": "Europe/Paris",
    }

    logger.info(f"Récupération météo ({latitude:.4f}, {longitude:.4f}) "
                f"du {start_date} au {end_date}...")

    responses = client.weather_api(
        "https://archive-api.open-meteo.com/v1/archive", params=params
    )
    response = responses[0]

    daily = response.Daily()
    data = {"date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left",
    )}

    for i, var in enumerate(DAILY_VARIABLES):
        data[var] = daily.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data)
    df["date"] = df["date"].dt.date
    logger.success(f"{len(df)} jours de données météo récupérés")
    return df


def fetch_forecast_weather(
    latitude: float,
    longitude: float,
    days_ahead: int = 7,
) -> pd.DataFrame:
    """
    Récupère les prévisions météo sur les prochains jours.

    Args:
        latitude: Latitude
        longitude: Longitude
        days_ahead: Nombre de jours de prévision (max 16)

    Returns:
        DataFrame de prévisions
    """
    client = get_openmeteo_client()

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": DAILY_VARIABLES,
        "forecast_days": min(days_ahead, 16),
        "timezone": "Europe/Paris",
    }

    logger.info(f"Récupération prévisions {days_ahead}j ({latitude:.4f}, {longitude:.4f})...")

    responses = client.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
    response = responses[0]

    daily = response.Daily()
    data = {"date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left",
    )}

    for i, var in enumerate(DAILY_VARIABLES):
        data[var] = daily.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data)
    df["date"] = df["date"].dt.date
    return df


def save_weather_data(df: pd.DataFrame, output_path: Path) -> None:
    """Sauvegarde les données météo en CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Données météo sauvegardées : {output_path}")
