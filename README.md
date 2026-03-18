# 🛰️ Parcelle Watch

> Surveillance satellite open source des parcelles agricoles — détection précoce de stress végétatif et prévision de rendement.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-en%20développement-orange)

---

## 🎯 Objectif

Parcelle Watch est un outil **gratuit, open source et local** qui permet à un agriculteur de :

- 📡 Récupérer automatiquement les images satellite Sentinel-2 sur ses parcelles
- 🌿 Calculer des indices de végétation (NDVI, NDRE, NDWI) sur la durée
- 🚨 Détecter automatiquement les zones de stress (hydrique, azoté, fongique)
- 📈 Prédire le rendement en fin de saison
- 📄 Générer un rapport PDF hebdomadaire clés-en-main

**Cible :** céréaliers, viticulteurs, arboriculteurs — exploitations > 2 ha.  
**Contrainte :** fonctionne entièrement en local, sans abonnement, sans cloud payant.

---

## 🏗️ Architecture

```
parcelle-watch/
│
├── data/
│   ├── raw/              # Images Sentinel-2 brutes (non versionnées)
│   ├── processed/        # Indices calculés, séries temporelles
│   └── external/         # RPG, météo Open-Meteo, rendements Agreste
│
├── notebooks/            # Exploration et prototypage (étapes 1 & 2)
│   ├── 01_sentinel2_exploration.ipynb
│   ├── 02_indices_calcul.ipynb
│   ├── 03_anomaly_detection.ipynb
│   └── 04_yield_prediction.ipynb
│
├── src/
│   ├── ingestion/        # Téléchargement Sentinel-2, RPG, météo
│   ├── indices/          # Calcul NDVI, NDRE, NDWI, EVI
│   ├── models/           # Modèles ML : détection anomalies + prédiction rendement
│   ├── viz/              # Cartes Folium, graphiques temporels
│   └── report/           # Génération PDF automatique
│
├── app/
│   └── streamlit_app.py  # Interface utilisateur principale
│
├── tests/                # Tests unitaires
├── docs/                 # Documentation technique
├── pyproject.toml        # Dépendances Poetry
└── .env.example          # Variables d'environnement (credentials Copernicus)
```

---

## 🔢 Stack technique

| Domaine | Librairies |
|---------|-----------|
| Images satellite | `sentinelhub`, `rasterio`, `shapely` |
| Calcul indices | `numpy`, `xarray` |
| Données géo | `geopandas`, `folium` |
| ML anomalies | `scikit-learn` (Isolation Forest) |
| ML rendement | `xgboost`, `scikit-learn` |
| Météo | `openmeteo-requests` |
| Dashboard | `streamlit`, `plotly` |
| Rapport PDF | `reportlab` ou `weasyprint` |
| Environnement | `python 3.11+`, `poetry` |

---

## 🗺️ Données utilisées

| Source | Contenu | Accès |
|--------|---------|-------|
| [Copernicus / Sentinel-2](https://dataspace.copernicus.eu/) | Images satellite multi-bandes, 10m/pixel, depuis 2015 | Gratuit, inscription requise |
| [Open-Meteo](https://open-meteo.com/) | Météo historique et prévisionnelle | Gratuit, sans clé |
| [RPG - data.gouv.fr](https://www.data.gouv.fr/fr/datasets/registre-parcellaire-graphique-rpg/) | Contours des parcelles agricoles françaises | Gratuit, open data |
| [Agreste](https://agreste.agriculture.gouv.fr/) | Rendements historiques par culture/département | Gratuit, open data |

---

## 🚀 Installation

```bash
git clone https://github.com/TON_PSEUDO/parcelle-watch.git
cd parcelle-watch

# Installer les dépendances
poetry install

# Configurer les credentials Copernicus
cp .env.example .env
# → Remplir COPERNICUS_CLIENT_ID et COPERNICUS_CLIENT_SECRET

# Lancer l'app
poetry run streamlit run app/streamlit_app.py
```

---

## 📅 Roadmap

- [x] Architecture & repo
- [ ] **Étape 1** — Ingestion Sentinel-2 + calcul NDVI sur parcelle réelle
- [ ] **Étape 2** — Séries temporelles + détection d'anomalies (Isolation Forest)
- [ ] **Étape 3** — Dashboard Streamlit interactif + rapport PDF
- [ ] **Étape 4** — Prédiction de rendement (XGBoost + météo)

---

## 📸 Screenshots

*À venir — étape 3*

---

## 🧑‍💻 Auteur

Projet portfolio — data scientist en recherche d'emploi. 
N'hésitez pas à ouvrir une issue ou à me contacter sur LinkedIn https://www.linkedin.com/in/mpdiquero/.

---

## 📄 Licence

MIT
