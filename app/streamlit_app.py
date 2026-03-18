"""
app/streamlit_app.py
--------------------
Interface utilisateur principale de Parcelle Watch.

Navigation :
    📡 Mes Parcelles   — sélection et visualisation des parcelles (RPG ou dessin manuel)
    🌿 Indices         — cartes NDVI/NDRE/NDWI interactives + série temporelle
    🚨 Alertes         — anomalies détectées, sévérité, localisation
    📈 Rendement       — prévision de rendement avec intervalles de confiance
    📄 Rapport         — génération et téléchargement du rapport PDF hebdo

Lancement :
    poetry run streamlit run app/streamlit_app.py
"""

import streamlit as st

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Parcelle Watch",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("docs/logo.png", width=180) if False else st.title("🛰️ Parcelle Watch")
    st.caption("Surveillance satellite de vos parcelles")
    st.divider()

    page = st.radio(
        "Navigation",
        options=[
            "📡 Mes Parcelles",
            "🌿 Indices & Végétation",
            "🚨 Alertes",
            "📈 Prévision Rendement",
            "📄 Rapport PDF",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Données : Sentinel-2 · Open-Meteo · RPG")

# ── Pages ─────────────────────────────────────────────────────────────────────

if page == "📡 Mes Parcelles":
    st.header("📡 Mes Parcelles")
    st.info("🚧 En construction — Étape 1 : sélection de parcelle via RPG ou coordonnées GPS")
    # TODO:
    # - Carte Folium pour sélectionner / dessiner une parcelle
    # - Chargement depuis RPG (GeoJSON)
    # - Saisie manuelle de coordonnées
    # - Lancement du téléchargement Sentinel-2

elif page == "🌿 Indices & Végétation":
    st.header("🌿 Indices de Végétation")
    st.info("🚧 En construction — Étape 2 : cartes NDVI + séries temporelles")
    # TODO:
    # - Sélecteur d'indice (NDVI / NDRE / NDWI / EVI)
    # - Carte Folium colorée par valeur d'indice
    # - Graphique temporel Plotly (évolution sur la saison)
    # - Comparaison avec saison précédente

elif page == "🚨 Alertes":
    st.header("🚨 Alertes Détectées")
    st.info("🚧 En construction — Étape 2 : détection d'anomalies Isolation Forest")
    # TODO:
    # - Tableau des alertes récentes (date, zone, sévérité, indice concerné)
    # - Carte des zones en anomalie
    # - Graphique : score d'anomalie dans le temps

elif page == "📈 Prévision Rendement":
    st.header("📈 Prévision de Rendement")
    st.info("🚧 En construction — Étape 4 : prédiction XGBoost")
    # TODO:
    # - Sélecteur culture + département
    # - Affichage prédiction centrale + intervalle de confiance
    # - Graphique comparaison avec moyenne régionale Agreste
    # - Importance des features

elif page == "📄 Rapport PDF":
    st.header("📄 Rapport Hebdomadaire")
    st.info("🚧 En construction — Étape 3 : génération PDF automatique")
    # TODO:
    # - Bouton "Générer le rapport"
    # - Aperçu des sections incluses
    # - Bouton de téléchargement
    # - Option d'envoi par email (futur)
