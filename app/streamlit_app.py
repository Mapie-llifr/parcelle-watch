"""
app/streamlit_app.py
--------------------
Point d'entree de l'application Parcelle Watch.
Streamlit multi-pages : chaque fichier dans app/pages/ est une page.

Pages disponibles :
  1_Mes_Parcelles.py  -> selection des parcelles sur carte
  2_Alertes.py        -> carte des alertes + tableau
  3_Rapport_PDF.py    -> generation et telechargement PDF

Lancement :
    poetry run streamlit run app/streamlit_app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Parcelle Watch",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛰️ Parcelle Watch")
st.markdown(
    "Surveillance satellite des parcelles agricoles — "
    "detection precoce de stress hydrique, risque ravageurs, prevision de rendement."
)
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🗺️ Mes Parcelles")
    st.markdown("Selectionnez vos parcelles agricoles sur la carte interactive.")
    if st.button("Acceder", key="btn_parcelles", use_container_width=True):
        st.switch_page("pages/1_Mes_Parcelles.py")

with col2:
    st.markdown("### 🚨 Alertes")
    st.markdown("Visualisez les alertes de stress hydrique par parcelle et par date.")
    if st.button("Acceder", key="btn_alertes", use_container_width=True):
        st.switch_page("pages/2_Alertes.py")

with col3:
    st.markdown("### 📄 Rapport PDF")
    st.markdown("Generez et telechargez le rapport hebdomadaire de vos parcelles.")
    if st.button("Acceder", key="btn_rapport", use_container_width=True):
        st.switch_page("pages/3_Rapport_PDF.py")

st.divider()
st.caption(
    "Donnees : Sentinel-2 (ESA/Copernicus) · Open-Meteo · "
    "Registre Parcellaire Graphique (IGN) · Modele Isolation Forest"
)
