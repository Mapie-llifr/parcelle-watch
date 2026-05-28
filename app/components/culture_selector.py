"""
app/components/culture_selector.py
------------------------------------
Composant réutilisable de sélection de culture.

Usage :
    from components.culture_selector import culture_selector

    code = culture_selector(key="dessin", default="BTH")
    # retourne le code culture sélectionné (str)
"""

from pathlib import Path
import pandas as pd
import streamlit as st

# ── Cultures courantes affichées directement dans le selectbox ───────────────
CULTURES_COURANTES = [
    "BTH",  # Blé tendre hiver
    "MIS",  # Maïs grain
    "CZH",  # Colza hiver
    "ORH",  # Orge hiver
    "ORP",  # Orge printemps
    "VRC",  # Vigne raisins de cuve
    "PPH",  # Prairie permanente
    "BFS",  # Betterave
    "FVL",  # Féverole
    "TRN",  # Tournesol 
    "SOJ",  # Soja 
    "JAC",  # Jachère
]

SENTINEL = "__AUTRE__"


@st.cache_data
def load_cultures_csv(csv_path: str) -> pd.DataFrame:
    """Charge le CSV des cultures. Résultat mis en cache."""
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8", dtype=str)
    # Normalisation des noms de colonnes (robustesse)
    df.columns = [c.strip() for c in df.columns]
    # Renommage flexible selon ce qu'on trouve
    col_map = {}
    for c in df.columns:
        if c.lower() in ("code", "code_cultu", "code culture"):
            col_map[c] = "code"
        elif c.lower() in ("libellé", "libelle", "label", "désignation", "designation"):
            col_map[c] = "libelle"
    df = df.rename(columns=col_map)
    df["code"]    = df["code"].str.strip().str.upper()
    df["libelle"] = df["libelle"].str.strip()
    return df.dropna(subset=["code", "libelle"])


def _get_all_cultures(csv_path: str) -> dict[str, str]:
    """Retourne {code: libelle} pour toutes les cultures du CSV."""
    df = load_cultures_csv(csv_path)
    return dict(zip(df["code"], df["libelle"]))


def _get_courantes(all_cultures: dict) -> dict[str, str]:
    """Filtre les cultures courantes présentes dans le CSV."""
    return {k: v for k, v in all_cultures.items() if k in CULTURES_COURANTES}


def culture_selector(
    key: str,
    csv_path: str | Path,
    default: str = "BTH",
    label: str = "Culture",
) -> str:
    """
    Composant de sélection de culture à deux niveaux.

    - Selectbox avec les cultures courantes + option "Autre culture..."
    - Si "Autre culture..." : text_input de recherche (code ou libellé)
      qui filtre dynamiquement le CSV et affiche un second selectbox.

    Args:
        key:       Clé unique Streamlit (ex: "dessin", "gps", "edit_pid123")
        csv_path:  Chemin vers le CSV des cultures (sep=";")
        default:   Code culture présélectionné
        label:     Libellé affiché au-dessus du selectbox

    Returns:
        Code culture sélectionné (str)
    """
    csv_path    = str(csv_path)
    all_cult    = _get_all_cultures(csv_path)
    courantes   = _get_courantes(all_cult)

    # Options du selectbox principal : cultures courantes + séparateur + Autre
    options_main = list(courantes.keys()) + [SENTINEL]

    def fmt_main(code):
        if code == SENTINEL:
            return "✦ Autre culture..."
        return f"{code} — {courantes.get(code, code)}"

    # Index par défaut : la culture courante si elle est dans la liste,
    # sinon on prépositionne sur SENTINEL
    if default in courantes:
        default_idx = options_main.index(default)
    else:
        default_idx = options_main.index(SENTINEL)

    selected_main = st.selectbox(
        label,
        options=options_main,
        index=default_idx,
        format_func=fmt_main,
        key=f"cult_main_{key}",
    )

    # ── Mode "Autre" : recherche dans le CSV complet ─────────────────────────
    if selected_main == SENTINEL:
        search = st.text_input(
            "Rechercher (code ou libellé)",
            placeholder="ex: TRN  ou  tournesol",
            key=f"cult_search_{key}",
        )

        query = search.strip().upper()
        if query:
            filtered = {
                code: lib
                for code, lib in all_cult.items()
                if query in code.upper() or query in lib.upper()
            }
        else:
            filtered = all_cult  # Tout afficher si champ vide

        if not filtered:
            st.warning("Aucune culture trouvée pour cette recherche.")
            # Retourner la valeur en session si elle existe, sinon BTH
            return st.session_state.get(f"cult_result_{key}", "BTH")

        options_other = list(filtered.keys())

        # Conserver la sélection précédente si elle est toujours dans les résultats
        prev = st.session_state.get(f"cult_result_{key}", options_other[0])
        other_idx = options_other.index(prev) if prev in options_other else 0

        selected_other = st.selectbox(
            f"Résultats ({len(filtered)})",
            options=options_other,
            index=other_idx,
            format_func=lambda c: f"{c} — {filtered[c]}",
            key=f"cult_other_{key}",
        )
        st.session_state[f"cult_result_{key}"] = selected_other
        return selected_other

    # ── Mode normal : culture courante sélectionnée ──────────────────────────
    st.session_state[f"cult_result_{key}"] = selected_main
    return selected_main
