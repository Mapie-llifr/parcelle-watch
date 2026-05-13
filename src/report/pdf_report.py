"""
src/report/pdf_report.py
------------------------
Génération du rapport PDF hebdomadaire Parcelle Watch.

Contenu du rapport :
    Page 1 — En-tête + résumé zone (date, nb parcelles, nb alertes)
    Page 2 — Tableau des alertes (critical + warning) triées par sévérité
    Page 3 — Graphique NDVI/NDWI temporel (toutes parcelles)
    Page 4 — Top 5 parcelles les plus en stress

Dépendances : reportlab, matplotlib, pandas
"""

from datetime import date
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # pas de display, génération en mémoire
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from loguru import logger
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Palette couleurs ──────────────────────────────────────────────────────────
COLOR_GREEN    = colors.HexColor("#2d6a4f")
COLOR_ORANGE   = colors.HexColor("#e07b00")
COLOR_RED      = colors.HexColor("#b5292a")
COLOR_DARK     = colors.HexColor("#1a1a2e")
COLOR_LIGHT_BG = colors.HexColor("#f8f9fa")
COLOR_BORDER   = colors.HexColor("#dee2e6")

SEV_COLORS = {
    "critical": COLOR_RED,
    "warning":  COLOR_ORANGE,
    "normal":   COLOR_GREEN,
}


# ── Styles texte ──────────────────────────────────────────────────────────────
def _get_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontSize=22,
            textColor=COLOR_DARK,
            spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontSize=11,
            textColor=colors.HexColor("#555555"),
            spaceAfter=16,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontSize=13,
            textColor=COLOR_DARK,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#666666"),
        ),
        "alert_critical": ParagraphStyle(
            "alert_critical",
            parent=base["Normal"],
            fontSize=9,
            textColor=COLOR_RED,
        ),
        "alert_warning": ParagraphStyle(
            "alert_warning",
            parent=base["Normal"],
            fontSize=9,
            textColor=COLOR_ORANGE,
        ),
    }


# ── Graphique matplotlib → bytes ──────────────────────────────────────────────
def _make_timeseries_chart(df: pd.DataFrame) -> BytesIO:
    """Graphique NDVI/NDWI moyen toutes parcelles confondues."""
    ts = (
        df.groupby("date")[["ndvi_mean", "ndwi_mean"]]
        .mean()
        .reset_index()
    )
    ts["date"] = pd.to_datetime(ts["date"])

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 4), sharex=True)
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_facecolor("#f8f9fa")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    axes[0].plot(ts["date"], ts["ndvi_mean"],
                 color="#2d6a4f", linewidth=2, marker="o", markersize=4)
    axes[0].fill_between(ts["date"], ts["ndvi_mean"], alpha=0.1, color="#2d6a4f")
    axes[0].set_ylabel("NDVI moyen", fontsize=8)
    axes[0].axhline(0.4, color="#aaaaaa", linestyle="--", linewidth=0.8)
    axes[0].set_title("Évolution NDVI / NDWI — Zone complète", fontsize=9, pad=8)

    axes[1].plot(ts["date"], ts["ndwi_mean"],
                 color="#1d6fa4", linewidth=2, marker="o", markersize=4)
    axes[1].fill_between(ts["date"], ts["ndwi_mean"], alpha=0.1, color="#1d6fa4")
    axes[1].axhline(-0.3, color="#e07b00", linestyle="--", linewidth=0.8,
                    label="Seuil stress")
    axes[1].set_ylabel("NDWI moyen", fontsize=8)
    axes[1].legend(fontsize=7)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    plt.tight_layout(pad=1.2)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _make_top5_chart(df_last: pd.DataFrame) -> BytesIO:
    """Barplot des 5 parcelles avec le score d'anomalie le plus bas (plus critique)."""
    top5 = (
        df_last[df_last["is_anomaly"]]
        .nsmallest(5, "anomaly_score")[
            ["parcelle_id", "code_cultu", "anomaly_score", "ndwi_mean"]
        ]
    )
    if top5.empty:
        # Pas d'anomalies — graphique vide
        fig, ax = plt.subplots(figsize=(7, 2))
        ax.text(0.5, 0.5, "Aucune anomalie détectée",
                ha="center", va="center", fontsize=11, color="#555")
        ax.axis("off")
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf

    labels = [f"{row.parcelle_id}\n({row.code_cultu})" for _, row in top5.iterrows()]
    bar_colors = [
        "#b5292a" if s < -0.12 else "#e07b00"
        for s in top5["anomaly_score"]
    ]

    fig, ax = plt.subplots(figsize=(7, 2.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")
    ax.spines[["top", "right"]].set_visible(False)

    bars = ax.barh(labels, top5["anomaly_score"], color=bar_colors, height=0.5)
    ax.axvline(0, color="#aaaaaa", linewidth=0.8)
    ax.set_xlabel("Score anomalie (plus négatif = plus critique)", fontsize=8)
    ax.set_title("Top 5 parcelles les plus en stress", fontsize=9, pad=8)
    ax.tick_params(labelsize=8)

    plt.tight_layout(pad=1.0)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ── Générateur principal ──────────────────────────────────────────────────────
def generate_report(
    df_anomalies: pd.DataFrame,
    report_date: date,
    zone_name: str = "Brie française",
    output_path: Path | None = None,
) -> bytes:
    """
    Génère le rapport PDF hebdomadaire.

    Args:
        df_anomalies : DataFrame complet (sortie du notebook 04)
        report_date  : Date de référence du rapport
        zone_name    : Nom affiché de la zone
        output_path  : Si fourni, sauvegarde le PDF sur disque en plus

    Returns:
        Contenu PDF en bytes (pour st.download_button)
    """
    styles = _get_styles()
    buf = BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    story = []
    W = A4[0] - 4 * cm  # largeur utile

    # ── Filtres pour la date du rapport ──────────────────────────────────────
    df_anomalies["date"] = pd.to_datetime(df_anomalies["date"])
    report_ts = pd.Timestamp(report_date)

    # Trouver la date disponible la plus proche de report_date
    available_dates = df_anomalies["date"].unique()
    closest_date = min(available_dates, key=lambda d: abs(d - report_ts))
    df_last = df_anomalies[df_anomalies["date"] == closest_date].copy()

    n_parcelles  = len(df_last)
    n_anomalies  = int(df_last["is_anomaly"].sum())
    n_critical   = int((df_last["severity"] == "critical").sum())
    n_warning    = int((df_last["severity"] == "warning").sum())

    # ── Page 1 : En-tête + résumé ─────────────────────────────────────────────
    story.append(Paragraph("🛰️ Parcelle Watch", styles["title"]))
    story.append(Paragraph(
        f"Rapport hebdomadaire — {zone_name} — "
        f"{closest_date.strftime('%d %B %Y')}",
        styles["subtitle"]
    ))
    story.append(HRFlowable(width=W, thickness=1, color=COLOR_BORDER))
    story.append(Spacer(1, 0.4 * cm))

    # Tableau de résumé 4 cases
    summary_data = [
        ["Parcelles analysées", "Anomalies détectées", "⚠ Attention", "🔴 Critique"],
        [
            str(n_parcelles),
            f"{n_anomalies} ({n_anomalies/n_parcelles*100:.0f}%)" if n_parcelles else "0",
            str(n_warning),
            str(n_critical),
        ],
    ]
    summary_table = Table(summary_data, colWidths=[W / 4] * 4)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  COLOR_DARK),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("BACKGROUND",   (0, 1), (-1, 1),  COLOR_LIGHT_BG),
        ("BACKGROUND",   (3, 1), (3, 1),   colors.HexColor("#fde8e8")),
        ("BACKGROUND",   (2, 1), (2, 1),   colors.HexColor("#fff3e0")),
        ("FONTSIZE",     (0, 0), (-1, -1), 10),
        ("FONTSIZE",     (0, 1), (-1, 1),  16),
        ("FONTNAME",     (0, 1), (-1, 1),  "Helvetica-Bold"),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), None),
        ("GRID",         (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.6 * cm))

    # Graphique temporel
    story.append(Paragraph("Évolution des indices sur la période", styles["h2"]))
    ts_chart = _make_timeseries_chart(df_anomalies)
    story.append(Image(ts_chart, width=W, height=W * 0.5))

    story.append(PageBreak())

    # ── Page 2 : Tableau des alertes ─────────────────────────────────────────
    story.append(Paragraph("Détail des alertes", styles["h2"]))

    df_alerts = df_last[df_last["is_anomaly"]].copy()
    df_alerts = df_alerts.sort_values("anomaly_score")  # plus critique en premier

    if df_alerts.empty:
        story.append(Paragraph("✅ Aucune anomalie détectée pour cette date.", styles["body"]))
    else:
        # En-tête du tableau
        table_data = [["Parcelle", "Culture", "Surface (ha)",
                        "NDWI", "Score", "Sévérité"]]

        for _, row in df_alerts.iterrows():
            sev = str(row.get("severity", "normal"))
            sev_style = styles.get(f"alert_{sev}", styles["body"])
            table_data.append([
                Paragraph(str(row["parcelle_id"]), styles["small"]),
                Paragraph(str(row.get("code_cultu", "?")), styles["body"]),
                Paragraph(f"{row.get('surf_parc', 0):.1f}", styles["body"]),
                Paragraph(f"{row.get('ndwi_mean', 0):.3f}", styles["body"]),
                Paragraph(f"{row.get('anomaly_score', 0):.3f}", styles["body"]),
                Paragraph(sev.upper(), sev_style),
            ])

        col_widths = [W * 0.25, W * 0.12, W * 0.15, W * 0.14, W * 0.14, W * 0.20]
        alert_table = Table(table_data, colWidths=col_widths, repeatRows=1)
        alert_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  COLOR_DARK),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
            ("FONTSIZE",      (0, 0), (-1, 0),  9),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("GRID",          (0, 0), (-1, -1), 0.4, COLOR_BORDER),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, COLOR_LIGHT_BG]),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(alert_table)

    story.append(Spacer(1, 0.6 * cm))

    # Top 5 graphique
    story.append(Paragraph("Parcelles les plus critiques", styles["h2"]))
    top5_chart = _make_top5_chart(df_last)
    story.append(Image(top5_chart, width=W, height=W * 0.33))

    # ── Pied de page ──────────────────────────────────────────────────────────
    story.append(Spacer(1, 1 * cm))
    story.append(HRFlowable(width=W, thickness=0.5, color=COLOR_BORDER))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        f"Rapport généré par Parcelle Watch — "
        f"Données Sentinel-2 / Copernicus (ESA) · Open-Meteo · RPG IGN — "
        f"Modèle Isolation Forest",
        styles["small"]
    ))

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(story)
    pdf_bytes = buf.getvalue()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(pdf_bytes)
        logger.success(f"Rapport PDF sauvegardé : {output_path}")

    return pdf_bytes
