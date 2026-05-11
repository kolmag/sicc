"""
app.py — Supplier Intelligence Command Center
App 4: AI-powered supplier risk scoring, APQP/NPI governance,
       grounded Q&A, executive oversight, and scenario simulation.
"""

import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.answer import answer as rag_answer, CHROMA_DB_PATH

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Supplier Intelligence Command Center",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1623;
    border-right: 1px solid #1e2d45;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label { color: #64748b !important; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; }

/* Page title */
.page-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
}
.page-subtitle {
    font-size: 0.85rem;
    color: #64748b;
    margin-bottom: 1.5rem;
}

/* KPI cards */
.kpi-card {
    background: #131c2e;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    min-height: 90px;
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.35rem;
}
.kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #f1f5f9;
    font-family: 'DM Mono', monospace;
    line-height: 1;
}
.kpi-delta {
    font-size: 0.75rem;
    margin-top: 0.3rem;
    font-family: 'DM Mono', monospace;
}
.delta-up   { color: #f87171; }
.delta-down { color: #34d399; }
.delta-flat { color: #64748b; }

/* Risk badges */
.badge-red    { background:#450a0a; color:#f87171; border:1px solid #7f1d1d; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:600; }
.badge-amber  { background:#451a03; color:#fb923c; border:1px solid #7c2d12; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:600; }
.badge-green  { background:#052e16; color:#34d399; border:1px solid #14532d; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:600; }
.badge-critical { background:#3b0764; color:#c084fc; border:1px solid #6b21a8; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:600; }

/* Section headers */
.section-header {
    font-size: 0.75rem;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid #1e2d45;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
    margin-top: 1.5rem;
}

/* Tables */
.stDataFrame { border: 1px solid #1e2d45; border-radius: 8px; overflow: hidden; }
thead tr th { background: #0f1623 !important; color: #64748b !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
tbody tr:hover { background: #131c2e !important; }

/* Alert cards */
.alert-card {
    background: #0f1623;
    border-left: 3px solid #f87171;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}
.alert-card.amber { border-left-color: #fb923c; }
.alert-card.green { border-left-color: #34d399; }
.alert-card.purple { border-left-color: #c084fc; }

/* AI summary card */
.ai-summary {
    background: linear-gradient(135deg, #0f1623 0%, #131c2e 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    position: relative;
}
.ai-badge {
    display: inline-block;
    background: #1e3a5f;
    color: #60a5fa;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 2px 8px;
    border-radius: 4px;
    margin-bottom: 0.75rem;
}

/* Nav branding */
.brand-header {
    padding: 1rem 1rem 0.5rem;
    border-bottom: 1px solid #1e2d45;
    margin-bottom: 1rem;
}
.brand-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.01em;
}
.brand-sub {
    font-size: 0.68rem;
    color: #475569;
    margin-top: 2px;
}

/* Plotly chart backgrounds */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────

DB_PATH = Path(__file__).parent / "data" / "supplier_portfolio.db"
ML_DIR  = Path(__file__).parent / "ml"

@st.cache_data(ttl=300)
def load_all_data():
    """Load all tables from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    tables = {}
    for table in ["suppliers", "supplier_kpis", "claims", "apqp_projects",
                  "audits", "risk_scores", "external_events"]:
        try:
            tables[table] = pd.read_sql(f"SELECT * FROM {table}", conn)
        except Exception:
            tables[table] = pd.DataFrame()
    conn.close()

    if not tables["supplier_kpis"].empty:
        tables["supplier_kpis"]["year_month"] = pd.to_datetime(
            tables["supplier_kpis"]["year_month"])

    return tables


@st.cache_resource
def load_ml_artefacts():
    """Load trained model + SHAP payload. Returns None if not yet trained."""
    model_path   = ML_DIR / "model.pkl"
    shap_path    = ML_DIR / "shap_values.pkl"
    metrics_path = ML_DIR / "model_metrics.json"

    if not model_path.exists():
        return None

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(shap_path, "rb") as f:
        shap_payload = pickle.load(f)

    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    # Normalise SHAP format — new SHAP returns (n_samples, n_features, n_classes)
    # convert to list of (n_samples, n_features) per class
    sv = shap_payload["shap_values"]
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        sv = [sv[:, :, i] for i in range(sv.shape[2])]

    return {
        "model":          model,
        "shap_values":    sv,
        "expected_value": shap_payload["expected_value"],
        "feature_names":  shap_payload["feature_names"],
        "X":              shap_payload["X"],
        "supplier_ids":   shap_payload["supplier_ids"],
        "y_pred":         shap_payload["y_pred"],
        "y_pred_proba":   shap_payload["y_pred_proba"],
        "label_order":    shap_payload["label_order"],
        "winner_name":    shap_payload.get("winner_name", "RandomForest"),
        "metrics":        metrics,
    }


# ── UI helpers ────────────────────────────────────────────────────────────────

def risk_color(label):
    colors = {"red": "#f87171", "amber": "#fb923c", "green": "#34d399"}
    return colors.get(str(label).lower(), "#94a3b8")


def risk_badge(label):
    label = str(label).lower()
    cls = {"red": "badge-red", "amber": "badge-amber", "green": "badge-green",
           "critical": "badge-critical"}.get(label, "badge-green")
    return f'<span class="{cls}">{label.upper()}</span>'


def severity_badge(sev):
    cls = {"Critical": "badge-critical", "High": "badge-red",
           "Medium": "badge-amber", "Low": "badge-green"}.get(sev, "badge-green")
    return f'<span class="{cls}">{sev}</span>'


def plotly_dark_layout(fig, height=300, margin=None):
    m = margin or dict(l=10, r=10, t=10, b=10)
    fig.update_layout(
        height=height, margin=m,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#94a3b8", size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        xaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45", zerolinecolor="#1e2d45"),
        yaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45", zerolinecolor="#1e2d45"),
    )
    return fig


def kpi_card(label, value, delta=None, delta_direction="flat"):
    delta_html = ""
    if delta is not None:
        cls   = {"up": "delta-up", "down": "delta-down", "flat": "delta-flat"}[delta_direction]
        arrow = {"up": "▲", "down": "▼", "flat": "—"}[delta_direction]
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {delta}</div>'
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>"""


# ── SHAP helpers ──────────────────────────────────────────────────────────────

def _fmt_feature_name(name: str) -> str:
    replacements = {
        "ppm_external": "PPM", "otd_pct": "OTD%", "audit_score": "Audit",
        "scar_count": "SCARs", "cost_of_poor_quality_eur": "COPQ",
        "ppap_first_time_pass_pct": "PPAP FTP", "ca_closure_rate_pct": "CA Closure",
        "oqd_pct": "OQD%", "_3m": " 3m", "_6m": " 6m", "_12m": " 12m",
        "_std_12m": " σ12m", "_trend": " trend", "_deterioration": " Δ",
        "_worst_3m": " worst", "months_ppm_above_500": "Mo PPM>500",
        "months_ppm_above_200": "Mo PPM>200", "months_otd_below_90": "Mo OTD<90",
        "months_otd_below_95": "Mo OTD<95", "months_audit_below_60": "Mo Audit<60",
        "months_audit_below_75": "Mo Audit<75", "spend_tier_enc": "Spend Tier",
        "strat_imp_enc": "Strategic Imp", "qual_status_enc": "Qual Status",
        "region_risk_enc": "Region Risk", "single_source_int": "Single Source",
        "years_active": "Yrs Active", "annual_spend_eur": "Annual Spend", "fam_": "Fam: ",
    }
    result = name
    for k, v in replacements.items():
        result = result.replace(k, v)
    return result[:32]


def make_shap_waterfall(ml, supplier_id: str, class_idx: int = 2, top_n: int = 12):
    """Bar chart of SHAP contributions for one supplier and one predicted class."""
    label_colors = {0: "#34d399", 1: "#fb923c", 2: "#f87171"}

    if ml is None or supplier_id not in ml["supplier_ids"]:
        return None

    idx        = ml["supplier_ids"].index(supplier_id)
    shap_vals  = ml["shap_values"][class_idx][idx]
    feat_names = ml["feature_names"]
    base_val   = (ml["expected_value"][class_idx]
                  if isinstance(ml["expected_value"], (list, np.ndarray))
                  else ml["expected_value"])

    order = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    vals  = shap_vals[order]
    names = [_fmt_feature_name(feat_names[i]) for i in order]

    bar_colors = [label_colors[2] if v > 0 else label_colors[0] for v in vals]

    fig = go.Figure(go.Bar(
        x=names, y=vals,
        marker_color=bar_colors,
        text=[f"{v:+.3f}" for v in vals],
        textposition="outside",
        textfont=dict(size=10, color="#94a3b8"),
    ))
    class_names = ["GREEN", "AMBER", "RED"]
    fig.update_layout(
        title=dict(
            text=f"SHAP contributions → <b>{class_names[class_idx]}</b> risk",
            font=dict(size=12, color="#94a3b8"),
        ),
        xaxis_tickangle=-38,
        yaxis_title="SHAP value",
        showlegend=False,
        shapes=[dict(type="line", xref="paper", x0=0, x1=1, y0=0, y1=0,
                     line=dict(color="#475569", width=1, dash="dot"))],
        annotations=[dict(text=f"Base: {base_val:.3f}", xref="paper", yref="paper",
                          x=1, y=1.02, showarrow=False,
                          font=dict(size=9, color="#475569"), xanchor="right")],
    )
    plotly_dark_layout(fig, height=320)
    return fig


def make_feature_importance_chart(ml, top_n: int = 20):
    """Global feature importance — mean |SHAP| across all suppliers (red class)."""
    if ml is None:
        return None

    shap_red      = ml["shap_values"][2]
    mean_abs_shap = np.abs(shap_red).mean(axis=0)
    feat_names    = ml["feature_names"]

    order = np.argsort(mean_abs_shap)[::-1][:top_n]
    vals  = mean_abs_shap[order][::-1]
    names = [_fmt_feature_name(feat_names[i]) for i in order][::-1]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color="#3b82f6", opacity=0.85,
    ))
    fig.update_layout(
        xaxis_title="Mean |SHAP value| — impact on RED risk",
        yaxis_title="", showlegend=False,
    )
    plotly_dark_layout(fig, height=440)
    return fig


def make_ml_metrics_html(ml) -> str:
    """Render model performance as KPI cards."""
    if ml is None or not ml["metrics"]:
        return ""
    m = ml["metrics"].get("winner_metrics", ml["metrics"])
    cards = [
        ("Accuracy",  f"{m.get('accuracy', 0)*100:.1f}%"),
        ("F1 Macro",  f"{m.get('f1_macro', 0):.3f}"),
        ("AUC (OvR)", f"{m.get('auc_ovr', 0):.3f}"),
        ("F1 Red",    f"{m.get('f1_red', 0):.3f}"),
        ("F1 Amber",  f"{m.get('f1_amber', 0):.3f}"),
        ("F1 Green",  f"{m.get('f1_green', 0):.3f}"),
    ]
    html = '<div style="display:flex; gap:0.75rem; flex-wrap:wrap; margin-bottom:1rem;">'
    for label, val in cards:
        html += f"""
        <div class="kpi-card" style="flex:1; min-width:90px;">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="font-size:1.3rem;">{val}</div>
        </div>"""
    html += "</div>"
    return html


def ml_predicted_badge(ml, supplier_id: str) -> str:
    """Inline ML prediction badge for a supplier."""
    if ml is None or supplier_id not in ml["supplier_ids"]:
        return ""
    idx       = ml["supplier_ids"].index(supplier_id)
    class_idx = ml["y_pred"][idx]
    label     = ml["label_order"][class_idx]
    proba     = ml["y_pred_proba"][idx][class_idx]
    return (f'<span style="font-size:0.72rem; color:#475569; margin-right:0.25rem;">ML:</span>'
            f'{risk_badge(label)} '
            f'<span style="font-size:0.7rem; color:#64748b; font-family:\'DM Mono\',monospace;">'
            f'{proba*100:.0f}% conf</span>')


# ── Load data ─────────────────────────────────────────────────────────────────

data        = load_all_data()
suppliers   = data["suppliers"]
kpis        = data["supplier_kpis"]
claims      = data["claims"]
apqp        = data["apqp_projects"]
audits      = data["audits"]
risk_scores = data["risk_scores"]
events      = data["external_events"]
ml          = load_ml_artefacts()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="brand-header">
        <div class="brand-title">⬡ SICC</div>
        <div class="brand-sub">Supplier Intelligence Command Center</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.selectbox(
        "Navigation",
        ["Executive Portfolio", "Risk Scoring Engine", "Supplier Profile",
         "APQP / NPI Tracker", "Supplier Q&A Agent", "What-If Simulator"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="section-header">Filters</div>', unsafe_allow_html=True)

    sel_families = st.multiselect("Product Family",
        options=sorted(suppliers["product_family"].unique()), default=[],
        placeholder="All families")
    sel_regions = st.multiselect("Region",
        options=sorted(suppliers["region"].unique()), default=[],
        placeholder="All regions")
    sel_tiers = st.multiselect("Spend Tier", options=["A", "B", "C"], default=[],
        placeholder="All tiers")

    filtered_suppliers = suppliers.copy()
    if sel_families:
        filtered_suppliers = filtered_suppliers[filtered_suppliers["product_family"].isin(sel_families)]
    if sel_regions:
        filtered_suppliers = filtered_suppliers[filtered_suppliers["region"].isin(sel_regions)]
    if sel_tiers:
        filtered_suppliers = filtered_suppliers[filtered_suppliers["spend_tier"].isin(sel_tiers)]

    filtered_ids  = set(filtered_suppliers["supplier_id"])
    filtered_risk = risk_scores[risk_scores["supplier_id"].isin(filtered_ids)]

    st.markdown("---")
    n_red   = len(filtered_risk[filtered_risk["risk_label"] == "red"])
    n_amber = len(filtered_risk[filtered_risk["risk_label"] == "amber"])
    n_green = len(filtered_risk[filtered_risk["risk_label"] == "green"])
    st.markdown(f"""
    <div style="font-size:0.72rem; color:#475569; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;">Portfolio Snapshot</div>
    <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
        <span class="badge-red">{n_red} RED</span>
        <span class="badge-amber">{n_amber} AMBER</span>
        <span class="badge-green">{n_green} GREEN</span>
    </div>
    """, unsafe_allow_html=True)

    if ml is not None:
        winner = ml.get("winner_name", "RandomForest")
        m = ml["metrics"].get("winner_metrics", {})
        st.markdown(f"""
        <div style="margin-top:1rem; font-size:0.68rem; color:#475569; border-top:1px solid #1e2d45; padding-top:0.75rem;">
            ⬡ ML · {winner}<br>
            AUC {m.get('auc_ovr', 0):.3f} · F1-Red {m.get('f1_red', 0):.3f}
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 1: EXECUTIVE PORTFOLIO VIEW
# ═══════════════════════════════════════════════════════════════════════

if page == "Executive Portfolio":

    st.markdown('<div class="page-title">Executive Portfolio</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Portfolio-level supplier risk, spend exposure, and concentration</div>', unsafe_allow_html=True)

    total_suppliers  = len(filtered_risk)
    high_risk_pct    = len(filtered_risk[filtered_risk["risk_label"] == "red"]) / max(total_suppliers, 1) * 100
    high_risk_spend  = filtered_risk[filtered_risk["risk_label"] == "red"]["annual_spend_eur"].sum()
    single_source_red = len(filtered_risk[(filtered_risk["risk_label"] == "red") &
                                           (filtered_risk["single_source"] == 1)])
    open_events      = len(events[events["supplier_id"].isin(filtered_ids) &
                                   events["status"].isin(["Open", "Under Review", "Escalated"])])
    apqp_filtered    = apqp[apqp["supplier_id"].isin(filtered_ids)]
    programs_at_risk = len(apqp_filtered[apqp_filtered["is_delayed"] == 1])

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(kpi_card("Suppliers Monitored", f"{total_suppliers:,}"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("High Risk %", f"{high_risk_pct:.1f}%",
                             delta="10% threshold",
                             delta_direction="up" if high_risk_pct > 10 else "down"),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("High-Risk Spend", f"€{high_risk_spend/1e6:.1f}M",
                             delta=f"{single_source_red} sole-source",
                             delta_direction="up" if single_source_red > 0 else "flat"),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Open Alerts", f"{open_events:,}",
                             delta="ESG + Sanctions + Geo",
                             delta_direction="up" if open_events > 50 else "flat"),
                    unsafe_allow_html=True)
    with c5:
        st.markdown(kpi_card("Programs at Risk", f"{programs_at_risk:,}",
                             delta="Delayed milestones",
                             delta_direction="up" if programs_at_risk > 5 else "flat"),
                    unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-header">Risk Distribution by Product Family</div>', unsafe_allow_html=True)
        risk_by_family = filtered_risk.drop(
            columns=["product_family"], errors="ignore"
        ).merge(
            filtered_suppliers[["supplier_id", "product_family"]], on="supplier_id"
        ).groupby(["product_family", "risk_label"]).size().reset_index(name="count")
        fig = px.bar(risk_by_family, x="product_family", y="count", color="risk_label",
                     color_discrete_map={"red": "#f87171", "amber": "#fb923c", "green": "#34d399"})
        fig.update_layout(xaxis_title="", yaxis_title="Suppliers", legend_title="Risk",
                          xaxis_tickangle=-35, bargap=0.3)
        plotly_dark_layout(fig, height=280)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Spend Exposure by Risk Tier</div>', unsafe_allow_html=True)
        spend_risk = filtered_risk.groupby("risk_label")["annual_spend_eur"].sum().reset_index()
        fig2 = px.pie(spend_risk, values="annual_spend_eur", names="risk_label",
                      color="risk_label",
                      color_discrete_map={"red": "#f87171", "amber": "#fb923c", "green": "#34d399"},
                      hole=0.55)
        fig2.update_traces(textinfo="label+percent", textfont_size=11)
        plotly_dark_layout(fig2, height=240)
        st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Top 10 Risk Suppliers</div>', unsafe_allow_html=True)
        top_risk = filtered_risk.drop(
            columns=["product_family"], errors="ignore"
        ).merge(
            filtered_suppliers[["supplier_id", "name", "product_family"]], on="supplier_id"
        ).sort_values("composite_risk_score").head(10)

        for _, row in top_risk.iterrows():
            score = row["composite_risk_score"]
            label = row["risk_label"]
            color = risk_color(label)
            ml_html = ml_predicted_badge(ml, row["supplier_id"]) if ml else ""
            st.markdown(f"""
            <div class="alert-card {'amber' if label=='amber' else ('green' if label=='green' else '')}">
                <div style="font-size:0.78rem; font-weight:600; color:#f1f5f9;">{row['name'][:32]}</div>
                <div style="font-size:0.7rem; color:#64748b;">{row['product_family']}</div>
                <div style="display:flex; justify-content:space-between; margin-top:0.3rem; align-items:center;">
                    <div>{risk_badge(label)} {ml_html}</div>
                    <span style="font-family:'DM Mono',monospace; font-size:0.78rem; color:{color};">{score:.0f}/100</span>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">Geographic Concentration</div>', unsafe_allow_html=True)
        country_count = filtered_suppliers["country"].value_counts().head(8).reset_index()
        country_count.columns = ["country", "count"]
        fig3 = px.bar(country_count, x="count", y="country", orientation="h",
                      color_discrete_sequence=["#3b82f6"])
        fig3.update_layout(xaxis_title="Suppliers", yaxis_title="")
        plotly_dark_layout(fig3, height=220)
        st.plotly_chart(fig3, use_container_width=True)

    # AI Executive Summary
    st.markdown('<div class="section-header">AI Executive Summary</div>', unsafe_allow_html=True)
    red_suppliers = filtered_risk[filtered_risk["risk_label"] == "red"]
    top_red = red_suppliers.drop(
        columns=["product_family"], errors="ignore"
    ).merge(
        filtered_suppliers[["supplier_id", "name", "product_family", "country"]], on="supplier_id"
    ).sort_values("composite_risk_score").head(3)
    top_names = ", ".join(top_red["name"].str[:20].tolist()) if len(top_red) > 0 else "none identified"

    summary_text = f"""
    The supplier portfolio currently spans **{total_suppliers:,} suppliers** across **{len(filtered_suppliers['region'].unique())} regions**.
    **{n_red} suppliers ({high_risk_pct:.1f}%)** are rated HIGH RISK, representing **€{high_risk_spend/1e6:.1f}M** in annual spend exposure.
    Of these, **{single_source_red} are sole-source** dependencies with no qualified alternative — these represent the highest business continuity risk.

    Top priority suppliers requiring immediate attention: **{top_names}**.

    **{open_events} external alerts** are currently open (ESG, sanctions, geopolitical, regulatory).
    **{programs_at_risk} NPI/APQP programmes** have delayed milestones with confirmed supplier impact.

    *Recommended actions: (1) Schedule for-cause audits for all RED sole-source suppliers within 30 days.
    (2) Initiate dual-sourcing feasibility for the top 3 single-source RED suppliers.
    (3) Review all open Critical/High external events for CAPA linkage.*
    """
    st.markdown(f"""
    <div class="ai-summary">
        <div class="ai-badge">AI Generated · Powered by portfolio data</div>
        <div style="font-size:0.85rem; color:#cbd5e1; line-height:1.65;">{summary_text}</div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 2: RISK SCORING ENGINE
# ═══════════════════════════════════════════════════════════════════════

elif page == "Risk Scoring Engine":
    st.markdown('<div class="page-title">Risk Scoring Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">ML-powered supplier risk ranking with SHAP explainability</div>', unsafe_allow_html=True)

    # ML model status banner
    if ml is None:
        st.warning("⚠ ML model not found. Run `uv run python ml/train_risk_model.py` to train. "
                   "Showing rule-based risk scores only.")
    else:
        winner  = ml.get("winner_name", "RandomForest")
        n_feat  = len(ml["feature_names"])
        n_sup   = len(ml["supplier_ids"])
        m       = ml["metrics"].get("winner_metrics", {})
        st.markdown(f"""
        <div style="background:#052e16; border:1px solid #14532d; border-radius:8px;
                    padding:0.6rem 1rem; margin-bottom:0.75rem; font-size:0.78rem; color:#34d399;">
            ✓ {winner} loaded · {n_feat} features · {n_sup} suppliers ·
            Accuracy {m.get('accuracy',0)*100:.1f}% · AUC {m.get('auc_ovr',0):.3f} ·
            F1-Red {m.get('f1_red',0):.3f}
        </div>""", unsafe_allow_html=True)

    # Filter / sort row
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1])
    with c1:
        risk_filter = st.selectbox("Risk Tier", ["All", "RED only", "AMBER only", "GREEN only"])
    with c2:
        sort_by = st.selectbox("Sort by", ["Risk Score ↑", "Spend ↓", "PPM ↓", "OTD ↑"])
    with c3:
        single_source_only = st.checkbox("Single source only", value=False)
    with c4:
        ml_diverge_only = st.checkbox("ML/rule mismatch", value=False,
                                      disabled=(ml is None),
                                      help="Suppliers where ML prediction differs from rule-based label")

    # Build display table
    display = filtered_risk.drop(
        columns=["product_family"], errors="ignore"
    ).merge(
        filtered_suppliers[["supplier_id", "name", "product_family", "country",
                             "spend_tier", "qualification_status"]], on="supplier_id"
    ).sort_values("composite_risk_score")

    # Add ML predictions
    if ml is not None:
        ml_pred_map  = dict(zip(ml["supplier_ids"], ml["y_pred"]))
        ml_proba_map = dict(zip(ml["supplier_ids"],
                                [p[i] for p, i in zip(ml["y_pred_proba"], ml["y_pred"])]))
        label_order  = ml["label_order"]
        display["ml_pred_label"] = display["supplier_id"].map(
            lambda s: label_order[ml_pred_map[s]] if s in ml_pred_map else None)
        display["ml_pred_conf"]  = display["supplier_id"].map(
            lambda s: ml_proba_map.get(s, None))
        display["ml_rule_mismatch"] = (
            display["ml_pred_label"].notna() &
            (display["ml_pred_label"] != display["risk_label"]))

    if risk_filter == "RED only":
        display = display[display["risk_label"] == "red"]
    elif risk_filter == "AMBER only":
        display = display[display["risk_label"] == "amber"]
    elif risk_filter == "GREEN only":
        display = display[display["risk_label"] == "green"]
    if single_source_only:
        display = display[display["single_source"] == 1]
    if ml_diverge_only and ml is not None and "ml_rule_mismatch" in display.columns:
        display = display[display["ml_rule_mismatch"]]

    sort_col_map = {
        "Risk Score ↑": ("composite_risk_score", True),
        "Spend ↓":      ("annual_spend_eur", False),
        "PPM ↓":        ("avg_ppm_3m", False),
        "OTD ↑":        ("avg_otd_3m", True),
    }
    sc, asc = sort_col_map[sort_by]
    display = display.sort_values(sc, ascending=asc)

    st.markdown(f'<div class="section-header">{len(display)} Suppliers</div>', unsafe_allow_html=True)

    # Two-column layout: table left, SHAP right
    col_table, col_shap = st.columns([3, 2])

    with col_table:
        top20 = display.head(20)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="PPM", x=top20["name"].str[:18],
                             y=(100 - top20["avg_ppm_3m"] / 25).clip(0, 100),
                             marker_color="#f87171", opacity=0.85))
        fig.add_trace(go.Bar(name="OTD", x=top20["name"].str[:18],
                             y=((top20["avg_otd_3m"] - 80) * 5).clip(0, 100),
                             marker_color="#fb923c", opacity=0.85))
        fig.add_trace(go.Bar(name="Audit", x=top20["name"].str[:18],
                             y=top20["avg_audit_score_3m"],
                             marker_color="#60a5fa", opacity=0.85))
        fig.update_layout(barmode="group", xaxis_tickangle=-40, legend_orientation="h")
        plotly_dark_layout(fig, height=240)
        st.plotly_chart(fig, use_container_width=True)

        # Supplier table
        table_cols = ["name", "product_family", "country", "risk_label",
                      "composite_risk_score", "avg_ppm_3m", "avg_otd_3m",
                      "avg_audit_score_3m", "annual_spend_eur", "single_source"]
        col_labels = ["Supplier", "Family", "Country", "Rule Risk",
                      "Score", "PPM", "OTD%", "Audit", "Spend €", "SS"]

        if ml is not None and "ml_pred_label" in display.columns:
            table_cols += ["ml_pred_label", "ml_pred_conf"]
            col_labels  += ["ML Risk", "ML Conf%"]

        table_display = display[table_cols].copy()
        table_display.columns = col_labels
        table_display["Score"]  = table_display["Score"].round(1)
        table_display["PPM"]    = table_display["PPM"].round(0)
        table_display["OTD%"]   = table_display["OTD%"].round(1)
        table_display["Audit"]  = table_display["Audit"].round(1)
        table_display["Spend €"] = (display["annual_spend_eur"] / 1000).round(0).astype(int).apply(lambda x: f"€{x:,}k")
        table_display["SS"]     = display["single_source"].map({1: "⚠", 0: "", True: "⚠", False: ""})
        if "ML Conf%" in table_display.columns:
            table_display["ML Conf%"] = (display["ml_pred_conf"] * 100).round(0)
        st.dataframe(table_display, use_container_width=True, height=400)

    with col_shap:
        st.markdown('<div class="section-header">SHAP Explainer</div>', unsafe_allow_html=True)

        if ml is None:
            st.info("Train the ML model to enable SHAP explanations.")
        else:
            shap_options = display[["supplier_id", "name"]].copy()
            shap_options["display"] = shap_options["name"].str[:28] + " (" + shap_options["supplier_id"] + ")"
            selected_shap = st.selectbox("Select supplier", shap_options["display"].tolist(),
                                         key="risk_shap_selector")
            sid_shap  = selected_shap.split("(")[-1].replace(")", "").strip()
            shap_class = st.radio("Explain class",
                                  ["Red Risk ↑", "Amber Risk", "Green Risk ↓"],
                                  horizontal=True, index=0)
            class_idx_map = {"Red Risk ↑": 2, "Amber Risk": 1, "Green Risk ↓": 0}
            class_idx = class_idx_map[shap_class]

            wf_fig = make_shap_waterfall(ml, sid_shap, class_idx=class_idx)
            if wf_fig:
                st.plotly_chart(wf_fig, use_container_width=True)

            # ML vs rule comparison
            if sid_shap in display["supplier_id"].values:
                rule_label = display[display["supplier_id"] == sid_shap]["risk_label"].iloc[0]
                ml_html    = ml_predicted_badge(ml, sid_shap)
                st.markdown(
                    f'<div style="font-size:0.78rem; color:#64748b; margin-top:0.25rem;">'
                    f'Rule-based: {risk_badge(rule_label)} &nbsp;|&nbsp; {ml_html}</div>',
                    unsafe_allow_html=True)

    # Global feature importance expander
    if ml is not None:
        with st.expander("📊 Global Feature Importance (mean |SHAP| — RED class)", expanded=False):
            st.markdown(make_ml_metrics_html(ml), unsafe_allow_html=True)
            fi_fig = make_feature_importance_chart(ml, top_n=20)
            if fi_fig:
                st.plotly_chart(fi_fig, use_container_width=True)
            m = ml["metrics"].get("winner_metrics", {})
            st.markdown(
                f'<div style="font-size:0.72rem; color:#475569; margin-top:0.5rem;">'
                f'{ml.get("winner_name","RandomForest")} · {len(ml["feature_names"])} features · '
                f'{m.get("n_train","—")} train / {m.get("n_test","—")} test · seed=42 · 12% label noise'
                f'</div>', unsafe_allow_html=True)

    # Recent alerts
    st.markdown('<div class="section-header">Recent High-Severity Alerts</div>', unsafe_allow_html=True)
    recent_alerts = events[
        (events["supplier_id"].isin(filtered_ids)) &
        (events["severity"].isin(["Critical", "High"])) &
        (events["status"].isin(["Open", "Under Review", "Escalated"]))
    ].sort_values("event_date", ascending=False).head(8)

    if not recent_alerts.empty:
        for _, evt in recent_alerts.iterrows():
            color_cls = "purple" if evt["severity"] == "Critical" else ""
            st.markdown(f"""
            <div class="alert-card {color_cls}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-size:0.78rem; font-weight:600; color:#f1f5f9;">{evt['supplier_name'][:35]}</span>
                        <span style="font-size:0.7rem; color:#64748b; margin-left:0.5rem;">{evt['event_type']}</span>
                    </div>
                    {severity_badge(evt['severity'])}
                </div>
                <div style="font-size:0.75rem; color:#94a3b8; margin-top:0.25rem;">{evt['description'][:80]}</div>
                <div style="font-size:0.68rem; color:#475569; margin-top:0.2rem;">{evt['event_date']} · {evt['status']}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No high-severity open alerts for current filter selection.")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 3: SUPPLIER PROFILE
# ═══════════════════════════════════════════════════════════════════════

elif page == "Supplier Profile":
    st.markdown('<div class="page-title">Supplier Profile</div>', unsafe_allow_html=True)

    supplier_options = filtered_suppliers[["supplier_id", "name"]].copy()
    supplier_options["display"] = supplier_options["name"] + " (" + supplier_options["supplier_id"] + ")"
    sel = st.selectbox("Select Supplier", supplier_options["display"].tolist(),
                       label_visibility="collapsed")

    sid        = sel.split("(")[-1].replace(")", "").strip()
    sup        = suppliers[suppliers["supplier_id"] == sid].iloc[0]
    sup_risk   = risk_scores[risk_scores["supplier_id"] == sid]
    sup_kpis   = kpis[kpis["supplier_id"] == sid].sort_values("year_month")
    sup_claims = claims[claims["supplier_id"] == sid]
    sup_audits = audits[audits["supplier_id"] == sid]
    sup_events = events[events["supplier_id"] == sid]
    sup_apqp   = apqp[apqp["supplier_id"] == sid]

    risk_row   = sup_risk.iloc[0] if not sup_risk.empty else {}
    risk_label = risk_row.get("risk_label", "unknown") if len(risk_row) else "unknown"
    risk_score = risk_row.get("composite_risk_score", 0) if len(risk_row) else 0

    # Overview card
    c1, c2 = st.columns([2, 1])
    with c1:
        ml_badge_html = ml_predicted_badge(ml, sid) if ml else ""
        st.markdown(f"""
        <div class="kpi-card" style="padding:1.25rem;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <div style="font-size:1.15rem; font-weight:700; color:#f1f5f9;">{sup['name']}</div>
                    <div style="font-size:0.8rem; color:#64748b; margin-top:0.2rem;">
                        {sup['product_family']} · {sup['subcategory']} · {sup['country']}
                    </div>
                    <div style="font-size:0.78rem; color:#475569; margin-top:0.4rem;">
                        {sup['certification']} · Spend Tier {sup['spend_tier']} ·
                        {'⚠ Single Source' if sup['single_source'] else 'Multi-source'} ·
                        {sup['qualification_status']}
                    </div>
                    <div style="margin-top:0.5rem;">{ml_badge_html}</div>
                </div>
                <div style="text-align:right;">
                    {risk_badge(risk_label)}
                    <div style="font-family:'DM Mono',monospace; font-size:1.5rem; color:{risk_color(risk_label)}; margin-top:0.3rem;">{risk_score:.0f}</div>
                    <div style="font-size:0.65rem; color:#475569;">Rule-based Score</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        recommended = risk_row.get("recommended_action", "—") if len(risk_row) else "—"
        st.markdown(f"""
        <div class="ai-summary" style="height:100%; min-height:90px;">
            <div class="ai-badge">Recommended Action</div>
            <div style="font-size:0.82rem; color:#cbd5e1;">{recommended}</div>
        </div>""", unsafe_allow_html=True)

    # KPI trends
    st.markdown('<div class="section-header">KPI Trends (36 months)</div>', unsafe_allow_html=True)
    if not sup_kpis.empty:
        c1, c2, c3 = st.columns(3)
        with c1:
            fig = px.line(sup_kpis, x="year_month", y="ppm_external",
                          color_discrete_sequence=["#f87171"])
            fig.add_hline(y=200, line_dash="dot", line_color="#475569",
                          annotation_text="Amber threshold")
            fig.add_hline(y=500, line_dash="dot", line_color="#7f1d1d",
                          annotation_text="Red threshold")
            fig.update_layout(xaxis_title="", yaxis_title="PPM")
            plotly_dark_layout(fig, height=200)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.line(sup_kpis, x="year_month", y="otd_pct",
                          color_discrete_sequence=["#34d399"])
            fig.add_hline(y=95, line_dash="dot", line_color="#475569")
            fig.add_hline(y=90, line_dash="dot", line_color="#7f1d1d")
            fig.update_layout(xaxis_title="", yaxis_title="OTD %", yaxis_range=[80, 100])
            plotly_dark_layout(fig, height=200)
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            fig = px.line(sup_kpis, x="year_month", y="audit_score",
                          color_discrete_sequence=["#60a5fa"])
            fig.add_hline(y=75, line_dash="dot", line_color="#475569")
            fig.add_hline(y=60, line_dash="dot", line_color="#7f1d1d")
            fig.update_layout(xaxis_title="", yaxis_title="Audit Score", yaxis_range=[40, 100])
            plotly_dark_layout(fig, height=200)
            st.plotly_chart(fig, use_container_width=True)

    # Tabs: Claims / Audits / Events / APQP / ML Explainer
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Claims", "Audits", "External Events", "APQP Programs", "ML Explainer"])

    with tab1:
        if not sup_claims.empty:
            st.dataframe(sup_claims[[
                "incident_number", "creation_date", "category", "status",
                "number_of_bad_parts", "chargeback", "chargeback_value_eur"
            ]].sort_values("creation_date", ascending=False).head(20),
            use_container_width=True)
        else:
            st.info("No claims on record.")

    with tab2:
        if not sup_audits.empty:
            st.dataframe(sup_audits[[
                "audit_id", "audit_date", "audit_type", "is_remote",
                "audit_score", "n_findings", "highest_finding_type", "status"
            ]].sort_values("audit_date", ascending=False), use_container_width=True)
        else:
            st.info("No audit records.")

    with tab3:
        if not sup_events.empty:
            st.dataframe(sup_events[[
                "event_id", "event_date", "event_type", "severity",
                "description", "status", "requires_capa", "capa_linked"
            ]].sort_values("event_date", ascending=False), use_container_width=True)
        else:
            st.info("No external events on record.")

    with tab4:
        if not sup_apqp.empty:
            st.dataframe(sup_apqp[[
                "project_id", "project_type", "status", "creation_date",
                "customer_sop_date", "completion_pct", "is_delayed"
            ]].sort_values("creation_date", ascending=False), use_container_width=True)
        else:
            st.info("No APQP projects.")

    with tab5:
        if ml is None:
            st.info("Train the ML model (`uv run python ml/train_risk_model.py`) to enable SHAP explanations.")
        elif sid not in ml["supplier_ids"]:
            st.warning(f"Supplier {sid} not found in ML model output.")
        else:
            idx         = ml["supplier_ids"].index(sid)
            proba       = ml["y_pred_proba"][idx]
            pred_idx    = ml["y_pred"][idx]
            pred_label  = ml["label_order"][pred_idx]
            prob_green  = proba[0] * 100
            prob_amber  = proba[1] * 100
            prob_red    = proba[2] * 100

            pc1, pc2, pc3, pc4 = st.columns(4)
            with pc1:
                st.markdown(kpi_card("ML Prediction", pred_label.upper(),
                                     delta=f"{proba[pred_idx]*100:.0f}% confidence",
                                     delta_direction="flat"), unsafe_allow_html=True)
            with pc2:
                st.markdown(kpi_card("P(Red)",   f"{prob_red:.1f}%",
                                     delta_direction="up" if prob_red > 30 else "down"),
                            unsafe_allow_html=True)
            with pc3:
                st.markdown(kpi_card("P(Amber)", f"{prob_amber:.1f}%",
                                     delta_direction="flat"), unsafe_allow_html=True)
            with pc4:
                st.markdown(kpi_card("P(Green)", f"{prob_green:.1f}%",
                                     delta_direction="down" if prob_green < 40 else "flat"),
                            unsafe_allow_html=True)

            # Probability bar chart
            fig_prob = go.Figure(go.Bar(
                x=["Green", "Amber", "Red"],
                y=[prob_green, prob_amber, prob_red],
                marker_color=["#34d399", "#fb923c", "#f87171"],
                text=[f"{v:.1f}%" for v in [prob_green, prob_amber, prob_red]],
                textposition="outside",
            ))
            fig_prob.update_layout(yaxis_title="Probability (%)", yaxis_range=[0, 115],
                                   showlegend=False)
            plotly_dark_layout(fig_prob, height=220)
            st.plotly_chart(fig_prob, use_container_width=True)

            # SHAP waterfall
            st.markdown('<div class="section-header">SHAP Feature Contributions</div>',
                        unsafe_allow_html=True)
            shap_class_sel = st.radio(
                "Explain class", ["Red Risk ↑", "Amber Risk", "Green Risk ↓"],
                horizontal=True, index=0, key="profile_shap_class")
            class_idx_map = {"Red Risk ↑": 2, "Amber Risk": 1, "Green Risk ↓": 0}
            wf_fig = make_shap_waterfall(ml, sid,
                                         class_idx=class_idx_map[shap_class_sel])
            if wf_fig:
                st.plotly_chart(wf_fig, use_container_width=True)

            st.markdown(
                '<div style="font-size:0.72rem; color:#475569; margin-top:0.5rem;">'
                'Red bars push toward higher predicted risk · '
                'Green bars push toward lower predicted risk · '
                'Values are additive contributions relative to the model base rate.'
                '</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 4: APQP / NPI TRACKER
# ═══════════════════════════════════════════════════════════════════════

elif page == "APQP / NPI Tracker":
    st.markdown('<div class="page-title">APQP / NPI Tracker</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Programme launch readiness and supplier-linked deliverables</div>', unsafe_allow_html=True)

    apqp_filtered = apqp[apqp["supplier_id"].isin(filtered_ids)].copy()
    apqp_merged   = apqp_filtered.drop(
        columns=["product_family", "country"], errors="ignore"
    ).merge(
        suppliers[["supplier_id", "name", "product_family", "country"]], on="supplier_id")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Active Programs",
                             f"{len(apqp_merged[apqp_merged['status']=='Active']):,}"),
                    unsafe_allow_html=True)
    with c2:
        delayed = apqp_merged[apqp_merged["is_delayed"] == 1]
        st.markdown(kpi_card("Delayed", f"{len(delayed):,}",
                             delta="Milestone overdue",
                             delta_direction="up" if len(delayed) > 0 else "flat"),
                    unsafe_allow_html=True)
    with c3:
        completed = apqp_merged[apqp_merged["status"] == "Completed"]
        st.markdown(kpi_card("Completed", f"{len(completed):,}"), unsafe_allow_html=True)
    with c4:
        avg_completion = apqp_merged["completion_pct"].mean()
        st.markdown(kpi_card("Avg Completion", f"{avg_completion:.0f}%"), unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns([3, 1])

    with col_l:
        st.markdown('<div class="section-header">Programme List</div>', unsafe_allow_html=True)
        status_filter = st.selectbox("Filter by status",
                                     ["All", "Active", "Delayed", "Completed", "On Hold"])
        table_data = apqp_merged if status_filter == "All" else apqp_merged[apqp_merged["status"] == status_filter]
        table = table_data[[
            "project_id", "name", "project_type", "status",
            "customer_sop_date", "completion_pct", "is_delayed", "product_family"
        ]].copy()
        table.columns = ["Project ID", "Supplier", "Type", "Status",
                         "SOP Date", "Completion %", "Delayed", "Family"]
        table["Delayed"] = table["Delayed"].map({1: "⚠ Yes", 0: "No", True: "⚠ Yes", False: "No"})
        st.dataframe(table.sort_values("Delayed", ascending=False),
                     use_container_width=True, height=400)

    with col_r:
        st.markdown('<div class="section-header">Status Breakdown</div>', unsafe_allow_html=True)
        status_counts = apqp_merged["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig = px.pie(status_counts, values="Count", names="Status",
                     color_discrete_sequence=["#34d399", "#60a5fa", "#f87171",
                                              "#fb923c", "#94a3b8"],
                     hole=0.5)
        plotly_dark_layout(fig, height=220)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Completion Distribution</div>',
                    unsafe_allow_html=True)
        fig2 = px.histogram(apqp_merged, x="completion_pct", nbins=10,
                            color_discrete_sequence=["#3b82f6"])
        fig2.update_layout(xaxis_title="Completion %", yaxis_title="Programs", bargap=0.1)
        plotly_dark_layout(fig2, height=200)
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 5: SUPPLIER Q&A AGENT — RAG CONNECTED
# ═══════════════════════════════════════════════════════════════════════
#
# HOW TO INTEGRATE:
# Replace the entire block from:
#   elif page == "Supplier Q&A Agent":
# through to (but not including):
#   st.markdown("""  ⬡ RAG integration ready ...
# with the block below.
#
# Also add this import near the top of app.py (after existing imports):
#   import sys, os
#   sys.path.insert(0, os.path.dirname(__file__))
#   from scripts.answer import answer as rag_answer, CHROMA_DB_PATH
# ═══════════════════════════════════════════════════════════════════════

elif page == "Supplier Q&A Agent":
    st.markdown('<div class="page-title">Supplier Q&A Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Two-layer intelligence: structured portfolio queries + grounded RAG over supplier quality knowledge base</div>', unsafe_allow_html=True)

    # ── Mode selector ─────────────────────────────────────────────────────────
    query_mode = st.radio(
        "Query mode",
        ["📊 Portfolio Data  (supplier KPIs, risk, claims, audits)",
         "📚 Knowledge Base  (PPAP, APQP, SCAR, audit standards, procedures)"],
        horizontal=True,
        label_visibility="collapsed",
    )
    is_rag_mode = "Knowledge Base" in query_mode

    st.markdown("---")

    # ── Suggested prompts — different per mode ────────────────────────────────
    if is_rag_mode:
        st.markdown('<div class="section-header">Suggested Knowledge Base Queries</div>',
                    unsafe_allow_html=True)
        prompts = [
            "What does PPAP Level 3 require?",
            "When is a for-cause audit mandatory?",
            "What are the RED tier KPI thresholds?",
            "What is the SCAR escalation process?",
            "What are the APQP Phase 4 pass criteria?",
            "What buffer stock is required for single-source suppliers?",
        ]
    else:
        st.markdown('<div class="section-header">Suggested Portfolio Queries</div>',
                    unsafe_allow_html=True)
        prompts = [
            "Which RED-risk suppliers have open major audit findings?",
            "Show sole-source suppliers with PPM > 300 in the last 3 months",
            "Which suppliers have Critical external events and no linked CAPA?",
            "What are the top recurring claim categories across Electronics suppliers?",
            "Which APQP programmes are delayed and linked to RED suppliers?",
            "Show suppliers in China with High or Critical geopolitical events",
        ]

    cols = st.columns(3)
    selected_prompt = None
    for i, prompt in enumerate(prompts):
        with cols[i % 3]:
            if st.button(prompt, key=f"prompt_{i}", use_container_width=True):
                selected_prompt = prompt

    st.markdown("---")

    query = st.text_area(
        "Ask a question",
        value=selected_prompt or "",
        height=80,
        placeholder=(
            "e.g. What does PPAP Level 3 require?" if is_rag_mode
            else "e.g. Which critical suppliers have open audit findings and no CAPA?"
        ),
    )

    # ── Filters (portfolio mode only) ─────────────────────────────────────────
    if not is_rag_mode:
        with st.expander("Portfolio Filters", expanded=False):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                filter_family = st.multiselect("Product Family",
                                               options=sorted(suppliers["product_family"].unique()))
            with fc2:
                filter_risk = st.multiselect("Risk Tier", options=["red", "amber", "green"])
            with fc3:
                filter_region = st.multiselect("Region",
                                               options=sorted(suppliers["region"].unique()))
    else:
        filter_family = []
        filter_risk   = []
        filter_region = []

    # ── Search button ─────────────────────────────────────────────────────────
    if st.button("Search", type="primary") and query:

        # ══════════════════════════════════════════════════════════════════════
        # LAYER 2 — RAG (Knowledge Base mode)
        # ══════════════════════════════════════════════════════════════════════
        if is_rag_mode:
            with st.spinner("Searching knowledge base..."):
                try:
                    result = rag_answer(
                        question=query,
                        db_path=CHROMA_DB_PATH,
                        session_id="streamlit_live",
                    )

                    # Confidence badge colour
                    conf_color = {
                        "high":   "#34d399",
                        "medium": "#fb923c",
                        "low":    "#f87171",
                    }.get(result.confidence, "#94a3b8")

                    # Action required badge
                    action_html = (
                        '<span class="badge-red">⚡ ACTION REQUIRED</span>'
                        if result.action_required else ""
                    )

                    # Insufficient evidence
                    if result.insufficient_evidence:
                        st.markdown(f"""
                        <div class="ai-summary">
                            <div class="ai-badge">Knowledge Base · Insufficient Evidence</div>
                            <div style="font-size:0.88rem; color:#f87171; margin-top:0.5rem;">
                                {result.answer}
                            </div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        # Answer card
                        sources_html = " · ".join(
                            f'<code style="font-size:0.68rem; color:#60a5fa;">{s}</code>'
                            for s in result.sources
                        )
                        st.markdown(f"""
                        <div class="ai-summary">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                                <div class="ai-badge">Knowledge Base · RAG Answer</div>
                                <div style="display:flex; gap:0.5rem; align-items:center;">
                                    {action_html}
                                    <span style="font-size:0.72rem; font-weight:600;
                                                 color:{conf_color}; text-transform:uppercase;
                                                 letter-spacing:0.06em;">
                                        {result.confidence} confidence
                                    </span>
                                </div>
                            </div>
                        </div>""", unsafe_allow_html=True)
                        st.markdown(result.answer)
                        st.markdown(f"""
                        <div style="margin-top:0.5rem; font-size:0.7rem; color:#475569;">
                            Sources: {sources_html}
                        </div>""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"RAG pipeline error: {e}")
                    st.info("Ensure `chroma_db/` exists and `scripts/ingest.py` has been run.")

        # ══════════════════════════════════════════════════════════════════════
        # LAYER 1 — Structured (Portfolio Data mode)
        # ══════════════════════════════════════════════════════════════════════
        else:
            with st.spinner("Searching supplier intelligence..."):
                result_df = risk_scores.merge(
                    suppliers[["supplier_id", "name", "product_family", "country",
                                "region", "single_source", "spend_tier",
                                "qualification_status", "certification"]], on="supplier_id")

                if filter_family:
                    result_df = result_df[result_df["product_family"].isin(filter_family)]
                if filter_risk:
                    result_df = result_df[result_df["risk_label"].isin(filter_risk)]
                if filter_region:
                    result_df = result_df[result_df["region"].isin(filter_region)]

                q_lower     = query.lower()
                answer_text = ""
                show_df     = None

                if "red" in q_lower or "high risk" in q_lower:
                    show_df     = result_df[result_df["risk_label"] == "red"].sort_values("composite_risk_score")
                    answer_text = f"Found **{len(show_df)} RED-risk suppliers** matching your criteria."
                elif "sole" in q_lower or "single source" in q_lower:
                    show_df     = result_df[result_df["single_source"].isin([1, True])].sort_values("risk_label")
                    answer_text = f"Found **{len(show_df)} single-source suppliers**. {len(show_df[show_df['risk_label']=='red'])} are RED risk."
                elif "ppm" in q_lower:
                    threshold   = 300
                    show_df     = result_df[result_df["avg_ppm_3m"] > threshold].sort_values("avg_ppm_3m", ascending=False)
                    answer_text = f"Found **{len(show_df)} suppliers** with PPM > {threshold} in the last 3 months."
                elif "audit" in q_lower or "finding" in q_lower:
                    audit_sup   = audits[audits["highest_finding_type"].isin(["Major NCR", "Critical NCR"])]["supplier_id"].unique()
                    show_df     = result_df[result_df["supplier_id"].isin(audit_sup)].sort_values("composite_risk_score")
                    answer_text = f"Found **{len(show_df)} suppliers** with open major or critical NCRs."
                elif "capa" in q_lower or "event" in q_lower or "alert" in q_lower:
                    capa_needed = events[
                        (events["requires_capa"] == True) & (events["capa_linked"] == False) &
                        (events["status"].isin(["Open", "Under Review"]))
                    ]["supplier_id"].unique()
                    show_df     = result_df[result_df["supplier_id"].isin(capa_needed)].sort_values("composite_risk_score")
                    answer_text = f"Found **{len(show_df)} suppliers** with open Critical/High alerts and no linked CAPA."
                elif "geopolit" in q_lower or any(
                    c.lower() in q_lower for c in suppliers["country"].unique()
                ):
                    geo_sups = events[
                        (events["event_type"] == "Geopolitical") &
                        (events["severity"].isin(["High", "Critical"])) &
                        (events["status"].isin(["Open", "Under Review", "Escalated"]))
                    ]["supplier_id"].unique()
                    geo_mask = result_df["supplier_id"].isin(geo_sups)

                    # Extract country from query if mentioned
                    mentioned_country = next(
                        (c for c in suppliers["country"].unique() if c.lower() in q_lower),
                        None
                    )
                    if mentioned_country:
                        country_mask = result_df["country"] == mentioned_country
                        show_df      = result_df[country_mask & geo_mask].sort_values("composite_risk_score")
                        answer_text  = f"Found **{len(show_df)} {mentioned_country}-based suppliers** with active High/Critical geopolitical events."
                    else:
                        show_df     = result_df[geo_mask].sort_values("composite_risk_score")
                        answer_text = f"Found **{len(show_df)} suppliers** with active High/Critical geopolitical events."
                elif "apqp" in q_lower or "delayed" in q_lower or "programme" in q_lower:
                    red_sups     = result_df[result_df["risk_label"] == "red"]["supplier_id"].unique()
                    delayed_apqp = apqp[apqp["is_delayed"] == 1].merge(
                        suppliers[["supplier_id", "name"]], on="supplier_id")
                    show_df      = delayed_apqp[delayed_apqp["supplier_id"].isin(red_sups)]
                    answer_text  = f"Found **{len(show_df)} delayed APQP programmes** linked to RED-risk suppliers."
                else:
                    show_df     = result_df.sort_values("composite_risk_score").head(20)
                    answer_text = f"Showing top {len(show_df)} suppliers matching your filters."

                st.markdown(f"""
                <div class="ai-summary">
                    <div class="ai-badge">Portfolio Query · Structured data retrieval</div>
                    <div style="font-size:0.9rem; color:#f1f5f9; font-weight:500;">{answer_text}</div>
                    <div style="font-size:0.75rem; color:#475569; margin-top:0.5rem;">
                        Source: supplier_kpis, risk_scores, audits, external_events ·
                        {len(result_df)} suppliers in scope
                    </div>
                </div>""", unsafe_allow_html=True)

                if show_df is not None and not show_df.empty:
                    display_cols = [c for c in [
                        "name", "product_family", "country", "risk_label",
                        "composite_risk_score", "avg_ppm_3m", "avg_otd_3m",
                        "single_source", "annual_spend_eur"
                    ] if c in show_df.columns]
                    st.dataframe(show_df[display_cols].head(25), use_container_width=True)

    # Footer
    st.markdown(f"""
    <div style="font-size:0.72rem; color:#475569; margin-top:2rem; padding:0.75rem;
                border:1px solid #1e2d45; border-radius:8px;">
        ⬡ <strong>Two-layer Q&A.</strong>
        Portfolio Data mode queries structured supplier KPIs, risk scores, audits, and events directly from SQLite.
        Knowledge Base mode uses hybrid RAG (BM25 + embedding + RRF) over {231} KB chunks
        (15 supplier quality documents) via ChromaDB · OSS-120B generator · OSS-20B groundedness checker.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 6: WHAT-IF SIMULATOR
# ═══════════════════════════════════════════════════════════════════════

elif page == "What-If Simulator":
    st.markdown('<div class="page-title">What-If Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Model disruption scenarios and evaluate mitigation impact</div>', unsafe_allow_html=True)

    col_cfg, col_result = st.columns([1, 2])

    with col_cfg:
        st.markdown('<div class="section-header">Scenario Builder</div>', unsafe_allow_html=True)
        scenario_type = st.selectbox("Scenario Type", [
            "Supplier Outage", "Production Delay", "Cost Increase",
            "Sole-Source Failure", "Quality Escape", "Region Disruption",
        ])
        supplier_options = filtered_suppliers[["supplier_id", "name"]].copy()
        supplier_options["display"] = (supplier_options["name"].str[:30]
                                       + " (" + supplier_options["supplier_id"] + ")")
        selected_sup = st.selectbox("Affected Supplier", supplier_options["display"].tolist())
        sid_sim  = selected_sup.split("(")[-1].replace(")", "").strip()
        sup_sim  = suppliers[suppliers["supplier_id"] == sid_sim].iloc[0]
        risk_sim = risk_scores[risk_scores["supplier_id"] == sid_sim]

        if scenario_type in ["Supplier Outage", "Production Delay"]:
            duration = st.slider("Duration (days)", 7, 180, 30)
        elif scenario_type == "Cost Increase":
            cost_pct = st.slider("Cost increase (%)", 5, 80, 20)
        elif scenario_type == "Quality Escape":
            escape_ppm = st.slider("Escape PPM", 50, 2000, 300)
        elif scenario_type == "Region Disruption":
            region_sel = st.selectbox("Affected Region", sorted(suppliers["region"].unique()))

        run_sim = st.button("Run Simulation", type="primary", use_container_width=True)

    with col_result:
        st.markdown('<div class="section-header">Simulation Results</div>', unsafe_allow_html=True)

        if run_sim:
            annual_spend    = sup_sim["annual_spend_eur"]
            is_single       = bool(sup_sim["single_source"])
            risk_label_sim  = risk_sim["risk_label"].iloc[0] if not risk_sim.empty else "amber"
            risk_score_sim  = risk_sim["composite_risk_score"].iloc[0] if not risk_sim.empty else 50

            if scenario_type == "Supplier Outage":
                daily_cost    = annual_spend / 365
                direct_cost   = daily_cost * duration
                expedite_cost = direct_cost * 0.35
                total_cost    = direct_cost + expedite_cost
                prog_impact   = len(apqp[apqp["supplier_id"] == sid_sim])
                risk_delta    = min(35, (duration / 30) * 12)
                new_risk_score = min(100, risk_score_sim + risk_delta)

                st.markdown(f"""
                <div class="kpi-card" style="margin-bottom:0.75rem;">
                    <div class="kpi-label">Scenario</div>
                    <div style="font-size:0.9rem; color:#f1f5f9; font-weight:600;">
                        {scenario_type} · {sup_sim['name'][:35]} · {duration} days
                    </div>
                    <div style="font-size:0.78rem; color:#64748b; margin-top:0.25rem;">
                        {sup_sim['product_family']} ·
                        {'⚠ SOLE SOURCE' if is_single else 'Multi-source'} ·
                        Current risk: {risk_badge(risk_label_sim)}
                    </div>
                </div>""", unsafe_allow_html=True)

                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    st.markdown(kpi_card("Direct Cost Impact", f"€{total_cost/1000:.0f}k",
                                         delta=f"€{daily_cost/1000:.1f}k/day",
                                         delta_direction="up"), unsafe_allow_html=True)
                with rc2:
                    st.markdown(kpi_card("Risk Score Delta", f"+{risk_delta:.0f}",
                                         delta=f"New score: {new_risk_score:.0f}",
                                         delta_direction="up"), unsafe_allow_html=True)
                with rc3:
                    st.markdown(kpi_card("Programmes Impacted", f"{prog_impact}",
                                         delta="NPI/APQP at risk",
                                         delta_direction="up" if prog_impact > 0 else "flat"),
                                unsafe_allow_html=True)

                st.markdown('<div class="section-header">Recommended Mitigations</div>',
                            unsafe_allow_html=True)
                mitigations = []
                if is_single:
                    mitigations.append(("🔴 CRITICAL",
                                        "Initiate emergency dual-sourcing — identify qualified alternative within 14 days",
                                        "High"))
                mitigations.append(("⚡ Immediate",
                                    f"Activate buffer stock — target {duration+14} days coverage from existing inventory",
                                    "High"))
                mitigations.append(("📋 30 days",
                                    "Issue formal SCAR to supplier — root cause analysis and recovery plan required",
                                    "Medium"))
                mitigations.append(("🔍 60 days",
                                    "Conduct for-cause audit on resumption — verify process stability before full ramp",
                                    "Medium"))
                if prog_impact > 0:
                    mitigations.append(("📅 NPI",
                                        f"Notify programme managers — {prog_impact} programme(s) require milestone review",
                                        "High"))

                for priority, action, _ in mitigations:
                    st.markdown(f"""
                    <div class="alert-card">
                        <div style="font-size:0.78rem; color:#60a5fa; font-weight:600;">{priority}</div>
                        <div style="font-size:0.82rem; color:#cbd5e1; margin-top:0.2rem;">{action}</div>
                    </div>""", unsafe_allow_html=True)

            elif scenario_type == "Cost Increase":
                cost_impact = annual_spend * (cost_pct / 100)
                st.markdown(f"""
                <div class="kpi-card" style="margin-bottom:0.75rem;">
                    <div class="kpi-label">Scenario</div>
                    <div style="font-size:0.9rem; color:#f1f5f9; font-weight:600;">
                        {cost_pct}% cost increase · {sup_sim['name'][:35]}
                    </div>
                </div>""", unsafe_allow_html=True)
                rc1, rc2 = st.columns(2)
                with rc1:
                    st.markdown(kpi_card("Annual Cost Impact", f"€{cost_impact/1000:.0f}k",
                                         delta=f"Base: €{annual_spend/1000:.0f}k/yr",
                                         delta_direction="up"), unsafe_allow_html=True)
                with rc2:
                    st.markdown(kpi_card("3-Year Exposure", f"€{cost_impact*3/1000:.0f}k",
                                         delta="If not mitigated",
                                         delta_direction="up"), unsafe_allow_html=True)
                st.markdown('<div class="section-header">Mitigations</div>', unsafe_allow_html=True)
                for action in [
                    "Re-negotiate contract — invoke price review clause if available",
                    "Request cost breakdown — identify material vs labour vs overhead drivers",
                    f"Initiate resourcing analysis — identify 2-3 alternative suppliers in {sup_sim['product_family']}",
                    "Evaluate design-to-cost opportunity — engage engineering for specification review",
                ]:
                    st.markdown(
                        f'<div class="alert-card amber"><div style="font-size:0.82rem; color:#cbd5e1;">{action}</div></div>',
                        unsafe_allow_html=True)

            elif scenario_type == "Region Disruption":
                affected      = suppliers[suppliers["region"] == region_sel]
                affected_risk = risk_scores[risk_scores["supplier_id"].isin(affected["supplier_id"])]
                affected_spend = affected.merge(
                    risk_scores[["supplier_id", "annual_spend_eur"]], on="supplier_id"
                )["annual_spend_eur"].sum()
                n_red    = len(affected_risk[affected_risk["risk_label"] == "red"])
                n_single = len(affected[affected["single_source"].isin([1, True])])

                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    st.markdown(kpi_card("Suppliers Affected", f"{len(affected):,}",
                                         delta=f"{n_red} RED risk",
                                         delta_direction="up" if n_red > 0 else "flat"),
                                unsafe_allow_html=True)
                with rc2:
                    st.markdown(kpi_card("Spend at Risk", f"€{affected_spend/1e6:.1f}M"),
                                unsafe_allow_html=True)
                with rc3:
                    st.markdown(kpi_card("Single Sources", f"{n_single}",
                                         delta="No alternative available",
                                         delta_direction="up" if n_single > 0 else "flat"),
                                unsafe_allow_html=True)

                family_breakdown = affected.groupby("product_family").size().reset_index(name="count")
                fig = px.bar(family_breakdown, x="product_family", y="count",
                             color_discrete_sequence=["#f87171"])
                fig.update_layout(xaxis_title="", yaxis_title="Suppliers", xaxis_tickangle=-35)
                plotly_dark_layout(fig, height=220)
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.info(f"Simulation for '{scenario_type}' — configure parameters and click Run Simulation.")

        else:
            st.markdown("""
            <div style="text-align:center; padding:3rem; color:#475569;">
                <div style="font-size:2rem; margin-bottom:1rem;">⬡</div>
                <div style="font-size:0.9rem;">Configure a scenario on the left and click Run Simulation</div>
            </div>""", unsafe_allow_html=True)
