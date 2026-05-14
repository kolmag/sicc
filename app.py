"""
app.py — Supplier Intelligence Command Center
App 4: AI-powered supplier risk scoring, APQP/NPI governance,
       grounded Q&A, executive oversight, and scenario simulation.
"""

import json
import pickle
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.answer import answer as rag_answer, CHROMA_DB_PATH
from scripts.supplier_intake_agent import (
    SupplierDevelopmentBrief,
    brief_to_markdown,
    generate_supplier_development_brief,
)
from scripts.supplier_alert_agent import build_supplier_trend_alerts
from scripts.scar_capa_agent import triage_claim, triage_manual_issue
from scripts.apqp_readiness_agent import assess_apqp_launch_readiness
from scripts.continuity_agent import build_continuity_watchlist, assess_single_source_continuity
from scripts.audit_planning_agent import build_audit_plan_watchlist, plan_supplier_audit
from scripts.agent_memory import (
    clear_stale_supplier_memory,
    finish_agent_run,
    get_supplier_agent_runs,
    get_supplier_memory,
    init_agent_memory,
    normalize_severity,
    record_agent_run_step,
    remember_agent_output,
    start_agent_run,
)

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

TABLE_COLUMNS = {
    "suppliers": [
        "supplier_id", "name", "country", "region", "city", "product_family",
        "subcategory", "certification", "spend_tier", "annual_spend_eur",
        "qualification_status", "single_source", "strategic_importance",
        "years_active", "onboarding_date", "archetype", "archetype_description",
        "primary_contact", "primary_contact_email", "account_manager",
    ],
    "supplier_kpis": [
        "kpi_id", "supplier_id", "year_month", "year", "month", "ppm_external",
        "ppm_internal", "otd_pct", "oqd_pct", "audit_score", "scar_count",
        "scar_open_days_avg", "ppap_first_time_pass_pct", "ca_closure_rate_pct",
        "cost_of_poor_quality_eur", "risk_label", "risk_label_true",
    ],
    "claims": [
        "incident_number", "supplier_id", "supplier_name", "creation_date",
        "category", "status", "number_of_bad_parts", "chargeback",
        "chargeback_value_eur", "product_family", "spend_tier",
    ],
    "apqp_projects": [
        "project_id", "supplier_id", "supplier_name", "project_type", "status",
        "creation_date", "customer_sop_date", "supplier_sop_date",
        "product_family", "spend_tier", "completion_pct", "is_delayed",
    ],
    "audits": [
        "audit_id", "supplier_id", "supplier_name", "audit_date", "audit_type",
        "is_remote", "audit_score", "n_findings", "highest_finding_type",
        "status", "product_family",
    ],
    "risk_scores": [
        "supplier_id", "avg_ppm_3m", "avg_otd_3m", "avg_audit_score_3m",
        "avg_scar_count_3m", "composite_risk_score", "spend_risk_priority",
        "risk_label", "risk_label_true", "recommended_action", "product_family",
        "spend_tier", "annual_spend_eur", "single_source",
        "strategic_importance", "qualification_status",
    ],
    "external_events": [
        "event_id", "supplier_id", "supplier_name", "country", "region",
        "event_type", "severity", "description", "event_date", "status",
        "response_due_date", "resolved_date", "product_family", "spend_tier",
        "annual_spend_eur", "single_source", "requires_capa", "capa_linked",
        "source",
    ],
}

@st.cache_data(ttl=300)
def load_all_data():
    """Load all tables from SQLite."""
    tables = {name: pd.DataFrame(columns=cols) for name, cols in TABLE_COLUMNS.items()}
    if not DB_PATH.exists():
        return tables

    conn = sqlite3.connect(DB_PATH)
    for table, columns in TABLE_COLUMNS.items():
        try:
            tables[table] = pd.read_sql(f"SELECT * FROM {table}", conn)
        except Exception:
            tables[table] = pd.DataFrame(columns=columns)
    conn.close()

    if not tables["supplier_kpis"].empty:
        tables["supplier_kpis"]["year_month"] = pd.to_datetime(
            tables["supplier_kpis"]["year_month"])

    for _tbl, _col in [
        ("claims",          "creation_date"),
        ("audits",          "audit_date"),
        ("apqp_projects",   "customer_sop_date"),
        ("apqp_projects",   "supplier_sop_date"),
        ("external_events", "event_date"),
        ("external_events", "response_due_date"),
    ]:
        if not tables[_tbl].empty and _col in tables[_tbl].columns:
            tables[_tbl][_col] = pd.to_datetime(tables[_tbl][_col], errors="coerce")

    return tables

@st.cache_resource(show_spinner=False)
def get_kb_chunk_count() -> int:
    """Return live ChromaDB chunk count; falls back to last-known value."""
    try:
        import chromadb
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH), settings=CHROMA_SETTINGS)
        return _client.get_collection("supplier_kb").count()
    except Exception:
        return 264


@st.cache_resource
def load_ml_artefacts():
    """Load trained model + SHAP payload. Returns None if not yet trained."""
    model_path   = ML_DIR / "model.pkl"
    shap_path    = ML_DIR / "shap_values.pkl"
    metrics_path = ML_DIR / "model_metrics.json"

    if not model_path.exists():
        return None

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(shap_path, "rb") as f:
            shap_payload = pickle.load(f)

        # Normalise SHAP format
        sv = shap_payload["shap_values"]
        if isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv = [sv[:, :, i] for i in range(sv.shape[2])]

        metrics = {}
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)

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
    except Exception as e:
        print(f"[ML] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None
        
@st.cache_data(ttl=3600, show_spinner=False)
def generate_executive_summary(
    n_suppliers: int,
    n_regions: int,
    n_red: int,
    n_amber: int,
    n_green: int,
    high_risk_pct: float,
    high_risk_spend: float,
    single_source_red: int,
    open_events: int,
    programs_at_risk: int,
    top_red_names: str,
) -> str:
    """Generate LLM executive summary. Cached per session per filter state."""
    from litellm import completion as litellm_completion
 
    prompt = f"""You are a Chief Procurement Officer writing a concise executive portfolio summary.
 
Portfolio snapshot:
- Total suppliers monitored: {n_suppliers:,}
- Regions covered: {n_regions}
- Risk distribution: {n_red} RED ({high_risk_pct:.1f}%), {n_amber} AMBER, {n_green} GREEN
- High-risk annual spend exposure: €{high_risk_spend/1e6:.1f}M
- Single-source RED suppliers (no qualified alternative): {single_source_red}
- Open external alerts (ESG, sanctions, geopolitical): {open_events}
- NPI/APQP programmes with delayed milestones: {programs_at_risk}
- Top priority suppliers: {top_red_names}
 
Write a 3-paragraph executive brief:
1. Portfolio risk status and headline numbers (2-3 sentences)
2. Most critical risks requiring immediate action, with specific supplier context (2-3 sentences)
3. Recommended actions with clear timelines and owners (3 bullet points)
 
Rules:
- Be direct and specific — no generic filler
- Use supplier quality terminology (SCAR, OTD, PPM, for-cause audit, dual-source)
- Quantify risks in business terms (spend exposure, supply continuity days)
- Actions must have timelines (e.g. "within 30 days") and owners (e.g. "Supply Chain Director")
- Do not mention that this is AI-generated"""
 
    try:
        response = litellm_completion(
            model="groq/openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return (
            f"The supplier portfolio currently spans **{n_suppliers:,} suppliers** across **{n_regions} regions**. "
            f"**{n_red} suppliers ({high_risk_pct:.1f}%)** are rated HIGH RISK, representing **€{high_risk_spend/1e6:.1f}M** in annual spend exposure. "
            f"Of these, **{single_source_red} are sole-source** dependencies with no qualified alternative.\n\n"
            f"Top priority suppliers requiring immediate attention: **{top_red_names}**. "
            f"**{open_events} external alerts** are currently open. "
            f"**{programs_at_risk} NPI/APQP programmes** have delayed milestones.\n\n"
            f"*Recommended actions: (1) Schedule for-cause audits for all RED sole-source suppliers within 30 days. "
            f"(2) Initiate dual-sourcing feasibility for top 3 single-source RED suppliers. "
            f"(3) Review all open Critical/High external events for CAPA linkage.*"
        )


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


def get_ml_pred_label(ml, supplier_id: str) -> str | None:
    """Return the ML-predicted risk label string, or None if unavailable."""
    if ml is None or supplier_id not in ml["supplier_ids"]:
        return None
    idx = ml["supplier_ids"].index(supplier_id)
    return ml["label_order"][ml["y_pred"][idx]]


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


def parse_agent_ts(ts: str):
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def memory_age_hours(item: dict) -> float | None:
    parsed = parse_agent_ts(item.get("updated_at", ""))
    if not parsed:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - parsed).total_seconds() / 3600


def is_memory_fresh(memory: list[dict], max_age_hours: int = 24) -> bool:
    if not memory:
        return False
    ages = [memory_age_hours(item) for item in memory]
    ages = [age for age in ages if age is not None]
    return bool(ages) and max(ages) <= max_age_hours


def severity_rank(severity: str) -> int:
    return {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}.get(normalize_severity(severity), 0)


def severity_badge_text(severity: str) -> str:
    return normalize_severity(severity).upper()


def operator_status_label(status: str) -> str:
    labels = {
        "fresh": "fresh",
        "stale": "stale - refresh recommended",
        "failed": "failed latest run",
        "skipped": "skipped latest run",
        "success": "fresh",
        "no data": "no current output",
        "info": "no current output",
    }
    return labels.get(str(status).strip().lower(), str(status))


def build_run_log_markdown(supplier_id: str, agent_runs: list[dict]) -> str:
    lines = [
        f"# SICC Agent Run Log: {supplier_id}",
        "",
        f"- Exported: {datetime.now().isoformat(timespec='seconds')}",
    ]
    if not agent_runs:
        lines.append("- No agent runs recorded.")
        return "\n".join(lines)

    for run in agent_runs:
        lines.extend([
            "",
            f"## Run {run['run_id']}",
            f"- Status: {run['status']}",
            f"- Started: {run['started_at']}",
            f"- Finished: {run.get('finished_at') or ''}",
            f"- Summary: {run['summary']}",
            "",
            "### Steps",
        ])
        for step in run.get("steps", []):
            lines.append(
                f"- {step['agent_name']}: {operator_status_label(step['status'])} "
                f"({normalize_severity(step['severity'])}) - {step['error'] or step['summary']}"
            )
    return "\n".join(lines)


def build_evidence_pack_markdown(
    supplier,
    risk_row,
    sup_kpis,
    sup_claims,
    sup_audits,
    sup_events,
    sup_apqp,
    memory: list[dict],
    agent_runs: list[dict] | None = None,
) -> str:
    supplier_name = supplier.get("name", supplier.get("supplier_name", "Unknown supplier"))
    risk_label = risk_row.get("risk_label", "unknown") if hasattr(risk_row, "get") else "unknown"
    lines = [
        f"# SICC Supplier Evidence Pack: {supplier_name}",
        "",
        f"- Supplier ID: {supplier.get('supplier_id', 'unknown')}",
        f"- Product family: {supplier.get('product_family', 'unknown')}",
        f"- Country: {supplier.get('country', 'unknown')}",
        f"- Single source: {bool(supplier.get('single_source', False))}",
        f"- Risk tier: {str(risk_label).upper()}",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## KPI Snapshot",
    ]
    for key in ["avg_ppm_3m", "avg_otd_3m", "avg_audit_score_3m", "avg_scar_count_3m", "composite_risk_score"]:
        if hasattr(risk_row, "get") and key in risk_row:
            lines.append(f"- {key}: {risk_row.get(key)}")

    lines.extend(["", "## Claims"])
    if sup_claims.empty:
        lines.append("- No claims on record.")
    else:
        for _, row in sup_claims.sort_values("creation_date", ascending=False).head(10).iterrows():
            lines.append(f"- {row.get('incident_number')}: {row.get('category')} · {row.get('status')} · bad parts {row.get('number_of_bad_parts')}")

    lines.extend(["", "## Audits"])
    if sup_audits.empty:
        lines.append("- No audits on record.")
    else:
        for _, row in sup_audits.sort_values("audit_date", ascending=False).head(10).iterrows():
            lines.append(f"- {row.get('audit_id')}: {row.get('audit_type')} · score {row.get('audit_score')} · {row.get('highest_finding_type')} · {row.get('status')}")

    lines.extend(["", "## External Events"])
    if sup_events.empty:
        lines.append("- No external events on record.")
    else:
        for _, row in sup_events.sort_values("event_date", ascending=False).head(10).iterrows():
            lines.append(f"- {row.get('event_id')}: {row.get('event_type')} · {row.get('severity')} · {row.get('status')}")

    lines.extend(["", "## APQP Projects"])
    if sup_apqp.empty:
        lines.append("- No APQP projects on record.")
    else:
        for _, row in sup_apqp.sort_values("creation_date", ascending=False).head(10).iterrows():
            lines.append(f"- {row.get('project_id')}: {row.get('project_type')} · {row.get('status')} · completion {row.get('completion_pct')}% · delayed {row.get('is_delayed')}")

    lines.extend(["", "## Agent Memory"])
    if not memory:
        lines.append("- No agent memory records.")
    else:
        for item in memory:
            lines.extend([
                f"### {item['agent_name']} ({severity_badge_text(item['severity'])})",
                f"- Subject: {item['subject_id']}",
                f"- Updated: {item['updated_at']}",
                f"- Summary: {item['summary']}",
            ])
            payload = item.get("payload", {})
            for key in ["primary_risk_drivers", "signals", "triggers", "blockers", "risks", "exposure_drivers", "mandatory_actions", "recovery_actions"]:
                values = payload.get(key)
                if isinstance(values, list) and values:
                    lines.append(f"- {key.replace('_', ' ').title()}:")
                    for value in values[:6]:
                        if isinstance(value, dict):
                            lines.append(f"  - {value.get('action') or value.get('issue') or value}")
                        else:
                            lines.append(f"  - {value}")
            docs = payload.get("source_documents", [])
            if docs:
                lines.append(f"- Source documents: {', '.join(docs)}")
            lines.append("")

    lines.extend(["", "## Agent Run History"])
    if not agent_runs:
        lines.append("- No agent runs recorded.")
    else:
        for run in agent_runs:
            lines.extend([
                f"### Run {run['run_id'][:10]} ({run['status']})",
                f"- Started: {run['started_at']}",
                f"- Finished: {run.get('finished_at') or ''}",
                f"- Summary: {run['summary']}",
            ])
            for step in run.get("steps", []):
                lines.append(
                    f"  - {step['agent_name']}: {operator_status_label(step['status'])} "
                    f"({normalize_severity(step['severity'])}) - {step['error'] or step['summary']}"
                )

    return "\n".join(lines)


@st.cache_data(ttl=300, show_spinner=False)
def classify_portfolio_intent(q: str) -> dict:
    """
    LLM intent classifier for portfolio Q&A queries.
    Top-level cached function — not redefined on every render cycle.
    Returns structured intent dict for routing to the correct data query.
    """
    from litellm import completion as _completion
    import json as _json
    prompt = f"""Classify this supplier portfolio query into a structured filter.

Query: {q}

Return ONLY a JSON object with these exact fields (no markdown, no explanation):
{{
  "intent": "red_risk" | "single_source" | "ppm_threshold" | "audit_findings" | "capa_events" | "geopolitical" | "apqp_delayed" | "general",
  "country": "country name or null",
  "ppm_threshold": number or null,
  "risk_tier": "red" | "amber" | "green" | null,
  "finding_type": "Major NCR" | "Critical NCR" | "Minor NCR" | null
}}"""
    try:
        resp = _completion(
            model="groq/openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=120,
        )
        text = resp.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return _json.loads(text)
    except Exception:
        return {"intent": "general", "country": None,
                "ppm_threshold": None, "risk_tier": None, "finding_type": None}


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

if suppliers.empty or risk_scores.empty:
    st.error("Supplier portfolio data is not available.")
    st.info("Generate the dataset first with `uv run python generate_supplier_data.py --out data/`.")
    st.stop()

init_agent_memory(DB_PATH)

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
        ["Executive Portfolio", "Agent Command Center", "Risk Scoring Engine", "Early Warning Agent", "SCAR/CAPA Triage", "APQP Readiness Agent", "Continuity Agent", "Audit Planning Agent", "Supplier Profile",
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

if filtered_suppliers.empty:
    st.info("No suppliers match the current sidebar filters. Clear or adjust the filters to continue.")
    st.stop()


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
        st.markdown(
            '<div style="font-size:0.72rem; color:#475569; margin:-0.3rem 0 0.6rem 0;">'
            'Ranked by rule-based composite score. '
            '<span style="font-family:\'DM Mono\',monospace;">ML:</span> badge = RandomForest prediction. '
            '<span style="color:#fb923c; font-weight:600;">⚠ diverges</span> '
            '= models disagree — investigate before acting.'
            '</div>',
            unsafe_allow_html=True,
        )
        top_risk = filtered_risk[
            filtered_risk["risk_label"].isin(["red", "amber"])
        ].drop(columns=["product_family"], errors="ignore").merge(
            filtered_suppliers[["supplier_id", "name", "product_family"]], on="supplier_id"
        ).sort_values(
            ["risk_label", "composite_risk_score"],
            ascending=[True, False],   # red sorts before amber ("red" < "amber" alphabetically = False, so True keeps red first)
            key=lambda col: col.map({"red": 0, "amber": 1}) if col.name == "risk_label" else col,
        ).head(10)

        for _, row in top_risk.iterrows():
            score    = row["composite_risk_score"]
            label    = row["risk_label"]
            color    = risk_color(label)
            ml_label = get_ml_pred_label(ml, row["supplier_id"]) if ml else None
            ml_html  = ml_predicted_badge(ml, row["supplier_id"]) if ml else ""
            diverges = ml_label is not None and ml_label != label
            diverge_html = (
                '<span style="font-size:0.68rem; color:#fb923c; font-weight:600; '
                'margin-left:0.4rem;">⚠ diverges</span>'
                if diverges else ""
            )
            st.markdown(f"""
            <div class="alert-card {'amber' if label=='amber' else ('green' if label=='green' else '')}">
                <div style="font-size:0.78rem; font-weight:600; color:#f1f5f9;">{row['name'][:32]}</div>
                <div style="font-size:0.7rem; color:#64748b;">{row['product_family']}</div>
                <div style="display:flex; justify-content:space-between; margin-top:0.3rem; align-items:center;">
                    <div>{risk_badge(label)} {ml_html}{diverge_html}</div>
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
    ).sort_values("composite_risk_score", ascending=False).head(3)
    top_names = ", ".join(top_red["name"].str[:20].tolist()) if len(top_red) > 0 else "none identified"

    with st.spinner("Generating executive brief..."):
        summary_text = generate_executive_summary(
            n_suppliers=total_suppliers,
            n_regions=len(filtered_suppliers["region"].unique()),
            n_red=n_red,
            n_amber=n_amber,
            n_green=n_green,
            high_risk_pct=high_risk_pct,
            high_risk_spend=high_risk_spend,
            single_source_red=single_source_red,
            open_events=open_events,
            programs_at_risk=programs_at_risk,
            top_red_names=top_names,
        )

    st.markdown(f"""
    <div class="ai-summary">
        <div class="ai-badge">AI Generated · OSS-120B · Portfolio data as of today</div>
        <div style="font-size:0.85rem; color:#cbd5e1; line-height:1.65;">
            {summary_text.replace(chr(10), '<br>')}
        </div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 2: AGENT COMMAND CENTER
# ═══════════════════════════════════════════════════════════════════════

elif page == "Agent Command Center":
    st.markdown('<div class="page-title">Agent Command Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Shared memory view across supplier workflow agents</div>', unsafe_allow_html=True)

    supplier_options = filtered_suppliers[["supplier_id", "name"]].copy()
    supplier_options["display"] = supplier_options["name"] + " (" + supplier_options["supplier_id"] + ")"
    selected_supplier = st.selectbox("Select supplier", supplier_options["display"].tolist())
    sid = selected_supplier.split("(")[-1].replace(")", "").strip()

    sup = suppliers[suppliers["supplier_id"] == sid].iloc[0]
    risk_match = risk_scores[risk_scores["supplier_id"] == sid]
    risk_row = risk_match.iloc[0] if not risk_match.empty else {}
    sup_kpis = kpis[kpis["supplier_id"] == sid].sort_values("year_month")
    sup_claims = claims[claims["supplier_id"] == sid]
    sup_audits = audits[audits["supplier_id"] == sid]
    sup_events = events[events["supplier_id"] == sid]
    sup_apqp = apqp[apqp["supplier_id"] == sid]
    memory = get_supplier_memory(DB_PATH, sid)
    fresh_memory = is_memory_fresh(memory)
    agent_runs = get_supplier_agent_runs(DB_PATH, sid, limit=5)
    stale_records = [item for item in memory if (memory_age_hours(item) is None or memory_age_hours(item) > 24)]

    sync_col, cleanup_col, export_col = st.columns([1, 1, 1])
    with sync_col:
        force_sync = st.checkbox("Force refresh", value=False)
        run_sync = st.button("Run Agent Sync", type="primary", disabled=(fresh_memory and not force_sync))
    with cleanup_col:
        clear_stale = st.button("Clear Stale Memory", disabled=not stale_records)
        if clear_stale:
            deleted = clear_stale_supplier_memory(DB_PATH, sid, max_age_hours=24)
            st.success(f"Cleared {deleted} stale memory record(s).")
            memory = get_supplier_memory(DB_PATH, sid)
            fresh_memory = is_memory_fresh(memory)
            stale_records = [item for item in memory if (memory_age_hours(item) is None or memory_age_hours(item) > 24)]
    with export_col:
        st.markdown(
            f'<div style="font-size:0.78rem; color:#64748b; padding-top:0.45rem;">'
            f'Memory status: {"fresh" if fresh_memory else "missing or stale"}'
            f'</div>',
            unsafe_allow_html=True,
        )

    if stale_records:
        stale_names = ", ".join(sorted({item["agent_name"] for item in stale_records}))
        st.warning(f"{len(stale_records)} stale agent memory record(s) detected: {stale_names}. Refresh or clear stale memory before using this supplier pack operationally.")

    if agent_runs:
        st.download_button(
            "Download Run Log",
            data=build_run_log_markdown(sid, agent_runs),
            file_name=f"{sid}_agent_run_log.md",
            mime="text/markdown",
            key=f"download_agent_run_log_{sid}",
        )

    if run_sync:
        run_id = start_agent_run(DB_PATH, sid)
        sync_results = []

        def log_step(agent_name, status, summary="", severity="info", error="", started_at=None):
            record_agent_run_step(
                DB_PATH,
                run_id=run_id,
                agent_name=agent_name,
                supplier_id=sid,
                status=status,
                summary=summary,
                severity=severity,
                error=error,
                started_at=started_at,
            )
            sync_results.append({
                "Agent": agent_name,
                "Status": status,
                "Severity": normalize_severity(severity),
                "Summary": summary,
                "Error": error,
            })

        def step_started_at():
            return datetime.now(timezone.utc).isoformat(timespec="seconds")

        started_at = step_started_at()
        try:
            brief = generate_supplier_development_brief(
                supplier=sup,
                risk_row=risk_row,
                kpis=sup_kpis,
                claims=sup_claims,
                audits=sup_audits,
                events=sup_events,
                apqp=sup_apqp,
                use_llm=False,
            )
            brief_severity = {"red": "high", "amber": "medium", "green": "low"}.get(brief.risk_level, "info")
            brief_summary = f"{brief.recommended_pathway}: {len(brief.development_actions)} action(s)"
            remember_agent_output(
                DB_PATH,
                agent_name="Supplier Intake Agent",
                supplier_id=sid,
                subject_id="development_brief",
                severity=brief_severity,
                summary=brief_summary,
                payload=brief.model_dump(),
            )
            log_step("Supplier Intake Agent", "success", brief_summary, brief_severity, started_at=started_at)
        except Exception as exc:
            log_step("Supplier Intake Agent", "failed", "Development brief failed", "critical", str(exc), started_at)

        started_at = step_started_at()
        try:
            alerts = build_supplier_trend_alerts(
                suppliers=suppliers,
                kpis=kpis,
                risk_scores=risk_scores,
                claims=claims,
                audits=audits,
                events=events,
                apqp=apqp,
                supplier_ids={sid},
                top_n=1,
            )
            if alerts:
                alert = alerts[0]
                alert_summary = f"{alert.direction.title()} detected, score {alert.trend_score:.1f}/100"
                remember_agent_output(
                    DB_PATH,
                    agent_name="Early Warning Agent",
                    supplier_id=sid,
                    subject_id="trend_alert",
                    severity=alert.alert_level,
                    summary=alert_summary,
                    payload=alert.model_dump(),
                )
                log_step("Early Warning Agent", "success", alert_summary, alert.alert_level, started_at=started_at)
            else:
                log_step("Early Warning Agent", "skipped", "No material trend alert detected", "info", started_at=started_at)
        except Exception as exc:
            log_step("Early Warning Agent", "failed", "Trend alert failed", "critical", str(exc), started_at)

        started_at = step_started_at()
        try:
            if bool(sup.get("single_source", False)):
                continuity = assess_single_source_continuity(
                    supplier=sup,
                    risk_row=risk_row,
                    supplier_claims=sup_claims,
                    supplier_events=sup_events,
                    supplier_apqp=sup_apqp,
                )
                continuity_summary = f"{continuity.continuity_level.title()} continuity exposure, buffer target {continuity.buffer_stock_target_days}"
                remember_agent_output(
                    DB_PATH,
                    agent_name="Continuity Agent",
                    supplier_id=sid,
                    subject_id="single_source_plan",
                    severity=continuity.continuity_level,
                    summary=continuity_summary,
                    payload=continuity.model_dump(),
                )
                log_step("Continuity Agent", "success", continuity_summary, continuity.continuity_level, started_at=started_at)
            else:
                log_step("Continuity Agent", "skipped", "Supplier is not marked single source", "info", started_at=started_at)
        except Exception as exc:
            log_step("Continuity Agent", "failed", "Continuity plan failed", "critical", str(exc), started_at)

        started_at = step_started_at()
        try:
            audit_plan = plan_supplier_audit(
                supplier=sup,
                risk_row=risk_row,
                supplier_kpis=sup_kpis,
                supplier_claims=sup_claims,
                supplier_audits=sup_audits,
                supplier_events=sup_events,
            )
            audit_severity = {
                "immediate": "critical",
                "high": "high",
                "medium": "medium",
                "scheduled": "low",
            }.get(audit_plan.urgency, "info")
            audit_summary = f"{audit_plan.audit_type} · {audit_plan.urgency} · {audit_plan.schedule_timeline}"
            remember_agent_output(
                DB_PATH,
                agent_name="Audit Planning Agent",
                supplier_id=sid,
                subject_id="audit_plan",
                severity=audit_severity,
                summary=audit_summary,
                payload=audit_plan.model_dump(),
            )
            log_step("Audit Planning Agent", "success", audit_summary, audit_severity, started_at=started_at)
        except Exception as exc:
            log_step("Audit Planning Agent", "failed", "Audit planning failed", "critical", str(exc), started_at)

        started_at = step_started_at()
        try:
            if not sup_apqp.empty:
                apqp_decisions = [
                    assess_apqp_launch_readiness(
                        project=row,
                        supplier=sup,
                        risk_row=risk_row,
                        supplier_claims=sup_claims,
                        supplier_events=sup_events,
                    )
                    for _, row in sup_apqp.iterrows()
                ]
                worst = sorted(apqp_decisions, key=lambda item: item.readiness_score)[0]
                apqp_severity = {
                    "HOLD": "critical",
                    "CONDITIONAL_GO": "medium",
                    "GO": "low",
                }.get(worst.launch_decision, "info")
                apqp_summary = f"{worst.launch_decision.replace('_', ' ')} for {worst.project_id}, score {worst.readiness_score:.1f}/100"
                remember_agent_output(
                    DB_PATH,
                    agent_name="APQP Readiness Agent",
                    supplier_id=sid,
                    subject_id=f"apqp_{worst.project_id}",
                    severity=apqp_severity,
                    summary=apqp_summary,
                    payload=worst.model_dump(),
                )
                log_step("APQP Readiness Agent", "success", apqp_summary, apqp_severity, started_at=started_at)
            else:
                log_step("APQP Readiness Agent", "skipped", "No APQP project found for supplier", "info", started_at=started_at)
        except Exception as exc:
            log_step("APQP Readiness Agent", "failed", "APQP readiness failed", "critical", str(exc), started_at)

        started_at = step_started_at()
        try:
            open_claims = sup_claims[sup_claims["status"].astype(str).str.lower() != "closed"] if not sup_claims.empty else pd.DataFrame()
            if not open_claims.empty:
                claim = open_claims.sort_values("creation_date", ascending=False).iloc[0]
                triage = triage_claim(
                    claim=claim,
                    supplier=sup,
                    risk_row=risk_row,
                    supplier_claims=sup_claims,
                    supplier_kpis=sup_kpis,
                    supplier_audits=sup_audits,
                    supplier_events=sup_events,
                )
                scar_severity = {
                    "Critical NCR": "critical",
                    "Major NCR": "high",
                    "Minor NCR": "medium",
                    "Observation": "low",
                }.get(triage.finding_grade, "info")
                scar_summary = f"{triage.finding_grade} · {triage.scar_escalation_level} · severity {triage.severity_score:.1f}/100"
                remember_agent_output(
                    DB_PATH,
                    agent_name="SCAR/CAPA Triage Agent",
                    supplier_id=sid,
                    subject_id=triage.incident_number,
                    severity=scar_severity,
                    summary=scar_summary,
                    payload=triage.model_dump(),
                )
                log_step("SCAR/CAPA Triage Agent", "success", scar_summary, scar_severity, started_at=started_at)
            else:
                log_step("SCAR/CAPA Triage Agent", "skipped", "No open claim available for triage", "info", started_at=started_at)
        except Exception as exc:
            log_step("SCAR/CAPA Triage Agent", "failed", "SCAR/CAPA triage failed", "critical", str(exc), started_at)

        finish_agent_run(DB_PATH, run_id)
        failed_steps = [row for row in sync_results if row["Status"] == "failed"]
        if failed_steps:
            st.warning(f"Agent sync completed with {len(failed_steps)} failed step(s). See Run History below.")
        else:
            st.success("Agent sync completed and outputs were saved to shared memory.")
        st.dataframe(pd.DataFrame(sync_results), use_container_width=True, hide_index=True)
        memory = get_supplier_memory(DB_PATH, sid)
        fresh_memory = is_memory_fresh(memory)
        agent_runs = get_supplier_agent_runs(DB_PATH, sid, limit=5)
        stale_records = [item for item in memory if (memory_age_hours(item) is None or memory_age_hours(item) > 24)]

    expected_agents = [
        "Supplier Intake Agent",
        "Early Warning Agent",
        "Continuity Agent",
        "Audit Planning Agent",
        "APQP Readiness Agent",
        "SCAR/CAPA Triage Agent",
    ]
    memory_by_agent = {}
    for item in memory:
        memory_by_agent.setdefault(item["agent_name"], []).append(item)
    agent_runs = get_supplier_agent_runs(DB_PATH, sid, limit=5)
    latest_steps = {}
    if agent_runs:
        latest_steps = {step["agent_name"]: step for step in agent_runs[0].get("steps", [])}
    worst_severity = max([severity_rank(item["severity"]) for item in memory], default=0)
    worst_label = {
        4: "critical",
        3: "high",
        2: "medium",
        1: "low",
        0: "info",
    }[worst_severity]

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(kpi_card("Memory Records", f"{len(memory):,}"), unsafe_allow_html=True)
    with m2:
        critical = sum(1 for item in memory if normalize_severity(item["severity"]) == "critical")
        st.markdown(kpi_card("Critical", f"{critical:,}", delta_direction="up" if critical else "flat"), unsafe_allow_html=True)
    with m3:
        high = sum(1 for item in memory if normalize_severity(item["severity"]) == "high")
        st.markdown(kpi_card("High", f"{high:,}", delta_direction="up" if high else "flat"), unsafe_allow_html=True)
    with m4:
        st.markdown(kpi_card("Worst Severity", worst_label.upper()), unsafe_allow_html=True)
    with m5:
        st.markdown(kpi_card("Supplier", sid), unsafe_allow_html=True)

    if agent_runs:
        st.markdown('<div class="section-header">Run History</div>', unsafe_allow_html=True)
        run_rows = [
            {
                "Started": run["started_at"],
                "Finished": run.get("finished_at") or "",
                "Status": operator_status_label(run["status"]),
                "Summary": run["summary"],
                "Run ID": run["run_id"][:10],
            }
            for run in agent_runs
        ]
        st.dataframe(pd.DataFrame(run_rows), use_container_width=True, hide_index=True)

        latest = agent_runs[0]
        failed_latest = [step for step in latest.get("steps", []) if step["status"] == "failed"]
        if failed_latest:
            st.error("Latest Agent Sync has failed steps.")
            for step in failed_latest:
                st.markdown(f"- **{step['agent_name']}**: {step['error'] or step['summary']}")

        with st.expander("Latest Run Step Details", expanded=bool(failed_latest)):
            step_rows = [
                {
                    "Agent": step["agent_name"],
                    "Status": operator_status_label(step["status"]),
                    "Severity": normalize_severity(step["severity"]),
                    "Summary": step["summary"],
                    "Error": step["error"],
                    "Finished": step["finished_at"],
                }
                for step in latest.get("steps", [])
            ]
            st.dataframe(pd.DataFrame(step_rows), use_container_width=True, hide_index=True)

    if not memory:
        st.info("No shared memory records for this supplier yet. Run Agent Sync to populate them.")
    else:
        st.markdown('<div class="section-header">Agent Status</div>', unsafe_allow_html=True)
        status_rows = []
        for agent in expected_agents:
            records = memory_by_agent.get(agent, [])
            latest_step = latest_steps.get(agent)
            if records:
                top = sorted(records, key=lambda item: severity_rank(item["severity"]), reverse=True)[0]
                age = memory_age_hours(top)
                status = "fresh" if age is not None and age <= 24 else "stale"
                if latest_step and latest_step["status"] == "failed":
                    status = "failed"
                status_rows.append({
                    "Agent": agent,
                    "Status": operator_status_label(status),
                    "Severity": normalize_severity(top["severity"]),
                    "Records": len(records),
                    "Last Updated": top["updated_at"],
                    "Summary": latest_step["error"] if latest_step and latest_step["status"] == "failed" else top["summary"],
                })
            else:
                status = "no data"
                summary = "No current memory record"
                if latest_step:
                    status = latest_step["status"]
                    summary = latest_step["error"] or latest_step["summary"]
                status_rows.append({
                    "Agent": agent,
                    "Status": operator_status_label(status),
                    "Severity": "info",
                    "Records": 0,
                    "Last Updated": "",
                    "Summary": summary,
                })
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">Shared Agent Memory</div>', unsafe_allow_html=True)
        mem_df = pd.DataFrame([
            {
                "Agent": item["agent_name"],
                "Severity": normalize_severity(item["severity"]),
                "Subject": item["subject_id"],
                "Summary": item["summary"],
                "Updated": item["updated_at"],
            }
            for item in memory
        ])
        st.dataframe(mem_df, use_container_width=True, hide_index=True)

        evidence_pack = build_evidence_pack_markdown(
            supplier=sup,
            risk_row=risk_row,
            sup_kpis=sup_kpis,
            sup_claims=sup_claims,
            sup_audits=sup_audits,
            sup_events=sup_events,
            sup_apqp=sup_apqp,
            memory=memory,
            agent_runs=agent_runs,
        )
        st.download_button(
            "Download Evidence Pack",
            data=evidence_pack,
            file_name=f"{sid}_agent_evidence_pack.md",
            mime="text/markdown",
            key=f"download_agent_pack_{sid}",
        )

        st.markdown('<div class="section-header">Memory Details</div>', unsafe_allow_html=True)
        for item in memory:
            item_severity = normalize_severity(item["severity"])
            with st.expander(f"{item_severity.upper()} · {item['agent_name']} · {item['summary']}", expanded=item_severity == "critical"):
                payload = item.get("payload", {})
                key_lists = [
                    "primary_risk_drivers", "signals", "triggers", "blockers",
                    "risks", "exposure_drivers", "mandatory_actions",
                    "recovery_actions", "development_actions",
                ]
                for key in key_lists:
                    value = payload.get(key)
                    if isinstance(value, list) and value:
                        st.markdown(f"**{key.replace('_', ' ').title()}**")
                        for entry in value[:8]:
                            if isinstance(entry, dict):
                                st.markdown(f"- {entry.get('action') or entry.get('issue') or entry}")
                            else:
                                st.markdown(f"- {entry}")
                docs = payload.get("source_documents", [])
                if docs:
                    st.markdown("**Source Documents**")
                    st.markdown(", ".join(f"`{doc}`" for doc in docs))


# ═══════════════════════════════════════════════════════════════════════
# PAGE 3: RISK SCORING ENGINE
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
        # Normalize all KPIs to 0–100 quality score (100 = best, 0 = worst)
        # PPM: 0 PPM → 100, 2500+ PPM → 0
        # OTD: 100% → 100, 60% → 0  (linear over 60–100 range)
        # Audit: raw 0–100 (already normalised)
        _ppm_score   = (1 - (top20["avg_ppm_3m"].clip(0, 2500) / 2500)) * 100
        _otd_score   = ((top20["avg_otd_3m"].clip(60, 100) - 60) / 40) * 100
        _audit_score = top20["avg_audit_score_3m"].clip(0, 100)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="PPM Quality (0=2500+, 100=0 PPM)",
                             x=top20["name"].str[:18], y=_ppm_score,
                             marker_color="#f87171", opacity=0.85))
        fig.add_trace(go.Bar(name="OTD Score (0=60%, 100=100%)",
                             x=top20["name"].str[:18], y=_otd_score,
                             marker_color="#fb923c", opacity=0.85))
        fig.add_trace(go.Bar(name="Audit Score (raw 0–100)",
                             x=top20["name"].str[:18], y=_audit_score,
                             marker_color="#60a5fa", opacity=0.85))
        fig.update_layout(barmode="group", xaxis_tickangle=-40,
                          yaxis_title="Quality Score (100 = best)",
                          legend=dict(orientation="h", y=1.12, font_size=10))
        plotly_dark_layout(fig, height=260)
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
        elif display.empty:
            st.info("No suppliers match the current risk scoring filters.")
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
        with st.expander("📊 Global Feature Importance & Model Comparison (mean |SHAP| — RED class)", expanded=False):

            # ── Model comparison table ────────────────────────────────────────
            metrics     = ml.get("metrics", {})
            rf_m        = metrics.get("rf_metrics", {})
            xgb_m       = metrics.get("xgb_metrics", {})
            winner_name = metrics.get("winner", ml.get("winner_name", "RandomForest"))

            if rf_m and xgb_m:
                st.markdown('<div class="section-header">RF vs XGBoost — Model Comparison</div>',
                            unsafe_allow_html=True)

                comparison_rows = [
                    ("Accuracy",  "accuracy"),
                    ("F1 Macro",  "f1_macro"),
                    ("AUC (OvR)", "auc_ovr"),
                    ("F1 Green",  "f1_green"),
                    ("F1 Amber",  "f1_amber"),
                    ("F1 Red ★",  "f1_red"),
                ]

                metrics_list = ["Metric", "RandomForest", "XGBoost", "Winner"]
                rows_data = []
                for label, key in comparison_rows:
                    rf_val  = rf_m.get(key, 0)
                    xgb_val = xgb_m.get(key, 0)
                    winner  = "RF ✓" if rf_val >= xgb_val else "XGB ✓"
                    is_primary = label == "F1 Red ★"
                    rows_data.append((label, rf_val, xgb_val, winner, is_primary))

                # Plotly grouped bar for comparison
                labels  = [r[0] for r in rows_data]
                rf_vals = [r[1] for r in rows_data]
                xgb_vals= [r[2] for r in rows_data]

                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Bar(
                    name="RandomForest",
                    x=labels, y=rf_vals,
                    marker_color="#3b82f6", opacity=0.85,
                    text=[f"{v:.3f}" for v in rf_vals],
                    textposition="outside", textfont=dict(size=9),
                ))
                fig_cmp.add_trace(go.Bar(
                    name="XGBoost",
                    x=labels, y=xgb_vals,
                    marker_color="#fb923c", opacity=0.85,
                    text=[f"{v:.3f}" for v in xgb_vals],
                    textposition="outside", textfont=dict(size=9),
                ))
                fig_cmp.update_layout(
                    barmode="group",
                    yaxis=dict(range=[0.80, 0.97], title="Score"),
                    legend=dict(orientation="h", y=1.12),
                    shapes=[dict(
                        type="rect", xref="x", yref="paper",
                        x0=4.5, x1=5.5, y0=0, y1=1,
                        fillcolor="#3b82f6", opacity=0.08,
                        line=dict(width=0),
                    )],
                    annotations=[dict(
                        x=5, y=1.08, xref="x", yref="paper",
                        text="★ Primary criterion", showarrow=False,
                        font=dict(size=9, color="#60a5fa"),
                    )],
                )
                plotly_dark_layout(fig_cmp, height=300)
                st.plotly_chart(fig_cmp, use_container_width=True)

                # Decision rationale
                st.markdown(f"""
                <div style="background:#0f1623; border:1px solid #1e2d45; border-radius:8px;
                            padding:0.75rem 1rem; margin-bottom:1rem; font-size:0.78rem; color:#94a3b8;">
                    <span style="color:#34d399; font-weight:600;">✓ Winner: {winner_name}</span>
                    &nbsp;·&nbsp;
                    Decision criterion: <span style="color:#60a5fa;">F1-Red = {rf_m.get('f1_red',0):.3f} (RF)
                    vs {xgb_m.get('f1_red',0):.3f} (XGB)</span>
                    &nbsp;·&nbsp;
                    Catching RED-risk suppliers is the priority — F1-Red is the tiebreaker.
                    &nbsp;·&nbsp;
                    12% label noise · 80/20 train-test split · seed=42
                </div>""", unsafe_allow_html=True)

            # ── Performance metrics cards ─────────────────────────────────────
            st.markdown(make_ml_metrics_html(ml), unsafe_allow_html=True)

            # ── Feature importance chart ──────────────────────────────────────
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
# PAGE 3: EARLY WARNING AGENT
# ═══════════════════════════════════════════════════════════════════════

elif page == "Early Warning Agent":
    st.markdown('<div class="page-title">Early Warning Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Supplier deterioration detection from KPI trends, events, claims, APQP delays, and continuity exposure</div>', unsafe_allow_html=True)

    alerts = build_supplier_trend_alerts(
        suppliers=suppliers,
        kpis=kpis,
        risk_scores=risk_scores,
        claims=claims,
        audits=audits,
        events=events,
        apqp=apqp,
        supplier_ids=filtered_ids,
        top_n=min(len(filtered_ids), 200),
    )

    alert_df = pd.DataFrame([a.model_dump() for a in alerts])

    ac1, ac2, ac3, ac4 = st.columns(4)
    with ac1:
        st.markdown(kpi_card("Active Alerts", f"{len(alerts):,}"), unsafe_allow_html=True)
    with ac2:
        critical_n = int((alert_df["alert_level"] == "critical").sum()) if not alert_df.empty else 0
        st.markdown(kpi_card("Critical", f"{critical_n:,}", delta_direction="up" if critical_n else "flat"), unsafe_allow_html=True)
    with ac3:
        early_n = int((alert_df["current_risk"] == "green").sum()) if not alert_df.empty else 0
        st.markdown(kpi_card("GREEN Drift", f"{early_n:,}", delta="early deterioration", delta_direction="up" if early_n else "flat"), unsafe_allow_html=True)
    with ac4:
        single_source_n = sum(
            1 for a in alerts
            if any("Single-source" in signal for signal in a.signals)
        )
        st.markdown(kpi_card("Single-Source Exposure", f"{single_source_n:,}", delta_direction="up" if single_source_n else "flat"), unsafe_allow_html=True)

    fc1, fc2 = st.columns([1, 1])
    with fc1:
        level_filter = st.multiselect(
            "Alert level",
            ["critical", "high", "medium", "watch"],
            default=["critical", "high", "medium"],
        )
    with fc2:
        risk_filter = st.multiselect(
            "Current risk",
            ["red", "amber", "green", "unknown"],
            default=[],
            placeholder="All risk tiers",
        )

    if alert_df.empty:
        st.info("No deteriorating suppliers detected for the current filters.")
        st.stop()

    filtered_alerts = alert_df.copy()
    if level_filter:
        filtered_alerts = filtered_alerts[filtered_alerts["alert_level"].isin(level_filter)]
    if risk_filter:
        filtered_alerts = filtered_alerts[filtered_alerts["current_risk"].isin(risk_filter)]

    st.markdown('<div class="section-header">Deterioration Watchlist</div>', unsafe_allow_html=True)
    table = filtered_alerts[[
        "supplier_name", "supplier_id", "current_risk", "alert_level",
        "trend_score", "direction", "escalation_owner", "recommended_action"
    ]].copy()
    table.columns = [
        "Supplier", "ID", "Current Risk", "Alert", "Trend Score",
        "Direction", "Owner", "Recommended Action",
    ]
    st.dataframe(table, use_container_width=True, height=360, hide_index=True)

    st.markdown('<div class="section-header">Alert Evidence</div>', unsafe_allow_html=True)
    selected_options = [
        f"{row.supplier_name} ({row.supplier_id})"
        for row in alerts
        if row.supplier_id in set(filtered_alerts["supplier_id"])
    ]
    if selected_options:
        selected_alert = st.selectbox("Select alert", selected_options)
        selected_id = selected_alert.split("(")[-1].replace(")", "").strip()
        alert = next(a for a in alerts if a.supplier_id == selected_id)

        ec1, ec2 = st.columns([1, 1])
        with ec1:
            st.markdown(f"""
            <div class="ai-summary">
                <div class="ai-badge">Agentic Early Warning</div>
                <div style="font-size:1rem; color:#f1f5f9; font-weight:700;">{alert.supplier_name}</div>
                <div style="font-size:0.8rem; color:#94a3b8; margin-top:0.35rem;">
                    {alert.direction.title()} · {alert.alert_level.upper()} · Score {alert.trend_score:.1f}/100
                </div>
                <div style="font-size:0.82rem; color:#cbd5e1; margin-top:0.8rem;">{alert.recommended_action}</div>
                <div style="font-size:0.72rem; color:#64748b; margin-top:0.5rem;">Owner: {alert.escalation_owner}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-header">Signals</div>', unsafe_allow_html=True)
            for signal in alert.signals:
                st.markdown(f"- {signal}")

        with ec2:
            evidence_df = pd.DataFrame(
                [{"Metric": key, "Value": value} for key, value in alert.evidence.items()]
            )
            evidence_df["Value"] = evidence_df["Value"].astype(str)
            st.dataframe(evidence_df, use_container_width=True, hide_index=True)
            st.markdown('<div class="section-header">SICC Source Documents</div>', unsafe_allow_html=True)
            st.markdown(", ".join(f"`{doc}`" for doc in alert.source_documents))


# ═══════════════════════════════════════════════════════════════════════
# PAGE 4: SCAR / CAPA TRIAGE
# ═══════════════════════════════════════════════════════════════════════

elif page == "SCAR/CAPA Triage":
    st.markdown('<div class="page-title">SCAR/CAPA Triage</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Operational triage for supplier quality escapes, SCAR escalation, CAPA evidence, and closure governance</div>', unsafe_allow_html=True)

    mode = st.radio("Triage mode", ["Existing claim", "Manual issue"], horizontal=True)

    supplier_options = filtered_suppliers[["supplier_id", "name"]].copy()
    supplier_options["display"] = supplier_options["name"] + " (" + supplier_options["supplier_id"] + ")"

    if mode == "Existing claim":
        claim_pool = claims[claims["supplier_id"].isin(filtered_ids)].copy()
        # Default to open/in-progress claims only
        open_statuses = ["Open", "In Progress", "Under Investigation", "Pending"]
        status_options = sorted(claim_pool["status"].dropna().unique())
        default_open = [s for s in open_statuses if s in status_options]
        claim_status_filter = st.multiselect(
            "Claim status", status_options,
            default=default_open or status_options[:1],
            key="scar_claim_status",
        )
        if claim_status_filter:
            claim_pool = claim_pool[claim_pool["status"].isin(claim_status_filter)]
        if claim_pool.empty:
            st.info("No claims match the selected status filter.")
            st.stop()

        claim_pool = claim_pool.sort_values("creation_date", ascending=False)
        claim_pool["display"] = (
            claim_pool["incident_number"].astype(str)
            + " | "
            + claim_pool["supplier_name"].astype(str).str[:34]
            + " | "
            + claim_pool["category"].astype(str)
        )
        # Build lookup dict keyed by display string to avoid fragile string splitting
        _claim_map = {row["display"]: row for _, row in claim_pool.iterrows()}
        selected_claim = st.selectbox("Select claim", list(_claim_map.keys()))
        claim = _claim_map[selected_claim]
        sid = claim["supplier_id"]
    else:
        selected_supplier = st.selectbox("Select supplier", supplier_options["display"].tolist())
        sid = selected_supplier.split("(")[-1].replace(")", "").strip()
        claim = None

    sup = suppliers[suppliers["supplier_id"] == sid].iloc[0]
    risk_match = risk_scores[risk_scores["supplier_id"] == sid]
    risk_row = risk_match.iloc[0] if not risk_match.empty else {}
    sup_claims = claims[claims["supplier_id"] == sid]
    sup_kpis = kpis[kpis["supplier_id"] == sid].sort_values("year_month")
    sup_audits = audits[audits["supplier_id"] == sid]
    sup_events = events[events["supplier_id"] == sid]

    _triage_key = f"scar_triage_{sid}_{mode}_{selected_claim if mode == 'Existing claim' else ''}"

    if mode == "Manual issue":
        mc1, mc2 = st.columns([2, 1])
        with mc1:
            issue_description = st.text_input("Issue", value="Quality escape")
        with mc2:
            detected_at_customer = st.checkbox("Customer / field detected", value=False)
        nc1, nc2, nc3 = st.columns(3)
        with nc1:
            bad_parts = st.number_input("Bad parts", min_value=0, value=50, step=10)
        with nc2:
            suspected_parts = st.number_input("Suspected parts", min_value=0, value=100, step=10)
        with nc3:
            recurrent = st.checkbox("Recurring issue", value=False)

    if st.button("Run Triage", type="primary"):
        with st.spinner("Triaging issue..."):
            if mode == "Manual issue":
                _result = triage_manual_issue(
                    supplier=sup,
                    risk_row=risk_row,
                    issue_description=issue_description,
                    bad_parts=int(bad_parts),
                    suspected_parts=int(suspected_parts),
                    detected_at_customer=detected_at_customer,
                    recurrent=recurrent,
                    supplier_claims=sup_claims,
                    supplier_kpis=sup_kpis,
                    supplier_audits=sup_audits,
                    supplier_events=sup_events,
                )
            else:
                _result = triage_claim(
                    claim=claim,
                    supplier=sup,
                    risk_row=risk_row,
                    supplier_claims=sup_claims,
                    supplier_kpis=sup_kpis,
                    supplier_audits=sup_audits,
                    supplier_events=sup_events,
                )
            st.session_state[_triage_key] = _result.model_dump()

    if _triage_key not in st.session_state:
        st.info("Select a claim or enter an issue above, then click **Run Triage**.")
        st.stop()

    triage = type("_T", (), st.session_state[_triage_key])()  # namespace access
    import types as _types
    triage = _types.SimpleNamespace(**st.session_state[_triage_key])
    # Restore list/dict fields (SimpleNamespace already has them from model_dump)

    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        st.markdown(kpi_card("Finding Grade", triage.finding_grade), unsafe_allow_html=True)
    with tc2:
        st.markdown(kpi_card("SCAR Level", triage.scar_escalation_level), unsafe_allow_html=True)
    with tc3:
        st.markdown(kpi_card("Severity", f"{triage.severity_score:.1f}/100"), unsafe_allow_html=True)
    with tc4:
        st.markdown(kpi_card("Owner", triage.owner[:24]), unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ai-summary">
        <div class="ai-badge">Agentic SCAR/CAPA Triage</div>
        <div style="font-size:1rem; color:#f1f5f9; font-weight:700;">{triage.supplier_name}</div>
        <div style="font-size:0.8rem; color:#94a3b8; margin-top:0.35rem;">
            {triage.incident_number} · {triage.issue_summary}
        </div>
    </div>
    """, unsafe_allow_html=True)

    tr1, tr2 = st.columns(2)
    with tr1:
        st.markdown('<div class="section-header">Escalation Triggers</div>', unsafe_allow_html=True)
        for item in triage.triggers:
            st.markdown(f"- {item}")

        st.markdown('<div class="section-header">Immediate Containment</div>', unsafe_allow_html=True)
        for item in triage.immediate_containment:
            st.markdown(f"- {item}")

    with tr2:
        st.markdown('<div class="section-header">Deadlines</div>', unsafe_allow_html=True)
        deadline_df = pd.DataFrame(
            [{"Phase": key.replace("_", " ").title(), "Timeline": value} for key, value in triage.deadlines.items()]
        )
        st.dataframe(deadline_df, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">Escalation Actions</div>', unsafe_allow_html=True)
        if triage.escalation_actions:
            for item in triage.escalation_actions:
                st.markdown(f"- {item}")
        else:
            st.markdown("- No escalation beyond standard monitoring required from current evidence.")

    ev1, ev2 = st.columns(2)
    with ev1:
        st.markdown('<div class="section-header">Required Evidence</div>', unsafe_allow_html=True)
        for item in triage.required_evidence:
            st.markdown(f"- {item}")
    with ev2:
        st.markdown('<div class="section-header">Closure Criteria</div>', unsafe_allow_html=True)
        for item in triage.closure_criteria:
            st.markdown(f"- {item}")
        st.markdown('<div class="section-header">SICC Source Documents</div>', unsafe_allow_html=True)
        st.markdown(", ".join(f"`{doc}`" for doc in triage.source_documents))

    import json as _json
    _triage_export = _json.dumps(st.session_state[_triage_key], indent=2, default=str)
    _fname = f"scar_triage_{getattr(triage, 'incident_number', sid).replace('/', '-')}.json"
    st.download_button("⬇ Download Triage Report", data=_triage_export,
                       file_name=_fname, mime="application/json")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 5: APQP READINESS AGENT
# ═══════════════════════════════════════════════════════════════════════

elif page == "APQP Readiness Agent":
    st.markdown('<div class="page-title">APQP Readiness Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Launch readiness decisioning from APQP gates, PPAP evidence, supplier risk, claims, and external events</div>', unsafe_allow_html=True)

    project_pool = apqp[apqp["supplier_id"].isin(filtered_ids)].copy()
    if project_pool.empty:
        st.info("No APQP projects available for the current filters.")
        st.stop()

    status_filter = st.multiselect(
        "Programme status",
        sorted(project_pool["status"].dropna().unique()),
        default=[],
        placeholder="All statuses",
    )
    if status_filter:
        project_pool = project_pool[project_pool["status"].isin(status_filter)]

    if project_pool.empty:
        st.info("No APQP projects match the selected status filter.")
        st.stop()

    project_pool["display"] = (
        project_pool["project_id"].astype(str)
        + " | "
        + project_pool["supplier_name"].astype(str).str[:34]
        + " | "
        + project_pool["project_type"].astype(str)
        + " | "
        + project_pool["status"].astype(str)
    )
    project_pool = project_pool.sort_values(["is_delayed", "completion_pct"], ascending=[False, True])
    _proj_map = {row["display"]: row for _, row in project_pool.iterrows()}
    selected_project = st.selectbox("Select APQP project", list(_proj_map.keys()))
    project = _proj_map[selected_project]
    sid = project["supplier_id"]

    sup = suppliers[suppliers["supplier_id"] == sid].iloc[0]
    risk_match = risk_scores[risk_scores["supplier_id"] == sid]
    risk_row = risk_match.iloc[0] if not risk_match.empty else {}
    sup_claims = claims[claims["supplier_id"] == sid]
    sup_events = events[events["supplier_id"] == sid]

    _apqp_key = f"apqp_decision_{project['project_id']}"

    if st.button("Assess Readiness", type="primary"):
        with st.spinner("Assessing launch readiness..."):
            _dec = assess_apqp_launch_readiness(
                project=project,
                supplier=sup,
                risk_row=risk_row,
                supplier_claims=sup_claims,
                supplier_events=sup_events,
            )
            st.session_state[_apqp_key] = _dec.model_dump()

    if _apqp_key not in st.session_state:
        st.info("Select a project above, then click **Assess Readiness**.")
        st.stop()

    import types as _types
    decision = _types.SimpleNamespace(**st.session_state[_apqp_key])
    decision.gate_findings = [
        _types.SimpleNamespace(**gf) for gf in (st.session_state[_apqp_key].get("gate_findings") or [])
    ]

    dc1, dc2, dc3, dc4 = st.columns(4)
    with dc1:
        st.markdown(kpi_card("Decision", decision.launch_decision.replace("_", " ")), unsafe_allow_html=True)
    with dc2:
        st.markdown(kpi_card("Readiness", f"{decision.readiness_score:.1f}/100"), unsafe_allow_html=True)
    with dc3:
        st.markdown(kpi_card("Blockers", f"{len(decision.blockers):,}", delta_direction="up" if decision.blockers else "flat"), unsafe_allow_html=True)
    with dc4:
        st.markdown(kpi_card("Owner", decision.owner[:24]), unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ai-summary">
        <div class="ai-badge">Agentic APQP Readiness</div>
        <div style="font-size:1rem; color:#f1f5f9; font-weight:700;">{decision.supplier_name}</div>
        <div style="font-size:0.8rem; color:#94a3b8; margin-top:0.35rem;">
            {decision.project_id} · {project['project_type']} · {project['status']} · completion {project['completion_pct']:.0f}%
        </div>
        <div style="font-size:0.84rem; color:#cbd5e1; margin-top:0.8rem;">{decision.decision_summary}</div>
    </div>
    """, unsafe_allow_html=True)

    ar1, ar2 = st.columns(2)
    with ar1:
        st.markdown('<div class="section-header">Launch Blockers</div>', unsafe_allow_html=True)
        if decision.blockers:
            for blocker in decision.blockers:
                st.markdown(f"- {blocker}")
        else:
            st.markdown("- No launch blockers detected.")

    with ar2:
        st.markdown('<div class="section-header">Launch Risks</div>', unsafe_allow_html=True)
        if decision.risks:
            for risk in decision.risks:
                st.markdown(f"- {risk}")
        else:
            st.markdown("- No conditional launch risks detected.")

    st.markdown('<div class="section-header">Gate Findings</div>', unsafe_allow_html=True)
    if decision.gate_findings:
        gate_df = pd.DataFrame([vars(f) for f in decision.gate_findings])
        gate_df.columns = ["Gate", "Status", "Issue", "Required Action"]
        st.dataframe(gate_df, use_container_width=True, hide_index=True)
    else:
        st.info("Required pre-SOP gates are validated.")

    rr1, rr2 = st.columns(2)
    with rr1:
        st.markdown('<div class="section-header">Recovery Actions</div>', unsafe_allow_html=True)
        for action in decision.recovery_actions:
            st.markdown(f"- {action}")
    with rr2:
        st.markdown('<div class="section-header">Required Evidence</div>', unsafe_allow_html=True)
        for item in decision.required_evidence:
            st.markdown(f"- {item}")
        st.markdown('<div class="section-header">SICC Source Documents</div>', unsafe_allow_html=True)
        st.markdown(", ".join(f"`{doc}`" for doc in decision.source_documents))


# ═══════════════════════════════════════════════════════════════════════
# PAGE 6: SINGLE-SOURCE CONTINUITY AGENT
# ═══════════════════════════════════════════════════════════════════════

elif page == "Continuity Agent":
    st.markdown('<div class="page-title">Continuity Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Single-source continuity mitigation, buffer stock targets, BCP controls, and dual-source urgency</div>', unsafe_allow_html=True)

    plans = build_continuity_watchlist(
        suppliers=suppliers,
        risk_scores=risk_scores,
        claims=claims,
        events=events,
        apqp=apqp,
        supplier_ids=filtered_ids,
        top_n=min(len(filtered_ids), 200),
    )
    plan_df = pd.DataFrame([p.model_dump() for p in plans])

    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1:
        st.markdown(kpi_card("Single-Source Watchlist", f"{len(plans):,}"), unsafe_allow_html=True)
    with cc2:
        critical_n = int((plan_df["continuity_level"] == "critical").sum()) if not plan_df.empty else 0
        st.markdown(kpi_card("Critical", f"{critical_n:,}", delta_direction="up" if critical_n else "flat"), unsafe_allow_html=True)
    with cc3:
        red_n = int((plan_df["risk_tier"] == "red").sum()) if not plan_df.empty else 0
        st.markdown(kpi_card("RED Single Source", f"{red_n:,}", delta_direction="up" if red_n else "flat"), unsafe_allow_html=True)
    with cc4:
        avg_score = plan_df["continuity_score"].mean() if not plan_df.empty else 0
        st.markdown(kpi_card("Avg Exposure", f"{avg_score:.1f}/100"), unsafe_allow_html=True)

    if plan_df.empty:
        st.info("No single-source suppliers match the current filters.")
        st.stop()

    cf1, cf2 = st.columns(2)
    with cf1:
        level_filter = st.multiselect(
            "Continuity level",
            ["critical", "high", "medium", "monitor"],
            default=["critical", "high", "medium"],
        )
    with cf2:
        risk_filter = st.multiselect(
            "Risk tier",
            ["red", "amber", "green", "unknown"],
            default=[],
            placeholder="All risk tiers",
        )

    filtered_plans = plan_df.copy()
    if level_filter:
        filtered_plans = filtered_plans[filtered_plans["continuity_level"].isin(level_filter)]
    if risk_filter:
        filtered_plans = filtered_plans[filtered_plans["risk_tier"].isin(risk_filter)]

    st.markdown('<div class="section-header">Single-Source Continuity Watchlist</div>', unsafe_allow_html=True)
    watchlist = filtered_plans[[
        "supplier_name", "supplier_id", "risk_tier", "continuity_level",
        "continuity_score", "buffer_stock_target_days", "assessment_frequency",
        "escalation_owner",
    ]].copy()
    watchlist.columns = [
        "Supplier", "ID", "Risk Tier", "Continuity", "Score",
        "Buffer Target", "Assessment", "Owner",
    ]
    st.dataframe(watchlist, use_container_width=True, height=340, hide_index=True)

    selected_options = [
        f"{plan.supplier_name} ({plan.supplier_id})"
        for plan in plans
        if plan.supplier_id in set(filtered_plans["supplier_id"])
    ]
    if not selected_options:
        st.info("No suppliers match the selected continuity filters.")
        st.stop()

    st.markdown('<div class="section-header">Mitigation Plan</div>', unsafe_allow_html=True)
    selected_plan = st.selectbox("Select supplier", selected_options)
    selected_id = selected_plan.split("(")[-1].replace(")", "").strip()
    plan = next(p for p in plans if p.supplier_id == selected_id)

    st.markdown(f"""
    <div class="ai-summary">
        <div class="ai-badge">Agentic Continuity Plan</div>
        <div style="font-size:1rem; color:#f1f5f9; font-weight:700;">{plan.supplier_name}</div>
        <div style="font-size:0.8rem; color:#94a3b8; margin-top:0.35rem;">
            {plan.continuity_level.upper()} · Risk {plan.risk_tier.upper()} · Score {plan.continuity_score:.1f}/100 · Buffer target {plan.buffer_stock_target_days}
        </div>
        <div style="font-size:0.84rem; color:#cbd5e1; margin-top:0.8rem;">{plan.decision_summary}</div>
        <div style="font-size:0.72rem; color:#64748b; margin-top:0.5rem;">Owner: {plan.escalation_owner} · Review: {plan.assessment_frequency}</div>
    </div>
    """, unsafe_allow_html=True)

    cp1, cp2 = st.columns(2)
    with cp1:
        st.markdown('<div class="section-header">Exposure Drivers</div>', unsafe_allow_html=True)
        for item in plan.exposure_drivers:
            st.markdown(f"- {item}")

        st.markdown('<div class="section-header">Mandatory Actions</div>', unsafe_allow_html=True)
        for item in plan.mandatory_actions:
            st.markdown(f"- {item}")

    with cp2:
        st.markdown('<div class="section-header">Dual-Sourcing Actions</div>', unsafe_allow_html=True)
        for item in plan.dual_sourcing_actions:
            st.markdown(f"- {item}")

        st.markdown('<div class="section-header">BCP Controls</div>', unsafe_allow_html=True)
        for item in plan.bcp_controls:
            st.markdown(f"- {item}")

        st.markdown('<div class="section-header">SICC Source Documents</div>', unsafe_allow_html=True)
        st.markdown(", ".join(f"`{doc}`" for doc in plan.source_documents))

    import json as _json
    _cont_export = _json.dumps(
        next(p for p in plans if p.supplier_id == selected_id).model_dump(),
        indent=2, default=str)
    st.download_button("⬇ Download Continuity Plan", data=_cont_export,
                       file_name=f"continuity_plan_{selected_id}.json",
                       mime="application/json")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 7: AUDIT PLANNING AGENT
# ═══════════════════════════════════════════════════════════════════════

elif page == "Audit Planning Agent":
    st.markdown('<div class="page-title">Audit Planning Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">For-cause audit trigger detection, scope planning, evidence requests, and audit scheduling guidance</div>', unsafe_allow_html=True)

    audit_plans = build_audit_plan_watchlist(
        suppliers=suppliers,
        risk_scores=risk_scores,
        kpis=kpis,
        claims=claims,
        audits=audits,
        events=events,
        supplier_ids=filtered_ids,
        top_n=min(len(filtered_ids), 200),
    )
    audit_df = pd.DataFrame([p.model_dump() for p in audit_plans])

    au1, au2, au3, au4 = st.columns(4)
    with au1:
        st.markdown(kpi_card("Audit Triggers", f"{len(audit_plans):,}"), unsafe_allow_html=True)
    with au2:
        immediate_n = int((audit_df["urgency"] == "immediate").sum()) if not audit_df.empty else 0
        st.markdown(kpi_card("Immediate", f"{immediate_n:,}", delta_direction="up" if immediate_n else "flat"), unsafe_allow_html=True)
    with au3:
        for_cause_n = int((audit_df["audit_type"] == "For-Cause Audit").sum()) if not audit_df.empty else 0
        st.markdown(kpi_card("For-Cause", f"{for_cause_n:,}", delta_direction="up" if for_cause_n else "flat"), unsafe_allow_html=True)
    with au4:
        red_related = sum(1 for p in audit_plans if any("RED" in t or "red" in t for t in p.triggers))
        st.markdown(kpi_card("RED Linked", f"{red_related:,}", delta_direction="up" if red_related else "flat"), unsafe_allow_html=True)

    if audit_df.empty:
        st.info("No audit triggers detected for the current filters.")
        st.stop()

    af1, af2 = st.columns(2)
    with af1:
        urgency_filter = st.multiselect(
            "Urgency",
            ["immediate", "high", "medium", "scheduled"],
            default=["immediate", "high", "medium"],
        )
    with af2:
        type_filter = st.multiselect(
            "Audit type",
            sorted(audit_df["audit_type"].unique()),
            default=[],
            placeholder="All audit types",
        )

    filtered_audits = audit_df.copy()
    if urgency_filter:
        filtered_audits = filtered_audits[filtered_audits["urgency"].isin(urgency_filter)]
    if type_filter:
        filtered_audits = filtered_audits[filtered_audits["audit_type"].isin(type_filter)]

    st.markdown('<div class="section-header">Audit Trigger Watchlist</div>', unsafe_allow_html=True)
    import re as _re
    from datetime import date as _date, timedelta as _td
    def _parse_target_date(timeline: str) -> str:
        _m = _re.search(r"(\d+)\s+day", timeline, _re.I)
        if _m:
            return (_date.today() + _td(days=int(_m.group(1)))).isoformat()
        _m = _re.search(r"(\d+)\s+week", timeline, _re.I)
        if _m:
            return (_date.today() + _td(weeks=int(_m.group(1)))).isoformat()
        _m = _re.search(r"(\d+)\s+month", timeline, _re.I)
        if _m:
            return (_date.today() + _td(days=int(_m.group(1)) * 30)).isoformat()
        return "—"
    table = filtered_audits[[
        "supplier_name", "supplier_id", "audit_type", "urgency",
        "schedule_timeline", "owner"
    ]].copy()
    table["target_date"] = table["schedule_timeline"].apply(_parse_target_date)
    table.columns = ["Supplier", "ID", "Audit Type", "Urgency", "Timeline", "Owner", "Target Date"]
    st.dataframe(table, use_container_width=True, height=340, hide_index=True)

    selected_options = [
        f"{plan.supplier_name} ({plan.supplier_id})"
        for plan in audit_plans
        if plan.supplier_id in set(filtered_audits["supplier_id"])
    ]
    if not selected_options:
        st.info("No suppliers match the selected audit filters.")
        st.stop()

    st.markdown('<div class="section-header">Audit Plan</div>', unsafe_allow_html=True)
    selected_plan = st.selectbox("Select supplier", selected_options)
    selected_id = selected_plan.split("(")[-1].replace(")", "").strip()
    plan = next(p for p in audit_plans if p.supplier_id == selected_id)

    st.markdown(f"""
    <div class="ai-summary">
        <div class="ai-badge">Agentic Audit Plan</div>
        <div style="font-size:1rem; color:#f1f5f9; font-weight:700;">{plan.supplier_name}</div>
        <div style="font-size:0.8rem; color:#94a3b8; margin-top:0.35rem;">
            {plan.audit_type} · {plan.urgency.upper()} · {plan.schedule_timeline}
        </div>
        <div style="font-size:0.72rem; color:#64748b; margin-top:0.5rem;">Owner: {plan.owner}</div>
    </div>
    """, unsafe_allow_html=True)

    ap1, ap2 = st.columns(2)
    with ap1:
        st.markdown('<div class="section-header">Triggers</div>', unsafe_allow_html=True)
        for trigger in plan.triggers:
            st.markdown(f"- {trigger}")

        st.markdown('<div class="section-header">Audit Scope</div>', unsafe_allow_html=True)
        for item in plan.audit_scope:
            st.markdown(f"- {item}")

        st.markdown('<div class="section-header">Checklist Focus</div>', unsafe_allow_html=True)
        for item in plan.checklist_focus:
            st.markdown(f"- {item}")

    with ap2:
        st.markdown('<div class="section-header">Evidence To Request</div>', unsafe_allow_html=True)
        for item in plan.evidence_to_request:
            st.markdown(f"- {item}")

        st.markdown('<div class="section-header">Expected Outputs</div>', unsafe_allow_html=True)
        for item in plan.expected_outputs:
            st.markdown(f"- {item}")

        st.markdown('<div class="section-header">SICC Source Documents</div>', unsafe_allow_html=True)
        st.markdown(", ".join(f"`{doc}`" for doc in plan.source_documents))

    import json as _json
    _audit_export = _json.dumps(
        next(p for p in audit_plans if p.supplier_id == selected_id).model_dump(),
        indent=2, default=str)
    st.download_button("⬇ Download Audit Plan", data=_audit_export,
                       file_name=f"audit_plan_{selected_id}.json",
                       mime="application/json")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 8: SUPPLIER PROFILE
# ═══════════════════════════════════════════════════════════════════════

elif page == "Supplier Profile":
    st.markdown('<div class="page-title">Supplier Profile</div>', unsafe_allow_html=True)

    supplier_options = filtered_suppliers[["supplier_id", "name"]].copy()
    if supplier_options.empty:
        st.info("No suppliers match the current sidebar filters.")
        st.stop()
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
            _otd_min = max(0, sup_kpis["otd_pct"].min() - 3)
            fig.update_layout(xaxis_title="", yaxis_title="OTD %",
                              yaxis_range=[min(_otd_min, 80), 101])
            plotly_dark_layout(fig, height=200)
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            fig = px.line(sup_kpis, x="year_month", y="audit_score",
                          color_discrete_sequence=["#60a5fa"])
            fig.add_hline(y=75, line_dash="dot", line_color="#475569")
            fig.add_hline(y=60, line_dash="dot", line_color="#7f1d1d")
            _aud_min = max(0, sup_kpis["audit_score"].min() - 5)
            fig.update_layout(xaxis_title="", yaxis_title="Audit Score",
                              yaxis_range=[min(_aud_min, 40), 101])
            plotly_dark_layout(fig, height=200)
            st.plotly_chart(fig, use_container_width=True)

    # Tabs: Intake agent / Claims / Audits / Events / APQP / ML Explainer
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Development Brief", "Claims", "Audits", "External Events", "APQP Programs", "ML Explainer"])

    with tab0:
        brief_key = f"supplier_development_brief_{sid}"
        c_action, c_meta = st.columns([1, 2])
        with c_action:
            if st.button("Generate Brief", type="primary", key=f"generate_brief_{sid}"):
                with st.spinner("Reviewing supplier evidence and SICC guidance..."):
                    brief = generate_supplier_development_brief(
                        supplier=sup,
                        risk_row=risk_row,
                        kpis=sup_kpis,
                        claims=sup_claims,
                        audits=sup_audits,
                        events=sup_events,
                        apqp=sup_apqp,
                        use_llm=bool(os.getenv("GROQ_API_KEY")),
                    )
                st.session_state[brief_key] = brief.model_dump()
        with c_meta:
            st.markdown(
                '<div style="font-size:0.78rem; color:#64748b; padding-top:0.45rem;">'
                'Creates a governed supplier development brief from portfolio evidence, KPI thresholds, and SICC policy guidance.'
                '</div>',
                unsafe_allow_html=True,
            )

        if brief_key not in st.session_state:
            brief = generate_supplier_development_brief(
                supplier=sup,
                risk_row=risk_row,
                kpis=sup_kpis,
                claims=sup_claims,
                audits=sup_audits,
                events=sup_events,
                apqp=sup_apqp,
                use_llm=False,
            )
            st.session_state[brief_key] = brief.model_dump()

        brief = SupplierDevelopmentBrief.model_validate(st.session_state[brief_key])

        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            st.markdown(kpi_card("Risk Level", brief.risk_level.upper()), unsafe_allow_html=True)
        with bc2:
            st.markdown(kpi_card("Pathway", brief.recommended_pathway[:34]), unsafe_allow_html=True)
        with bc3:
            st.markdown(kpi_card("Generation", brief.generation_mode.replace("_", " ").title()), unsafe_allow_html=True)

        st.markdown('<div class="section-header">Situation Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ai-summary"><div style="font-size:0.84rem; color:#cbd5e1;">{brief.situation_summary}</div></div>', unsafe_allow_html=True)

        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown('<div class="section-header">Risk Drivers</div>', unsafe_allow_html=True)
            for driver in brief.primary_risk_drivers:
                st.markdown(f"- {driver}")
        with dc2:
            st.markdown('<div class="section-header">Identified Gaps</div>', unsafe_allow_html=True)
            for gap in brief.identified_gaps:
                st.markdown(f"- {gap}")

        st.markdown('<div class="section-header">Development Actions</div>', unsafe_allow_html=True)
        action_rows = [
            {
                "priority": action.priority,
                "owner": action.owner,
                "due_date": action.due_date,
                "action": action.action,
                "evidence_required": "; ".join(action.evidence_required),
            }
            for action in brief.development_actions
        ]
        st.dataframe(pd.DataFrame(action_rows), use_container_width=True, hide_index=True)

        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown('<div class="section-header">Escalation Triggers</div>', unsafe_allow_html=True)
            for trigger in brief.escalation_triggers:
                st.markdown(f"- {trigger}")
        with ec2:
            st.markdown('<div class="section-header">Exit Criteria</div>', unsafe_allow_html=True)
            for criterion in brief.exit_criteria:
                st.markdown(f"- {criterion}")

        st.markdown('<div class="section-header">SICC Source Documents</div>', unsafe_allow_html=True)
        st.markdown(", ".join(f"`{doc}`" for doc in brief.source_documents))
        st.download_button(
            "Download Brief",
            data=brief_to_markdown(brief),
            file_name=f"{sid}_supplier_development_brief.md",
            mime="text/markdown",
            key=f"download_brief_{sid}",
        )

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
        avg_str = f"{avg_completion:.0f}%" if not pd.isna(avg_completion) else "—"
        st.markdown(kpi_card("Avg Completion", avg_str), unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns([3, 1])

    with col_l:
        st.markdown('<div class="section-header">Programme List</div>', unsafe_allow_html=True)
        status_filter = st.selectbox("Filter by status",
                                     ["All", "Active", "Delayed", "Completed", "On Hold"])
        if status_filter == "All":
            table_data = apqp_merged
        elif status_filter == "Delayed":
            table_data = apqp_merged[apqp_merged["is_delayed"].isin([1, True])]
        else:
            table_data = apqp_merged[apqp_merged["status"] == status_filter]
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

    # ── APQP Gate Matrix ──────────────────────────────────────────────────────
    with st.expander("📊 APQP Gate Matrix — Phase completion heatmap", expanded=True):
        PHASES = [
            ("supplier_selection",           "1. Supplier\nSelection"),
            ("supplier_nomination",          "2. Supplier\nNomination"),
            ("design_validation_of_process", "3. Design\nValidation"),
            ("process_validation",           "4. Process\nValidation"),
            ("initial_sample_validation",    "5. Initial\nSample"),
            ("start_of_production",          "6. SOP"),
            ("pqa_management",               "7. PQA\nMgmt"),
            ("yearly_is_submission",         "8. Yearly IS\nSubmission"),
            ("ppap_update",                  "9. PPAP\nUpdate"),
        ]
 
        STATUS_SCORE = {
            "Validated":    4,
            "Submitted":    3,
            "In Progress":  2,
            "Overdue":      1,
            "Not Started":  0,
        }
 
        STATUS_COLOR = {
            "Validated":   "#34d399",
            "Submitted":   "#60a5fa",
            "In Progress": "#fb923c",
            "Overdue":     "#f87171",
            "Not Started": "#1e2d45",
        }
 
        # Filter controls
        mc1, mc2, mc3 = st.columns([1, 1, 2])
        with mc1:
            matrix_status = st.selectbox("Programme status",
                ["All", "Active", "Delayed", "On Hold"], key="matrix_status")
        with mc2:
            matrix_family = st.selectbox("Product family",
                ["All"] + sorted(apqp_merged["product_family"].unique().tolist()),
                key="matrix_family")
        with mc3:
            matrix_search = st.text_input("Search supplier name", key="matrix_search",
                                           placeholder="Type to filter...")
 
        matrix_df = apqp_merged.copy()
        if matrix_status != "All":
            matrix_df = matrix_df[matrix_df["status"] == matrix_status]
        if matrix_family != "All":
            matrix_df = matrix_df[matrix_df["product_family"] == matrix_family]
        if matrix_search:
            matrix_df = matrix_df[matrix_df["name"].str.contains(
                matrix_search, case=False, na=False)]
 
        matrix_df = matrix_df.head(40)  # cap at 40 rows for readability
 
        if matrix_df.empty:
            st.info("No programmes match the current filter.")
        else:
            # Build z (score), text, and color matrices
            phase_keys  = [p[0] for p in PHASES]
            phase_labels = [p[1] for p in PHASES]
 
            z_matrix    = []
            text_matrix = []
            color_matrix = []
 
            y_labels = []
            _n_phases = len(phase_keys)
            for _, row in matrix_df.iterrows():
                z_row    = []
                text_row = []
                col_row  = []
                # Derive phase completion from completion_pct + is_delayed
                # since individual phase status columns are not stored in the DB.
                _pct = float(row["completion_pct"]) if row["completion_pct"] <= 1 \
                       else float(row["completion_pct"]) / 100
                _n_done   = max(0, min(_n_phases, round(_pct * _n_phases)))
                _is_delay = bool(row["is_delayed"])
                _is_complete = row.get("status", "") == "Completed"
                for i, pk in enumerate(phase_keys):
                    if _is_complete or i < _n_done - 1:
                        status = "Validated"
                    elif i == _n_done - 1 and _n_done > 0:
                        status = "Overdue" if _is_delay else "Submitted"
                    elif i == _n_done:
                        status = "Overdue" if _is_delay else "In Progress"
                    else:
                        status = "Not Started"
                    z_row.append(STATUS_SCORE.get(status, 0))
                    text_row.append(status[:3].upper() if status != "Not Started" else "—")
                    col_row.append(STATUS_COLOR.get(status, "#1e2d45"))
                z_matrix.append(z_row)
                text_matrix.append(text_row)
                color_matrix.append(col_row)
 
                # Y label: supplier name + project type + risk badge
                risk_row_data = risk_scores[risk_scores["supplier_id"] == row["supplier_id"]]
                risk_lbl = risk_row_data["risk_label"].iloc[0] if not risk_row_data.empty else "green"
                risk_icon = {"red": "🔴", "amber": "🟡", "green": "🟢"}.get(risk_lbl, "⚪")
                y_labels.append(f"{risk_icon} {row['name'][:22]} · {row['project_type'][:12]}")
 
            # Plotly heatmap
            colorscale = [
                [0.00, "#1e2d45"],   # Not Started
                [0.25, "#f87171"],   # Overdue
                [0.50, "#fb923c"],   # In Progress
                [0.75, "#60a5fa"],   # Submitted
                [1.00, "#34d399"],   # Validated
            ]
 
            fig_matrix = go.Figure(go.Heatmap(
                z=z_matrix,
                x=phase_labels,
                y=y_labels,
                text=text_matrix,
                texttemplate="%{text}",
                textfont=dict(size=9, color="#0f1923"),
                colorscale=colorscale,
                zmin=0, zmax=4,
                showscale=False,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Phase: %{x}<br>"
                    "Status: %{text}<extra></extra>"
                ),
                xgap=2,
                ygap=1,
            ))
 
            fig_matrix.update_layout(
                height=max(300, len(matrix_df) * 28 + 80),
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="#0f1923",
                plot_bgcolor="#0f1923",
                font=dict(color="#94a3b8", size=10, family="DM Sans"),
                xaxis=dict(side="top", tickfont=dict(size=9), tickangle=-20),
                yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
            )
            st.plotly_chart(fig_matrix, use_container_width=True)
 
            # Legend
            st.markdown(
                '<div style="display:flex; gap:1rem; font-size:0.7rem; color:#64748b; margin-top:-0.5rem;">'
                + "".join([
                    f'<span><span style="background:{c}; padding:1px 6px; border-radius:3px; '
                    f'color:#0f1923; font-size:0.68rem;">{s[:3].upper()}</span> {s}</span>'
                    for s, c in STATUS_COLOR.items()
                ])
                + "</div>",
                unsafe_allow_html=True,
            )

    # ── SOP Timeline Gantt ─────────────────────────────────────────────────────
    with st.expander("📅 SOP Timeline — Supplier vs Customer dates", expanded=False):
        _gantt_df = apqp_merged[
            apqp_merged["status"].isin(["Active", "On Hold"]) &
            apqp_merged["supplier_sop_date"].notna() &
            apqp_merged["customer_sop_date"].notna()
        ].copy().head(30)
        if _gantt_df.empty:
            st.info("No active programmes with SOP dates available.")
        else:
            _gantt_fig = go.Figure()
            for _, _row in _gantt_df.iterrows():
                _color = "#f87171" if bool(_row["is_delayed"]) else "#34d399"
                _label = f"{_row['name'][:22]} · {_row['project_id']}"
                # Supplier target bar
                _gantt_fig.add_trace(go.Bar(
                    name="Supplier SOP", orientation="h",
                    x=[(_row["supplier_sop_date"] - _row["supplier_sop_date"]).days + 1],
                    base=[_row["supplier_sop_date"]],
                    y=[_label],
                    marker_color=_color, width=0.4,
                    showlegend=False,
                ))
                # Customer deadline marker
                _gantt_fig.add_vline(
                    x=_row["customer_sop_date"].timestamp() * 1000,
                    line_dash="dot", line_color="#60a5fa", line_width=1,
                )
            # Build as scatter timeline instead for clarity
            _gantt_fig = go.Figure()
            _labels, _sup_dates, _cust_dates, _colors = [], [], [], []
            for _, _row in _gantt_df.iterrows():
                _labels.append(f"{_row['name'][:22]} · {_row['project_id']}")
                _sup_dates.append(_row["supplier_sop_date"])
                _cust_dates.append(_row["customer_sop_date"])
                _colors.append("#f87171" if bool(_row["is_delayed"]) else "#34d399")
            _gantt_fig.add_trace(go.Scatter(
                x=_sup_dates, y=_labels, mode="markers",
                name="Supplier SOP",
                marker=dict(color=_colors, size=12, symbol="diamond"),
            ))
            _gantt_fig.add_trace(go.Scatter(
                x=_cust_dates, y=_labels, mode="markers",
                name="Customer SOP deadline",
                marker=dict(color="#60a5fa", size=10, symbol="line-ns-open"),
            ))
            for _sup, _cust, _lbl in zip(_sup_dates, _cust_dates, _labels):
                _gantt_fig.add_shape(type="line",
                    x0=_sup, x1=_cust, y0=_lbl, y1=_lbl,
                    line=dict(color="#475569", width=1.5, dash="dot"))
            _gantt_fig.update_layout(
                xaxis_title="Date", yaxis_title="",
                legend=dict(orientation="h", y=1.08),
                yaxis=dict(autorange="reversed"),
            )
            plotly_dark_layout(_gantt_fig, height=max(280, len(_gantt_df) * 30 + 80))
            st.plotly_chart(_gantt_fig, use_container_width=True)
            st.caption("🔴 Delayed  🟢 On track  🔵 Customer SOP deadline")


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
                            <div style="font-size:0.85rem; color:#cbd5e1; line-height:1.6;">
                                {result.answer}
                            </div>
                            <div style="margin-top:0.75rem; font-size:0.7rem; color:#475569;">
                                Sources: {sources_html}
                            </div>
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

                # Apply sidebar filters first so all intent branches respect them
                result_df = result_df[result_df["supplier_id"].isin(filtered_ids)]

                if filter_family:
                    result_df = result_df[result_df["product_family"].isin(filter_family)]
                if filter_risk:
                    result_df = result_df[result_df["risk_label"].isin(filter_risk)]
                if filter_region:
                    result_df = result_df[result_df["region"].isin(filter_region)]

                intent = classify_portfolio_intent(query)

                answer_text = ""
                show_df     = None

                if intent["intent"] == "red_risk" or intent.get("risk_tier") == "red":
                    tier    = intent.get("risk_tier") or "red"
                    show_df = result_df[result_df["risk_label"] == tier].sort_values("composite_risk_score")
                    answer_text = f"Found **{len(show_df)} {tier.upper()}-risk suppliers** matching your criteria."

                elif intent["intent"] == "single_source":
                    show_df = result_df[result_df["single_source"].isin([1, True])].sort_values("risk_label")
                    answer_text = f"Found **{len(show_df)} single-source suppliers**. {len(show_df[show_df['risk_label']=='red'])} are RED risk."

                elif intent["intent"] == "ppm_threshold":
                    threshold = intent.get("ppm_threshold") or 300
                    show_df   = result_df[result_df["avg_ppm_3m"] > threshold].sort_values("avg_ppm_3m", ascending=False)
                    answer_text = f"Found **{len(show_df)} suppliers** with PPM > {threshold:.0f} in the last 3 months."

                elif intent["intent"] == "audit_findings":
                    finding = intent.get("finding_type") or "Major NCR"
                    if finding not in ["Major NCR", "Critical NCR", "Minor NCR"]:
                        finding = "Major NCR"
                    audit_sup = audits[audits["highest_finding_type"] == finding]["supplier_id"].unique()
                    show_df   = result_df[result_df["supplier_id"].isin(audit_sup)].sort_values("composite_risk_score")
                    answer_text = f"Found **{len(show_df)} suppliers** with **{finding}** audit findings."

                elif intent["intent"] == "claim_categories":
                    _family   = intent.get("product_family")
                    _cl_scope = claims[claims["supplier_id"].isin(filtered_ids)]
                    if _family:
                        _cl_scope = _cl_scope[_cl_scope["product_family"] == _family]
                    _cat_counts = (
                        _cl_scope.groupby("category").size()
                        .reset_index(name="count")
                        .sort_values("count", ascending=False)
                    )
                    show_df = _cat_counts
                    _fam_str = f" for {_family} suppliers" if _family else ""
                    answer_text = f"Top {len(show_df)} recurring claim categories{_fam_str} across {len(_cl_scope):,} claims."

                elif intent["intent"] == "capa_events":
                    capa_needed = events[
                        events["requires_capa"].isin([True, 1]) &
                        ~events["capa_linked"].isin([True, 1]) &
                        events["status"].isin(["Open", "Under Review"])
                    ]["supplier_id"].unique()
                    show_df = result_df[result_df["supplier_id"].isin(capa_needed)].sort_values("composite_risk_score")
                    answer_text = f"Found **{len(show_df)} suppliers** with open alerts and no linked CAPA."

                elif intent["intent"] == "geopolitical":
                    geo_sups = events[
                        (events["event_type"] == "Geopolitical") &
                        (events["severity"].isin(["High", "Critical"])) &
                        (events["status"].isin(["Open", "Under Review", "Escalated"]))
                    ]["supplier_id"].unique()
                    geo_mask = result_df["supplier_id"].isin(geo_sups)
                    country  = intent.get("country")
                    if country:
                        country_mask = result_df["country"].str.lower() == country.lower()
                        show_df      = result_df[geo_mask & country_mask].sort_values("composite_risk_score")
                        answer_text  = f"Found **{len(show_df)} {country}-based suppliers** with active High/Critical geopolitical events."
                    else:
                        show_df     = result_df[geo_mask].sort_values("composite_risk_score")
                        answer_text = f"Found **{len(show_df)} suppliers** with active High/Critical geopolitical events."

                elif intent["intent"] == "apqp_delayed":
                    red_sups     = result_df[result_df["risk_label"] == "red"]["supplier_id"].unique()
                    delayed_apqp = apqp[apqp["is_delayed"] == 1].merge(
                        suppliers[["supplier_id", "name"]], on="supplier_id")
                    show_df     = delayed_apqp[delayed_apqp["supplier_id"].isin(red_sups)]
                    answer_text = f"Found **{len(show_df)} delayed APQP programmes** linked to RED-risk suppliers."

                else:
                    show_df     = result_df.sort_values("composite_risk_score").head(20)
                    answer_text = f"Showing top {len(show_df)} suppliers by risk score. Refine your query for a specific filter."

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
        Knowledge Base mode uses hybrid RAG (BM25 + embedding + RRF) over {get_kb_chunk_count()} KB chunks
        (16 supplier quality documents) via ChromaDB · OSS-120B generator · OSS-20B groundedness checker.
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
        if supplier_options.empty:
            st.info("No suppliers match the current sidebar filters.")
            st.stop()
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
                _active_apqp  = apqp[(apqp["supplier_id"] == sid_sim) & (~apqp["status"].isin(["Completed", "Cancelled"]))]
                prog_impact   = len(_active_apqp)
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

                # Cost escalation chart — shows line-down penalty kicking in after buffer exhausted
                st.markdown('<div class="section-header">Cost Escalation Timeline</div>',
                            unsafe_allow_html=True)
                _buf_days = {"A": 35, "B": 21, "C": 14}.get(str(sup_sim["spend_tier"]), 21)
                _ld_mult  = 5.0 if is_single else 2.5  # line-down penalty multiplier
                _plot_end = duration + 45
                _step     = max(1, _plot_end // 14)
                _day_pts  = sorted(set(
                    [0, _buf_days, duration, _plot_end]
                    + list(range(0, _plot_end + _step, _step))
                ))
                _cum_no_mit, _cum_mit = [], []
                for _d in _day_pts:
                    # No mitigation: normal cost until buffer exhausted, then line-down penalty
                    if _d <= _buf_days:
                        _c_no = daily_cost * _d
                    else:
                        _c_no = daily_cost * _buf_days + daily_cost * _ld_mult * (_d - _buf_days)
                    # With emergency sourcing: 1.35× expedite premium during outage, normal after
                    _c_mit = daily_cost * 1.35 * min(_d, duration) + daily_cost * max(0, _d - duration)
                    _cum_no_mit.append(_c_no / 1000)
                    _cum_mit.append(_c_mit / 1000)
                fig_ot = go.Figure()
                fig_ot.add_trace(go.Scatter(x=_day_pts, y=_cum_no_mit, name="No mitigation",
                                             line=dict(color="#f87171", dash="dash")))
                fig_ot.add_trace(go.Scatter(x=_day_pts, y=_cum_mit, name="With emergency sourcing",
                                             line=dict(color="#34d399")))
                fig_ot.add_vline(x=_buf_days, line_dash="dot", line_color="#fb923c",
                                  annotation_text="Buffer exhausted")
                if duration > _buf_days:
                    fig_ot.add_vline(x=duration, line_dash="dot", line_color="#475569",
                                      annotation_text="Supplier resumes")
                fig_ot.update_layout(xaxis_title="Days from outage start",
                                      yaxis_title="Cumulative cost (€k)",
                                      legend=dict(orientation="h", y=1.1))
                plotly_dark_layout(fig_ot, height=240)
                st.plotly_chart(fig_ot, use_container_width=True)

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

                # 3-year spend projection chart
                st.markdown('<div class="section-header">3-Year Spend Projection</div>',
                            unsafe_allow_html=True)
                _years  = ["Year 1", "Year 2", "Year 3"]
                _base_k = annual_spend / 1000
                # Unmitigated: full increase persists all 3 years
                _inc_k  = [annual_spend * (1 + cost_pct / 100) / 1000] * 3
                # Mitigated: Year 1 full (too late), Year 2 renegotiation achieves 50% offset,
                #            Year 3 resourcing alternative achieves 80% offset
                _mit_k  = [
                    annual_spend * (1 + cost_pct / 100) / 1000,
                    annual_spend * (1 + cost_pct / 100 * 0.50) / 1000,
                    annual_spend * (1 + cost_pct / 100 * 0.20) / 1000,
                ]
                fig_ci = go.Figure()
                fig_ci.add_trace(go.Bar(name="Baseline", x=_years,
                                         y=[_base_k] * 3, marker_color="#3b82f6"))
                fig_ci.add_trace(go.Bar(name="Unmitigated increase", x=_years,
                                         y=_inc_k, marker_color="#f87171"))
                fig_ci.add_trace(go.Bar(name="With renegotiation / resourcing", x=_years,
                                         y=_mit_k, marker_color="#34d399"))
                fig_ci.update_layout(barmode="group", xaxis_title="",
                                      yaxis_title="Annual spend (€k)",
                                      legend=dict(orientation="h", y=1.1))
                plotly_dark_layout(fig_ci, height=240)
                st.plotly_chart(fig_ci, use_container_width=True)

            elif scenario_type == "Region Disruption":
                affected       = suppliers[suppliers["region"] == region_sel]
                affected_risk  = risk_scores[risk_scores["supplier_id"].isin(affected["supplier_id"])]
                affected_spend = affected["annual_spend_eur"].sum()  # already in suppliers table
                n_red          = len(affected_risk[affected_risk["risk_label"] == "red"])
                n_single       = len(affected[affected["single_source"].isin([1, True])])
                n_red_single   = len(affected[
                    affected["single_source"].isin([1, True]) &
                    affected["supplier_id"].isin(
                        affected_risk[affected_risk["risk_label"] == "red"]["supplier_id"]
                    )
                ])

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

                st.markdown('<div class="section-header">Recommended Mitigations</div>',
                            unsafe_allow_html=True)
                _region_actions = [
                    ("⚡ Immediate",
                     f"Activate regional crisis team — assess full exposure across {len(affected)} suppliers in {region_sel}"),
                    ("📋 48 hours",
                     f"Contact all {n_red} RED-risk supplier{'s' if n_red != 1 else ''} for delivery status confirmation and contingency plan"),
                ]
                if n_single > 0:
                    _region_actions.append((
                        "🔴 CRITICAL",
                        f"Escalate {n_single} sole-source supplier{'s' if n_single != 1 else ''} to VP level — no alternative supply available"
                    ))
                if n_red_single > 0:
                    _region_actions.append((
                        "⚡ 7 days",
                        f"Emergency buffer stock for {n_red_single} sole-source RED supplier{'s' if n_red_single != 1 else ''} — target 60-day coverage minimum"
                    ))
                _region_actions.extend([
                    ("📋 14 days",
                     f"Identify alternative sourcing options for sole-source parts — initiate emergency supplier qualification in {region_sel}"),
                    ("🔍 30 days",
                     f"Review safety stock targets for all {region_sel} suppliers and re-evaluate regional concentration risk in supply strategy"),
                ])
                for _priority, _action in _region_actions:
                    st.markdown(f"""
                    <div class="alert-card">
                        <div style="font-size:0.78rem; color:#60a5fa; font-weight:600;">{_priority}</div>
                        <div style="font-size:0.82rem; color:#cbd5e1; margin-top:0.2rem;">{_action}</div>
                    </div>""", unsafe_allow_html=True)

            elif scenario_type == "Production Delay":
                daily_cost    = annual_spend / 365
                direct_cost   = daily_cost * duration
                expedite_cost = direct_cost * 0.25
                total_cost    = direct_cost + expedite_cost
                _active_apqp  = apqp[(apqp["supplier_id"] == sid_sim) & (~apqp["status"].isin(["Completed", "Cancelled"]))]
                prog_impact   = len(_active_apqp)
                schedule_slip = int(duration * 1.3)

                st.markdown(f"""
                <div class="kpi-card" style="margin-bottom:0.75rem;">
                    <div class="kpi-label">Scenario</div>
                    <div style="font-size:0.9rem; color:#f1f5f9; font-weight:600;">
                        Production Delay · {sup_sim['name'][:35]} · {duration} days
                    </div>
                    <div style="font-size:0.78rem; color:#64748b; margin-top:0.25rem;">
                        {sup_sim['product_family']} ·
                        {'⚠ SOLE SOURCE' if is_single else 'Multi-source'} ·
                        Current risk: {risk_badge(risk_label_sim)}
                    </div>
                </div>""", unsafe_allow_html=True)

                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    st.markdown(kpi_card("Cost Impact", f"€{total_cost/1000:.0f}k",
                                         delta=f"Incl. €{expedite_cost/1000:.0f}k expedite",
                                         delta_direction="up"), unsafe_allow_html=True)
                with rc2:
                    st.markdown(kpi_card("Schedule Slip", f"{schedule_slip} days",
                                         delta=f"{prog_impact} programmes affected",
                                         delta_direction="up"), unsafe_allow_html=True)
                with rc3:
                    st.markdown(kpi_card("SOP Risk", f"{prog_impact} progs",
                                         delta="NPI milestones at risk",
                                         delta_direction="up" if prog_impact > 0 else "flat"),
                                unsafe_allow_html=True)

                for action in [
                    f"⚡ Immediate — issue formal delay notification to customer programme managers ({prog_impact} programmes)",
                    "📅 7 days — request updated delivery schedule with weekly milestone confirmation",
                    "📋 14 days — assess expedite options: airfreight, weekend shift, sub-contracting",
                    "🔍 30 days — root cause review to prevent recurrence; update APQP milestone buffer",
                ]:
                    st.markdown(
                        f'<div class="alert-card amber"><div style="font-size:0.82rem; color:#cbd5e1;">{action}</div></div>',
                        unsafe_allow_html=True)

                # Programme exposure chart — completion % coloured by delay status
                if prog_impact > 0 and not _active_apqp.empty:
                    st.markdown('<div class="section-header">Active Programme Exposure</div>',
                                unsafe_allow_html=True)
                    _prog_show = _active_apqp.head(8).copy()
                    _pct_vals  = [
                        float(r["completion_pct"]) * 100 if float(r["completion_pct"]) <= 1
                        else float(r["completion_pct"])
                        for _, r in _prog_show.iterrows()
                    ]
                    _colors = [
                        "#f87171" if bool(r["is_delayed"]) else "#fb923c" if pct > 60 else "#3b82f6"
                        for (_, r), pct in zip(_prog_show.iterrows(), _pct_vals)
                    ]
                    fig_prog = go.Figure(go.Bar(
                        x=[r["project_id"] for _, r in _prog_show.iterrows()],
                        y=_pct_vals,
                        marker_color=_colors,
                        text=[f"{p:.0f}%" for p in _pct_vals],
                        textposition="auto",
                    ))
                    fig_prog.add_hline(y=75, line_dash="dot", line_color="#475569",
                                       annotation_text=f"SOP risk zone — add {schedule_slip}d slip")
                    fig_prog.update_layout(xaxis_title="Programme", yaxis_title="Completion %",
                                           xaxis_tickangle=-30, showlegend=False,
                                           yaxis=dict(range=[0, 105]))
                    plotly_dark_layout(fig_prog, height=230)
                    st.plotly_chart(fig_prog, use_container_width=True)
                    st.caption("🔴 Already delayed  🟠 >60% complete (SOP risk)  🔵 On track")

            elif scenario_type == "Sole-Source Failure":
                # Derive buffer days from spend tier + strategic importance
                _buf_base   = {"A": 35, "B": 22, "C": 14}.get(str(sup_sim["spend_tier"]), 22)
                _imp_adj    = {"Critical": 10, "Preferred": 5,
                               "Approved": 0, "Conditional": -5}.get(
                    str(sup_sim["strategic_importance"]), 0)
                buffer_days = max(7, _buf_base + _imp_adj)

                # Derive recovery days from product family qualification complexity
                _recovery_by_family = {
                    "Software/Firmware": 300, "Optical & Precision": 270,
                    "Electromechanics": 210, "Electronics": 180,
                    "Cables & Harness": 150, "Mechanics - Metal": 150,
                    "Surface Treatment": 135, "Mechanics - Plastic": 120,
                    "Raw Materials": 90,      "Services": 60,
                }
                recovery_days = _recovery_by_family.get(str(sup_sim["product_family"]), 180)

                if not is_single:
                    st.warning(
                        f"⚠ {sup_sim['name'][:40]} is not flagged as sole-source in the portfolio. "
                        "Results below model a sole-source failure for illustration purposes."
                    )

                st.markdown(f"""
                <div class="kpi-card" style="margin-bottom:0.75rem;">
                    <div class="kpi-label">Scenario</div>
                    <div style="font-size:0.9rem; color:#f1f5f9; font-weight:600;">
                        Sole-Source Failure · {sup_sim['name'][:35]}
                    </div>
                    <div style="font-size:0.78rem; color:#64748b; margin-top:0.25rem;">
                        {sup_sim['product_family']} · Annual spend: €{annual_spend/1000:.0f}k ·
                        Current risk: {risk_badge(risk_label_sim)}
                    </div>
                </div>""", unsafe_allow_html=True)

                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    st.markdown(kpi_card("Buffer Stock", f"{buffer_days} days",
                                         delta="Estimated coverage",
                                         delta_direction="up" if buffer_days < 30 else "flat"),
                                unsafe_allow_html=True)
                with rc2:
                    st.markdown(kpi_card("Recovery Timeline", f"{recovery_days} days",
                                         delta="Alt supplier qualification",
                                         delta_direction="up"), unsafe_allow_html=True)
                with rc3:
                    gap_days = max(0, recovery_days - buffer_days)
                    st.markdown(kpi_card("Supply Gap", f"{gap_days} days",
                                         delta="Without emergency action",
                                         delta_direction="up" if gap_days > 0 else "down"),
                                unsafe_allow_html=True)
                with rc4:
                    spend_at_risk = annual_spend * (recovery_days / 365)
                    st.markdown(kpi_card("Spend at Risk", f"€{spend_at_risk/1000:.0f}k",
                                         delta_direction="up"), unsafe_allow_html=True)

                scenarios_compare = {
                    "No action":         {"supply_days": buffer_days,       "cost_k": 0},
                    "Emergency stock":   {"supply_days": buffer_days + 45,  "cost_k": int(annual_spend * 0.12 / 1000)},
                    "Alt supplier fast": {"supply_days": buffer_days + 90,  "cost_k": int(annual_spend * 0.20 / 1000)},
                    "In-house qualify":  {"supply_days": buffer_days + 180, "cost_k": int(annual_spend * 0.35 / 1000)},
                }
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Bar(
                    name="Supply coverage (days)",
                    x=list(scenarios_compare.keys()),
                    y=[v["supply_days"] for v in scenarios_compare.values()],
                    marker_color="#3b82f6", yaxis="y",
                ))
                fig_cmp.add_trace(go.Scatter(
                    name="Mitigation cost (€k)",
                    x=list(scenarios_compare.keys()),
                    y=[v["cost_k"] for v in scenarios_compare.values()],
                    marker_color="#fb923c", mode="lines+markers",
                    yaxis="y2",
                ))
                fig_cmp.update_layout(
                    yaxis=dict(title="Supply coverage (days)", gridcolor="#1e2d45"),
                    yaxis2=dict(title="Cost (€k)", overlaying="y", side="right", gridcolor="#1e2d45"),
                    legend=dict(orientation="h", y=1.1),
                    barmode="group",
                )
                plotly_dark_layout(fig_cmp, height=260)
                st.plotly_chart(fig_cmp, use_container_width=True)

                for action in [
                    "🔴 CRITICAL — VP Operations and Supply Chain Director notification within 24 hours",
                    f"⚡ 48 hours — emergency buffer stock purchase targeting {buffer_days + 45} days coverage",
                    f"📋 15 days — dual-sourcing feasibility: identify 3 alternative suppliers in {sup_sim['product_family']}",
                    "🔧 30 days — engineering assessment: can design be modified to enable alternative sourcing?",
                    "📅 6 months — target: qualified alternative supplier, first PPAP approved",
                ]:
                    st.markdown(
                        f'<div class="alert-card"><div style="font-size:0.82rem; color:#cbd5e1;">{action}</div></div>',
                        unsafe_allow_html=True)

            elif scenario_type == "Quality Escape":
                # Part cost varies by product family; affects parts-at-risk and scrap value
                _part_cost_by_family = {
                    "Electronics": 8,       "Electromechanics": 45,
                    "Mechanics - Metal": 35, "Mechanics - Plastic": 12,
                    "Raw Materials": 5,      "Cables & Harness": 25,
                    "Surface Treatment": 20, "Optical & Precision": 150,
                    "Software/Firmware": 15, "Services": 15,
                }
                part_cost_eur = _part_cost_by_family.get(str(sup_sim["product_family"]), 15)
                # Scrap rate scales with escape severity (3% at 200 PPM → 20% at 2000 PPM)
                scrap_pct     = min(0.20, max(0.03, escape_ppm / 8000))
                parts_at_risk = int(annual_spend / 365 * 30 / part_cost_eur)
                sort_cost     = parts_at_risk * 2.5
                scrap_cost    = parts_at_risk * scrap_pct * part_cost_eur
                total_cost    = sort_cost + scrap_cost

                st.markdown(f"""
                <div class="kpi-card" style="margin-bottom:0.75rem;">
                    <div class="kpi-label">Scenario</div>
                    <div style="font-size:0.9rem; color:#f1f5f9; font-weight:600;">
                        Quality Escape · {sup_sim['name'][:35]} · {escape_ppm} PPM
                    </div>
                    <div style="font-size:0.78rem; color:#64748b; margin-top:0.25rem;">
                        {sup_sim['product_family']} · Current risk: {risk_badge(risk_label_sim)}
                    </div>
                </div>""", unsafe_allow_html=True)

                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    st.markdown(kpi_card("Parts at Risk", f"{parts_at_risk:,}",
                                         delta="30-day stock estimate",
                                         delta_direction="up"), unsafe_allow_html=True)
                with rc2:
                    st.markdown(kpi_card("Sort Cost", f"€{sort_cost/1000:.1f}k",
                                         delta="100% inspection labour",
                                         delta_direction="up"), unsafe_allow_html=True)
                with rc3:
                    st.markdown(kpi_card("Est. Scrap", f"€{scrap_cost/1000:.1f}k",
                                         delta=f"{scrap_pct*100:.1f}% defect rate",
                                         delta_direction="up"), unsafe_allow_html=True)
                with rc4:
                    new_ppm_tier = "RED" if escape_ppm > 500 else "AMBER" if escape_ppm > 200 else "GREEN"
                    _tier_dir    = "up" if escape_ppm > 200 else "flat"
                    st.markdown(kpi_card("Risk Impact", new_ppm_tier,
                                         delta=f"From {escape_ppm} PPM escape",
                                         delta_direction=_tier_dir), unsafe_allow_html=True)

                months   = list(range(-3, 7))
                sim_kpis_hist = kpis[kpis["supplier_id"] == sid_sim].sort_values("year_month").tail(3)
                if len(sim_kpis_hist) >= 3:
                    ppm_base = [round(float(v)) for v in sim_kpis_hist["ppm_external"].tolist()]
                else:
                    ppm_base = [max(50, escape_ppm * (0.3 + 0.1 * i)) for i in range(3)]
                ppm_before = ppm_base + [escape_ppm, escape_ppm, escape_ppm * 0.9, escape_ppm * 0.85, escape_ppm * 0.8, escape_ppm * 0.75, escape_ppm * 0.7]
                ppm_after  = ppm_base + [escape_ppm, escape_ppm * 0.5, escape_ppm * 0.15, 80, 50, 40, 35]

                fig_ppm = go.Figure()
                fig_ppm.add_trace(go.Scatter(x=months, y=ppm_before,
                                              name="Without containment",
                                              line=dict(color="#f87171", dash="dash")))
                fig_ppm.add_trace(go.Scatter(x=months, y=ppm_after,
                                              name="With immediate containment",
                                              line=dict(color="#34d399")))
                fig_ppm.add_vline(x=0, line_dash="dot", line_color="#475569",
                                   annotation_text="Escape detected")
                fig_ppm.add_hline(y=500, line_dash="dot", line_color="#7f1d1d",
                                   annotation_text="RED threshold")
                fig_ppm.add_hline(y=200, line_dash="dot", line_color="#92400e",
                                   annotation_text="AMBER threshold")
                fig_ppm.update_layout(xaxis_title="Month (0 = detection)",
                                       yaxis_title="PPM", legend_orientation="h")
                plotly_dark_layout(fig_ppm, height=240)
                st.plotly_chart(fig_ppm, use_container_width=True)

                for action in [
                    "⚡ 24 hours — 100% sort of all suspect stock at supplier, in transit, and at customer goods-in",
                    "📋 5 days — issue SCAR; require containment report and interim supply of conforming parts",
                    "🔍 30 days — root cause analysis (5-Why minimum); corrective action plan with evidence",
                    f"📅 60 days — effectiveness verification: {escape_ppm // 5} PPM target sustained for 30 days before SCAR closure",
                ]:
                    st.markdown(
                        f'<div class="alert-card"><div style="font-size:0.82rem; color:#cbd5e1;">{action}</div></div>',
                        unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="text-align:center; padding:3rem; color:#475569;">
                <div style="font-size:2rem; margin-bottom:1rem;">⬡</div>
                <div style="font-size:0.9rem;">Configure a scenario on the left and click Run Simulation</div>
            </div>""", unsafe_allow_html=True)
