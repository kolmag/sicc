"""
app.py — Supplier Intelligence Command Center
Orchestrator: page config, CSS, data loading, sidebar, and routing.
All page logic lives in sicc_pages/; shared helpers in utils/.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from utils.config import DB_PATH
from utils.data import load_all_data, get_kb_chunk_count
from utils.ml import load_ml_artefacts
from utils.ui import inject_css
from scripts.agent_memory import init_agent_memory

import sicc_pages.executive_portfolio    as _pg_executive
import sicc_pages.agent_command_center   as _pg_agent_cmd
import sicc_pages.risk_scoring_engine    as _pg_risk
import sicc_pages.early_warning_agent    as _pg_early_warning
import sicc_pages.scar_capa_triage       as _pg_scar
import sicc_pages.apqp_readiness_agent   as _pg_apqp_ready
import sicc_pages.continuity_agent       as _pg_continuity
import sicc_pages.audit_planning_agent   as _pg_audit
import sicc_pages.supplier_profile       as _pg_profile
import sicc_pages.apqp_npi_tracker       as _pg_apqp_npi
import sicc_pages.supplier_qa_agent      as _pg_qa
import sicc_pages.what_if_simulator      as _pg_what_if

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Supplier Intelligence Command Center",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ── Data ──────────────────────────────────────────────────────────────────────

tables      = load_all_data()
suppliers   = tables["suppliers"]
kpis        = tables["supplier_kpis"]
claims      = tables["claims"]
apqp        = tables["apqp_projects"]
audits      = tables["audits"]
risk_scores = tables["risk_scores"]
events      = tables["external_events"]
ml          = load_ml_artefacts()

if suppliers.empty or risk_scores.empty:
    st.error("Supplier portfolio data is not available.")
    st.info("Generate the dataset first with `uv run python scripts/generate_supplier_data.py --out data/`.")
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
        [
            "Executive Portfolio",
            "Agent Command Center",
            "Risk Scoring Engine",
            "Early Warning Agent",
            "SCAR/CAPA Triage",
            "APQP Readiness Agent",
            "Continuity Agent",
            "Audit Planning Agent",
            "Supplier Profile",
            "APQP / NPI Tracker",
            "Supplier Q&A Agent",
            "What-If Simulator",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-header">Filters</div>', unsafe_allow_html=True)

    sel_families = st.multiselect(
        "Product Family",
        options=sorted(suppliers["product_family"].unique()),
        default=[],
        placeholder="All families",
    )
    sel_regions = st.multiselect(
        "Region",
        options=sorted(suppliers["region"].unique()),
        default=[],
        placeholder="All regions",
    )
    sel_tiers = st.multiselect(
        "Spend Tier", options=["A", "B", "C"], default=[], placeholder="All tiers"
    )

    filtered_suppliers = suppliers.copy()
    if sel_families:
        filtered_suppliers = filtered_suppliers[
            filtered_suppliers["product_family"].isin(sel_families)
        ]
    if sel_regions:
        filtered_suppliers = filtered_suppliers[
            filtered_suppliers["region"].isin(sel_regions)
        ]
    if sel_tiers:
        filtered_suppliers = filtered_suppliers[
            filtered_suppliers["spend_tier"].isin(sel_tiers)
        ]

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

# ── Routing ───────────────────────────────────────────────────────────────────

_ctx = dict(
    tables=tables,
    filtered_suppliers=filtered_suppliers,
    filtered_ids=filtered_ids,
    filtered_risk=filtered_risk,
    ml=ml,
)

_pages = {
    "Executive Portfolio":  _pg_executive,
    "Agent Command Center": _pg_agent_cmd,
    "Risk Scoring Engine":  _pg_risk,
    "Early Warning Agent":  _pg_early_warning,
    "SCAR/CAPA Triage":     _pg_scar,
    "APQP Readiness Agent": _pg_apqp_ready,
    "Continuity Agent":     _pg_continuity,
    "Audit Planning Agent": _pg_audit,
    "Supplier Profile":     _pg_profile,
    "APQP / NPI Tracker":   _pg_apqp_npi,
    "Supplier Q&A Agent":   _pg_qa,
    "What-If Simulator":    _pg_what_if,
}

_pages[page].render(**_ctx)
