import json as _json
import pandas as pd
import streamlit as st

from utils.ui import kpi_card
from scripts.continuity_agent import build_continuity_watchlist, assess_single_source_continuity


def render(tables, filtered_suppliers, filtered_ids, filtered_risk, ml):
    suppliers   = tables["suppliers"]
    kpis        = tables["supplier_kpis"]
    claims      = tables["claims"]
    apqp        = tables["apqp_projects"]
    audits      = tables["audits"]
    risk_scores = tables["risk_scores"]
    events      = tables["external_events"]

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

    _cont_export = _json.dumps(
        next(p for p in plans if p.supplier_id == selected_id).model_dump(),
        indent=2, default=str)
    st.download_button("⬇ Download Continuity Plan", data=_cont_export,
                       file_name=f"continuity_plan_{selected_id}.json",
                       mime="application/json")
