import json as _json
import re as _re
from datetime import date as _date, timedelta as _td
import pandas as pd
import streamlit as st

from utils.ui import kpi_card
from scripts.audit_planning_agent import build_audit_plan_watchlist, plan_supplier_audit


def render(tables, filtered_suppliers, filtered_ids, filtered_risk, ml):
    suppliers   = tables["suppliers"]
    kpis        = tables["supplier_kpis"]
    claims      = tables["claims"]
    apqp        = tables["apqp_projects"]
    audits      = tables["audits"]
    risk_scores = tables["risk_scores"]
    events      = tables["external_events"]

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

    _audit_export = _json.dumps(
        next(p for p in audit_plans if p.supplier_id == selected_id).model_dump(),
        indent=2, default=str)
    st.download_button("⬇ Download Audit Plan", data=_audit_export,
                       file_name=f"audit_plan_{selected_id}.json",
                       mime="application/json")
