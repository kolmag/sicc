import pandas as pd
import streamlit as st

from utils.ui import kpi_card
from scripts.apqp_readiness_agent import assess_apqp_launch_readiness


def render(tables, filtered_suppliers, filtered_ids, filtered_risk, ml):
    suppliers   = tables["suppliers"]
    kpis        = tables["supplier_kpis"]
    claims      = tables["claims"]
    apqp        = tables["apqp_projects"]
    audits      = tables["audits"]
    risk_scores = tables["risk_scores"]
    events      = tables["external_events"]

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
