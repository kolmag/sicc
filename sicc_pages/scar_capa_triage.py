import json as _json
import pandas as pd
import streamlit as st

from utils.ui import kpi_card
from scripts.scar_capa_agent import triage_claim, triage_manual_issue


def render(tables, filtered_suppliers, filtered_ids, filtered_risk, ml):
    suppliers   = tables["suppliers"]
    kpis        = tables["supplier_kpis"]
    claims      = tables["claims"]
    apqp        = tables["apqp_projects"]
    audits      = tables["audits"]
    risk_scores = tables["risk_scores"]
    events      = tables["external_events"]

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

    _triage_export = _json.dumps(st.session_state[_triage_key], indent=2, default=str)
    _fname = f"scar_triage_{getattr(triage, 'incident_number', sid).replace('/', '-')}.json"
    st.download_button("⬇ Download Triage Report", data=_triage_export,
                       file_name=_fname, mime="application/json")
