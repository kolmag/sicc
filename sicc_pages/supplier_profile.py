import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.ui import kpi_card, plotly_dark_layout, risk_badge, risk_color
from utils.ml import make_shap_waterfall, ml_predicted_badge
from scripts.supplier_intake_agent import (
    SupplierDevelopmentBrief,
    brief_to_markdown,
    generate_supplier_development_brief,
)


def render(tables, filtered_suppliers, filtered_ids, filtered_risk, ml):
    suppliers   = tables["suppliers"]
    kpis        = tables["supplier_kpis"]
    claims      = tables["claims"]
    apqp        = tables["apqp_projects"]
    audits      = tables["audits"]
    risk_scores = tables["risk_scores"]
    events      = tables["external_events"]

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
