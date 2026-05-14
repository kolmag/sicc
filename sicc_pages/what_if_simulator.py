import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.ui import kpi_card, plotly_dark_layout, risk_badge


def render(tables, filtered_suppliers, filtered_ids, filtered_risk, ml):
    suppliers   = tables["suppliers"]
    kpis        = tables["supplier_kpis"]
    claims      = tables["claims"]
    apqp        = tables["apqp_projects"]
    audits      = tables["audits"]
    risk_scores = tables["risk_scores"]
    events      = tables["external_events"]

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
