import plotly.express as px
import streamlit as st

from utils.ui import kpi_card, plotly_dark_layout, risk_badge, risk_color
from utils.ml import get_ml_pred_label, ml_predicted_badge
from utils.intent import generate_executive_summary


def render(tables, filtered_suppliers, filtered_ids, filtered_risk, ml):
    suppliers   = tables["suppliers"]
    kpis        = tables["supplier_kpis"]
    claims      = tables["claims"]
    apqp        = tables["apqp_projects"]
    audits      = tables["audits"]
    risk_scores = tables["risk_scores"]
    events      = tables["external_events"]

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

    n_red   = len(filtered_risk[filtered_risk["risk_label"] == "red"])
    n_amber = len(filtered_risk[filtered_risk["risk_label"] == "amber"])
    n_green = len(filtered_risk[filtered_risk["risk_label"] == "green"])

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
