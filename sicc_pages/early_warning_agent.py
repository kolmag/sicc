import pandas as pd
import streamlit as st

from utils.ui import kpi_card
from scripts.supplier_alert_agent import build_supplier_trend_alerts


def render(tables, filtered_suppliers, filtered_ids, filtered_risk, ml):
    suppliers   = tables["suppliers"]
    kpis        = tables["supplier_kpis"]
    claims      = tables["claims"]
    apqp        = tables["apqp_projects"]
    audits      = tables["audits"]
    risk_scores = tables["risk_scores"]
    events      = tables["external_events"]

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
