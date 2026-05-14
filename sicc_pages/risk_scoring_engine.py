import plotly.graph_objects as go
import streamlit as st

from utils.ui import kpi_card, plotly_dark_layout, risk_badge, severity_badge
from utils.ml import (
    make_feature_importance_chart,
    make_ml_metrics_html,
    make_shap_waterfall,
    ml_predicted_badge,
)


def render(tables, filtered_suppliers, filtered_ids, filtered_risk, ml):
    suppliers   = tables["suppliers"]
    kpis        = tables["supplier_kpis"]
    claims      = tables["claims"]
    apqp        = tables["apqp_projects"]
    audits      = tables["audits"]
    risk_scores = tables["risk_scores"]
    events      = tables["external_events"]

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
