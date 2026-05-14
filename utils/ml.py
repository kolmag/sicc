import json
import pickle

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.config import ML_DIR
from utils.ui import plotly_dark_layout, risk_badge


@st.cache_resource
def load_ml_artefacts():
    """Load trained model + SHAP payload. Returns None if not yet trained."""
    model_path   = ML_DIR / "model.pkl"
    shap_path    = ML_DIR / "shap_values.pkl"
    metrics_path = ML_DIR / "model_metrics.json"

    if not model_path.exists():
        return None

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(shap_path, "rb") as f:
            shap_payload = pickle.load(f)

        # Normalise SHAP format
        sv = shap_payload["shap_values"]
        if isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv = [sv[:, :, i] for i in range(sv.shape[2])]

        metrics = {}
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)

        return {
            "model":          model,
            "shap_values":    sv,
            "expected_value": shap_payload["expected_value"],
            "feature_names":  shap_payload["feature_names"],
            "X":              shap_payload["X"],
            "supplier_ids":   shap_payload["supplier_ids"],
            "y_pred":         shap_payload["y_pred"],
            "y_pred_proba":   shap_payload["y_pred_proba"],
            "label_order":    shap_payload["label_order"],
            "winner_name":    shap_payload.get("winner_name", "RandomForest"),
            "metrics":        metrics,
        }
    except Exception as e:
        print(f"[ML] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def _fmt_feature_name(name: str) -> str:
    replacements = {
        "ppm_external": "PPM", "otd_pct": "OTD%", "audit_score": "Audit",
        "scar_count": "SCARs", "cost_of_poor_quality_eur": "COPQ",
        "ppap_first_time_pass_pct": "PPAP FTP", "ca_closure_rate_pct": "CA Closure",
        "oqd_pct": "OQD%", "_3m": " 3m", "_6m": " 6m", "_12m": " 12m",
        "_std_12m": " σ12m", "_trend": " trend", "_deterioration": " Δ",
        "_worst_3m": " worst", "months_ppm_above_500": "Mo PPM>500",
        "months_ppm_above_200": "Mo PPM>200", "months_otd_below_90": "Mo OTD<90",
        "months_otd_below_95": "Mo OTD<95", "months_audit_below_60": "Mo Audit<60",
        "months_audit_below_75": "Mo Audit<75", "spend_tier_enc": "Spend Tier",
        "strat_imp_enc": "Strategic Imp", "qual_status_enc": "Qual Status",
        "region_risk_enc": "Region Risk", "single_source_int": "Single Source",
        "years_active": "Yrs Active", "annual_spend_eur": "Annual Spend", "fam_": "Fam: ",
    }
    result = name
    for k, v in replacements.items():
        result = result.replace(k, v)
    return result[:32]


def make_shap_waterfall(ml, supplier_id: str, class_idx: int = 2, top_n: int = 12):
    """Bar chart of SHAP contributions for one supplier and one predicted class."""
    label_colors = {0: "#34d399", 1: "#fb923c", 2: "#f87171"}

    if ml is None or supplier_id not in ml["supplier_ids"]:
        return None

    idx        = ml["supplier_ids"].index(supplier_id)
    shap_vals  = ml["shap_values"][class_idx][idx]
    feat_names = ml["feature_names"]
    base_val   = (ml["expected_value"][class_idx]
                  if isinstance(ml["expected_value"], (list, np.ndarray))
                  else ml["expected_value"])

    order = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    vals  = shap_vals[order]
    names = [_fmt_feature_name(feat_names[i]) for i in order]

    bar_colors = [label_colors[2] if v > 0 else label_colors[0] for v in vals]

    fig = go.Figure(go.Bar(
        x=names, y=vals,
        marker_color=bar_colors,
        text=[f"{v:+.3f}" for v in vals],
        textposition="outside",
        textfont=dict(size=10, color="#94a3b8"),
    ))
    class_names = ["GREEN", "AMBER", "RED"]
    fig.update_layout(
        title=dict(
            text=f"SHAP contributions → <b>{class_names[class_idx]}</b> risk",
            font=dict(size=12, color="#94a3b8"),
        ),
        xaxis_tickangle=-38,
        yaxis_title="SHAP value",
        showlegend=False,
        shapes=[dict(type="line", xref="paper", x0=0, x1=1, y0=0, y1=0,
                     line=dict(color="#475569", width=1, dash="dot"))],
        annotations=[dict(text=f"Base: {base_val:.3f}", xref="paper", yref="paper",
                          x=1, y=1.02, showarrow=False,
                          font=dict(size=9, color="#475569"), xanchor="right")],
    )
    plotly_dark_layout(fig, height=320)
    return fig


def make_feature_importance_chart(ml, top_n: int = 20):
    """Global feature importance — mean |SHAP| across all suppliers (red class)."""
    if ml is None:
        return None

    shap_red      = ml["shap_values"][2]
    mean_abs_shap = np.abs(shap_red).mean(axis=0)
    feat_names    = ml["feature_names"]

    order = np.argsort(mean_abs_shap)[::-1][:top_n]
    vals  = mean_abs_shap[order][::-1]
    names = [_fmt_feature_name(feat_names[i]) for i in order][::-1]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color="#3b82f6", opacity=0.85,
    ))
    fig.update_layout(
        xaxis_title="Mean |SHAP value| — impact on RED risk",
        yaxis_title="", showlegend=False,
    )
    plotly_dark_layout(fig, height=440)
    return fig


def make_ml_metrics_html(ml) -> str:
    """Render model performance as KPI cards."""
    if ml is None or not ml["metrics"]:
        return ""
    m = ml["metrics"].get("winner_metrics", ml["metrics"])
    cards = [
        ("Accuracy",  f"{m.get('accuracy', 0)*100:.1f}%"),
        ("F1 Macro",  f"{m.get('f1_macro', 0):.3f}"),
        ("AUC (OvR)", f"{m.get('auc_ovr', 0):.3f}"),
        ("F1 Red",    f"{m.get('f1_red', 0):.3f}"),
        ("F1 Amber",  f"{m.get('f1_amber', 0):.3f}"),
        ("F1 Green",  f"{m.get('f1_green', 0):.3f}"),
    ]
    html = '<div style="display:flex; gap:0.75rem; flex-wrap:wrap; margin-bottom:1rem;">'
    for label, val in cards:
        html += f"""
        <div class="kpi-card" style="flex:1; min-width:90px;">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="font-size:1.3rem;">{val}</div>
        </div>"""
    html += "</div>"
    return html


def get_ml_pred_label(ml, supplier_id: str) -> str | None:
    """Return the ML-predicted risk label string, or None if unavailable."""
    if ml is None or supplier_id not in ml["supplier_ids"]:
        return None
    idx = ml["supplier_ids"].index(supplier_id)
    return ml["label_order"][ml["y_pred"][idx]]


def ml_predicted_badge(ml, supplier_id: str) -> str:
    """Inline ML prediction badge for a supplier."""
    if ml is None or supplier_id not in ml["supplier_ids"]:
        return ""
    idx       = ml["supplier_ids"].index(supplier_id)
    class_idx = ml["y_pred"][idx]
    label     = ml["label_order"][class_idx]
    proba     = ml["y_pred_proba"][idx][class_idx]
    return (f'<span style="font-size:0.72rem; color:#475569; margin-right:0.25rem;">ML:</span>'
            f'{risk_badge(label)} '
            f'<span style="font-size:0.7rem; color:#64748b; font-family:\'DM Mono\',monospace;">'
            f'{proba*100:.0f}% conf</span>')
