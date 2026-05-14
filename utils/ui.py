import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1623;
    border-right: 1px solid #1e2d45;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label { color: #64748b !important; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; }

/* Page title */
.page-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
}
.page-subtitle {
    font-size: 0.85rem;
    color: #64748b;
    margin-bottom: 1.5rem;
}

/* KPI cards */
.kpi-card {
    background: #131c2e;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    min-height: 90px;
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.35rem;
}
.kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #f1f5f9;
    font-family: 'DM Mono', monospace;
    line-height: 1;
}
.kpi-delta {
    font-size: 0.75rem;
    margin-top: 0.3rem;
    font-family: 'DM Mono', monospace;
}
.delta-up   { color: #f87171; }
.delta-down { color: #34d399; }
.delta-flat { color: #64748b; }

/* Risk badges */
.badge-red    { background:#450a0a; color:#f87171; border:1px solid #7f1d1d; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:600; }
.badge-amber  { background:#451a03; color:#fb923c; border:1px solid #7c2d12; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:600; }
.badge-green  { background:#052e16; color:#34d399; border:1px solid #14532d; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:600; }
.badge-critical { background:#3b0764; color:#c084fc; border:1px solid #6b21a8; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:600; }

/* Section headers */
.section-header {
    font-size: 0.75rem;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid #1e2d45;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
    margin-top: 1.5rem;
}

/* Tables */
.stDataFrame { border: 1px solid #1e2d45; border-radius: 8px; overflow: hidden; }
thead tr th { background: #0f1623 !important; color: #64748b !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
tbody tr:hover { background: #131c2e !important; }

/* Alert cards */
.alert-card {
    background: #0f1623;
    border-left: 3px solid #f87171;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}
.alert-card.amber { border-left-color: #fb923c; }
.alert-card.green { border-left-color: #34d399; }
.alert-card.purple { border-left-color: #c084fc; }

/* AI summary card */
.ai-summary {
    background: linear-gradient(135deg, #0f1623 0%, #131c2e 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    position: relative;
}
.ai-badge {
    display: inline-block;
    background: #1e3a5f;
    color: #60a5fa;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 2px 8px;
    border-radius: 4px;
    margin-bottom: 0.75rem;
}

/* Nav branding */
.brand-header {
    padding: 1rem 1rem 0.5rem;
    border-bottom: 1px solid #1e2d45;
    margin-bottom: 1rem;
}
.brand-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.01em;
}
.brand-sub {
    font-size: 0.68rem;
    color: #475569;
    margin-top: 2px;
}

/* Plotly chart backgrounds */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def risk_color(label):
    colors = {"red": "#f87171", "amber": "#fb923c", "green": "#34d399"}
    return colors.get(str(label).lower(), "#94a3b8")


def risk_badge(label):
    label = str(label).lower()
    cls = {"red": "badge-red", "amber": "badge-amber", "green": "badge-green",
           "critical": "badge-critical"}.get(label, "badge-green")
    return f'<span class="{cls}">{label.upper()}</span>'


def severity_badge(sev):
    cls = {"Critical": "badge-critical", "High": "badge-red",
           "Medium": "badge-amber", "Low": "badge-green"}.get(sev, "badge-green")
    return f'<span class="{cls}">{sev}</span>'


def plotly_dark_layout(fig, height=300, margin=None):
    m = margin or dict(l=10, r=10, t=10, b=10)
    fig.update_layout(
        height=height, margin=m,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#94a3b8", size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        xaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45", zerolinecolor="#1e2d45"),
        yaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45", zerolinecolor="#1e2d45"),
    )
    return fig


def kpi_card(label, value, delta=None, delta_direction="flat"):
    delta_html = ""
    if delta is not None:
        cls   = {"up": "delta-up", "down": "delta-down", "flat": "delta-flat"}[delta_direction]
        arrow = {"up": "▲", "down": "▼", "flat": "—"}[delta_direction]
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {delta}</div>'
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>"""
