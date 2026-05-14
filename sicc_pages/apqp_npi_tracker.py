import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.ui import kpi_card, plotly_dark_layout


def render(tables, filtered_suppliers, filtered_ids, filtered_risk, ml):
    suppliers   = tables["suppliers"]
    kpis        = tables["supplier_kpis"]
    claims      = tables["claims"]
    apqp        = tables["apqp_projects"]
    audits      = tables["audits"]
    risk_scores = tables["risk_scores"]
    events      = tables["external_events"]

    st.markdown('<div class="page-title">APQP / NPI Tracker</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Programme launch readiness and supplier-linked deliverables</div>', unsafe_allow_html=True)

    apqp_filtered = apqp[apqp["supplier_id"].isin(filtered_ids)].copy()
    apqp_merged   = apqp_filtered.drop(
        columns=["product_family", "country"], errors="ignore"
    ).merge(
        suppliers[["supplier_id", "name", "product_family", "country"]], on="supplier_id")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Active Programs",
                             f"{len(apqp_merged[apqp_merged['status']=='Active']):,}"),
                    unsafe_allow_html=True)
    with c2:
        delayed = apqp_merged[apqp_merged["is_delayed"] == 1]
        st.markdown(kpi_card("Delayed", f"{len(delayed):,}",
                             delta="Milestone overdue",
                             delta_direction="up" if len(delayed) > 0 else "flat"),
                    unsafe_allow_html=True)
    with c3:
        completed = apqp_merged[apqp_merged["status"] == "Completed"]
        st.markdown(kpi_card("Completed", f"{len(completed):,}"), unsafe_allow_html=True)
    with c4:
        avg_completion = apqp_merged["completion_pct"].mean()
        avg_str = f"{avg_completion:.0f}%" if not pd.isna(avg_completion) else "—"
        st.markdown(kpi_card("Avg Completion", avg_str), unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns([3, 1])

    with col_l:
        st.markdown('<div class="section-header">Programme List</div>', unsafe_allow_html=True)
        status_filter = st.selectbox("Filter by status",
                                     ["All", "Active", "Delayed", "Completed", "On Hold"])
        if status_filter == "All":
            table_data = apqp_merged
        elif status_filter == "Delayed":
            table_data = apqp_merged[apqp_merged["is_delayed"].isin([1, True])]
        else:
            table_data = apqp_merged[apqp_merged["status"] == status_filter]
        table = table_data[[
            "project_id", "name", "project_type", "status",
            "customer_sop_date", "completion_pct", "is_delayed", "product_family"
        ]].copy()
        table.columns = ["Project ID", "Supplier", "Type", "Status",
                         "SOP Date", "Completion %", "Delayed", "Family"]
        table["Delayed"] = table["Delayed"].map({1: "⚠ Yes", 0: "No", True: "⚠ Yes", False: "No"})
        st.dataframe(table.sort_values("Delayed", ascending=False),
                     use_container_width=True, height=400)

    with col_r:
        st.markdown('<div class="section-header">Status Breakdown</div>', unsafe_allow_html=True)
        status_counts = apqp_merged["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig = px.pie(status_counts, values="Count", names="Status",
                     color_discrete_sequence=["#34d399", "#60a5fa", "#f87171",
                                              "#fb923c", "#94a3b8"],
                     hole=0.5)
        plotly_dark_layout(fig, height=220)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Completion Distribution</div>',
                    unsafe_allow_html=True)
        fig2 = px.histogram(apqp_merged, x="completion_pct", nbins=10,
                            color_discrete_sequence=["#3b82f6"])
        fig2.update_layout(xaxis_title="Completion %", yaxis_title="Programs", bargap=0.1)
        plotly_dark_layout(fig2, height=200)
        st.plotly_chart(fig2, use_container_width=True)

    # ── APQP Gate Matrix ──────────────────────────────────────────────────────
    with st.expander("📊 APQP Gate Matrix — Phase completion heatmap", expanded=True):
        PHASES = [
            ("supplier_selection",           "1. Supplier\nSelection"),
            ("supplier_nomination",          "2. Supplier\nNomination"),
            ("design_validation_of_process", "3. Design\nValidation"),
            ("process_validation",           "4. Process\nValidation"),
            ("initial_sample_validation",    "5. Initial\nSample"),
            ("start_of_production",          "6. SOP"),
            ("pqa_management",               "7. PQA\nMgmt"),
            ("yearly_is_submission",         "8. Yearly IS\nSubmission"),
            ("ppap_update",                  "9. PPAP\nUpdate"),
        ]

        STATUS_SCORE = {
            "Validated":    4,
            "Submitted":    3,
            "In Progress":  2,
            "Overdue":      1,
            "Not Started":  0,
        }

        STATUS_COLOR = {
            "Validated":   "#34d399",
            "Submitted":   "#60a5fa",
            "In Progress": "#fb923c",
            "Overdue":     "#f87171",
            "Not Started": "#1e2d45",
        }

        # Filter controls
        mc1, mc2, mc3 = st.columns([1, 1, 2])
        with mc1:
            matrix_status = st.selectbox("Programme status",
                ["All", "Active", "Delayed", "On Hold"], key="matrix_status")
        with mc2:
            matrix_family = st.selectbox("Product family",
                ["All"] + sorted(apqp_merged["product_family"].unique().tolist()),
                key="matrix_family")
        with mc3:
            matrix_search = st.text_input("Search supplier name", key="matrix_search",
                                           placeholder="Type to filter...")

        matrix_df = apqp_merged.copy()
        if matrix_status != "All":
            matrix_df = matrix_df[matrix_df["status"] == matrix_status]
        if matrix_family != "All":
            matrix_df = matrix_df[matrix_df["product_family"] == matrix_family]
        if matrix_search:
            matrix_df = matrix_df[matrix_df["name"].str.contains(
                matrix_search, case=False, na=False)]

        matrix_df = matrix_df.head(40)  # cap at 40 rows for readability

        if matrix_df.empty:
            st.info("No programmes match the current filter.")
        else:
            # Build z (score), text, and color matrices
            phase_keys  = [p[0] for p in PHASES]
            phase_labels = [p[1] for p in PHASES]

            z_matrix    = []
            text_matrix = []
            color_matrix = []

            y_labels = []
            _n_phases = len(phase_keys)
            for _, row in matrix_df.iterrows():
                z_row    = []
                text_row = []
                col_row  = []
                # Derive phase completion from completion_pct + is_delayed
                # since individual phase status columns are not stored in the DB.
                _pct = float(row["completion_pct"]) if row["completion_pct"] <= 1 \
                       else float(row["completion_pct"]) / 100
                _n_done   = max(0, min(_n_phases, round(_pct * _n_phases)))
                _is_delay = bool(row["is_delayed"])
                _is_complete = row.get("status", "") == "Completed"
                for i, pk in enumerate(phase_keys):
                    if _is_complete or i < _n_done - 1:
                        status = "Validated"
                    elif i == _n_done - 1 and _n_done > 0:
                        status = "Overdue" if _is_delay else "Submitted"
                    elif i == _n_done:
                        status = "Overdue" if _is_delay else "In Progress"
                    else:
                        status = "Not Started"
                    z_row.append(STATUS_SCORE.get(status, 0))
                    text_row.append(status[:3].upper() if status != "Not Started" else "—")
                    col_row.append(STATUS_COLOR.get(status, "#1e2d45"))
                z_matrix.append(z_row)
                text_matrix.append(text_row)
                color_matrix.append(col_row)

                # Y label: supplier name + project type + risk badge
                risk_row_data = risk_scores[risk_scores["supplier_id"] == row["supplier_id"]]
                risk_lbl = risk_row_data["risk_label"].iloc[0] if not risk_row_data.empty else "green"
                risk_icon = {"red": "🔴", "amber": "🟡", "green": "🟢"}.get(risk_lbl, "⚪")
                y_labels.append(f"{risk_icon} {row['name'][:22]} · {row['project_type'][:12]}")

            # Plotly heatmap
            colorscale = [
                [0.00, "#1e2d45"],   # Not Started
                [0.25, "#f87171"],   # Overdue
                [0.50, "#fb923c"],   # In Progress
                [0.75, "#60a5fa"],   # Submitted
                [1.00, "#34d399"],   # Validated
            ]

            fig_matrix = go.Figure(go.Heatmap(
                z=z_matrix,
                x=phase_labels,
                y=y_labels,
                text=text_matrix,
                texttemplate="%{text}",
                textfont=dict(size=9, color="#0f1923"),
                colorscale=colorscale,
                zmin=0, zmax=4,
                showscale=False,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Phase: %{x}<br>"
                    "Status: %{text}<extra></extra>"
                ),
                xgap=2,
                ygap=1,
            ))

            fig_matrix.update_layout(
                height=max(300, len(matrix_df) * 28 + 80),
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="#0f1923",
                plot_bgcolor="#0f1923",
                font=dict(color="#94a3b8", size=10, family="DM Sans"),
                xaxis=dict(side="top", tickfont=dict(size=9), tickangle=-20),
                yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
            )
            st.plotly_chart(fig_matrix, use_container_width=True)

            # Legend
            st.markdown(
                '<div style="display:flex; gap:1rem; font-size:0.7rem; color:#64748b; margin-top:-0.5rem;">'
                + "".join([
                    f'<span><span style="background:{c}; padding:1px 6px; border-radius:3px; '
                    f'color:#0f1923; font-size:0.68rem;">{s[:3].upper()}</span> {s}</span>'
                    for s, c in STATUS_COLOR.items()
                ])
                + "</div>",
                unsafe_allow_html=True,
            )

    # ── SOP Timeline Gantt ─────────────────────────────────────────────────────
    with st.expander("📅 SOP Timeline — Supplier vs Customer dates", expanded=False):
        _gantt_df = apqp_merged[
            apqp_merged["status"].isin(["Active", "On Hold"]) &
            apqp_merged["supplier_sop_date"].notna() &
            apqp_merged["customer_sop_date"].notna()
        ].copy().head(30)
        if _gantt_df.empty:
            st.info("No active programmes with SOP dates available.")
        else:
            _gantt_fig = go.Figure()
            for _, _row in _gantt_df.iterrows():
                _color = "#f87171" if bool(_row["is_delayed"]) else "#34d399"
                _label = f"{_row['name'][:22]} · {_row['project_id']}"
                # Supplier target bar
                _gantt_fig.add_trace(go.Bar(
                    name="Supplier SOP", orientation="h",
                    x=[(_row["supplier_sop_date"] - _row["supplier_sop_date"]).days + 1],
                    base=[_row["supplier_sop_date"]],
                    y=[_label],
                    marker_color=_color, width=0.4,
                    showlegend=False,
                ))
                # Customer deadline marker
                _gantt_fig.add_vline(
                    x=_row["customer_sop_date"].timestamp() * 1000,
                    line_dash="dot", line_color="#60a5fa", line_width=1,
                )
            # Build as scatter timeline instead for clarity
            _gantt_fig = go.Figure()
            _labels, _sup_dates, _cust_dates, _colors = [], [], [], []
            for _, _row in _gantt_df.iterrows():
                _labels.append(f"{_row['name'][:22]} · {_row['project_id']}")
                _sup_dates.append(_row["supplier_sop_date"])
                _cust_dates.append(_row["customer_sop_date"])
                _colors.append("#f87171" if bool(_row["is_delayed"]) else "#34d399")
            _gantt_fig.add_trace(go.Scatter(
                x=_sup_dates, y=_labels, mode="markers",
                name="Supplier SOP",
                marker=dict(color=_colors, size=12, symbol="diamond"),
            ))
            _gantt_fig.add_trace(go.Scatter(
                x=_cust_dates, y=_labels, mode="markers",
                name="Customer SOP deadline",
                marker=dict(color="#60a5fa", size=10, symbol="line-ns-open"),
            ))
            for _sup, _cust, _lbl in zip(_sup_dates, _cust_dates, _labels):
                _gantt_fig.add_shape(type="line",
                    x0=_sup, x1=_cust, y0=_lbl, y1=_lbl,
                    line=dict(color="#475569", width=1.5, dash="dot"))
            _gantt_fig.update_layout(
                xaxis_title="Date", yaxis_title="",
                legend=dict(orientation="h", y=1.08),
                yaxis=dict(autorange="reversed"),
            )
            plotly_dark_layout(_gantt_fig, height=max(280, len(_gantt_df) * 30 + 80))
            st.plotly_chart(_gantt_fig, use_container_width=True)
            st.caption("🔴 Delayed  🟢 On track  🔵 Customer SOP deadline")
