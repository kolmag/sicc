import streamlit as st

from utils.data import get_kb_chunk_count
from utils.intent import classify_portfolio_intent
from scripts.answer import answer as rag_answer, CHROMA_DB_PATH


def render(tables, filtered_suppliers, filtered_ids, filtered_risk, ml):
    suppliers   = tables["suppliers"]
    kpis        = tables["supplier_kpis"]
    claims      = tables["claims"]
    apqp        = tables["apqp_projects"]
    audits      = tables["audits"]
    risk_scores = tables["risk_scores"]
    events      = tables["external_events"]

    st.markdown('<div class="page-title">Supplier Q&A Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Two-layer intelligence: structured portfolio queries + grounded RAG over supplier quality knowledge base</div>', unsafe_allow_html=True)

    # ── Mode selector ─────────────────────────────────────────────────────────
    query_mode = st.radio(
        "Query mode",
        ["📊 Portfolio Data  (supplier KPIs, risk, claims, audits)",
         "📚 Knowledge Base  (PPAP, APQP, SCAR, audit standards, procedures)"],
        horizontal=True,
        label_visibility="collapsed",
    )
    is_rag_mode = "Knowledge Base" in query_mode

    st.markdown("---")

    # ── Suggested prompts — different per mode ────────────────────────────────
    if is_rag_mode:
        st.markdown('<div class="section-header">Suggested Knowledge Base Queries</div>',
                    unsafe_allow_html=True)
        prompts = [
            "What does PPAP Level 3 require?",
            "When is a for-cause audit mandatory?",
            "What are the RED tier KPI thresholds?",
            "What is the SCAR escalation process?",
            "What are the APQP Phase 4 pass criteria?",
            "What buffer stock is required for single-source suppliers?",
        ]
    else:
        st.markdown('<div class="section-header">Suggested Portfolio Queries</div>',
                    unsafe_allow_html=True)
        prompts = [
            "Which RED-risk suppliers have open major audit findings?",
            "Show sole-source suppliers with PPM > 300 in the last 3 months",
            "Which suppliers have Critical external events and no linked CAPA?",
            "What are the top recurring claim categories across Electronics suppliers?",
            "Which APQP programmes are delayed and linked to RED suppliers?",
            "Show suppliers in China with High or Critical geopolitical events",
        ]

    cols = st.columns(3)
    selected_prompt = None
    for i, prompt in enumerate(prompts):
        with cols[i % 3]:
            if st.button(prompt, key=f"prompt_{i}", use_container_width=True):
                selected_prompt = prompt

    st.markdown("---")

    query = st.text_area(
        "Ask a question",
        value=selected_prompt or "",
        height=80,
        placeholder=(
            "e.g. What does PPAP Level 3 require?" if is_rag_mode
            else "e.g. Which critical suppliers have open audit findings and no CAPA?"
        ),
    )

    # ── Filters (portfolio mode only) ─────────────────────────────────────────
    if not is_rag_mode:
        with st.expander("Portfolio Filters", expanded=False):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                filter_family = st.multiselect("Product Family",
                                               options=sorted(suppliers["product_family"].unique()))
            with fc2:
                filter_risk = st.multiselect("Risk Tier", options=["red", "amber", "green"])
            with fc3:
                filter_region = st.multiselect("Region",
                                               options=sorted(suppliers["region"].unique()))
    else:
        filter_family = []
        filter_risk   = []
        filter_region = []

    # ── Search button ─────────────────────────────────────────────────────────
    if st.button("Search", type="primary") and query:

        # ══════════════════════════════════════════════════════════════════════
        # LAYER 2 — RAG (Knowledge Base mode)
        # ══════════════════════════════════════════════════════════════════════
        if is_rag_mode:
            with st.spinner("Searching knowledge base..."):
                try:
                    result = rag_answer(
                        question=query,
                        db_path=CHROMA_DB_PATH,
                        session_id="streamlit_live",
                    )

                    # Confidence badge colour
                    conf_color = {
                        "high":   "#34d399",
                        "medium": "#fb923c",
                        "low":    "#f87171",
                    }.get(result.confidence, "#94a3b8")

                    # Action required badge
                    action_html = (
                        '<span class="badge-red">⚡ ACTION REQUIRED</span>'
                        if result.action_required else ""
                    )

                    # Insufficient evidence
                    if result.insufficient_evidence:
                        st.markdown(f"""
                        <div class="ai-summary">
                            <div class="ai-badge">Knowledge Base · Insufficient Evidence</div>
                            <div style="font-size:0.88rem; color:#f87171; margin-top:0.5rem;">
                                {result.answer}
                            </div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        # Answer card
                        sources_html = " · ".join(
                            f'<code style="font-size:0.68rem; color:#60a5fa;">{s}</code>'
                            for s in result.sources
                        )
                        st.markdown(f"""
                        <div class="ai-summary">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                                <div class="ai-badge">Knowledge Base · RAG Answer</div>
                                <div style="display:flex; gap:0.5rem; align-items:center;">
                                    {action_html}
                                    <span style="font-size:0.72rem; font-weight:600;
                                                 color:{conf_color}; text-transform:uppercase;
                                                 letter-spacing:0.06em;">
                                        {result.confidence} confidence
                                    </span>
                                </div>
                            </div>
                            <div style="font-size:0.85rem; color:#cbd5e1; line-height:1.6;">
                                {result.answer}
                            </div>
                            <div style="margin-top:0.75rem; font-size:0.7rem; color:#475569;">
                                Sources: {sources_html}
                            </div>
                        </div>""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"RAG pipeline error: {e}")
                    st.info("Ensure `chroma_db/` exists and `scripts/ingest.py` has been run.")

        # ══════════════════════════════════════════════════════════════════════
        # LAYER 1 — Structured (Portfolio Data mode)
        # ══════════════════════════════════════════════════════════════════════
        else:
            with st.spinner("Searching supplier intelligence..."):
                result_df = risk_scores.merge(
                    suppliers[["supplier_id", "name", "product_family", "country",
                                "region", "single_source", "spend_tier",
                                "qualification_status", "certification"]], on="supplier_id")

                # Apply sidebar filters first so all intent branches respect them
                result_df = result_df[result_df["supplier_id"].isin(filtered_ids)]

                if filter_family:
                    result_df = result_df[result_df["product_family"].isin(filter_family)]
                if filter_risk:
                    result_df = result_df[result_df["risk_label"].isin(filter_risk)]
                if filter_region:
                    result_df = result_df[result_df["region"].isin(filter_region)]

                intent = classify_portfolio_intent(query)

                answer_text = ""
                show_df     = None

                if intent["intent"] == "red_risk" or intent.get("risk_tier") == "red":
                    tier    = intent.get("risk_tier") or "red"
                    show_df = result_df[result_df["risk_label"] == tier].sort_values("composite_risk_score")
                    answer_text = f"Found **{len(show_df)} {tier.upper()}-risk suppliers** matching your criteria."

                elif intent["intent"] == "single_source":
                    show_df = result_df[result_df["single_source"].isin([1, True])].sort_values("risk_label")
                    answer_text = f"Found **{len(show_df)} single-source suppliers**. {len(show_df[show_df['risk_label']=='red'])} are RED risk."

                elif intent["intent"] == "ppm_threshold":
                    threshold = intent.get("ppm_threshold") or 300
                    show_df   = result_df[result_df["avg_ppm_3m"] > threshold].sort_values("avg_ppm_3m", ascending=False)
                    answer_text = f"Found **{len(show_df)} suppliers** with PPM > {threshold:.0f} in the last 3 months."

                elif intent["intent"] == "audit_findings":
                    finding = intent.get("finding_type") or "Major NCR"
                    if finding not in ["Major NCR", "Critical NCR", "Minor NCR"]:
                        finding = "Major NCR"
                    audit_sup = audits[audits["highest_finding_type"] == finding]["supplier_id"].unique()
                    show_df   = result_df[result_df["supplier_id"].isin(audit_sup)].sort_values("composite_risk_score")
                    answer_text = f"Found **{len(show_df)} suppliers** with **{finding}** audit findings."

                elif intent["intent"] == "claim_categories":
                    _family   = intent.get("product_family")
                    _cl_scope = claims[claims["supplier_id"].isin(filtered_ids)]
                    if _family:
                        _cl_scope = _cl_scope[_cl_scope["product_family"] == _family]
                    _cat_counts = (
                        _cl_scope.groupby("category").size()
                        .reset_index(name="count")
                        .sort_values("count", ascending=False)
                    )
                    show_df = _cat_counts
                    _fam_str = f" for {_family} suppliers" if _family else ""
                    answer_text = f"Top {len(show_df)} recurring claim categories{_fam_str} across {len(_cl_scope):,} claims."

                elif intent["intent"] == "capa_events":
                    capa_needed = events[
                        events["requires_capa"].isin([True, 1]) &
                        ~events["capa_linked"].isin([True, 1]) &
                        events["status"].isin(["Open", "Under Review"])
                    ]["supplier_id"].unique()
                    show_df = result_df[result_df["supplier_id"].isin(capa_needed)].sort_values("composite_risk_score")
                    answer_text = f"Found **{len(show_df)} suppliers** with open alerts and no linked CAPA."

                elif intent["intent"] == "geopolitical":
                    geo_sups = events[
                        (events["event_type"] == "Geopolitical") &
                        (events["severity"].isin(["High", "Critical"])) &
                        (events["status"].isin(["Open", "Under Review", "Escalated"]))
                    ]["supplier_id"].unique()
                    geo_mask = result_df["supplier_id"].isin(geo_sups)
                    country  = intent.get("country")
                    if country:
                        country_mask = result_df["country"].str.lower() == country.lower()
                        show_df      = result_df[geo_mask & country_mask].sort_values("composite_risk_score")
                        answer_text  = f"Found **{len(show_df)} {country}-based suppliers** with active High/Critical geopolitical events."
                    else:
                        show_df     = result_df[geo_mask].sort_values("composite_risk_score")
                        answer_text = f"Found **{len(show_df)} suppliers** with active High/Critical geopolitical events."

                elif intent["intent"] == "apqp_delayed":
                    red_sups     = result_df[result_df["risk_label"] == "red"]["supplier_id"].unique()
                    delayed_apqp = apqp[apqp["is_delayed"] == 1].merge(
                        suppliers[["supplier_id", "name"]], on="supplier_id")
                    show_df     = delayed_apqp[delayed_apqp["supplier_id"].isin(red_sups)]
                    answer_text = f"Found **{len(show_df)} delayed APQP programmes** linked to RED-risk suppliers."

                else:
                    show_df     = result_df.sort_values("composite_risk_score").head(20)
                    answer_text = f"Showing top {len(show_df)} suppliers by risk score. Refine your query for a specific filter."

                st.markdown(f"""
                <div class="ai-summary">
                    <div class="ai-badge">Portfolio Query · Structured data retrieval</div>
                    <div style="font-size:0.9rem; color:#f1f5f9; font-weight:500;">{answer_text}</div>
                    <div style="font-size:0.75rem; color:#475569; margin-top:0.5rem;">
                        Source: supplier_kpis, risk_scores, audits, external_events ·
                        {len(result_df)} suppliers in scope
                    </div>
                </div>""", unsafe_allow_html=True)

                if show_df is not None and not show_df.empty:
                    display_cols = [c for c in [
                        "name", "product_family", "country", "risk_label",
                        "composite_risk_score", "avg_ppm_3m", "avg_otd_3m",
                        "single_source", "annual_spend_eur"
                    ] if c in show_df.columns]
                    st.dataframe(show_df[display_cols].head(25), use_container_width=True)

    # Footer
    st.markdown(f"""
    <div style="font-size:0.72rem; color:#475569; margin-top:2rem; padding:0.75rem;
                border:1px solid #1e2d45; border-radius:8px;">
        ⬡ <strong>Two-layer Q&A.</strong>
        Portfolio Data mode queries structured supplier KPIs, risk scores, audits, and events directly from SQLite.
        Knowledge Base mode uses hybrid RAG (BM25 + embedding + RRF) over {get_kb_chunk_count()} KB chunks
        (16 supplier quality documents) via ChromaDB · OSS-120B generator · OSS-20B groundedness checker.
    </div>""", unsafe_allow_html=True)
