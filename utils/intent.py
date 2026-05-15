import streamlit as st


@st.cache_data(ttl=3600, show_spinner=False)
def generate_executive_summary(
    n_suppliers: int,
    n_regions: int,
    n_red: int,
    n_amber: int,
    n_green: int,
    high_risk_pct: float,
    high_risk_spend: float,
    single_source_red: int,
    open_events: int,
    programs_at_risk: int,
    top_red_names: str,
) -> str:
    """Generate LLM executive summary. Cached per session per filter state."""
    from litellm import completion as litellm_completion

    prompt = f"""You are a Chief Procurement Officer writing a concise executive portfolio summary.

Portfolio snapshot:
- Total suppliers monitored: {n_suppliers:,}
- Regions covered: {n_regions}
- Risk distribution: {n_red} RED ({high_risk_pct:.1f}%), {n_amber} AMBER, {n_green} GREEN
- High-risk annual spend exposure: €{high_risk_spend/1e6:.1f}M
- Single-source RED suppliers (no qualified alternative): {single_source_red}
- Open external alerts (ESG, sanctions, geopolitical): {open_events}
- NPI/APQP programmes with delayed milestones: {programs_at_risk}
- Top priority suppliers: {top_red_names}

Write a 3-paragraph executive brief:
1. Portfolio risk status and headline numbers (2-3 sentences)
2. Most critical risks requiring immediate action, with specific supplier context (2-3 sentences)
3. Recommended actions with clear timelines and owners (3 bullet points)

Rules:
- Be direct and specific — no generic filler
- Use supplier quality terminology (SCAR, OTD, PPM, for-cause audit, dual-source)
- Quantify risks in business terms (spend exposure, supply continuity days)
- Actions must have timelines (e.g. "within 30 days") and owners (e.g. "Supply Chain Director")
- Do not mention that this is AI-generated"""

    try:
        response = litellm_completion(
            model="groq/openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return (
            f"The supplier portfolio currently spans **{n_suppliers:,} suppliers** across **{n_regions} regions**. "
            f"**{n_red} suppliers ({high_risk_pct:.1f}%)** are rated HIGH RISK, representing **€{high_risk_spend/1e6:.1f}M** in annual spend exposure. "
            f"Of these, **{single_source_red} are sole-source** dependencies with no qualified alternative.\n\n"
            f"Top priority suppliers requiring immediate attention: **{top_red_names}**. "
            f"**{open_events} external alerts** are currently open. "
            f"**{programs_at_risk} NPI/APQP programmes** have delayed milestones.\n\n"
            f"*Recommended actions: (1) Schedule for-cause audits for all RED sole-source suppliers within 30 days. "
            f"(2) Initiate dual-sourcing feasibility for top 3 single-source RED suppliers. "
            f"(3) Review all open Critical/High external events for CAPA linkage.*"
        )


_INTENT_FALLBACK = {
    "intent": "general", "country": None, "ppm_threshold": None,
    "risk_tier": None, "finding_type": None, "product_family": None,
}


@st.cache_data(ttl=300, show_spinner=False)
def classify_portfolio_intent(q: str) -> dict:
    """
    LLM intent classifier for portfolio Q&A queries.
    Top-level cached function — not redefined on every render cycle.
    Returns structured intent dict for routing to the correct data query.
    """
    from litellm import completion as _completion
    import json as _json
    import re as _re

    prompt = f"""Classify this supplier portfolio query into a structured filter.

Query: {q}

Return ONLY a JSON object with these exact fields (no markdown, no explanation):
{{
  "intent": "red_risk" | "single_source" | "ppm_threshold" | "audit_findings" | "claim_categories" | "capa_events" | "geopolitical" | "apqp_delayed" | "general",
  "country": "country name or null",
  "ppm_threshold": number or null,
  "risk_tier": "red" | "amber" | "green" | null,
  "finding_type": "Major NCR" | "Critical NCR" | "Minor NCR" | null,
  "product_family": "product family name or null"
}}"""
    try:
        resp = _completion(
            model="groq/openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=150,
        )
        text = resp.choices[0].message.content.strip()
        # Extract the first JSON object even if the model wraps it in extra text
        match = _re.search(r'\{.*?\}', text, _re.DOTALL)
        if not match:
            return _INTENT_FALLBACK
        return _json.loads(match.group())
    except Exception:
        return _INTENT_FALLBACK
