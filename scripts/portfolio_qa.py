"""
scripts/portfolio_qa.py — Streamlit-free portfolio data Q&A.

Extracts the two-layer logic from sicc_pages/supplier_qa_agent.py and
utils/intent.py so it can be called from the FastAPI layer.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "supplier_portfolio.db"

_INTENT_FALLBACK: dict[str, Any] = {
    "intent": "general", "metric": None, "sort_order": None,
    "country": None, "ppm_threshold": None, "otd_threshold": None,
    "risk_tier": None, "finding_type": None, "product_family": None, "limit": None,
}

_INTENT_PROMPT = """Classify this supplier portfolio query into a structured filter.

Query: {q}

Return ONLY a JSON object with these exact fields (no markdown, no explanation):
{{
  "intent": "red_risk" | "single_source" | "ppm_threshold" | "otd_performance" | "metric_ranking" | "audit_findings" | "claim_categories" | "capa_events" | "geopolitical" | "apqp_delayed" | "general",
  "metric": "ppm" | "otd" | "audit_score" | "scar_count" | "risk_score" | "spend" | null,
  "sort_order": "asc" | "desc" | null,
  "country": "country name or null",
  "ppm_threshold": number or null,
  "otd_threshold": number or null,
  "risk_tier": "red" | "amber" | "green" | null,
  "finding_type": "Major NCR" | "Critical NCR" | "Minor NCR" | null,
  "product_family": "product family name or null",
  "limit": number or null
}}

Intent guide:
- metric_ranking: ANY ranking/comparison/"who is worst/best" query — most common.
  metric: ppm, otd, audit_score, scar_count, risk_score (default), spend
  sort_order: asc = lowest first (worst OTD/audit); desc = highest first (worst PPM/SCARs)
  Default: metric=risk_score, sort_order=asc, limit=10
- otd_performance: OTD queries with a specific % threshold ("OTD below 85%")
- ppm_threshold: PPM queries with a specific numeric threshold ("PPM > 500")
- red_risk: suppliers specifically in the RED risk tier
- apqp_delayed: delayed APQP/NPI programmes
- audit_findings: suppliers with specific audit finding types
- claim_categories: breakdown of claim/NCR categories
- capa_events: open external events with no linked CAPA
- geopolitical: geopolitical or ESG risk events
- single_source: sole-source supplier queries"""


def classify_portfolio_intent(question: str) -> dict[str, Any]:
    """LLM intent classifier — no Streamlit dependency."""
    prompt = _INTENT_PROMPT.format(q=question)
    try:
        from groq import Groq
        client = Groq()
        resp = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are a supplier portfolio query classifier. Return only valid JSON, no markdown."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=200,
        )
        text = resp.choices[0].message.content.strip()
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if not match:
            return _INTENT_FALLBACK
        return json.loads(match.group())
    except Exception:
        return _INTENT_FALLBACK


def _load_tables() -> dict[str, pd.DataFrame]:
    conn = sqlite3.connect(_DB_PATH)
    tables = {}
    for name in ["suppliers", "supplier_kpis", "risk_scores", "claims", "audits", "apqp_projects", "external_events"]:
        try:
            tables[name] = pd.read_sql(f"SELECT * FROM {name}", conn)
        except Exception:
            tables[name] = pd.DataFrame()
    conn.close()
    return tables


_METRIC_MAP = {
    "ppm":         ("avg_ppm_3m",           False),
    "otd":         ("avg_otd_3m",           True),
    "audit_score": ("avg_audit_score_3m",   True),
    "scar_count":  ("avg_scar_count_3m",    False),
    "risk_score":  ("composite_risk_score", True),
    "spend":       ("annual_spend_eur",     False),
}

_DISPLAY_COLS = [
    "name", "product_family", "country", "risk_label",
    "composite_risk_score", "avg_ppm_3m", "avg_otd_3m",
    "avg_audit_score_3m", "single_source", "annual_spend_eur",
    "recommended_action",
]


def run_portfolio_query(
    question: str,
    family: str | None = None,
    region: str | None = None,
    risk: str | None = None,
) -> dict[str, Any]:
    """
    Classify the question and run the appropriate structured query.
    Returns { answer, rows, columns, intent, scope_count }.
    """
    tables = _load_tables()
    suppliers   = tables["suppliers"]
    risk_scores = tables["risk_scores"]
    claims      = tables["claims"]
    audits      = tables["audits"]
    apqp        = tables["apqp_projects"]
    events      = tables["external_events"]

    # Build merged base dataframe
    base = risk_scores.merge(
        suppliers[["supplier_id", "name", "product_family", "country",
                   "region", "single_source", "spend_tier",
                   "qualification_status", "certification"]],
        on="supplier_id",
        how="left",
    )

    # Apply filters
    if family:
        base = base[base["product_family"] == family]
    if region:
        base = base[base["region"] == region]
    if risk:
        base = base[base["risk_label"] == risk.lower()]

    scope_count = len(base)
    intent = classify_portfolio_intent(question)

    answer_text = ""
    show_df: pd.DataFrame | None = None

    i = intent.get("intent", "general")

    if i in ("red_risk",) or intent.get("risk_tier") == "red":
        tier = str(intent.get("risk_tier") or "red")
        limit = int(intent.get("limit") or 0)
        show_df = base[base["risk_label"] == tier].sort_values("composite_risk_score")
        if limit:
            show_df = show_df.head(limit)
        answer_text = f"Found {len(show_df)} {tier.upper()}-risk suppliers matching your criteria."

    elif i == "single_source":
        show_df = base[base["single_source"].isin([1, True, "True", "1"])].sort_values("risk_label")
        red_count = len(show_df[show_df["risk_label"] == "red"])
        answer_text = f"Found {len(show_df)} single-source suppliers. {red_count} are RED risk."

    elif i == "ppm_threshold":
        threshold = intent.get("ppm_threshold")
        limit = int(intent.get("limit") or 0)
        if threshold is not None:
            show_df = base[base["avg_ppm_3m"] > float(threshold)].sort_values("avg_ppm_3m", ascending=False)
            answer_text = f"Found {len(show_df)} suppliers with PPM > {float(threshold):.0f} (3-month average)."
        else:
            n = limit or 10
            show_df = base.sort_values("avg_ppm_3m", ascending=False).head(n)
            answer_text = f"{len(show_df)} worst-performing suppliers by PPM (3-month average)."
        if limit and threshold is not None:
            show_df = show_df.head(limit)

    elif i == "otd_performance":
        threshold = intent.get("otd_threshold")
        limit = int(intent.get("limit") or 0)
        if threshold is not None:
            show_df = base[base["avg_otd_3m"] < float(threshold)].sort_values("avg_otd_3m")
            answer_text = f"Found {len(show_df)} suppliers with OTD below {float(threshold):.0f}% (3-month average)."
        else:
            n = limit or 10
            show_df = base.sort_values("avg_otd_3m").head(n)
            answer_text = f"{len(show_df)} worst-performing suppliers by OTD."

    elif i == "metric_ranking":
        metric = str(intent.get("metric") or "risk_score")
        sort_order = str(intent.get("sort_order") or "asc")
        num_match = re.search(r"\b(\d+)\b", question)
        limit = int(intent.get("limit") or 0) or (int(num_match.group(1)) if num_match else 10)
        col, default_asc = _METRIC_MAP.get(metric, ("composite_risk_score", True))
        ascending = sort_order == "asc"
        if col in base.columns:
            show_df = base.sort_values(col, ascending=ascending).head(limit)
            direction = "lowest" if ascending else "highest"
            label = col.replace("avg_", "").replace("_3m", "").replace("_", " ").upper()
            answer_text = f"{len(show_df)} suppliers — {direction} {label} (3-month average)."
        else:
            show_df = base.sort_values("composite_risk_score").head(limit)
            answer_text = f"{len(show_df)} highest-risk suppliers by composite score."

    elif i == "audit_findings":
        finding = str(intent.get("finding_type") or "Major NCR")
        if finding not in ("Major NCR", "Critical NCR", "Minor NCR"):
            finding = "Major NCR"
        audit_sups = audits[audits["highest_finding_type"] == finding]["supplier_id"].unique()
        show_df = base[base["supplier_id"].isin(audit_sups)].sort_values("composite_risk_score")
        answer_text = f"Found {len(show_df)} suppliers with {finding} audit findings."

    elif i == "claim_categories":
        pfam = intent.get("product_family")
        scope = claims[claims["supplier_id"].isin(base["supplier_id"])]
        if pfam:
            scope = scope[scope["product_family"] == pfam]
        show_df = (
            scope.groupby("category").size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        fam_str = f" for {pfam} suppliers" if pfam else ""
        answer_text = f"Top {len(show_df)} claim categories{fam_str} across {len(scope):,} claims."

    elif i == "capa_events":
        open_no_capa = events[
            events["requires_capa"].isin([True, 1, "True", "1"]) &
            ~events["capa_linked"].isin([True, 1, "True", "1"]) &
            events["status"].isin(["Open", "Under Review"])
        ]["supplier_id"].unique()
        show_df = base[base["supplier_id"].isin(open_no_capa)].sort_values("composite_risk_score")
        answer_text = f"Found {len(show_df)} suppliers with open alerts and no linked CAPA."

    elif i == "geopolitical":
        geo_sups = events[
            (events["event_type"] == "Geopolitical") &
            (events["severity"].isin(["High", "Critical"])) &
            (events["status"].isin(["Open", "Under Review", "Escalated"]))
        ]["supplier_id"].unique()
        geo_mask = base["supplier_id"].isin(geo_sups)
        country = intent.get("country")
        if country:
            country_mask = base["country"].str.lower() == str(country).lower()
            show_df = base[geo_mask & country_mask].sort_values("composite_risk_score")
            answer_text = f"Found {len(show_df)} {country}-based suppliers with active High/Critical geopolitical events."
        else:
            show_df = base[geo_mask].sort_values("composite_risk_score")
            answer_text = f"Found {len(show_df)} suppliers with active High/Critical geopolitical events."

    elif i == "apqp_delayed":
        red_sups = base[base["risk_label"] == "red"]["supplier_id"].unique()
        delayed = apqp[apqp["is_delayed"].isin([True, 1, "True", "1"])].merge(
            suppliers[["supplier_id", "name"]], on="supplier_id"
        )
        show_df = delayed[delayed["supplier_id"].isin(red_sups)][
            ["project_id", "supplier_id", "name", "project_type", "status",
             "customer_sop_date", "completion_pct", "is_delayed"]
        ]
        answer_text = f"Found {len(show_df)} delayed APQP programmes linked to RED-risk suppliers."

    else:
        # Keyword fallback
        ql = question.lower()
        num_match = re.search(r"\b(\d+)\b", question)
        n = int(num_match.group(1)) if num_match else 10
        if any(w in ql for w in ("otd", "on-time", "delivery")):
            show_df = base.sort_values("avg_otd_3m").head(n)
            answer_text = f"{len(show_df)} worst-performing suppliers by OTD."
        elif any(w in ql for w in ("ppm", "defect")):
            show_df = base.sort_values("avg_ppm_3m", ascending=False).head(n)
            answer_text = f"{len(show_df)} worst-performing suppliers by PPM."
        elif any(w in ql for w in ("audit",)):
            show_df = base.sort_values("avg_audit_score_3m").head(n)
            answer_text = f"{len(show_df)} lowest-scoring suppliers by audit score."
        elif any(w in ql for w in ("scar", "corrective")):
            show_df = base.sort_values("avg_scar_count_3m", ascending=False).head(n)
            answer_text = f"{len(show_df)} suppliers with most SCARs."
        elif any(w in ql for w in ("spend", "cost")):
            show_df = base.sort_values("annual_spend_eur", ascending=False).head(n)
            answer_text = f"{len(show_df)} highest-spend suppliers."
        elif any(w in ql for w in ("ppap", "delayed", "delay", "apqp")):
            delayed = apqp[apqp["is_delayed"].isin([True, 1, "True", "1"])].merge(
                suppliers[["supplier_id", "name"]], on="supplier_id"
            )
            show_df = delayed[["project_id", "supplier_id", "name", "project_type",
                                "status", "customer_sop_date", "completion_pct"]]
            answer_text = f"Found {len(show_df)} delayed APQP/PPAP programmes."
        else:
            show_df = base.sort_values("composite_risk_score").head(n)
            answer_text = f"{len(show_df)} highest-risk suppliers by composite score."

    # Serialise result
    rows: list[dict] = []
    columns: list[str] = []
    if show_df is not None and not show_df.empty:
        disp_cols = [c for c in _DISPLAY_COLS if c in show_df.columns]
        if len(disp_cols) < 3:
            # Specialised result (APQP, claims, etc.) — show all meaningful columns
            disp_cols = [c for c in show_df.columns if c != "supplier_id"]
        out = show_df[disp_cols].head(25).copy()
        for col in out.select_dtypes("float").columns:
            out[col] = out[col].round(2)
        rows_raw = out.fillna("").to_dict(orient="records")
        # Always embed supplier_id in each row for UI navigation (not shown as column)
        if "supplier_id" in show_df.columns:
            sids = show_df["supplier_id"].head(25).tolist()
            for row, sid in zip(rows_raw, sids):
                row["supplier_id"] = sid
        rows = rows_raw
        columns = list(out.columns)

    return {
        "answer": answer_text,
        "intent": intent.get("intent", "general"),
        "rows": rows,
        "columns": columns,
        "scope_count": scope_count,
    }
