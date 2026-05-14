"""
Supplier early-warning alert agent.

This module looks for suppliers moving in the wrong direction by comparing
recent KPI windows, open risk events, APQP delays, claims, and current risk
context. It is intentionally deterministic so alerts are stable, auditable,
and cheap to run in Streamlit without burning model calls.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field


AlertLevel = Literal["critical", "high", "medium", "watch"]


class SupplierTrendAlert(BaseModel):
    supplier_id: str
    supplier_name: str
    current_risk: str
    alert_level: AlertLevel
    trend_score: float
    direction: str
    signals: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    recommended_action: str
    escalation_owner: str
    source_documents: list[str] = Field(default_factory=list)


SOURCE_MAP = {
    "kpi": ["supplier_kpi_definitions.md", "risk_tier_definitions.md"],
    "quality": ["scar_process_escalation.md", "corrective_action_closure_requirements.md"],
    "delivery": ["supplier_kpi_definitions.md", "supplier_development_methodology.md"],
    "audit": ["for_cause_audit_trigger_criteria.md", "audit_finding_classification.md"],
    "external": ["external_risk_event_response.md"],
    "apqp": ["apqp_phase_gate_guide.md", "ppap_submission_checklist.md"],
    "single_source": ["single_source_risk_management.md"],
}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _pct_change(recent: float, prior: float) -> float:
    if abs(prior) < 1e-9:
        return 0.0
    return (recent - prior) / abs(prior) * 100


def _alert_level(score: float) -> AlertLevel:
    if score >= 85:
        return "critical"
    if score >= 65:
        return "high"
    if score >= 40:
        return "medium"
    return "watch"


def _recommended_action(level: AlertLevel, risk: str, signals: list[str]) -> tuple[str, str]:
    text = " ".join(signals).lower()
    if level == "critical":
        return (
            "Open an immediate supplier risk review, assign accountable owners, and decide containment or escalation within 5 business days.",
            "Supplier Quality Manager",
        )
    if "single-source" in text:
        return (
            "Start a continuity mitigation review and confirm buffer stock, recovery plan, and alternate-source feasibility.",
            "Category Manager",
        )
    if "apqp" in text or "launch" in text:
        return (
            "Run an APQP recovery checkpoint and freeze launch readiness assumptions until delayed deliverables are recovered.",
            "APQP Program Manager",
        )
    if "audit" in text:
        return (
            "Schedule a focused process review or for-cause audit against the deteriorating control area.",
            "Lead Auditor",
        )
    if risk == "green":
        return (
            "Move supplier to enhanced monitoring for the next review cycle and request a short corrective trend response.",
            "Supplier Quality Engineer",
        )
    return (
        "Create a targeted supplier development action plan with weekly KPI review until the trend stabilises.",
        "Supplier Quality Engineer",
    )


def build_supplier_trend_alerts(
    suppliers: pd.DataFrame,
    kpis: pd.DataFrame,
    risk_scores: pd.DataFrame,
    claims: pd.DataFrame,
    audits: pd.DataFrame,
    events: pd.DataFrame,
    apqp: pd.DataFrame,
    supplier_ids: set[str] | None = None,
    top_n: int = 30,
) -> list[SupplierTrendAlert]:
    if suppliers.empty or kpis.empty:
        return []

    ids = supplier_ids or set(suppliers["supplier_id"])
    supplier_lookup = suppliers.set_index("supplier_id").to_dict("index")
    risk_lookup = risk_scores.set_index("supplier_id").to_dict("index") if not risk_scores.empty else {}

    alerts: list[SupplierTrendAlert] = []

    for sid, group in kpis[kpis["supplier_id"].isin(ids)].groupby("supplier_id"):
        if len(group) < 6 or sid not in supplier_lookup:
            continue

        group = group.sort_values("year_month")
        recent = group.tail(3)
        prior = group.tail(6).head(3)
        if prior.empty:
            continue

        supplier = supplier_lookup[sid]
        risk = str(risk_lookup.get(sid, {}).get("risk_label", "unknown")).lower()
        score = 0.0
        signals: list[str] = []
        docs: set[str] = set()

        ppm_recent = _num(recent["ppm_external"].mean())
        ppm_prior = _num(prior["ppm_external"].mean())
        ppm_delta = _pct_change(ppm_recent, ppm_prior)
        if ppm_delta >= 25 and ppm_recent - ppm_prior >= 50:
            add = min(25, 10 + ppm_delta / 4)
            score += add
            signals.append(f"External PPM is rising: {ppm_prior:.0f} -> {ppm_recent:.0f} ({ppm_delta:+.0f}%).")
            docs.update(SOURCE_MAP["kpi"] + SOURCE_MAP["quality"])

        otd_recent = _num(recent["otd_pct"].mean(), 100.0)
        otd_prior = _num(prior["otd_pct"].mean(), 100.0)
        otd_delta = otd_recent - otd_prior
        if otd_delta <= -2.0:
            score += min(25, 10 + abs(otd_delta) * 3)
            signals.append(f"OTD is deteriorating: {otd_prior:.1f}% -> {otd_recent:.1f}% ({otd_delta:+.1f} pts).")
            docs.update(SOURCE_MAP["kpi"] + SOURCE_MAP["delivery"])

        audit_recent = _num(recent["audit_score"].mean(), 100.0)
        audit_prior = _num(prior["audit_score"].mean(), 100.0)
        audit_delta = audit_recent - audit_prior
        if audit_delta <= -4.0 or audit_recent < 75:
            score += min(20, 8 + abs(min(audit_delta, 0)) * 2)
            signals.append(f"Audit score trend is weakening: {audit_prior:.1f} -> {audit_recent:.1f}.")
            docs.update(SOURCE_MAP["audit"])

        scar_recent = _num(recent["scar_count"].mean())
        scar_prior = _num(prior["scar_count"].mean())
        if scar_recent >= 2 and scar_recent > scar_prior:
            score += min(18, 8 + (scar_recent - scar_prior) * 5)
            signals.append(f"SCAR burden is increasing: {scar_prior:.1f} -> {scar_recent:.1f}.")
            docs.update(SOURCE_MAP["quality"])

        sid_claims = claims[claims["supplier_id"] == sid] if not claims.empty else pd.DataFrame()
        open_claims = sid_claims[sid_claims["status"].astype(str).str.lower() != "closed"] if not sid_claims.empty else pd.DataFrame()
        if len(open_claims) >= 2:
            score += min(12, 4 + len(open_claims) * 2)
            signals.append(f"{len(open_claims)} open claim(s) are still unresolved.")
            docs.update(SOURCE_MAP["quality"])

        sid_events = events[events["supplier_id"] == sid] if not events.empty else pd.DataFrame()
        open_events = sid_events[sid_events["status"].astype(str).str.lower().isin(["open", "under review", "escalated"])] if not sid_events.empty else pd.DataFrame()
        high_events = open_events[open_events["severity"].isin(["Critical", "High"])] if not open_events.empty and "severity" in open_events else pd.DataFrame()
        if not high_events.empty:
            score += min(20, 8 + len(high_events) * 5)
            signals.append(f"{len(high_events)} high/critical external event(s) are open.")
            docs.update(SOURCE_MAP["external"])

        sid_apqp = apqp[apqp["supplier_id"] == sid] if not apqp.empty else pd.DataFrame()
        delayed = sid_apqp[sid_apqp["is_delayed"].astype(int) == 1] if not sid_apqp.empty and "is_delayed" in sid_apqp else pd.DataFrame()
        if not delayed.empty:
            score += min(18, 8 + len(delayed) * 3)
            signals.append(f"{len(delayed)} APQP/NPI project(s) are delayed.")
            docs.update(SOURCE_MAP["apqp"])

        single_source = bool(supplier.get("single_source", risk_lookup.get(sid, {}).get("single_source", False)))
        if single_source and score >= 30:
            score += 10
            signals.append("Single-source status amplifies the deterioration risk.")
            docs.update(SOURCE_MAP["single_source"])

        if risk == "green" and score >= 25:
            score += 8
            signals.append("Supplier is still GREEN, but leading indicators are moving the wrong way.")
        elif risk == "amber" and score >= 35:
            score += 6
            signals.append("Supplier is AMBER and trending toward RED controls if not stabilised.")

        if score < 25 or not signals:
            continue

        level = _alert_level(score)
        action, owner = _recommended_action(level, risk, signals)
        direction = "deteriorating"
        if risk == "green":
            direction = "early deterioration"
        elif risk == "amber":
            direction = "red-risk drift"

        alerts.append(SupplierTrendAlert(
            supplier_id=sid,
            supplier_name=str(supplier.get("name", sid)),
            current_risk=risk,
            alert_level=level,
            trend_score=round(min(score, 100), 1),
            direction=direction,
            signals=signals,
            evidence={
                "ppm_recent_3m": round(ppm_recent, 1),
                "ppm_prior_3m": round(ppm_prior, 1),
                "otd_recent_3m": round(otd_recent, 1),
                "otd_prior_3m": round(otd_prior, 1),
                "audit_recent_3m": round(audit_recent, 1),
                "audit_prior_3m": round(audit_prior, 1),
                "scar_recent_3m": round(scar_recent, 1),
                "scar_prior_3m": round(scar_prior, 1),
                "open_claims": int(len(open_claims)),
                "open_high_events": int(len(high_events)),
                "delayed_apqp": int(len(delayed)),
                "generated_on": date.today().isoformat(),
            },
            recommended_action=action,
            escalation_owner=owner,
            source_documents=sorted(docs or set(SOURCE_MAP["kpi"])),
        ))

    alerts.sort(key=lambda alert: alert.trend_score, reverse=True)
    return alerts[:top_n]
