"""
Single-source continuity agent.

Assesses continuity exposure for single-source suppliers and recommends buffer
stock, dual-source urgency, BCP/tooling controls, escalation owners, and
executive actions.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field


ContinuityLevel = Literal["critical", "high", "medium", "monitor"]


class ContinuityPlan(BaseModel):
    supplier_id: str
    supplier_name: str
    continuity_level: ContinuityLevel
    continuity_score: float
    risk_tier: str
    buffer_stock_target_days: str
    assessment_frequency: str
    decision_summary: str
    exposure_drivers: list[str] = Field(default_factory=list)
    mandatory_actions: list[str] = Field(default_factory=list)
    dual_sourcing_actions: list[str] = Field(default_factory=list)
    bcp_controls: list[str] = Field(default_factory=list)
    escalation_owner: str
    source_documents: list[str] = Field(default_factory=list)


SOURCE_DOCUMENTS = [
    "single_source_risk_management.md",
    "risk_tier_definitions.md",
    "supplier_qualification_procedure.md",
    "for_cause_audit_trigger_criteria.md",
    "external_risk_event_response.md",
]


def _as_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, dict):
        return row
    if hasattr(row, "to_dict"):
        return row.to_dict()
    return dict(row)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _buffer_target(risk_tier: str) -> str:
    if risk_tier == "red":
        return "60-90 days"
    if risk_tier == "amber":
        return "45 days"
    if risk_tier == "green":
        return "30 days"
    return "30-45 days until risk tier is confirmed"


def _assessment_frequency(risk_tier: str) -> str:
    if risk_tier == "red":
        return "Immediate assessment + monthly review"
    if risk_tier == "amber":
        return "Semi-annual assessment"
    if risk_tier == "green":
        return "Annual assessment"
    return "Quarterly review until tier is confirmed"


def assess_single_source_continuity(
    supplier: Any,
    risk_row: Any,
    supplier_claims: pd.DataFrame,
    supplier_events: pd.DataFrame,
    supplier_apqp: pd.DataFrame,
) -> ContinuityPlan:
    supplier_data = _as_dict(supplier)
    risk_data = _as_dict(risk_row)
    supplier_id = str(supplier_data.get("supplier_id", risk_data.get("supplier_id", "unknown")))
    supplier_name = str(supplier_data.get("name", "Unknown supplier"))
    risk_tier = str(risk_data.get("risk_label", "unknown")).lower()
    single_source = _bool(supplier_data.get("single_source", risk_data.get("single_source", False)))

    score = 0.0
    drivers: list[str] = []

    if single_source:
        score += 25
        drivers.append("Supplier is flagged single-source in SICC.")
    else:
        drivers.append("Supplier is not flagged single-source; continuity controls are advisory.")

    if risk_tier == "red":
        score += 35
        drivers.append("Supplier is RED risk, activating single-source emergency protocol.")
    elif risk_tier == "amber":
        score += 20
        drivers.append("Supplier is AMBER risk and requires enhanced single-source mitigation.")
    elif risk_tier == "green":
        score += 8
        drivers.append("Supplier is GREEN risk but still requires annual single-source controls.")

    ppm = _num(risk_data.get("avg_ppm_3m"))
    otd = _num(risk_data.get("avg_otd_3m"), 100.0)
    audit = _num(risk_data.get("avg_audit_score_3m"), 100.0)
    scar = _num(risk_data.get("avg_scar_count_3m"))
    if ppm > 500:
        score += 10
        drivers.append(f"PPM is RED-level at {ppm:.0f}.")
    if otd < 90:
        score += 10
        drivers.append(f"OTD is RED-level at {otd:.1f}%.")
    if audit < 60:
        score += 8
        drivers.append(f"Audit score is RED-level at {audit:.1f}.")
    if scar >= 4:
        score += 8
        drivers.append(f"Open/recent SCAR burden is RED-level at {scar:.1f}.")

    open_claims = pd.DataFrame()
    if supplier_claims is not None and not supplier_claims.empty and "status" in supplier_claims:
        open_claims = supplier_claims[supplier_claims["status"].astype(str).str.lower() != "closed"]
        if len(open_claims) >= 5:
            score += 10
            drivers.append(f"{len(open_claims)} open claims/SCARs increase continuity risk.")
        elif len(open_claims) >= 2:
            score += 5
            drivers.append(f"{len(open_claims)} open claims/SCARs require monitoring.")

    open_high_events = pd.DataFrame()
    if supplier_events is not None and not supplier_events.empty and {"status", "severity"}.issubset(supplier_events.columns):
        open_high_events = supplier_events[
            supplier_events["status"].astype(str).str.lower().isin(["open", "under review", "escalated"])
            & supplier_events["severity"].isin(["Critical", "High"])
        ]
        if not open_high_events.empty:
            score += 12
            drivers.append(f"{len(open_high_events)} high/critical external event(s) are open.")

    delayed_apqp = pd.DataFrame()
    if supplier_apqp is not None and not supplier_apqp.empty and "is_delayed" in supplier_apqp:
        delayed_apqp = supplier_apqp[supplier_apqp["is_delayed"].astype(int) == 1]
        if not delayed_apqp.empty:
            score += 8
            drivers.append(f"{len(delayed_apqp)} delayed APQP/NPI programme(s) may constrain alternate-source readiness.")

    score = round(max(0, min(score, 100)), 1)
    if score >= 85:
        level: ContinuityLevel = "critical"
    elif score >= 65:
        level = "high"
    elif score >= 40:
        level = "medium"
    else:
        level = "monitor"

    mandatory_actions = [
        "Confirm supplier is listed on the Single-Source Risk Register.",
        "Verify documented Business Continuity Plan including recovery time objective and communication escalation.",
        f"Confirm buffer stock location, ownership, and coverage against target: {_buffer_target(risk_tier)}.",
        "Verify tooling ownership, insurance, access rights, and tooling rescue clause where applicable.",
        "Schedule required second-party audit at the required single-source frequency.",
    ]
    dual_sourcing_actions = []
    if risk_tier in {"amber", "red"}:
        dual_sourcing_actions = [
            "Start dual-sourcing feasibility assessment and identify qualified alternative candidates.",
            "Assess technical feasibility, qualification timeline, and PPAP/APQP impact.",
            "Present proceed / accept risk / design-out decision to Supply Chain Director and Procurement Director.",
            "If proceeding, initiate alternate supplier qualification with Supplier Quality Engineer resource assigned.",
        ]
    else:
        dual_sourcing_actions = [
            "Review alternate-source feasibility annually or when risk tier worsens.",
            "Document why single-source status remains acceptable under current risk conditions.",
        ]

    if risk_tier == "red" and single_source:
        mandatory_actions = [
            "Notify VP Operations within 24 hours.",
            "Notify customer programme manager within 48 hours if SOP/customer supply is at risk.",
            "Initiate emergency buffer stock purchase within 48 hours if below target.",
            "Schedule for-cause audit within 14 days.",
            "Fast-track dual-sourcing feasibility: 15-day feasibility and 10-day decision.",
            "Set weekly executive review until GREEN performance is restored.",
            "Create contingency production plan within 7 days.",
        ] + mandatory_actions

    bcp_controls = [
        "Alternative production site or subcontracting arrangement documented.",
        "Maximum recovery time objective documented and reviewed.",
        "Supplier disruption communication escalation procedure tested.",
        "Buffer stock or safety stock arrangement documented.",
        "Sub-tier critical input risks reviewed.",
    ]

    if level == "critical":
        owner = "Supply Chain Director"
        summary = "Critical single-source exposure requires emergency continuity governance and executive review."
    elif level == "high":
        owner = "Category Manager"
        summary = "High single-source exposure requires active mitigation and documented dual-source feasibility."
    elif level == "medium":
        owner = "Supplier Quality Engineer"
        summary = "Moderate single-source exposure requires enhanced monitoring and continuity evidence review."
    else:
        owner = "Procurement Manager"
        summary = "Single-source exposure is currently monitor-level but annual controls remain mandatory."

    return ContinuityPlan(
        supplier_id=supplier_id,
        supplier_name=supplier_name,
        continuity_level=level,
        continuity_score=score,
        risk_tier=risk_tier,
        buffer_stock_target_days=_buffer_target(risk_tier),
        assessment_frequency=_assessment_frequency(risk_tier),
        decision_summary=summary,
        exposure_drivers=drivers,
        mandatory_actions=mandatory_actions,
        dual_sourcing_actions=dual_sourcing_actions,
        bcp_controls=bcp_controls,
        escalation_owner=owner,
        source_documents=SOURCE_DOCUMENTS,
    )


def build_continuity_watchlist(
    suppliers: pd.DataFrame,
    risk_scores: pd.DataFrame,
    claims: pd.DataFrame,
    events: pd.DataFrame,
    apqp: pd.DataFrame,
    supplier_ids: set[str] | None = None,
    top_n: int = 50,
) -> list[ContinuityPlan]:
    ids = supplier_ids or set(suppliers["supplier_id"])
    plans: list[ContinuityPlan] = []
    risk_lookup = risk_scores.set_index("supplier_id").to_dict("index") if not risk_scores.empty else {}

    single_source_suppliers = suppliers[
        suppliers["supplier_id"].isin(ids) & suppliers["single_source"].astype(bool)
    ]
    for _, supplier in single_source_suppliers.iterrows():
        sid = supplier["supplier_id"]
        plan = assess_single_source_continuity(
            supplier=supplier,
            risk_row=risk_lookup.get(sid, {}),
            supplier_claims=claims[claims["supplier_id"] == sid] if not claims.empty else pd.DataFrame(),
            supplier_events=events[events["supplier_id"] == sid] if not events.empty else pd.DataFrame(),
            supplier_apqp=apqp[apqp["supplier_id"] == sid] if not apqp.empty else pd.DataFrame(),
        )
        plans.append(plan)

    plans.sort(key=lambda item: item.continuity_score, reverse=True)
    return plans[:top_n]
