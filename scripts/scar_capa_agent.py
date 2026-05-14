"""
SCAR / CAPA triage agent.

The agent converts a concrete supplier claim or quality issue into an
operational decision: finding grade, SCAR escalation level, deadlines,
required evidence, containment controls, and closure criteria.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field


FindingGrade = Literal["Observation", "Minor NCR", "Major NCR", "Critical NCR"]
EscalationLevel = Literal["Level 0", "Level 1", "Level 2", "Level 3"]


class ScarCapaTriage(BaseModel):
    incident_number: str
    supplier_id: str
    supplier_name: str
    finding_grade: FindingGrade
    scar_escalation_level: EscalationLevel
    severity_score: float
    issue_summary: str
    triggers: list[str] = Field(default_factory=list)
    immediate_containment: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    deadlines: dict[str, str] = Field(default_factory=dict)
    escalation_actions: list[str] = Field(default_factory=list)
    closure_criteria: list[str] = Field(default_factory=list)
    owner: str
    source_documents: list[str] = Field(default_factory=list)


SOURCE_DOCUMENTS = [
    "scar_process_escalation.md",
    "corrective_action_closure_requirements.md",
    "risk_tier_definitions.md",
    "for_cause_audit_trigger_criteria.md",
]


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


def _as_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, dict):
        return row
    if hasattr(row, "to_dict"):
        return row.to_dict()
    return dict(row)


def _due(days: int) -> str:
    return (date.today() + timedelta(days=days)).isoformat()


def _is_late(expected: Any, submitted: Any, status: Any) -> bool:
    if str(status).strip().lower() in {"submitted", "closed", "approved"}:
        return False
    try:
        expected_date = pd.to_datetime(expected).date()
    except Exception:
        return False
    if pd.isna(expected_date):
        return False
    return expected_date < date.today() and not str(submitted or "").strip()


def _grade_and_level(score: float, triggers: list[str]) -> tuple[FindingGrade, EscalationLevel]:
    joined = " ".join(triggers).lower()
    if score >= 85 or "field escape" in joined or "customer complaint" in joined:
        return "Critical NCR", "Level 3"
    if score >= 60 or "recurring" in joined or "3 or more open" in joined:
        return "Major NCR", "Level 2"
    if score >= 35:
        return "Minor NCR", "Level 1"
    return "Observation", "Level 0"


def triage_claim(
    claim: Any,
    supplier: Any,
    risk_row: Any,
    supplier_claims: pd.DataFrame,
    supplier_kpis: pd.DataFrame,
    supplier_audits: pd.DataFrame,
    supplier_events: pd.DataFrame,
) -> ScarCapaTriage:
    claim_data = _as_dict(claim)
    supplier_data = _as_dict(supplier)
    risk_data = _as_dict(risk_row)

    bad_parts = _num(claim_data.get("number_of_bad_parts"))
    suspected = _num(claim_data.get("number_of_suspected_parts"))
    risk_label = str(risk_data.get("risk_label", "unknown")).lower()
    single_source = _bool(supplier_data.get("single_source", risk_data.get("single_source", False)))
    detected = " ".join([
        str(claim_data.get("how_detected", "")),
        str(claim_data.get("who_detected", "")),
        str(claim_data.get("where_detected", "")),
    ]).lower()

    score = 0.0
    triggers: list[str] = []

    if bad_parts >= 400:
        score += 30
        triggers.append(f"Large defect quantity: {bad_parts:.0f} confirmed bad parts.")
    elif bad_parts >= 50:
        score += 18
        triggers.append(f"Quality escape exceeds AMBER SCAR trigger scale: {bad_parts:.0f} bad parts.")
    elif bad_parts > 0:
        score += 8
        triggers.append(f"Confirmed defective quantity: {bad_parts:.0f} bad parts.")

    if suspected >= max(100, bad_parts * 3):
        score += 10
        triggers.append(f"Suspect population is material: {suspected:.0f} suspect parts.")

    if any(term in detected for term in ["customer", "field", "return"]):
        score += 28
        triggers.append("Field escape or customer complaint traced to supplier defect.")
    elif "assembly" in detected or "final test" in detected:
        score += 12
        triggers.append("Issue escaped supplier controls and was detected downstream.")

    if _bool(claim_data.get("is_recurrent")) or _bool(claim_data.get("is_recurring_incident")):
        score += 18
        triggers.append("Recurring defect mode or repeat incident.")

    open_claims = pd.DataFrame()
    if supplier_claims is not None and not supplier_claims.empty and "status" in supplier_claims:
        open_claims = supplier_claims[supplier_claims["status"].astype(str).str.lower() != "closed"]
        if len(open_claims) >= 3:
            score += 16
            triggers.append(f"Supplier has 3 or more open SCAR/claim records ({len(open_claims)} open).")

    late_phases = []
    for prefix, label in [("qr", "containment"), ("pd", "problem definition"), ("ca", "root cause"), ("ci", "implementation")]:
        if _is_late(claim_data.get(f"{prefix}_expected_date"), claim_data.get(f"{prefix}_submitted_date"), claim_data.get(f"{prefix}_status")):
            late_phases.append(label)
    if late_phases:
        score += 20
        triggers.append(f"Overdue SCAR phase response: {', '.join(late_phases)}.")

    if risk_label == "red":
        score += 15
        triggers.append("Supplier is currently RED risk.")
    elif risk_label == "amber":
        score += 8
        triggers.append("Supplier is currently AMBER risk.")

    if single_source and score >= 35:
        score += 10
        triggers.append("Single-source exposure increases containment and continuity risk.")

    if supplier_events is not None and not supplier_events.empty:
        open_high = supplier_events[
            supplier_events["status"].astype(str).str.lower().isin(["open", "under review", "escalated"])
            & supplier_events["severity"].isin(["Critical", "High"])
        ] if {"status", "severity"}.issubset(supplier_events.columns) else pd.DataFrame()
        if not open_high.empty:
            score += 10
            triggers.append(f"{len(open_high)} high/critical external risk event(s) are open.")

    grade, level = _grade_and_level(score, triggers)

    containment_due = {
        "Critical NCR": "48 hours",
        "Major NCR": "5 business days",
        "Minor NCR": "10 business days",
        "Observation": "next scheduled review",
    }[grade]

    deadlines = {
        "containment": containment_due,
        "problem_definition": "15 business days from SCAR issue",
        "root_cause": "30 business days from SCAR issue",
        "implementation": "60 business days from SCAR issue",
        "effectiveness": "30 business days after implementation",
    }
    if grade == "Critical NCR":
        deadlines.update({
            "containment": "48 hours",
            "ca_plan": "5 business days",
            "implementation": "30 business days",
            "closure_evidence": "35 business days",
            "on_site_reaudit": "within 14 days after closure evidence",
        })
    elif grade == "Major NCR":
        deadlines["closure_evidence"] = "65 business days"
        deadlines["follow_up_audit"] = "within 90 days of closure"
    elif grade == "Minor NCR":
        deadlines["closure_evidence"] = "30 business days"

    containment = [
        "Stop shipment of affected part numbers pending inspection disposition.",
        "Segregate and identify all suspect stock at supplier, in transit, and at receiving.",
        "Perform 100% sort or inspection of suspect population.",
        "Confirm interim supply of conforming replacement parts.",
        "Submit written containment confirmation to Supplier Quality Engineer.",
    ]

    evidence = [
        "Problem definition with affected part numbers, batches, dates, and quantities.",
        "Validated root cause analysis using 5-Why, fishbone, Is/Is-Not, or fault-tree as applicable.",
        "Corrective action implementation evidence: photos, updated work instructions, control plan, PFMEA, and training records.",
        "Effectiveness evidence showing no recurrence during the required monitoring period.",
    ]
    if grade in {"Major NCR", "Critical NCR"}:
        evidence.extend([
            "8D sections D4-D7 or equivalent structured corrective action report.",
            "Horizontal deployment evidence for similar processes or products.",
            "Follow-up audit or re-audit evidence before closure.",
        ])
    if grade == "Critical NCR":
        evidence.extend([
            "Independent containment verification such as third-party sort or 100% inspection data.",
            "Field impact assessment and customer notification evidence if exposure is confirmed.",
            "Supplier Quality Director or executive sign-off on the 8D report.",
        ])

    escalation_actions = []
    if level == "Level 1":
        escalation_actions = [
            "Issue formal written escalation notice to supplier quality manager.",
            "Notify procurement for commercial visibility.",
            "Schedule weekly follow-up until response is received.",
        ]
    elif level == "Level 2":
        escalation_actions = [
            "Place supplier on controlled shipping or 100% inspection at supplier cost.",
            "Notify procurement and suspend new business award pending recovery.",
            "Initiate supplier development plan and director-level review.",
        ]
    elif level == "Level 3":
        escalation_actions = [
            "Schedule emergency for-cause audit within 14 days.",
            "Suspend affected shipments pending audit and containment decision.",
            "Notify legal/compliance and customer where production or field impact exists.",
            "Brief Supply Chain Director and VP Operations within 24 hours.",
            "Initiate disqualification review if audit result does not support reinstatement.",
        ]

    closure_criteria = [
        "All required 8D/CAPA sections are complete and accepted by Supplier Quality Engineer.",
        "Root cause is validated and corrective action is implemented, not only planned.",
        "Affected documents and training records are updated at current revision.",
        "Effectiveness is verified through monitoring, inspection data, or audit evidence.",
        "No recurrence of the same defect mode during the verification period.",
    ]

    owner = str(claim_data.get("sqa_engineer") or "Supplier Quality Engineer")
    incident = str(claim_data.get("incident_number", "manual_triage"))
    supplier_name = str(supplier_data.get("name", claim_data.get("supplier_name", "Unknown supplier")))
    category = str(claim_data.get("category", "quality issue"))
    part = str(claim_data.get("part_number", "affected part"))

    return ScarCapaTriage(
        incident_number=incident,
        supplier_id=str(supplier_data.get("supplier_id", claim_data.get("supplier_id", "unknown"))),
        supplier_name=supplier_name,
        finding_grade=grade,
        scar_escalation_level=level,
        severity_score=round(min(score, 100), 1),
        issue_summary=f"{category} on {part}: {bad_parts:.0f} bad parts, {suspected:.0f} suspected.",
        triggers=triggers or ["No mandatory SCAR escalation trigger detected from current evidence."],
        immediate_containment=containment,
        required_evidence=evidence,
        deadlines=deadlines,
        escalation_actions=escalation_actions,
        closure_criteria=closure_criteria,
        owner=owner,
        source_documents=SOURCE_DOCUMENTS,
    )


def triage_manual_issue(
    supplier: Any,
    risk_row: Any,
    issue_description: str,
    bad_parts: int,
    suspected_parts: int,
    detected_at_customer: bool,
    recurrent: bool,
    supplier_claims: pd.DataFrame,
    supplier_kpis: pd.DataFrame,
    supplier_audits: pd.DataFrame,
    supplier_events: pd.DataFrame,
) -> ScarCapaTriage:
    supplier_data = _as_dict(supplier)
    claim = {
        "incident_number": f"MANUAL-{date.today().isoformat()}",
        "supplier_id": supplier_data.get("supplier_id", "unknown"),
        "supplier_name": supplier_data.get("name", "Unknown supplier"),
        "category": issue_description or "Manual quality issue",
        "part_number": "manual input",
        "number_of_bad_parts": bad_parts,
        "number_of_suspected_parts": suspected_parts,
        "how_detected": "Customer complaint" if detected_at_customer else "Internal detection",
        "who_detected": "Customer" if detected_at_customer else "SICC user",
        "where_detected": "Field" if detected_at_customer else "Incoming/assembly",
        "is_recurrent": recurrent,
        "is_recurring_incident": recurrent,
        "status": "Open",
        "sqa_engineer": "Supplier Quality Engineer",
    }
    return triage_claim(
        claim=claim,
        supplier=supplier,
        risk_row=risk_row,
        supplier_claims=supplier_claims,
        supplier_kpis=supplier_kpis,
        supplier_audits=supplier_audits,
        supplier_events=supplier_events,
    )
