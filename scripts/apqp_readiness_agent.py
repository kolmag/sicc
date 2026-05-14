"""
APQP launch readiness agent.

Turns APQP gate status, PPAP readiness, supplier risk, open claims, and external
events into an operational launch decision: GO, CONDITIONAL GO, or HOLD.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field


LaunchDecision = Literal["GO", "CONDITIONAL_GO", "HOLD"]

GATES = [
    ("supplier_selection", "Supplier Selection"),
    ("supplier_nomination", "Supplier Nomination"),
    ("design_validation_of_process", "Design Validation of Process"),
    ("process_validation", "Process Validation"),
    ("initial_sample_validation", "Initial Sample Validation"),
    ("start_of_production", "Start of Production"),
    ("pqa_management", "PQA Management"),
    ("yearly_is_submission", "Yearly IS Submission"),
    ("ppap_update", "PPAP Update"),
]


class GateFinding(BaseModel):
    gate: str
    status: str
    issue: str
    required_action: str


class ApqpReadinessDecision(BaseModel):
    project_id: str
    supplier_id: str
    supplier_name: str
    launch_decision: LaunchDecision
    readiness_score: float
    decision_summary: str
    blockers: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    gate_findings: list[GateFinding] = Field(default_factory=list)
    recovery_actions: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    owner: str
    source_documents: list[str] = Field(default_factory=list)


SOURCE_DOCUMENTS = [
    "apqp_phase_gate_guide.md",
    "ppap_submission_checklist.md",
    "ppap_level_requirements.md",
    "risk_tier_definitions.md",
    "scar_process_escalation.md",
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


def _past_due(value: Any) -> bool:
    try:
        parsed = pd.to_datetime(value)
        if pd.isna(parsed):
            return False
        return parsed.date() < date.today()
    except Exception:
        return False


def _gate_status(project: dict[str, Any], gate_key: str) -> str:
    return str(project.get(f"{gate_key}_status", "Unknown") or "Unknown")


def _gate_validated(project: dict[str, Any], gate_key: str) -> bool:
    value = str(project.get(f"{gate_key}_validated", "") or "").strip()
    return bool(value) or _gate_status(project, gate_key).lower() == "validated"


def assess_apqp_launch_readiness(
    project: Any,
    supplier: Any,
    risk_row: Any,
    supplier_claims: pd.DataFrame,
    supplier_events: pd.DataFrame,
) -> ApqpReadinessDecision:
    project_data = _as_dict(project)
    supplier_data = _as_dict(supplier)
    risk_data = _as_dict(risk_row)

    score = 100.0
    blockers: list[str] = []
    risks: list[str] = []
    findings: list[GateFinding] = []

    project_status = str(project_data.get("status", "Unknown"))
    completion = _num(project_data.get("completion_pct"))
    if completion < 50:
        score -= 25
        blockers.append(f"Programme completion is only {completion:.0f}%.")
    elif completion < 80:
        score -= 12
        risks.append(f"Programme completion is {completion:.0f}%, below robust launch-readiness level.")

    if _bool(project_data.get("is_delayed")) or project_status.lower() == "delayed":
        score -= 20
        blockers.append("APQP programme is delayed.")
    if project_status.lower() == "on hold":
        score -= 35
        blockers.append("APQP programme is on hold.")
    if project_status.lower() == "cancelled":
        score = 0
        blockers.append("APQP programme is cancelled.")

    required_pre_sop = [
        "supplier_selection",
        "supplier_nomination",
        "design_validation_of_process",
        "process_validation",
        "initial_sample_validation",
        "start_of_production",
    ]
    for gate_key, gate_name in GATES:
        status = _gate_status(project_data, gate_key)
        expected_validation = project_data.get(f"{gate_key}_expected_validation")
        is_required = gate_key in required_pre_sop
        validated = _gate_validated(project_data, gate_key)
        if is_required and not validated:
            if status.lower() in {"not started", "unknown", ""}:
                penalty = 14
                issue = f"{gate_name} is not started or not evidenced."
                blockers.append(issue)
            elif status.lower() in {"in progress", "submitted"}:
                penalty = 8
                issue = f"{gate_name} is {status.lower()} and not validated."
                risks.append(issue)
            else:
                penalty = 10
                issue = f"{gate_name} is not validated."
                risks.append(issue)
            score -= penalty
            findings.append(GateFinding(
                gate=gate_name,
                status=status,
                issue=issue,
                required_action=f"Complete and validate {gate_name} deliverables before launch decision.",
            ))
        elif _past_due(expected_validation) and not validated:
            score -= 6
            issue = f"{gate_name} expected validation date is overdue."
            risks.append(issue)
            findings.append(GateFinding(
                gate=gate_name,
                status=status,
                issue=issue,
                required_action=f"Recover overdue {gate_name} validation and update APQP timing plan.",
            ))

    risk_label = str(risk_data.get("risk_label", "unknown")).lower()
    if risk_label == "red":
        score -= 25
        blockers.append("Supplier is RED risk; launch cannot proceed without executive risk acceptance and containment.")
    elif risk_label == "amber":
        score -= 10
        risks.append("Supplier is AMBER risk and requires enhanced launch monitoring.")

    if _bool(supplier_data.get("single_source", risk_data.get("single_source", False))) and risk_label in {"red", "amber"}:
        score -= 10
        risks.append("Single-source exposure increases launch continuity risk.")

    open_claims = pd.DataFrame()
    if supplier_claims is not None and not supplier_claims.empty and "status" in supplier_claims:
        open_claims = supplier_claims[supplier_claims["status"].astype(str).str.lower() != "closed"]
        if len(open_claims) >= 5:
            score -= 15
            blockers.append(f"{len(open_claims)} open claims/SCARs exist for the supplier.")
        elif len(open_claims) >= 2:
            score -= 8
            risks.append(f"{len(open_claims)} open claims/SCARs exist for the supplier.")

    open_high_events = pd.DataFrame()
    if supplier_events is not None and not supplier_events.empty and {"status", "severity"}.issubset(supplier_events.columns):
        open_high_events = supplier_events[
            supplier_events["status"].astype(str).str.lower().isin(["open", "under review", "escalated"])
            & supplier_events["severity"].isin(["Critical", "High"])
        ]
        if not open_high_events.empty:
            score -= 15
            blockers.append(f"{len(open_high_events)} high/critical external event(s) are open.")

    score = round(max(0, min(score, 100)), 1)
    if blockers or score < 55:
        decision: LaunchDecision = "HOLD"
    elif risks or score < 85:
        decision = "CONDITIONAL_GO"
    else:
        decision = "GO"

    if decision == "GO":
        summary = "Launch readiness is acceptable; continue standard APQP governance and PQA monitoring."
    elif decision == "CONDITIONAL_GO":
        summary = "Launch may proceed only with documented risk acceptance, enhanced monitoring, and closure of listed risks."
    else:
        summary = "Launch should be held until blockers are closed or formally accepted through escalation governance."

    recovery_actions = [
        "Update APQP timing plan with owner, due date, and recovery status for every open gate.",
        "Confirm PPAP submission status, PSW approval, and customer approval evidence.",
        "Run launch-readiness review with Supplier Quality Engineer, programme manager, supplier quality manager, and procurement.",
        "Apply GP-12 or enhanced incoming inspection for first 90 days where launch proceeds conditionally.",
    ]
    if decision == "HOLD":
        recovery_actions.insert(0, "Freeze launch readiness decision and escalate blockers to programme steering review.")
    if risk_label == "red":
        recovery_actions.append("Require Supply Chain Director approval before any launch release.")

    required_evidence = [
        "Validated APQP gate records through Start of Production.",
        "Complete PPAP package per agreed level, including PSW/customer approval.",
        "Capacity confirmation against programme volume.",
        "Updated control plan, PFMEA, MSA, dimensional and capability evidence.",
        "Open issue log with containment, owners, due dates, and risk acceptance where applicable.",
    ]
    if not open_claims.empty:
        required_evidence.append("SCAR/CAPA closure or containment evidence for open claims affecting launch risk.")
    if not open_high_events.empty:
        required_evidence.append("External-event mitigation record and CAPA linkage where required.")

    return ApqpReadinessDecision(
        project_id=str(project_data.get("project_id", "unknown")),
        supplier_id=str(project_data.get("supplier_id", supplier_data.get("supplier_id", "unknown"))),
        supplier_name=str(project_data.get("supplier_name", supplier_data.get("name", "Unknown supplier"))),
        launch_decision=decision,
        readiness_score=score,
        decision_summary=summary,
        blockers=blockers,
        risks=risks,
        gate_findings=findings,
        recovery_actions=recovery_actions,
        required_evidence=required_evidence,
        owner=str(project_data.get("project_manager", "APQP Program Manager")),
        source_documents=SOURCE_DOCUMENTS,
    )
