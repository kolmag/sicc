"""
Audit planning agent.

Creates a targeted audit plan from supplier risk, KPI triggers, audit history,
open SCAR/claim burden, external events, and single-source status.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field


AuditUrgency = Literal["immediate", "high", "medium", "scheduled"]
AuditType = Literal["For-Cause Audit", "Follow-Up Audit", "Process Audit", "System Audit", "Remote Assessment"]


class AuditPlan(BaseModel):
    supplier_id: str
    supplier_name: str
    audit_type: AuditType
    urgency: AuditUrgency
    schedule_timeline: str
    audit_scope: list[str] = Field(default_factory=list)
    triggers: list[str] = Field(default_factory=list)
    checklist_focus: list[str] = Field(default_factory=list)
    evidence_to_request: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    owner: str
    source_documents: list[str] = Field(default_factory=list)


SOURCE_DOCUMENTS = [
    "for_cause_audit_trigger_criteria.md",
    "audit_finding_classification.md",
    "risk_tier_definitions.md",
    "scar_process_escalation.md",
    "single_source_risk_management.md",
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


def _scope_for_trigger(trigger_type: str) -> list[str]:
    scopes = {
        "quality": [
            "Affected manufacturing process",
            "Detection controls and reaction plans",
            "PFMEA and control plan coverage",
            "Operator training and process discipline",
        ],
        "delivery": [
            "Production planning and scheduling process",
            "Capacity versus forecast demand",
            "Sub-tier supply continuity",
            "Recovery plan governance",
        ],
        "scar": [
            "Corrective action system",
            "Root cause methodology",
            "SCAR response timeliness",
            "Closure verification and recurrence prevention",
        ],
        "external": [
            "Event-specific compliance controls",
            "Supplier response plan and CAPA linkage",
            "Business continuity and escalation process",
            "Customer/regulatory notification evidence",
        ],
        "audit": [
            "Prior finding closure evidence",
            "Effectiveness verification",
            "Systemic recurrence checks",
            "Follow-up audit readiness",
        ],
        "single_source": [
            "Business continuity plan",
            "Capacity and recovery time objective",
            "Tooling ownership and rescue clause",
            "Sub-tier risk review",
        ],
    }
    return scopes.get(trigger_type, [])


def plan_supplier_audit(
    supplier: Any,
    risk_row: Any,
    supplier_kpis: pd.DataFrame,
    supplier_claims: pd.DataFrame,
    supplier_audits: pd.DataFrame,
    supplier_events: pd.DataFrame,
) -> AuditPlan:
    supplier_data = _as_dict(supplier)
    risk_data = _as_dict(risk_row)
    sid = str(supplier_data.get("supplier_id", risk_data.get("supplier_id", "unknown")))
    name = str(supplier_data.get("name", "Unknown supplier"))
    risk = str(risk_data.get("risk_label", "unknown")).lower()
    single_source = _bool(supplier_data.get("single_source", risk_data.get("single_source", False)))

    triggers: list[str] = []
    trigger_types: set[str] = set()
    urgency_score = 0
    timeline = "Next scheduled audit cycle"
    audit_type: AuditType = "System Audit"

    ppm = _num(risk_data.get("avg_ppm_3m"))
    otd = _num(risk_data.get("avg_otd_3m"), 100.0)
    audit_score = _num(risk_data.get("avg_audit_score_3m"), 100.0)
    scar_count = _num(risk_data.get("avg_scar_count_3m"))

    if ppm > 500:
        triggers.append(f"PPM external exceeds RED trigger level ({ppm:.0f}); for-cause quality audit required within 30 days.")
        trigger_types.add("quality")
        urgency_score += 25
        timeline = "Schedule within 30 days"
        audit_type = "For-Cause Audit"
    if otd < 90:
        triggers.append(f"OTD is below 90% ({otd:.1f}); delivery failure audit required within 45 days if sustained.")
        trigger_types.add("delivery")
        urgency_score += 18
        if audit_type != "For-Cause Audit":
            audit_type = "For-Cause Audit"
            timeline = "Schedule within 45 days"
    if scar_count >= 4:
        triggers.append(f"Supplier has RED-level SCAR burden ({scar_count:.1f}); multiple open SCAR trigger.")
        trigger_types.add("scar")
        urgency_score += 22
        audit_type = "For-Cause Audit"
        timeline = "Schedule within 30 days"
    if audit_score < 60:
        triggers.append(f"Audit score is below 60 ({audit_score:.1f}); re-audit required within 90 days.")
        trigger_types.add("audit")
        urgency_score += 20
        if audit_type == "System Audit":
            audit_type = "Follow-Up Audit"
            timeline = "Schedule within 90 days"

    if risk == "red":
        triggers.append("Supplier is RED tier; for-cause audit is mandatory within 30 days.")
        urgency_score += 20
        audit_type = "For-Cause Audit"
        timeline = "Schedule within 30 days"
    if risk == "red" and single_source:
        triggers.append("Supplier is single-source RED; audit timeline accelerates to 14 days.")
        trigger_types.add("single_source")
        urgency_score += 25
        audit_type = "For-Cause Audit"
        timeline = "Schedule within 14 days"

    if supplier_claims is not None and not supplier_claims.empty and "status" in supplier_claims:
        open_claims = supplier_claims[supplier_claims["status"].astype(str).str.lower() != "closed"]
        if len(open_claims) >= 4:
            triggers.append(f"{len(open_claims)} open claims/SCARs; SCAR non-compliance audit trigger.")
            trigger_types.add("scar")
            urgency_score += 20
            audit_type = "For-Cause Audit"
            timeline = "Schedule within 30 days"
        field_claims = supplier_claims[
            supplier_claims["who_detected"].astype(str).str.contains("Customer", case=False, na=False)
            | supplier_claims["how_detected"].astype(str).str.contains("Field|Customer", case=False, na=False)
        ] if {"who_detected", "how_detected"}.issubset(supplier_claims.columns) else pd.DataFrame()
        if not field_claims.empty:
            triggers.append(f"{len(field_claims)} customer/field-detected claim(s); field-return/customer complaint trigger.")
            trigger_types.add("quality")
            urgency_score += 15
            audit_type = "For-Cause Audit"
            if "14 days" not in timeline:
                timeline = "Schedule within 14-21 days"

    if supplier_audits is not None and not supplier_audits.empty:
        recent = supplier_audits.sort_values("audit_date", ascending=False).head(3)
        critical_or_major = recent[
            recent["highest_finding_type"].isin(["Critical NCR", "Major NCR"])
        ] if "highest_finding_type" in recent else pd.DataFrame()
        if not critical_or_major.empty:
            triggers.append("Recent audit history includes Major/Critical NCR findings requiring follow-up verification.")
            trigger_types.add("audit")
            urgency_score += 15
            if audit_type == "System Audit":
                audit_type = "Follow-Up Audit"
                timeline = "Schedule per finding closure timeline"
        follow_up = recent[recent["follow_up_required"].astype(int) == 1] if "follow_up_required" in recent else pd.DataFrame()
        if not follow_up.empty:
            triggers.append("Prior audit record requires follow-up.")
            trigger_types.add("audit")
            urgency_score += 8

    if supplier_events is not None and not supplier_events.empty and {"status", "severity", "event_type"}.issubset(supplier_events.columns):
        open_events = supplier_events[
            supplier_events["status"].astype(str).str.lower().isin(["open", "under review", "escalated"])
            & supplier_events["severity"].isin(["Critical", "High"])
        ]
        if not open_events.empty:
            event_types = ", ".join(sorted(open_events["event_type"].dropna().astype(str).unique())[:3])
            triggers.append(f"{len(open_events)} high/critical external event(s) open ({event_types}).")
            trigger_types.add("external")
            urgency_score += 20
            audit_type = "For-Cause Audit"
            timeline = "Schedule within 14-30 days depending on event type"
            event_text = (
                open_events["event_type"].astype(str) + " " + open_events["description"].astype(str)
                if "description" in open_events.columns else open_events["event_type"].astype(str)
            )
            if any(event_text.str.contains("natural disaster|flood|earthquake|conflict|facility access", case=False, na=False)):
                audit_type = "Remote Assessment"
                timeline = "Remote assessment within 7 days; on-site as soon as accessible"

    if not triggers:
        if single_source:
            triggers.append("Single-source supplier requires annual on-site audit and continuity/capacity assessment.")
            trigger_types.add("single_source")
            audit_type = "Process Audit"
            timeline = "Annual single-source audit cycle"
            urgency_score += 8
        else:
            triggers.append("No for-cause trigger detected; keep supplier on planned audit schedule.")

    if urgency_score >= 70:
        urgency: AuditUrgency = "immediate"
    elif urgency_score >= 45:
        urgency = "high"
    elif urgency_score >= 20:
        urgency = "medium"
    else:
        urgency = "scheduled"

    scope: list[str] = []
    for trigger_type in sorted(trigger_types):
        for item in _scope_for_trigger(trigger_type):
            if item not in scope:
                scope.append(item)
    if not scope:
        scope = ["Quality management system", "Manufacturing process controls", "Corrective action status"]

    checklist = [
        "Opening meeting: confirm trigger, scope, affected products, and audit boundaries.",
        "Process observation for the specific operation/process linked to the trigger.",
        "Document review: PFMEA, control plan, work instructions, training, inspection and SCAR records.",
        "Operator/process engineer/quality manager interviews.",
        "Closing meeting with preliminary findings and factual correction window.",
    ]
    evidence = [
        "Last 12 months KPI trend: PPM, OTD, audit score, SCAR count.",
        "Open SCAR and CAPA list with due dates and closure evidence.",
        "Prior audit reports and follow-up closure records.",
        "PFMEA, control plan, work instructions, training records, and inspection logs for scoped processes.",
    ]
    if "external" in trigger_types:
        evidence.append("External-event response, CAPA linkage, regulatory/customer notification evidence.")
    if "single_source" in trigger_types:
        evidence.append("Business continuity plan, recovery time objective, buffer stock evidence, tooling ownership records.")

    outputs = [
        "Draft audit report within 5 business days.",
        "Final audit report within 10 business days.",
        "Findings classified as Observation, Minor NCR, Major NCR, or Critical NCR.",
        "Supplier corrective action response due per finding grade.",
    ]
    if urgency == "immediate":
        outputs.append("Director approval and escalation record attached to audit file.")

    owner = "Lead Auditor"
    if urgency in {"immediate", "high"}:
        owner = "Supplier Quality Lead Auditor"
    if risk == "red" and single_source:
        owner = "Supplier Quality Lead Auditor + Supply Chain Director"

    return AuditPlan(
        supplier_id=sid,
        supplier_name=name,
        audit_type=audit_type,
        urgency=urgency,
        schedule_timeline=timeline,
        audit_scope=scope,
        triggers=triggers,
        checklist_focus=checklist,
        evidence_to_request=evidence,
        expected_outputs=outputs,
        owner=owner,
        source_documents=SOURCE_DOCUMENTS,
    )


def build_audit_plan_watchlist(
    suppliers: pd.DataFrame,
    risk_scores: pd.DataFrame,
    kpis: pd.DataFrame,
    claims: pd.DataFrame,
    audits: pd.DataFrame,
    events: pd.DataFrame,
    supplier_ids: set[str] | None = None,
    top_n: int = 50,
) -> list[AuditPlan]:
    ids = supplier_ids or set(suppliers["supplier_id"])
    risk_lookup = risk_scores.set_index("supplier_id").to_dict("index") if not risk_scores.empty else {}
    plans: list[AuditPlan] = []
    for _, supplier in suppliers[suppliers["supplier_id"].isin(ids)].iterrows():
        sid = supplier["supplier_id"]
        plan = plan_supplier_audit(
            supplier=supplier,
            risk_row=risk_lookup.get(sid, {}),
            supplier_kpis=kpis[kpis["supplier_id"] == sid] if not kpis.empty else pd.DataFrame(),
            supplier_claims=claims[claims["supplier_id"] == sid] if not claims.empty else pd.DataFrame(),
            supplier_audits=audits[audits["supplier_id"] == sid] if not audits.empty else pd.DataFrame(),
            supplier_events=events[events["supplier_id"] == sid] if not events.empty else pd.DataFrame(),
        )
        if plan.urgency != "scheduled" or plan.audit_type != "System Audit":
            plans.append(plan)

    order = {"immediate": 0, "high": 1, "medium": 2, "scheduled": 3}
    plans.sort(key=lambda item: (order[item.urgency], item.schedule_timeline, item.supplier_name))
    return plans[:top_n]
