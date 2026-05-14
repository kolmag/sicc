"""
Agentic supplier intake and development brief generator.

The agent first builds a structured supplier dossier from portfolio data,
then maps observed gaps to SICC policy guidance, and optionally asks an LLM
to turn that evidence into a governed development brief. A deterministic
brief is returned when model keys are unavailable or the LLM response fails
validation, so the Streamlit app remains usable in demos and offline review.
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from typing import Any, Literal

import pandas as pd
from litellm import completion
from pydantic import BaseModel, Field, ValidationError


Priority = Literal["critical", "high", "medium", "low"]
RiskLevel = Literal["red", "amber", "green", "unknown"]
Confidence = Literal["high", "medium", "low"]


class DevelopmentAction(BaseModel):
    priority: Priority
    action: str
    owner: str
    due_date: str
    evidence_required: list[str] = Field(default_factory=list)
    rationale: str


class SupplierDevelopmentBrief(BaseModel):
    supplier_id: str
    supplier_name: str
    risk_level: RiskLevel
    recommended_pathway: str
    situation_summary: str
    primary_risk_drivers: list[str] = Field(default_factory=list)
    identified_gaps: list[str] = Field(default_factory=list)
    development_actions: list[DevelopmentAction] = Field(default_factory=list)
    escalation_triggers: list[str] = Field(default_factory=list)
    exit_criteria: list[str] = Field(default_factory=list)
    source_documents: list[str] = Field(default_factory=list)
    confidence: Confidence = "medium"
    generation_mode: Literal["agentic_llm", "deterministic_fallback"] = "deterministic_fallback"


POLICY_GUIDANCE = {
    "red_risk": {
        "documents": ["risk_tier_definitions.md", "supplier_development_methodology.md"],
        "gap": "Supplier meets RED risk conditions and requires a formal recovery path.",
    },
    "single_source": {
        "documents": ["single_source_risk_management.md", "risk_tier_definitions.md"],
        "gap": "Single-source exposure limits immediate containment options.",
    },
    "quality": {
        "documents": [
            "supplier_kpi_definitions.md",
            "corrective_action_closure_requirements.md",
            "scar_process_escalation.md",
        ],
        "gap": "Quality performance is outside SICC control thresholds.",
    },
    "delivery": {
        "documents": ["supplier_kpi_definitions.md", "supplier_development_methodology.md"],
        "gap": "Delivery performance is below expected continuity threshold.",
    },
    "audit": {
        "documents": ["for_cause_audit_trigger_criteria.md", "supplier_development_methodology.md"],
        "gap": "Audit performance indicates process-system weakness.",
    },
    "external": {
        "documents": ["external_risk_event_response.md", "single_source_risk_management.md"],
        "gap": "Open external risk events require response governance.",
    },
    "apqp": {
        "documents": ["apqp_phase_gate_guide.md", "ppap_submission_checklist.md"],
        "gap": "APQP/NPI status indicates launch-readiness risk.",
    },
}


def _records(df: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    clean = df.copy().head(limit)
    return json.loads(clean.to_json(orient="records", date_format="iso"))


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
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _due(days: int) -> str:
    return (date.today() + timedelta(days=days)).isoformat()


def build_supplier_intake_context(
    supplier: Any,
    risk_row: Any,
    kpis: pd.DataFrame,
    claims: pd.DataFrame,
    audits: pd.DataFrame,
    events: pd.DataFrame,
    apqp: pd.DataFrame,
) -> dict[str, Any]:
    supplier_data = _as_dict(supplier)
    risk_data = _as_dict(risk_row)

    latest_kpi = {}
    if kpis is not None and not kpis.empty:
        latest_kpi = kpis.sort_values("year_month").tail(1).iloc[0].to_dict()

    open_claims = claims[claims["status"].astype(str).str.lower() != "closed"] if claims is not None and not claims.empty else pd.DataFrame()
    open_events = events[events["status"].astype(str).str.lower().isin(["open", "under review", "escalated"])] if events is not None and not events.empty else pd.DataFrame()
    delayed_apqp = apqp[apqp["is_delayed"].astype(int) == 1] if apqp is not None and not apqp.empty and "is_delayed" in apqp else pd.DataFrame()

    drivers: list[str] = []
    policy_keys: set[str] = set()
    risk_level = str(risk_data.get("risk_label", "unknown")).lower()
    ppm = _num(risk_data.get("avg_ppm_3m", latest_kpi.get("ppm_external")))
    otd = _num(risk_data.get("avg_otd_3m", latest_kpi.get("otd_pct")), 100.0)
    audit_score = _num(risk_data.get("avg_audit_score_3m", latest_kpi.get("audit_score")), 100.0)
    scar_count = _num(risk_data.get("avg_scar_count_3m", latest_kpi.get("scar_count")))
    single_source = _bool(supplier_data.get("single_source", risk_data.get("single_source", False)))

    if risk_level == "red":
        drivers.append("Overall supplier state is RED under SICC risk tiering.")
        policy_keys.add("red_risk")
    if single_source:
        drivers.append("Supplier is single-source, increasing continuity and containment exposure.")
        policy_keys.add("single_source")
    if ppm >= 500:
        drivers.append(f"External PPM is at RED level ({ppm:.0f} ppm, 3-month average).")
        policy_keys.add("quality")
    elif ppm >= 200:
        drivers.append(f"External PPM is at AMBER level ({ppm:.0f} ppm, 3-month average).")
        policy_keys.add("quality")
    if otd < 90:
        drivers.append(f"On-time delivery is at RED level ({otd:.1f}%).")
        policy_keys.add("delivery")
    elif otd < 95:
        drivers.append(f"On-time delivery is below target ({otd:.1f}%).")
        policy_keys.add("delivery")
    if audit_score < 60:
        drivers.append(f"Audit score is at RED level ({audit_score:.1f}).")
        policy_keys.add("audit")
    elif audit_score < 75:
        drivers.append(f"Audit score is below target ({audit_score:.1f}).")
        policy_keys.add("audit")
    if scar_count >= 2:
        drivers.append(f"Recent SCAR burden is elevated ({scar_count:.1f} average count).")
        policy_keys.add("quality")
    if not open_claims.empty:
        drivers.append(f"{len(open_claims)} claim(s) remain open or unresolved.")
        policy_keys.add("quality")
    if not open_events.empty:
        severe = open_events[open_events["severity"].isin(["Critical", "High"])] if "severity" in open_events else open_events
        drivers.append(f"{len(open_events)} open external event(s), including {len(severe)} high/critical.")
        policy_keys.add("external")
    if not delayed_apqp.empty:
        drivers.append(f"{len(delayed_apqp)} APQP/NPI project(s) are delayed.")
        policy_keys.add("apqp")

    if not drivers:
        drivers.append("No material threshold breach found in current supplier evidence.")

    source_documents = sorted({doc for key in policy_keys for doc in POLICY_GUIDANCE[key]["documents"]})
    gaps = [POLICY_GUIDANCE[key]["gap"] for key in sorted(policy_keys)]

    return {
        "supplier": supplier_data,
        "risk": risk_data,
        "latest_kpi": latest_kpi,
        "claims": _records(claims.sort_values("creation_date", ascending=False) if claims is not None and not claims.empty and "creation_date" in claims else claims),
        "audits": _records(audits.sort_values("audit_date", ascending=False) if audits is not None and not audits.empty and "audit_date" in audits else audits),
        "events": _records(events.sort_values("event_date", ascending=False) if events is not None and not events.empty and "event_date" in events else events),
        "apqp": _records(apqp.sort_values("creation_date", ascending=False) if apqp is not None and not apqp.empty and "creation_date" in apqp else apqp),
        "risk_drivers": drivers,
        "identified_gaps": gaps,
        "source_documents": source_documents or ["supplier_development_methodology.md", "risk_tier_definitions.md"],
    }


def _deterministic_brief(context: dict[str, Any]) -> SupplierDevelopmentBrief:
    supplier = context["supplier"]
    risk = context["risk"]
    risk_level = str(risk.get("risk_label", "unknown")).lower()
    if risk_level not in {"red", "amber", "green"}:
        risk_level = "unknown"

    single_source = _bool(supplier.get("single_source", risk.get("single_source", False)))
    drivers = context["risk_drivers"]
    gaps = context["identified_gaps"]
    source_docs = context["source_documents"]

    if risk_level == "red" and single_source:
        pathway = "Single-source RED recovery protocol"
        confidence: Confidence = "high"
    elif risk_level == "red":
        pathway = "RED supplier development recovery plan"
        confidence = "high"
    elif risk_level == "amber":
        pathway = "AMBER supplier development plan"
        confidence = "medium"
    elif risk_level == "green":
        pathway = "Preventive monitoring and sustainment"
        confidence = "medium"
    else:
        pathway = "Evidence completion before supplier development decision"
        confidence = "low"

    actions: list[DevelopmentAction] = []
    joined = " ".join(drivers).lower()
    if "ppm" in joined or "scar" in joined or "claim" in joined:
        actions.append(DevelopmentAction(
            priority="critical" if risk_level == "red" else "high",
            action="Open a supplier quality recovery track covering defect containment, SCAR aging, and verified corrective action closure.",
            owner="Supplier Quality Engineer",
            due_date=_due(14),
            evidence_required=["8D/CAPA plan", "containment records", "SCAR closure evidence", "updated defect Pareto"],
            rationale="Quality indicators exceed SICC thresholds or show unresolved claims/SCAR exposure.",
        ))
    if "delivery" in joined or "on-time" in joined:
        actions.append(DevelopmentAction(
            priority="high",
            action="Run delivery recovery review covering capacity, schedule adherence, and shipment escalation.",
            owner="Supply Chain Manager",
            due_date=_due(21),
            evidence_required=["capacity plan", "recovery schedule", "expedite log", "weekly OTD tracker"],
            rationale="Delivery performance is below SICC continuity expectations.",
        ))
    if "audit" in joined:
        actions.append(DevelopmentAction(
            priority="high",
            action="Trigger for-cause audit or focused process assessment against the failed control areas.",
            owner="Lead Auditor",
            due_date=_due(30),
            evidence_required=["audit agenda", "finding register", "corrective action plan", "effectiveness check"],
            rationale="Audit score indicates process-system weakness that needs independent verification.",
        ))
    if "single-source" in joined:
        actions.append(DevelopmentAction(
            priority="critical" if risk_level == "red" else "high",
            action="Create continuity mitigation plan for single-source exposure, including buffer stock and alternate-source feasibility.",
            owner="Category Manager",
            due_date=_due(21),
            evidence_required=["business continuity plan", "buffer stock decision", "alternate-source feasibility note"],
            rationale="Single-source status amplifies supplier disruption impact.",
        ))
    if "external event" in joined:
        actions.append(DevelopmentAction(
            priority="high",
            action="Link open external risk events to CAPA or continuity actions and confirm owner accountability.",
            owner="Risk & Compliance Manager",
            due_date=_due(10),
            evidence_required=["event assessment", "CAPA linkage", "supplier response", "risk acceptance or mitigation record"],
            rationale="Open high-severity external events require governed response.",
        ))
    if "apqp" in joined or "delayed" in joined:
        actions.append(DevelopmentAction(
            priority="high",
            action="Hold APQP gate recovery review and freeze launch readiness assumptions until delayed deliverables are recovered.",
            owner="APQP Program Manager",
            due_date=_due(14),
            evidence_required=["updated APQP timing plan", "open deliverable log", "PPAP readiness evidence", "launch risk decision"],
            rationale="Delayed APQP/NPI work creates launch-readiness risk.",
        ))

    if not actions:
        actions.append(DevelopmentAction(
            priority="medium",
            action="Keep supplier on preventive monitoring with monthly KPI review and documented exit criteria.",
            owner="Supplier Quality Engineer",
            due_date=_due(30),
            evidence_required=["monthly KPI snapshot", "risk review note"],
            rationale="Current evidence does not require a formal recovery track.",
        ))

    return SupplierDevelopmentBrief(
        supplier_id=str(supplier.get("supplier_id", "unknown")),
        supplier_name=str(supplier.get("name", supplier.get("supplier_name", "Unknown supplier"))),
        risk_level=risk_level,  # type: ignore[arg-type]
        recommended_pathway=pathway,
        situation_summary=(
            f"{supplier.get('name', 'Supplier')} is classified as {risk_level.upper()} with "
            f"{len(drivers)} active risk driver(s). The proposed pathway is {pathway}."
        ),
        primary_risk_drivers=drivers,
        identified_gaps=gaps or ["No formal development gap detected from current thresholds."],
        development_actions=actions,
        escalation_triggers=[
            "New critical external event or customer disruption is reported.",
            "Any critical/high action misses its due date without approved risk acceptance.",
            "PPM, OTD, audit score, or SCAR trend deteriorates for two consecutive review cycles.",
        ],
        exit_criteria=[
            "All critical/high actions closed with evidence accepted by SICC owner.",
            "Supplier returns to AMBER or GREEN risk tier for two consecutive review cycles.",
            "No open high-severity external events remain without CAPA or continuity decision.",
        ],
        source_documents=source_docs,
        confidence=confidence,
    )


def _llm_brief(context: dict[str, Any], fallback: SupplierDevelopmentBrief) -> SupplierDevelopmentBrief:
    if not os.getenv("GROQ_API_KEY"):
        return fallback

    schema_hint = SupplierDevelopmentBrief.model_json_schema()
    prompt = f"""
You are the SICC supplier intake agent. Create a supplier development brief.

Use only the supplied supplier dossier and SICC source-document list. Do not invent facts.
Return JSON only, matching this schema:
{json.dumps(schema_hint, indent=2)}

Supplier dossier:
{json.dumps(context, indent=2, default=str)}

Deterministic draft to refine:
{fallback.model_dump_json(indent=2)}
"""
    try:
        response = completion(
            model=os.getenv("SICC_INTAKE_AGENT_MODEL", "groq/openai/gpt-oss-120b"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        brief = SupplierDevelopmentBrief.model_validate(parsed)
        brief.generation_mode = "agentic_llm"
        return brief
    except (ValidationError, json.JSONDecodeError, Exception):
        return fallback


def generate_supplier_development_brief(
    supplier: Any,
    risk_row: Any,
    kpis: pd.DataFrame,
    claims: pd.DataFrame,
    audits: pd.DataFrame,
    events: pd.DataFrame,
    apqp: pd.DataFrame,
    use_llm: bool = True,
) -> SupplierDevelopmentBrief:
    context = build_supplier_intake_context(supplier, risk_row, kpis, claims, audits, events, apqp)
    fallback = _deterministic_brief(context)
    if use_llm:
        return _llm_brief(context, fallback)
    return fallback


def brief_to_markdown(brief: SupplierDevelopmentBrief) -> str:
    lines = [
        f"# Supplier Development Brief: {brief.supplier_name}",
        "",
        f"- Supplier ID: {brief.supplier_id}",
        f"- Risk level: {brief.risk_level.upper()}",
        f"- Pathway: {brief.recommended_pathway}",
        f"- Confidence: {brief.confidence}",
        f"- Generation mode: {brief.generation_mode}",
        "",
        "## Situation Summary",
        brief.situation_summary,
        "",
        "## Primary Risk Drivers",
        *[f"- {driver}" for driver in brief.primary_risk_drivers],
        "",
        "## Identified Gaps",
        *[f"- {gap}" for gap in brief.identified_gaps],
        "",
        "## Development Actions",
    ]
    for idx, action in enumerate(brief.development_actions, start=1):
        lines.extend([
            f"{idx}. [{action.priority.upper()}] {action.action}",
            f"   - Owner: {action.owner}",
            f"   - Due: {action.due_date}",
            f"   - Evidence: {', '.join(action.evidence_required)}",
            f"   - Rationale: {action.rationale}",
        ])
    lines.extend([
        "",
        "## Escalation Triggers",
        *[f"- {trigger}" for trigger in brief.escalation_triggers],
        "",
        "## Exit Criteria",
        *[f"- {criterion}" for criterion in brief.exit_criteria],
        "",
        "## SICC Source Documents",
        *[f"- {doc}" for doc in brief.source_documents],
    ])
    return "\n".join(lines)
