from datetime import datetime, timezone

from scripts.agent_memory import normalize_severity


def parse_agent_ts(ts: str):
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def memory_age_hours(item: dict) -> float | None:
    parsed = parse_agent_ts(item.get("updated_at", ""))
    if not parsed:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - parsed).total_seconds() / 3600


def is_memory_fresh(memory: list[dict], max_age_hours: int = 24) -> bool:
    if not memory:
        return False
    ages = [memory_age_hours(item) for item in memory]
    ages = [age for age in ages if age is not None]
    return bool(ages) and max(ages) <= max_age_hours


def severity_rank(severity: str) -> int:
    return {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}.get(normalize_severity(severity), 0)


def severity_badge_text(severity: str) -> str:
    return normalize_severity(severity).upper()


def operator_status_label(status: str) -> str:
    labels = {
        "fresh": "fresh",
        "stale": "stale - refresh recommended",
        "failed": "failed latest run",
        "skipped": "skipped latest run",
        "success": "fresh",
        "no data": "no current output",
        "info": "no current output",
    }
    return labels.get(str(status).strip().lower(), str(status))


def build_run_log_markdown(supplier_id: str, agent_runs: list[dict]) -> str:
    lines = [
        f"# SICC Agent Run Log: {supplier_id}",
        "",
        f"- Exported: {datetime.now().isoformat(timespec='seconds')}",
    ]
    if not agent_runs:
        lines.append("- No agent runs recorded.")
        return "\n".join(lines)

    for run in agent_runs:
        lines.extend([
            "",
            f"## Run {run['run_id']}",
            f"- Status: {run['status']}",
            f"- Started: {run['started_at']}",
            f"- Finished: {run.get('finished_at') or ''}",
            f"- Summary: {run['summary']}",
            "",
            "### Steps",
        ])
        for step in run.get("steps", []):
            lines.append(
                f"- {step['agent_name']}: {operator_status_label(step['status'])} "
                f"({normalize_severity(step['severity'])}) - {step['error'] or step['summary']}"
            )
    return "\n".join(lines)


def build_evidence_pack_markdown(
    supplier,
    risk_row,
    sup_kpis,
    sup_claims,
    sup_audits,
    sup_events,
    sup_apqp,
    memory: list[dict],
    agent_runs: list[dict] | None = None,
) -> str:
    supplier_name = supplier.get("name", supplier.get("supplier_name", "Unknown supplier"))
    risk_label = risk_row.get("risk_label", "unknown") if hasattr(risk_row, "get") else "unknown"
    lines = [
        f"# SICC Supplier Evidence Pack: {supplier_name}",
        "",
        f"- Supplier ID: {supplier.get('supplier_id', 'unknown')}",
        f"- Product family: {supplier.get('product_family', 'unknown')}",
        f"- Country: {supplier.get('country', 'unknown')}",
        f"- Single source: {bool(supplier.get('single_source', False))}",
        f"- Risk tier: {str(risk_label).upper()}",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## KPI Snapshot",
    ]
    for key in ["avg_ppm_3m", "avg_otd_3m", "avg_audit_score_3m", "avg_scar_count_3m", "composite_risk_score"]:
        if hasattr(risk_row, "get") and key in risk_row:
            lines.append(f"- {key}: {risk_row.get(key)}")

    lines.extend(["", "## Claims"])
    if sup_claims.empty:
        lines.append("- No claims on record.")
    else:
        for _, row in sup_claims.sort_values("creation_date", ascending=False).head(10).iterrows():
            lines.append(f"- {row.get('incident_number')}: {row.get('category')} · {row.get('status')} · bad parts {row.get('number_of_bad_parts')}")

    lines.extend(["", "## Audits"])
    if sup_audits.empty:
        lines.append("- No audits on record.")
    else:
        for _, row in sup_audits.sort_values("audit_date", ascending=False).head(10).iterrows():
            lines.append(f"- {row.get('audit_id')}: {row.get('audit_type')} · score {row.get('audit_score')} · {row.get('highest_finding_type')} · {row.get('status')}")

    lines.extend(["", "## External Events"])
    if sup_events.empty:
        lines.append("- No external events on record.")
    else:
        for _, row in sup_events.sort_values("event_date", ascending=False).head(10).iterrows():
            lines.append(f"- {row.get('event_id')}: {row.get('event_type')} · {row.get('severity')} · {row.get('status')}")

    lines.extend(["", "## APQP Projects"])
    if sup_apqp.empty:
        lines.append("- No APQP projects on record.")
    else:
        for _, row in sup_apqp.sort_values("creation_date", ascending=False).head(10).iterrows():
            lines.append(f"- {row.get('project_id')}: {row.get('project_type')} · {row.get('status')} · completion {row.get('completion_pct')}% · delayed {row.get('is_delayed')}")

    lines.extend(["", "## Agent Memory"])
    if not memory:
        lines.append("- No agent memory records.")
    else:
        for item in memory:
            lines.extend([
                f"### {item['agent_name']} ({severity_badge_text(item['severity'])})",
                f"- Subject: {item['subject_id']}",
                f"- Updated: {item['updated_at']}",
                f"- Summary: {item['summary']}",
            ])
            payload = item.get("payload", {})
            for key in ["primary_risk_drivers", "signals", "triggers", "blockers", "risks", "exposure_drivers", "mandatory_actions", "recovery_actions"]:
                values = payload.get(key)
                if isinstance(values, list) and values:
                    lines.append(f"- {key.replace('_', ' ').title()}:")
                    for value in values[:6]:
                        if isinstance(value, dict):
                            lines.append(f"  - {value.get('action') or value.get('issue') or value}")
                        else:
                            lines.append(f"  - {value}")
            docs = payload.get("source_documents", [])
            if docs:
                lines.append(f"- Source documents: {', '.join(docs)}")
            lines.append("")

    lines.extend(["", "## Agent Run History"])
    if not agent_runs:
        lines.append("- No agent runs recorded.")
    else:
        for run in agent_runs:
            lines.extend([
                f"### Run {run['run_id'][:10]} ({run['status']})",
                f"- Started: {run['started_at']}",
                f"- Finished: {run.get('finished_at') or ''}",
                f"- Summary: {run['summary']}",
            ])
            for step in run.get("steps", []):
                lines.append(
                    f"  - {step['agent_name']}: {operator_status_label(step['status'])} "
                    f"({normalize_severity(step['severity'])}) - {step['error'] or step['summary']}"
                )

    return "\n".join(lines)
