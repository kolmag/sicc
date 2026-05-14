"""
Shared agent memory for SICC workflow agents.

The memory table stores compact, structured outputs from independent agents so
the app can show cross-agent context without passing large payloads through the
LLM context window or recomputing every page.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SEVERITY_ALIASES = {
    "critical": "critical",
    "high": "high",
    "medium": "medium",
    "low": "low",
    "watch": "low",
    "monitor": "low",
    "info": "info",
}

MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_memory (
    memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    supplier_id TEXT NOT NULL,
    subject_id TEXT NOT NULL DEFAULT '',
    severity TEXT NOT NULL DEFAULT 'info',
    summary TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(agent_name, supplier_id, subject_id)
);
"""

RUN_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id TEXT PRIMARY KEY,
    supplier_id TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    summary TEXT NOT NULL DEFAULT ''
);
"""

RUN_STEP_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_run_steps (
    step_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    supplier_id TEXT NOT NULL,
    status TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'info',
    summary TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES agent_runs(run_id)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_severity(severity: str | None) -> str:
    """Map agent-specific severity words to the shared memory taxonomy."""
    if severity is None:
        return "info"
    return SEVERITY_ALIASES.get(str(severity).strip().lower(), "info")


def _payload_model_for(agent_name: str, subject_id: str):
    if agent_name == "Supplier Intake Agent" and subject_id == "development_brief":
        from scripts.supplier_intake_agent import SupplierDevelopmentBrief
        return SupplierDevelopmentBrief
    if agent_name == "Early Warning Agent" and subject_id == "trend_alert":
        from scripts.supplier_alert_agent import SupplierTrendAlert
        return SupplierTrendAlert
    if agent_name == "Continuity Agent" and subject_id == "single_source_plan":
        from scripts.continuity_agent import ContinuityPlan
        return ContinuityPlan
    if agent_name == "Audit Planning Agent" and subject_id == "audit_plan":
        from scripts.audit_planning_agent import AuditPlan
        return AuditPlan
    if agent_name == "APQP Readiness Agent" and subject_id.startswith("apqp_"):
        from scripts.apqp_readiness_agent import ApqpReadinessDecision
        return ApqpReadinessDecision
    if agent_name == "SCAR/CAPA Triage Agent":
        from scripts.scar_capa_agent import ScarCapaTriage
        return ScarCapaTriage
    return None


def validate_agent_payload(
    agent_name: str,
    subject_id: str,
    supplier_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Validate and canonicalize known agent memory payloads before storage."""
    if not isinstance(payload, dict):
        raise ValueError(f"{agent_name} payload must be a dict")

    model = _payload_model_for(agent_name, subject_id)
    if model is None:
        return payload

    try:
        validated = model.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"{agent_name} payload contract failed: {exc}") from exc

    payload_supplier_id = getattr(validated, "supplier_id", None)
    if payload_supplier_id is not None and str(payload_supplier_id) != str(supplier_id):
        raise ValueError(
            f"{agent_name} payload supplier_id {payload_supplier_id!r} does not match memory supplier_id {supplier_id!r}"
        )

    if agent_name == "APQP Readiness Agent":
        project_id = getattr(validated, "project_id", "")
        expected_subject = f"apqp_{project_id}"
        if project_id and subject_id != expected_subject:
            raise ValueError(
                f"APQP Readiness Agent subject_id {subject_id!r} must match project_id as {expected_subject!r}"
            )
    if agent_name == "SCAR/CAPA Triage Agent":
        incident_number = getattr(validated, "incident_number", "")
        if incident_number and subject_id != str(incident_number):
            raise ValueError(
                f"SCAR/CAPA Triage Agent subject_id {subject_id!r} must match incident_number {incident_number!r}"
            )

    return validated.model_dump()


def init_agent_memory(db_path: str | Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(MEMORY_SCHEMA)
        conn.execute(RUN_SCHEMA)
        conn.execute(RUN_STEP_SCHEMA)
        conn.execute(
            """
            UPDATE agent_memory
            SET severity = CASE lower(severity)
                WHEN 'critical' THEN 'critical'
                WHEN 'high' THEN 'high'
                WHEN 'medium' THEN 'medium'
                WHEN 'low' THEN 'low'
                WHEN 'watch' THEN 'low'
                WHEN 'monitor' THEN 'low'
                ELSE 'info'
            END
            """
        )
        conn.commit()


def start_agent_run(db_path: str | Path, supplier_id: str) -> str:
    init_agent_memory(db_path)
    run_id = uuid.uuid4().hex
    ts = _now()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO agent_runs (run_id, supplier_id, status, started_at, summary)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, supplier_id, "running", ts, "Agent sync running"),
        )
        conn.commit()
    return run_id


def record_agent_run_step(
    db_path: str | Path,
    run_id: str,
    agent_name: str,
    supplier_id: str,
    status: str,
    summary: str = "",
    severity: str = "info",
    error: str = "",
    started_at: str | None = None,
) -> None:
    init_agent_memory(db_path)
    finished_at = _now()
    started_at = started_at or finished_at
    status = str(status).strip().lower()
    if status not in {"success", "skipped", "failed"}:
        status = "failed"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO agent_run_steps (
                run_id, agent_name, supplier_id, status, severity, summary,
                error, started_at, finished_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                agent_name,
                supplier_id,
                status,
                normalize_severity(severity),
                summary,
                error,
                started_at,
                finished_at,
            ),
        )
        conn.commit()


def finish_agent_run(db_path: str | Path, run_id: str) -> None:
    init_agent_memory(db_path)
    finished_at = _now()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success_count,
                SUM(CASE WHEN status = 'skipped' THEN 1 ELSE 0 END) AS skipped_count,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_count,
                COUNT(*) AS total_count
            FROM agent_run_steps
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        success_count, skipped_count, failed_count, total_count = [int(value or 0) for value in row]
        status = "failed" if failed_count else "completed"
        if total_count == 0:
            status = "failed"
        summary = f"{success_count} succeeded, {skipped_count} skipped, {failed_count} failed"
        conn.execute(
            """
            UPDATE agent_runs
            SET status = ?, finished_at = ?, summary = ?
            WHERE run_id = ?
            """,
            (status, finished_at, summary, run_id),
        )
        conn.commit()


def get_supplier_agent_runs(db_path: str | Path, supplier_id: str, limit: int = 10) -> list[dict[str, Any]]:
    init_agent_memory(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        runs = conn.execute(
            """
            SELECT run_id, supplier_id, status, started_at, finished_at, summary
            FROM agent_runs
            WHERE supplier_id = ?
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (supplier_id, limit),
        ).fetchall()
        records = []
        for run in runs:
            run_item = dict(run)
            steps = conn.execute(
                """
                SELECT agent_name, supplier_id, status, severity, summary,
                       error, started_at, finished_at
                FROM agent_run_steps
                WHERE run_id = ?
                ORDER BY step_id ASC
                """,
                (run_item["run_id"],),
            ).fetchall()
            run_item["steps"] = [dict(step) for step in steps]
            records.append(run_item)
    return records


def remember_agent_output(
    db_path: str | Path,
    agent_name: str,
    supplier_id: str,
    summary: str,
    payload: dict[str, Any],
    severity: str = "info",
    subject_id: str = "",
    validate_payload: bool = True,
) -> None:
    init_agent_memory(db_path)
    ts = _now()
    severity = normalize_severity(severity)
    if validate_payload:
        payload = validate_agent_payload(agent_name, subject_id, supplier_id, payload)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO agent_memory (
                agent_name, supplier_id, subject_id, severity, summary,
                payload_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_name, supplier_id, subject_id)
            DO UPDATE SET
                severity=excluded.severity,
                summary=excluded.summary,
                payload_json=excluded.payload_json,
                updated_at=excluded.updated_at
            """,
            (
                agent_name,
                supplier_id,
                subject_id,
                severity,
                summary,
                json.dumps(payload, default=str),
                ts,
                ts,
            ),
        )
        conn.commit()


def get_supplier_memory(db_path: str | Path, supplier_id: str) -> list[dict[str, Any]]:
    init_agent_memory(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT agent_name, supplier_id, subject_id, severity, summary,
                   payload_json, created_at, updated_at
            FROM agent_memory
            WHERE supplier_id = ?
            ORDER BY
                CASE severity
                    WHEN 'critical' THEN 0
                    WHEN 'high' THEN 1
                    WHEN 'medium' THEN 2
                    WHEN 'low' THEN 3
                    WHEN 'monitor' THEN 3
                    WHEN 'watch' THEN 3
                    ELSE 4
                END,
                updated_at DESC
            """,
            (supplier_id,),
        ).fetchall()

    records = []
    for row in rows:
        item = dict(row)
        try:
            item["payload"] = json.loads(item.pop("payload_json"))
        except json.JSONDecodeError:
            item["payload"] = {}
        records.append(item)
    return records


def clear_stale_supplier_memory(db_path: str | Path, supplier_id: str, max_age_hours: int = 24) -> int:
    init_agent_memory(db_path)
    cutoff = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
    records = get_supplier_memory(db_path, supplier_id)
    stale_keys = []
    for item in records:
        try:
            updated = datetime.fromisoformat(str(item["updated_at"]).replace("Z", "+00:00"))
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
        except Exception:
            stale_keys.append((item["agent_name"], item["subject_id"]))
            continue
        if updated.timestamp() < cutoff:
            stale_keys.append((item["agent_name"], item["subject_id"]))

    if not stale_keys:
        return 0

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            DELETE FROM agent_memory
            WHERE supplier_id = ? AND agent_name = ? AND subject_id = ?
            """,
            [(supplier_id, agent_name, subject_id) for agent_name, subject_id in stale_keys],
        )
        conn.commit()
    return len(stale_keys)


def get_recent_memory(db_path: str | Path, limit: int = 100) -> list[dict[str, Any]]:
    init_agent_memory(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT agent_name, supplier_id, subject_id, severity, summary,
                   payload_json, created_at, updated_at
            FROM agent_memory
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    records = []
    for row in rows:
        item = dict(row)
        try:
            item["payload"] = json.loads(item.pop("payload_json"))
        except json.JSONDecodeError:
            item["payload"] = {}
        records.append(item)
    return records
