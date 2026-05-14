import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.agent_memory import (
    clear_stale_supplier_memory,
    finish_agent_run,
    get_supplier_agent_runs,
    get_supplier_memory,
    init_agent_memory,
    record_agent_run_step,
    remember_agent_output,
    start_agent_run,
    validate_agent_payload,
)


class AgentMemoryTests(unittest.TestCase):
    def test_remember_agent_output_normalizes_agent_specific_severities(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.db"

            remember_agent_output(
                db_path,
                agent_name="Early Warning Agent",
                supplier_id="SUP001",
                subject_id="watch",
                severity="watch",
                summary="Watch alert",
                payload={"alert_level": "watch"},
            )
            remember_agent_output(
                db_path,
                agent_name="Continuity Agent",
                supplier_id="SUP001",
                subject_id="monitor",
                severity="monitor",
                summary="Monitor continuity exposure",
                payload={"continuity_level": "monitor"},
            )
            remember_agent_output(
                db_path,
                agent_name="Unknown Agent",
                supplier_id="SUP001",
                subject_id="unknown",
                severity="unexpected",
                summary="Unknown severity",
                payload={},
            )

            severities = {item["subject_id"]: item["severity"] for item in get_supplier_memory(db_path, "SUP001")}

        self.assertEqual(severities["watch"], "low")
        self.assertEqual(severities["monitor"], "low")
        self.assertEqual(severities["unknown"], "info")

    def test_init_agent_memory_migrates_existing_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.db"
            init_agent_memory(db_path)
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO agent_memory (
                        agent_name, supplier_id, subject_id, severity, summary,
                        payload_json, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "Continuity Agent",
                        "SUP001",
                        "legacy",
                        "monitor",
                        "Legacy monitor record",
                        "{}",
                        "2026-05-14T00:00:00+00:00",
                        "2026-05-14T00:00:00+00:00",
                    ),
                )
                conn.commit()

            init_agent_memory(db_path)
            records = get_supplier_memory(db_path, "SUP001")

        self.assertEqual(records[0]["severity"], "low")

    def test_agent_run_history_tracks_success_skips_and_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.db"
            run_id = start_agent_run(db_path, "SUP001")

            record_agent_run_step(
                db_path,
                run_id,
                "Supplier Intake Agent",
                "SUP001",
                "success",
                summary="Brief created",
                severity="high",
            )
            record_agent_run_step(
                db_path,
                run_id,
                "Continuity Agent",
                "SUP001",
                "skipped",
                summary="Supplier is not single source",
            )
            record_agent_run_step(
                db_path,
                run_id,
                "APQP Readiness Agent",
                "SUP001",
                "failed",
                summary="APQP failed",
                severity="critical",
                error="missing project field",
            )
            finish_agent_run(db_path, run_id)

            runs = get_supplier_agent_runs(db_path, "SUP001")

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["status"], "failed")
        self.assertEqual(runs[0]["summary"], "1 succeeded, 1 skipped, 1 failed")
        self.assertEqual([step["status"] for step in runs[0]["steps"]], ["success", "skipped", "failed"])
        self.assertEqual(runs[0]["steps"][2]["error"], "missing project field")

    def test_known_agent_payloads_are_validated_before_storage(self):
        payload = {
            "supplier_id": "SUP001",
            "supplier_name": "Acme Components",
            "current_risk": "green",
            "alert_level": "watch",
            "trend_score": 25.0,
            "direction": "stable",
            "signals": [],
            "evidence": {},
            "recommended_action": "Monitor trend.",
            "escalation_owner": "Supplier Quality Engineer",
            "source_documents": ["supplier_kpi_definitions.md"],
        }

        validated = validate_agent_payload(
            "Early Warning Agent",
            "trend_alert",
            "SUP001",
            payload,
        )

        self.assertEqual(validated["supplier_id"], "SUP001")
        self.assertEqual(validated["alert_level"], "watch")

    def test_malformed_known_agent_payload_is_rejected_and_not_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.db"

            with self.assertRaises(ValueError):
                remember_agent_output(
                    db_path,
                    agent_name="Early Warning Agent",
                    supplier_id="SUP001",
                    subject_id="trend_alert",
                    severity="watch",
                    summary="Malformed alert",
                    payload={"supplier_id": "SUP001", "alert_level": "watch"},
                )

            records = get_supplier_memory(db_path, "SUP001")

        self.assertEqual(records, [])

    def test_payload_supplier_id_must_match_memory_supplier_id(self):
        payload = {
            "supplier_id": "SUP999",
            "supplier_name": "Acme Components",
            "current_risk": "green",
            "alert_level": "watch",
            "trend_score": 25.0,
            "direction": "stable",
            "signals": [],
            "evidence": {},
            "recommended_action": "Monitor trend.",
            "escalation_owner": "Supplier Quality Engineer",
            "source_documents": [],
        }

        with self.assertRaises(ValueError):
            validate_agent_payload("Early Warning Agent", "trend_alert", "SUP001", payload)

    def test_clear_stale_supplier_memory_removes_only_old_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.db"
            init_agent_memory(db_path)
            with sqlite3.connect(db_path) as conn:
                for subject_id, updated_at in [
                    ("old", "2026-05-12T00:00:00+00:00"),
                    ("fresh", "2999-01-01T00:00:00+00:00"),
                ]:
                    conn.execute(
                        """
                        INSERT INTO agent_memory (
                            agent_name, supplier_id, subject_id, severity, summary,
                            payload_json, created_at, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            "Unknown Agent",
                            "SUP001",
                            subject_id,
                            "info",
                            subject_id,
                            "{}",
                            updated_at,
                            updated_at,
                        ),
                    )
                conn.commit()

            deleted = clear_stale_supplier_memory(db_path, "SUP001", max_age_hours=24)
            remaining = {item["subject_id"] for item in get_supplier_memory(db_path, "SUP001")}

        self.assertEqual(deleted, 1)
        self.assertEqual(remaining, {"fresh"})


if __name__ == "__main__":
    unittest.main()
