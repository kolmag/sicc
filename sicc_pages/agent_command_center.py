import pandas as pd
import streamlit as st
from datetime import datetime, timezone

from utils.config import DB_PATH
from utils.ui import kpi_card
from utils.agent_helpers import (
    build_evidence_pack_markdown,
    build_run_log_markdown,
    is_memory_fresh,
    memory_age_hours,
    operator_status_label,
    severity_rank,
)
from scripts.agent_memory import (
    clear_stale_supplier_memory,
    finish_agent_run,
    get_supplier_agent_runs,
    get_supplier_memory,
    normalize_severity,
    record_agent_run_step,
    remember_agent_output,
    start_agent_run,
)
from scripts.supplier_intake_agent import (
    generate_supplier_development_brief,
)
from scripts.supplier_alert_agent import build_supplier_trend_alerts
from scripts.continuity_agent import assess_single_source_continuity
from scripts.audit_planning_agent import plan_supplier_audit
from scripts.apqp_readiness_agent import assess_apqp_launch_readiness
from scripts.scar_capa_agent import triage_claim


def render(tables, filtered_suppliers, filtered_ids, filtered_risk, ml):
    suppliers   = tables["suppliers"]
    kpis        = tables["supplier_kpis"]
    claims      = tables["claims"]
    apqp        = tables["apqp_projects"]
    audits      = tables["audits"]
    risk_scores = tables["risk_scores"]
    events      = tables["external_events"]

    st.markdown('<div class="page-title">Agent Command Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Shared memory view across supplier workflow agents</div>', unsafe_allow_html=True)

    supplier_options = filtered_suppliers[["supplier_id", "name"]].copy()
    supplier_options["display"] = supplier_options["name"] + " (" + supplier_options["supplier_id"] + ")"
    selected_supplier = st.selectbox("Select supplier", supplier_options["display"].tolist())
    sid = selected_supplier.split("(")[-1].replace(")", "").strip()

    sup = suppliers[suppliers["supplier_id"] == sid].iloc[0]
    risk_match = risk_scores[risk_scores["supplier_id"] == sid]
    risk_row = risk_match.iloc[0] if not risk_match.empty else {}
    sup_kpis = kpis[kpis["supplier_id"] == sid].sort_values("year_month")
    sup_claims = claims[claims["supplier_id"] == sid]
    sup_audits = audits[audits["supplier_id"] == sid]
    sup_events = events[events["supplier_id"] == sid]
    sup_apqp = apqp[apqp["supplier_id"] == sid]
    memory = get_supplier_memory(DB_PATH, sid)
    fresh_memory = is_memory_fresh(memory)
    agent_runs = get_supplier_agent_runs(DB_PATH, sid, limit=5)
    stale_records = [item for item in memory if (memory_age_hours(item) is None or memory_age_hours(item) > 24)]

    sync_col, cleanup_col, export_col = st.columns([1, 1, 1])
    with sync_col:
        force_sync = st.checkbox("Force refresh", value=False)
        run_sync = st.button("Run Agent Sync", type="primary", disabled=(fresh_memory and not force_sync))
    with cleanup_col:
        clear_stale = st.button("Clear Stale Memory", disabled=not stale_records)
        if clear_stale:
            deleted = clear_stale_supplier_memory(DB_PATH, sid, max_age_hours=24)
            st.success(f"Cleared {deleted} stale memory record(s).")
            memory = get_supplier_memory(DB_PATH, sid)
            fresh_memory = is_memory_fresh(memory)
            stale_records = [item for item in memory if (memory_age_hours(item) is None or memory_age_hours(item) > 24)]
    with export_col:
        st.markdown(
            f'<div style="font-size:0.78rem; color:#64748b; padding-top:0.45rem;">'
            f'Memory status: {"fresh" if fresh_memory else "missing or stale"}'
            f'</div>',
            unsafe_allow_html=True,
        )

    if stale_records:
        stale_names = ", ".join(sorted({item["agent_name"] for item in stale_records}))
        st.warning(f"{len(stale_records)} stale agent memory record(s) detected: {stale_names}. Refresh or clear stale memory before using this supplier pack operationally.")

    if agent_runs:
        st.download_button(
            "Download Run Log",
            data=build_run_log_markdown(sid, agent_runs),
            file_name=f"{sid}_agent_run_log.md",
            mime="text/markdown",
            key=f"download_agent_run_log_{sid}",
        )

    if run_sync:
        run_id = start_agent_run(DB_PATH, sid)
        sync_results = []

        def log_step(agent_name, status, summary="", severity="info", error="", started_at=None):
            record_agent_run_step(
                DB_PATH,
                run_id=run_id,
                agent_name=agent_name,
                supplier_id=sid,
                status=status,
                summary=summary,
                severity=severity,
                error=error,
                started_at=started_at,
            )
            sync_results.append({
                "Agent": agent_name,
                "Status": status,
                "Severity": normalize_severity(severity),
                "Summary": summary,
                "Error": error,
            })

        def step_started_at():
            return datetime.now(timezone.utc).isoformat(timespec="seconds")

        started_at = step_started_at()
        try:
            brief = generate_supplier_development_brief(
                supplier=sup,
                risk_row=risk_row,
                kpis=sup_kpis,
                claims=sup_claims,
                audits=sup_audits,
                events=sup_events,
                apqp=sup_apqp,
                use_llm=False,
            )
            brief_severity = {"red": "high", "amber": "medium", "green": "low"}.get(brief.risk_level, "info")
            brief_summary = f"{brief.recommended_pathway}: {len(brief.development_actions)} action(s)"
            remember_agent_output(
                DB_PATH,
                agent_name="Supplier Intake Agent",
                supplier_id=sid,
                subject_id="development_brief",
                severity=brief_severity,
                summary=brief_summary,
                payload=brief.model_dump(),
            )
            log_step("Supplier Intake Agent", "success", brief_summary, brief_severity, started_at=started_at)
        except Exception as exc:
            log_step("Supplier Intake Agent", "failed", "Development brief failed", "critical", str(exc), started_at)

        started_at = step_started_at()
        try:
            alerts = build_supplier_trend_alerts(
                suppliers=suppliers,
                kpis=kpis,
                risk_scores=risk_scores,
                claims=claims,
                audits=audits,
                events=events,
                apqp=apqp,
                supplier_ids={sid},
                top_n=1,
            )
            if alerts:
                alert = alerts[0]
                alert_summary = f"{alert.direction.title()} detected, score {alert.trend_score:.1f}/100"
                remember_agent_output(
                    DB_PATH,
                    agent_name="Early Warning Agent",
                    supplier_id=sid,
                    subject_id="trend_alert",
                    severity=alert.alert_level,
                    summary=alert_summary,
                    payload=alert.model_dump(),
                )
                log_step("Early Warning Agent", "success", alert_summary, alert.alert_level, started_at=started_at)
            else:
                log_step("Early Warning Agent", "skipped", "No material trend alert detected", "info", started_at=started_at)
        except Exception as exc:
            log_step("Early Warning Agent", "failed", "Trend alert failed", "critical", str(exc), started_at)

        started_at = step_started_at()
        try:
            if bool(sup.get("single_source", False)):
                continuity = assess_single_source_continuity(
                    supplier=sup,
                    risk_row=risk_row,
                    supplier_claims=sup_claims,
                    supplier_events=sup_events,
                    supplier_apqp=sup_apqp,
                )
                continuity_summary = f"{continuity.continuity_level.title()} continuity exposure, buffer target {continuity.buffer_stock_target_days}"
                remember_agent_output(
                    DB_PATH,
                    agent_name="Continuity Agent",
                    supplier_id=sid,
                    subject_id="single_source_plan",
                    severity=continuity.continuity_level,
                    summary=continuity_summary,
                    payload=continuity.model_dump(),
                )
                log_step("Continuity Agent", "success", continuity_summary, continuity.continuity_level, started_at=started_at)
            else:
                log_step("Continuity Agent", "skipped", "Supplier is not marked single source", "info", started_at=started_at)
        except Exception as exc:
            log_step("Continuity Agent", "failed", "Continuity plan failed", "critical", str(exc), started_at)

        started_at = step_started_at()
        try:
            audit_plan = plan_supplier_audit(
                supplier=sup,
                risk_row=risk_row,
                supplier_kpis=sup_kpis,
                supplier_claims=sup_claims,
                supplier_audits=sup_audits,
                supplier_events=sup_events,
            )
            audit_severity = {
                "immediate": "critical",
                "high": "high",
                "medium": "medium",
                "scheduled": "low",
            }.get(audit_plan.urgency, "info")
            audit_summary = f"{audit_plan.audit_type} · {audit_plan.urgency} · {audit_plan.schedule_timeline}"
            remember_agent_output(
                DB_PATH,
                agent_name="Audit Planning Agent",
                supplier_id=sid,
                subject_id="audit_plan",
                severity=audit_severity,
                summary=audit_summary,
                payload=audit_plan.model_dump(),
            )
            log_step("Audit Planning Agent", "success", audit_summary, audit_severity, started_at=started_at)
        except Exception as exc:
            log_step("Audit Planning Agent", "failed", "Audit planning failed", "critical", str(exc), started_at)

        started_at = step_started_at()
        try:
            if not sup_apqp.empty:
                apqp_decisions = [
                    assess_apqp_launch_readiness(
                        project=row,
                        supplier=sup,
                        risk_row=risk_row,
                        supplier_claims=sup_claims,
                        supplier_events=sup_events,
                    )
                    for _, row in sup_apqp.iterrows()
                ]
                worst = sorted(apqp_decisions, key=lambda item: item.readiness_score)[0]
                apqp_severity = {
                    "HOLD": "critical",
                    "CONDITIONAL_GO": "medium",
                    "GO": "low",
                }.get(worst.launch_decision, "info")
                apqp_summary = f"{worst.launch_decision.replace('_', ' ')} for {worst.project_id}, score {worst.readiness_score:.1f}/100"
                remember_agent_output(
                    DB_PATH,
                    agent_name="APQP Readiness Agent",
                    supplier_id=sid,
                    subject_id=f"apqp_{worst.project_id}",
                    severity=apqp_severity,
                    summary=apqp_summary,
                    payload=worst.model_dump(),
                )
                log_step("APQP Readiness Agent", "success", apqp_summary, apqp_severity, started_at=started_at)
            else:
                log_step("APQP Readiness Agent", "skipped", "No APQP project found for supplier", "info", started_at=started_at)
        except Exception as exc:
            log_step("APQP Readiness Agent", "failed", "APQP readiness failed", "critical", str(exc), started_at)

        started_at = step_started_at()
        try:
            open_claims = sup_claims[sup_claims["status"].astype(str).str.lower() != "closed"] if not sup_claims.empty else pd.DataFrame()
            if not open_claims.empty:
                claim = open_claims.sort_values("creation_date", ascending=False).iloc[0]
                triage = triage_claim(
                    claim=claim,
                    supplier=sup,
                    risk_row=risk_row,
                    supplier_claims=sup_claims,
                    supplier_kpis=sup_kpis,
                    supplier_audits=sup_audits,
                    supplier_events=sup_events,
                )
                scar_severity = {
                    "Critical NCR": "critical",
                    "Major NCR": "high",
                    "Minor NCR": "medium",
                    "Observation": "low",
                }.get(triage.finding_grade, "info")
                scar_summary = f"{triage.finding_grade} · {triage.scar_escalation_level} · severity {triage.severity_score:.1f}/100"
                remember_agent_output(
                    DB_PATH,
                    agent_name="SCAR/CAPA Triage Agent",
                    supplier_id=sid,
                    subject_id=triage.incident_number,
                    severity=scar_severity,
                    summary=scar_summary,
                    payload=triage.model_dump(),
                )
                log_step("SCAR/CAPA Triage Agent", "success", scar_summary, scar_severity, started_at=started_at)
            else:
                log_step("SCAR/CAPA Triage Agent", "skipped", "No open claim available for triage", "info", started_at=started_at)
        except Exception as exc:
            log_step("SCAR/CAPA Triage Agent", "failed", "SCAR/CAPA triage failed", "critical", str(exc), started_at)

        finish_agent_run(DB_PATH, run_id)
        failed_steps = [row for row in sync_results if row["Status"] == "failed"]
        if failed_steps:
            st.warning(f"Agent sync completed with {len(failed_steps)} failed step(s). See Run History below.")
        else:
            st.success("Agent sync completed and outputs were saved to shared memory.")
        st.dataframe(pd.DataFrame(sync_results), use_container_width=True, hide_index=True)
        memory = get_supplier_memory(DB_PATH, sid)
        fresh_memory = is_memory_fresh(memory)
        agent_runs = get_supplier_agent_runs(DB_PATH, sid, limit=5)
        stale_records = [item for item in memory if (memory_age_hours(item) is None or memory_age_hours(item) > 24)]

    expected_agents = [
        "Supplier Intake Agent",
        "Early Warning Agent",
        "Continuity Agent",
        "Audit Planning Agent",
        "APQP Readiness Agent",
        "SCAR/CAPA Triage Agent",
    ]
    memory_by_agent = {}
    for item in memory:
        memory_by_agent.setdefault(item["agent_name"], []).append(item)
    agent_runs = get_supplier_agent_runs(DB_PATH, sid, limit=5)
    latest_steps = {}
    if agent_runs:
        latest_steps = {step["agent_name"]: step for step in agent_runs[0].get("steps", [])}
    worst_severity = max([severity_rank(item["severity"]) for item in memory], default=0)
    worst_label = {
        4: "critical",
        3: "high",
        2: "medium",
        1: "low",
        0: "info",
    }[worst_severity]

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(kpi_card("Memory Records", f"{len(memory):,}"), unsafe_allow_html=True)
    with m2:
        critical = sum(1 for item in memory if normalize_severity(item["severity"]) == "critical")
        st.markdown(kpi_card("Critical", f"{critical:,}", delta_direction="up" if critical else "flat"), unsafe_allow_html=True)
    with m3:
        high = sum(1 for item in memory if normalize_severity(item["severity"]) == "high")
        st.markdown(kpi_card("High", f"{high:,}", delta_direction="up" if high else "flat"), unsafe_allow_html=True)
    with m4:
        st.markdown(kpi_card("Worst Severity", worst_label.upper()), unsafe_allow_html=True)
    with m5:
        st.markdown(kpi_card("Supplier", sid), unsafe_allow_html=True)

    if agent_runs:
        st.markdown('<div class="section-header">Run History</div>', unsafe_allow_html=True)
        run_rows = [
            {
                "Started": run["started_at"],
                "Finished": run.get("finished_at") or "",
                "Status": operator_status_label(run["status"]),
                "Summary": run["summary"],
                "Run ID": run["run_id"][:10],
            }
            for run in agent_runs
        ]
        st.dataframe(pd.DataFrame(run_rows), use_container_width=True, hide_index=True)

        latest = agent_runs[0]
        failed_latest = [step for step in latest.get("steps", []) if step["status"] == "failed"]
        if failed_latest:
            st.error("Latest Agent Sync has failed steps.")
            for step in failed_latest:
                st.markdown(f"- **{step['agent_name']}**: {step['error'] or step['summary']}")

        with st.expander("Latest Run Step Details", expanded=bool(failed_latest)):
            step_rows = [
                {
                    "Agent": step["agent_name"],
                    "Status": operator_status_label(step["status"]),
                    "Severity": normalize_severity(step["severity"]),
                    "Summary": step["summary"],
                    "Error": step["error"],
                    "Finished": step["finished_at"],
                }
                for step in latest.get("steps", [])
            ]
            st.dataframe(pd.DataFrame(step_rows), use_container_width=True, hide_index=True)

    if not memory:
        st.info("No shared memory records for this supplier yet. Run Agent Sync to populate them.")
    else:
        st.markdown('<div class="section-header">Agent Status</div>', unsafe_allow_html=True)
        status_rows = []
        for agent in expected_agents:
            records = memory_by_agent.get(agent, [])
            latest_step = latest_steps.get(agent)
            if records:
                top = sorted(records, key=lambda item: severity_rank(item["severity"]), reverse=True)[0]
                age = memory_age_hours(top)
                status = "fresh" if age is not None and age <= 24 else "stale"
                if latest_step and latest_step["status"] == "failed":
                    status = "failed"
                status_rows.append({
                    "Agent": agent,
                    "Status": operator_status_label(status),
                    "Severity": normalize_severity(top["severity"]),
                    "Records": len(records),
                    "Last Updated": top["updated_at"],
                    "Summary": latest_step["error"] if latest_step and latest_step["status"] == "failed" else top["summary"],
                })
            else:
                status = "no data"
                summary = "No current memory record"
                if latest_step:
                    status = latest_step["status"]
                    summary = latest_step["error"] or latest_step["summary"]
                status_rows.append({
                    "Agent": agent,
                    "Status": operator_status_label(status),
                    "Severity": "info",
                    "Records": 0,
                    "Last Updated": "",
                    "Summary": summary,
                })
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">Shared Agent Memory</div>', unsafe_allow_html=True)
        mem_df = pd.DataFrame([
            {
                "Agent": item["agent_name"],
                "Severity": normalize_severity(item["severity"]),
                "Subject": item["subject_id"],
                "Summary": item["summary"],
                "Updated": item["updated_at"],
            }
            for item in memory
        ])
        st.dataframe(mem_df, use_container_width=True, hide_index=True)

        evidence_pack = build_evidence_pack_markdown(
            supplier=sup,
            risk_row=risk_row,
            sup_kpis=sup_kpis,
            sup_claims=sup_claims,
            sup_audits=sup_audits,
            sup_events=sup_events,
            sup_apqp=sup_apqp,
            memory=memory,
            agent_runs=agent_runs,
        )
        st.download_button(
            "Download Evidence Pack",
            data=evidence_pack,
            file_name=f"{sid}_agent_evidence_pack.md",
            mime="text/markdown",
            key=f"download_agent_pack_{sid}",
        )

        st.markdown('<div class="section-header">Memory Details</div>', unsafe_allow_html=True)
        for item in memory:
            item_severity = normalize_severity(item["severity"])
            with st.expander(f"{item_severity.upper()} · {item['agent_name']} · {item['summary']}", expanded=item_severity == "critical"):
                payload = item.get("payload", {})
                key_lists = [
                    "primary_risk_drivers", "signals", "triggers", "blockers",
                    "risks", "exposure_drivers", "mandatory_actions",
                    "recovery_actions", "development_actions",
                ]
                for key in key_lists:
                    value = payload.get(key)
                    if isinstance(value, list) and value:
                        st.markdown(f"**{key.replace('_', ' ').title()}**")
                        for entry in value[:8]:
                            if isinstance(entry, dict):
                                st.markdown(f"- {entry.get('action') or entry.get('issue') or entry}")
                            else:
                                st.markdown(f"- {entry}")
                docs = payload.get("source_documents", [])
                if docs:
                    st.markdown("**Source Documents**")
                    st.markdown(", ".join(f"`{doc}`" for doc in docs))
