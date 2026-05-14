from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "supplier_portfolio.db"
ML_DIR  = Path(__file__).parent.parent / "ml"

from scripts.answer import CHROMA_DB_PATH, CHROMA_SETTINGS  # noqa: E402

TABLE_COLUMNS = {
    "suppliers": [
        "supplier_id", "name", "country", "region", "city", "product_family",
        "subcategory", "certification", "spend_tier", "annual_spend_eur",
        "qualification_status", "single_source", "strategic_importance",
        "years_active", "onboarding_date", "archetype", "archetype_description",
        "primary_contact", "primary_contact_email", "account_manager",
    ],
    "supplier_kpis": [
        "kpi_id", "supplier_id", "year_month", "year", "month", "ppm_external",
        "ppm_internal", "otd_pct", "oqd_pct", "audit_score", "scar_count",
        "scar_open_days_avg", "ppap_first_time_pass_pct", "ca_closure_rate_pct",
        "cost_of_poor_quality_eur", "risk_label", "risk_label_true",
    ],
    "claims": [
        "incident_number", "supplier_id", "supplier_name", "creation_date",
        "category", "status", "number_of_bad_parts", "chargeback",
        "chargeback_value_eur", "product_family", "spend_tier",
    ],
    "apqp_projects": [
        "project_id", "supplier_id", "supplier_name", "project_type", "status",
        "creation_date", "customer_sop_date", "supplier_sop_date",
        "product_family", "spend_tier", "completion_pct", "is_delayed",
    ],
    "audits": [
        "audit_id", "supplier_id", "supplier_name", "audit_date", "audit_type",
        "is_remote", "audit_score", "n_findings", "highest_finding_type",
        "status", "product_family",
    ],
    "risk_scores": [
        "supplier_id", "avg_ppm_3m", "avg_otd_3m", "avg_audit_score_3m",
        "avg_scar_count_3m", "composite_risk_score", "spend_risk_priority",
        "risk_label", "risk_label_true", "recommended_action", "product_family",
        "spend_tier", "annual_spend_eur", "single_source",
        "strategic_importance", "qualification_status",
    ],
    "external_events": [
        "event_id", "supplier_id", "supplier_name", "country", "region",
        "event_type", "severity", "description", "event_date", "status",
        "response_due_date", "resolved_date", "product_family", "spend_tier",
        "annual_spend_eur", "single_source", "requires_capa", "capa_linked",
        "source",
    ],
}
