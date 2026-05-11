"""
generate_supplier_data.py — Synthetic Supplier Portfolio Data Generator
App 4: Supplier Portfolio Intelligence

Generates realistic multi-category supplier data across 36 months with:
- 1,200 suppliers across 10 product families
- Monthly KPI time series (43,200 rows)
- APQP/NPI projects based on PQA schema
- Claims based on 8D claims schema
- Audit records
- ML risk labels (3-tier: green/amber/red) derived from KPIs with noise

Usage:
    python generate_supplier_data.py --out data/
    python generate_supplier_data.py --out data/ --suppliers 1200 --seed 42
"""

import argparse
import json
import math
import os
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()

# ── Constants ─────────────────────────────────────────────────────────────────

N_SUPPLIERS     = 1200
N_MONTHS        = 36
START_DATE      = date(2022, 1, 1)
NOISE_RATE      = 0.12   # 12% of risk labels randomly flipped — realistic messiness
SEED            = 42

# ── Product families and subcategories ───────────────────────────────────────

PRODUCT_FAMILIES = {
    "Electronics":         ["PCB", "PCBA", "IC Active", "IC Passive", "Connectors", "Sensors"],
    "Electromechanics":    ["Motors", "Actuators", "Solenoids", "Relays", "Switches"],
    "Mechanics - Metal":   ["CNC Machining", "Stamping", "Casting", "Forging", "Welding"],
    "Mechanics - Plastic": ["Injection Moulding", "Extrusion", "Blow Moulding", "Thermoforming"],
    "Raw Materials":       ["Sheet Metal", "Bar & Tube", "Granulate", "Chemicals", "Gases"],
    "Cables & Harness":    ["Wire Harness", "Power Cables", "Signal Cables", "Fibre Optic"],
    "Surface Treatment":   ["Plating", "Coating", "Anodising", "Heat Treatment", "Painting"],
    "Optical & Precision": ["Lenses", "Encoders", "Precision Mechanics", "Optics"],
    "Software/Firmware":   ["Embedded SW", "Safety-Critical SW", "Firmware", "FPGA"],
    "Services":            ["Calibration Lab", "Testing House", "Logistics", "Tooling"],
}

# Family weights — electronics and mechanics most common
FAMILY_WEIGHTS = [0.18, 0.12, 0.18, 0.12, 0.10, 0.08, 0.07, 0.06, 0.04, 0.05]

# Certifications by family
CERT_MAP = {
    "Electronics":         ["ISO 9001", "IATF 16949", "ISO 9001", "ISO 9001+AS9100D"],
    "Electromechanics":    ["ISO 9001", "IATF 16949", "ISO 9001"],
    "Mechanics - Metal":   ["ISO 9001", "IATF 16949", "AS9100D", "ISO 9001"],
    "Mechanics - Plastic": ["ISO 9001", "IATF 16949", "ISO 9001"],
    "Raw Materials":       ["ISO 9001", "ISO 14001", "ISO 9001", "None"],
    "Cables & Harness":    ["ISO 9001", "IATF 16949", "ISO 9001"],
    "Surface Treatment":   ["ISO 9001", "NADCAP", "ISO 9001", "AS9100D"],
    "Optical & Precision": ["ISO 9001", "AS9100D", "ISO 9001"],
    "Software/Firmware":   ["ISO 9001", "ISO/IEC 27001", "CMMI", "None"],
    "Services":            ["ISO 17025", "ISO 9001", "ISO 9001", "None"],
}

# Geography — weighted toward manufacturing regions
COUNTRIES = {
    "Germany":        0.12, "China":          0.14, "Poland":         0.08,
    "Czech Republic": 0.06, "Romania":        0.05, "Mexico":         0.06,
    "USA":            0.08, "Japan":          0.05, "South Korea":    0.04,
    "India":          0.06, "France":         0.05, "Italy":          0.05,
    "Netherlands":    0.04, "Spain":          0.03, "Hungary":        0.03,
    "Slovakia":       0.03, "Taiwan":         0.03, "Thailand":       0.03,
    "Turkey":         0.02, "Brazil":         0.02, "United Kingdom": 0.02,
}

REGIONS = {
    "Germany": "Europe", "Poland": "Europe", "Czech Republic": "Europe",
    "Romania": "Europe", "France": "Europe", "Italy": "Europe",
    "Netherlands": "Europe", "Spain": "Europe", "Hungary": "Europe",
    "Slovakia": "Europe", "United Kingdom": "Europe",
    "China": "Asia Pacific", "Japan": "Asia Pacific", "South Korea": "Asia Pacific",
    "India": "Asia Pacific", "Taiwan": "Asia Pacific", "Thailand": "Asia Pacific",
    "USA": "Americas", "Mexico": "Americas", "Brazil": "Americas",
    "Turkey": "Middle East & Africa",
}

# ── Supplier archetypes ───────────────────────────────────────────────────────

ARCHETYPES = {
    "stable_green": {
        "weight": 0.25,
        "ppm_base": 45, "ppm_trend": 0.0, "ppm_noise": 15,
        "otd_base": 97.5, "otd_trend": 0.0, "otd_noise": 1.2,
        "audit_base": 88, "audit_trend": 0.0, "audit_noise": 3,
        "scar_rate": 0.3, "true_risk": "green",
        "description": "Tier A stable reference supplier"
    },
    "slow_decline": {
        "weight": 0.12,
        "ppm_base": 120, "ppm_trend": 8.0, "ppm_noise": 30,
        "otd_base": 95.0, "otd_trend": -0.08, "otd_noise": 2.0,
        "audit_base": 78, "audit_trend": -0.3, "audit_noise": 4,
        "scar_rate": 1.2, "true_risk": "amber",
        "description": "Slow deterioration — intervention needed"
    },
    "improving": {
        "weight": 0.12,
        "ppm_base": 380, "ppm_trend": -12.0, "ppm_noise": 40,
        "otd_base": 91.0, "otd_trend": 0.15, "otd_noise": 2.5,
        "audit_base": 68, "audit_trend": 0.5, "audit_noise": 5,
        "scar_rate": 2.0, "true_risk": "amber",
        "description": "Development plan in progress — improving"
    },
    "new_supplier": {
        "weight": 0.10,
        "ppm_base": 200, "ppm_trend": -5.0, "ppm_noise": 60,
        "otd_base": 93.0, "otd_trend": 0.10, "otd_noise": 3.5,
        "audit_base": 74, "audit_trend": 0.4, "audit_noise": 6,
        "scar_rate": 1.5, "true_risk": "amber",
        "description": "New supplier — limited history, high uncertainty"
    },
    "critical_single_source": {
        "weight": 0.08,
        "ppm_base": 180, "ppm_trend": 2.0, "ppm_noise": 35,
        "otd_base": 94.5, "otd_trend": -0.05, "otd_noise": 2.0,
        "audit_base": 76, "audit_trend": 0.0, "audit_noise": 4,
        "scar_rate": 1.0, "true_risk": "amber",
        "description": "Moderate risk — no alternative source"
    },
    "chronic_underperformer": {
        "weight": 0.10,
        "ppm_base": 320, "ppm_trend": 1.0, "ppm_noise": 50,
        "otd_base": 92.0, "otd_trend": 0.0, "otd_noise": 3.0,
        "audit_base": 71, "audit_trend": 0.1, "audit_noise": 5,
        "scar_rate": 2.5, "true_risk": "amber",
        "description": "Persistently amber — never reaches red threshold"
    },
    "recovery": {
        "weight": 0.08,
        "ppm_base": 650, "ppm_trend": -18.0, "ppm_noise": 60,
        "otd_base": 88.0, "otd_trend": 0.25, "otd_noise": 3.5,
        "audit_base": 58, "audit_trend": 0.8, "audit_noise": 6,
        "scar_rate": 3.5, "true_risk": "red",  # starts red
        "description": "Was red — recovery programme active"
    },
    "high_risk": {
        "weight": 0.08,
        "ppm_base": 750, "ppm_trend": 15.0, "ppm_noise": 80,
        "otd_base": 86.0, "otd_trend": -0.20, "otd_noise": 4.0,
        "audit_base": 55, "audit_trend": -0.4, "audit_noise": 7,
        "scar_rate": 4.0, "true_risk": "red",
        "description": "High spend, high risk — escalation required"
    },
    "high_spend_stable": {
        "weight": 0.07,
        "ppm_base": 80, "ppm_trend": 0.5, "ppm_noise": 20,
        "otd_base": 96.5, "otd_trend": 0.0, "otd_noise": 1.5,
        "audit_base": 84, "audit_trend": 0.0, "audit_noise": 3,
        "scar_rate": 0.6, "true_risk": "green",
        "description": "Strategic high-spend supplier — well managed"
    },
}

# ── Helper functions ──────────────────────────────────────────────────────────

def weighted_choice(options: dict) -> str:
    keys = list(options.keys())
    weights = list(options.values())
    return random.choices(keys, weights=weights, k=1)[0]


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def months_range(start: date, n: int):
    months = []
    d = start.replace(day=1)
    for _ in range(n):
        months.append(d)
        # advance one month
        if d.month == 12:
            d = d.replace(year=d.year + 1, month=1)
        else:
            d = d.replace(month=d.month + 1)
    return months


def derive_risk_label(ppm, otd, audit_score, scar_count):
    """Derive three-tier risk label from KPI thresholds."""
    red_flags = 0
    amber_flags = 0

    if ppm > 500:   red_flags += 1
    elif ppm > 200: amber_flags += 1

    if otd < 90:    red_flags += 1
    elif otd < 95:  amber_flags += 1

    if audit_score < 60:  red_flags += 1
    elif audit_score < 75: amber_flags += 1

    if scar_count >= 4:   red_flags += 1
    elif scar_count >= 2: amber_flags += 1

    if red_flags >= 2:
        return "red"
    elif red_flags == 1 or amber_flags >= 2:
        return "amber"
    else:
        return "green"


def apply_noise(label, noise_rate=NOISE_RATE):
    """Randomly flip label to simulate real-world messiness."""
    if random.random() < noise_rate:
        options = ["green", "amber", "red"]
        options.remove(label)
        return random.choice(options)
    return label


# ── Generator functions ───────────────────────────────────────────────────────

def generate_suppliers(n: int) -> pd.DataFrame:
    """Generate supplier master table."""
    random.seed(SEED)
    np.random.seed(SEED)
    fake.seed_instance(SEED)

    families = list(PRODUCT_FAMILIES.keys())
    archetype_keys = list(ARCHETYPES.keys())
    archetype_weights = [ARCHETYPES[k]["weight"] for k in archetype_keys]
    country_list = list(COUNTRIES.keys())
    country_weights = list(COUNTRIES.values())

    rows = []
    for i in range(1, n + 1):
        family = random.choices(families, weights=FAMILY_WEIGHTS, k=1)[0]
        subcategory = random.choice(PRODUCT_FAMILIES[family])
        country = random.choices(country_list, weights=country_weights, k=1)[0]
        archetype = random.choices(archetype_keys, weights=archetype_weights, k=1)[0]
        cert_options = CERT_MAP.get(family, ["ISO 9001"])
        certification = random.choice(cert_options)
        spend_tier = random.choices(["A", "B", "C"], weights=[0.20, 0.35, 0.45], k=1)[0]
        annual_spend = {
            "A": round(random.uniform(500_000, 5_000_000), -3),
            "B": round(random.uniform(50_000, 500_000), -3),
            "C": round(random.uniform(5_000, 50_000), -3),
        }[spend_tier]
        qual_status = random.choices(
            ["Approved", "Conditionally Approved", "Development", "New", "Suspended"],
            weights=[0.65, 0.15, 0.10, 0.07, 0.03], k=1)[0]
        is_single_source = (archetype == "critical_single_source") or random.random() < 0.12
        strategic_importance = random.choices(
            ["Critical", "Preferred", "Approved", "Conditional"],
            weights=[0.15, 0.30, 0.40, 0.15], k=1)[0]
        years_active = random.randint(1, 18) if archetype != "new_supplier" else random.randint(0, 2)
        onboarding_date = (START_DATE - timedelta(days=365 * years_active + random.randint(0, 180)))

        rows.append({
            "supplier_id": f"SUP{i:04d}",
            "name": f"{fake.company()} {random.choice(['GmbH', 'S.A.', 'Ltd', 'Inc', 'Co.', 'SRL', 'BV', 'AG', 'SpA', ''])}".strip(),
            "country": country,
            "region": REGIONS.get(country, "Other"),
            "city": fake.city(),
            "product_family": family,
            "subcategory": subcategory,
            "certification": certification,
            "spend_tier": spend_tier,
            "annual_spend_eur": annual_spend,
            "qualification_status": qual_status,
            "single_source": is_single_source,
            "strategic_importance": strategic_importance,
            "years_active": years_active,
            "onboarding_date": onboarding_date.isoformat(),
            "archetype": archetype,
            "archetype_description": ARCHETYPES[archetype]["description"],
            "primary_contact": fake.name(),
            "primary_contact_email": fake.company_email(),
            "account_manager": fake.name(),
        })

    return pd.DataFrame(rows)


def generate_kpis(suppliers: pd.DataFrame) -> pd.DataFrame:
    """Generate 36-month KPI time series per supplier."""
    months = months_range(START_DATE, N_MONTHS)
    rows = []

    for _, sup in suppliers.iterrows():
        arch = ARCHETYPES[sup["archetype"]]
        sid = sup["supplier_id"]

        # Per-supplier random seed for reproducibility
        rng = np.random.RandomState(int(sid[3:]) + SEED)

        for m_idx, month in enumerate(months):
            t = m_idx  # time index 0-35

            # PPM — log-normal base with trend and noise
            ppm_mean = arch["ppm_base"] + arch["ppm_trend"] * t
            ppm_mean = max(5, ppm_mean)
            ppm = float(rng.normal(ppm_mean, arch["ppm_noise"]))
            ppm = clamp(ppm, 0, 2500)
            ppm = round(ppm, 1)

            # OTD %
            otd_mean = arch["otd_base"] + arch["otd_trend"] * t
            otd = float(rng.normal(otd_mean, arch["otd_noise"]))
            otd = clamp(otd, 60.0, 100.0)
            otd = round(otd, 1)

            # Audit score — not every month, interpolate
            audit_score_mean = arch["audit_base"] + arch["audit_trend"] * t
            audit_score = float(rng.normal(audit_score_mean, arch["audit_noise"]))
            audit_score = clamp(audit_score, 30.0, 100.0)
            audit_score = round(audit_score, 1)

            # SCAR count — Poisson
            scar_lambda = arch["scar_rate"] * (1 + 0.02 * t if arch["ppm_trend"] > 0 else 1)
            scar_count = int(rng.poisson(scar_lambda))

            # Derived metrics
            oqd = clamp(otd - float(rng.normal(1.5, 0.8)), 60, 100)  # slightly worse than OTD
            ppap_ftp = clamp(float(rng.normal(85, 8)), 50, 100)       # PPAP first-time pass %
            ca_closure = clamp(float(rng.normal(78, 12)), 30, 100)    # CA closure rate %

            # Cost of poor quality — correlated with PPM
            copq = round(ppm * float(rng.uniform(8, 25)), 2)  # EUR per PPM unit

            # Internal PPM (typically lower than external)
            ppm_internal = round(clamp(ppm * float(rng.uniform(0.3, 0.8)), 0, 1000), 1)

            # Risk label derived + noise
            true_label = derive_risk_label(ppm, otd, audit_score, scar_count)
            risk_label = apply_noise(true_label)

            rows.append({
                "kpi_id": f"{sid}_{month.strftime('%Y%m')}",
                "supplier_id": sid,
                "year_month": month.strftime("%Y-%m"),
                "year": month.year,
                "month": month.month,
                "ppm_external": ppm,
                "ppm_internal": ppm_internal,
                "otd_pct": otd,
                "oqd_pct": round(oqd, 1),
                "audit_score": audit_score,
                "scar_count": scar_count,
                "scar_open_days_avg": round(float(rng.normal(28, 12)), 0) if scar_count > 0 else 0,
                "ppap_first_time_pass_pct": round(ppap_ftp, 1),
                "ca_closure_rate_pct": round(ca_closure, 1),
                "cost_of_poor_quality_eur": copq,
                "risk_label": risk_label,
                "risk_label_true": true_label,
            })

    return pd.DataFrame(rows)


def generate_claims(suppliers: pd.DataFrame, kpis: pd.DataFrame) -> pd.DataFrame:
    """Generate claims based on claims.csv schema — correlated with PPM."""
    months = months_range(START_DATE, N_MONTHS)
    rows = []
    claim_id = 1

    categories = ["Dimensional", "Functional", "Visual/Cosmetic", "Material",
                  "Documentation", "Delivery", "Packaging", "Contamination",
                  "Wrong Part", "Missing Part"]
    detection_methods = ["Incoming Inspection", "In-Process", "End of Line", "Customer Return",
                         "Audit", "Periodic Test", "Field Complaint"]
    who_detected = ["Incoming QC", "Production", "Customer", "Quality Engineer",
                    "Auditor", "Lab", "Field Service"]
    statuses = ["Open", "QR Submitted", "PD Submitted", "CA Submitted", "Closed", "Escalated"]
    status_weights = [0.05, 0.08, 0.12, 0.20, 0.50, 0.05]

    for _, sup in suppliers.iterrows():
        sid = sup["supplier_id"]
        arch = ARCHETYPES[sup["archetype"]]

        # Expected claims per month based on PPM and spend tier
        base_claim_rate = arch["ppm_base"] / 500  # ~1 claim per 500 PPM
        if sup["spend_tier"] == "A":
            base_claim_rate *= 1.5
        elif sup["spend_tier"] == "C":
            base_claim_rate *= 0.6

        rng = np.random.RandomState(int(sid[3:]) + SEED + 1000)

        for month in months:
            n_claims = int(rng.poisson(max(0.1, base_claim_rate)))
            for _ in range(n_claims):
                creation_date = month + timedelta(days=int(rng.randint(0, 28)))
                status = random.choices(statuses, weights=status_weights, k=1)[0]
                n_bad_parts = int(rng.lognormal(3, 1.5))
                n_suspected = n_bad_parts * int(rng.randint(1, 8))
                is_recurrent = random.random() < 0.18
                chargeback = random.random() < 0.35
                chargeback_value = round(float(rng.lognormal(5, 1.2)), 2) if chargeback else 0

                # Phase dates — QR → PD → CA → CI
                qr_expected = (creation_date + timedelta(days=5)).isoformat()
                qr_submitted = (creation_date + timedelta(days=int(rng.randint(3, 12)))).isoformat() if status != "Open" else ""
                pd_expected = (creation_date + timedelta(days=15)).isoformat()
                pd_submitted = (creation_date + timedelta(days=int(rng.randint(10, 25)))).isoformat() if status in ["PD Submitted", "CA Submitted", "Closed"] else ""
                ca_expected = (creation_date + timedelta(days=30)).isoformat()
                ca_submitted = (creation_date + timedelta(days=int(rng.randint(25, 60)))).isoformat() if status in ["CA Submitted", "Closed"] else ""
                ci_expected = (creation_date + timedelta(days=60)).isoformat()
                ci_submitted = (creation_date + timedelta(days=int(rng.randint(55, 90)))).isoformat() if status == "Closed" else ""

                rows.append({
                    "incident_number": f"INC{claim_id:05d}",
                    "supplier_id": sid,
                    "supplier_name": sup["name"],
                    "sqa_engineer": fake.name(),
                    "category": random.choice(categories),
                    "is_recurrent": is_recurrent,
                    "site_impacted": random.choice(["Site A", "Site B", "Site C", "Site D"]),
                    "part_number": f"PN-{rng.randint(10000, 99999)}",
                    "part_description": fake.bs().title()[:40],
                    "defective_component": random.choice(["Housing", "PCB", "Connector", "Seal",
                                                          "Shaft", "Spring", "Coating", "Label"]),
                    "creation_date": creation_date.isoformat(),
                    "status": status,
                    "number_of_bad_parts": n_bad_parts,
                    "number_of_suspected_parts": n_suspected,
                    "how_detected": random.choice(detection_methods),
                    "who_detected": random.choice(who_detected),
                    "where_detected": random.choice(["Goods In", "Assembly Line", "Final Test",
                                                     "Warehouse", "Customer Site"]),
                    "qr_expected_date": qr_expected,
                    "qr_submitted_date": qr_submitted,
                    "qr_status": "Submitted" if qr_submitted else "Pending",
                    "pd_expected_date": pd_expected,
                    "pd_submitted_date": pd_submitted,
                    "pd_status": "Submitted" if pd_submitted else "Pending",
                    "ca_expected_date": ca_expected,
                    "ca_submitted_date": ca_submitted,
                    "ca_status": "Submitted" if ca_submitted else "Pending",
                    "ci_expected_date": ci_expected,
                    "ci_submitted_date": ci_submitted,
                    "ci_status": "Submitted" if ci_submitted else "Pending",
                    "is_recurring_incident": is_recurrent,
                    "chargeback": chargeback,
                    "chargeback_value_eur": chargeback_value,
                    "claim_is_pqa": random.random() < 0.20,
                    "probationary_period": random.random() < 0.10,
                    "costs_charged_back": chargeback,
                    "product_family": sup["product_family"],
                    "spend_tier": sup["spend_tier"],
                })
                claim_id += 1

    return pd.DataFrame(rows)


def generate_apqp_projects(suppliers: pd.DataFrame) -> pd.DataFrame:
    """Generate APQP/NPI projects based on PQA schema."""
    rows = []
    project_id = 1
    project_types = ["New Part", "Design Change", "Process Change",
                     "New Supplier", "Transfer", "Annual Requalification"]
    type_weights = [0.30, 0.20, 0.15, 0.15, 0.10, 0.10]
    statuses = ["Active", "On Hold", "Completed", "Cancelled", "Delayed"]

    phases = [
        "Supplier Selection",
        "Supplier Nomination",
        "Design Validation of Process",
        "Process Validation",
        "Initial Sample Validation",
        "Start of Production",
        "PQA Management",
        "Yearly IS Submission",
        "PPAP Update",
    ]

    # Only certain supplier types have APQP projects
    active_families = ["Electronics", "Electromechanics", "Mechanics - Metal",
                       "Mechanics - Plastic", "Cables & Harness", "Optical & Precision"]

    for _, sup in suppliers.iterrows():
        if sup["product_family"] not in active_families:
            continue
        if random.random() > 0.65:  # not all suppliers have active NPI
            continue

        n_projects = random.randint(1, 4)
        sid = sup["supplier_id"]

        for _ in range(n_projects):
            proj_type = random.choices(project_types, weights=type_weights, k=1)[0]
            creation_offset = random.randint(-900, -30)
            creation_date = (START_DATE + timedelta(days=creation_offset + N_MONTHS * 30))
            sop_offset = random.randint(90, 540)
            customer_sop = creation_date + timedelta(days=sop_offset)
            ushin_sop = customer_sop - timedelta(days=random.randint(10, 30))
            status = random.choices(statuses, weights=[0.35, 0.10, 0.45, 0.05, 0.05], k=1)[0]

            phase_data = {}
            cum_days = 0
            for phase in phases:
                phase_duration = random.randint(14, 90)
                cum_days += phase_duration
                expected_start = (creation_date + timedelta(days=cum_days - phase_duration)).isoformat()
                expected_validation = (creation_date + timedelta(days=cum_days)).isoformat()
                phase_status = random.choices(
                    ["Not Started", "In Progress", "Submitted", "Validated", "Overdue"],
                    weights=[0.15, 0.20, 0.15, 0.40, 0.10], k=1)[0]
                submission_date = ""
                validation_date = ""
                if phase_status in ["Submitted", "Validated"]:
                    submission_date = (creation_date + timedelta(days=cum_days + random.randint(-5, 15))).isoformat()
                if phase_status == "Validated":
                    validation_date = (creation_date + timedelta(days=cum_days + random.randint(0, 20))).isoformat()

                phase_key = phase.lower().replace(" ", "_").replace("/", "_")
                phase_data[f"{phase_key}_status"] = phase_status
                phase_data[f"{phase_key}_expected_start"] = expected_start
                phase_data[f"{phase_key}_expected_validation"] = expected_validation
                phase_data[f"{phase_key}_submitted"] = submission_date
                phase_data[f"{phase_key}_validated"] = validation_date

            # Calculate completion %
            validated = sum(1 for p in phases if phase_data.get(
                f"{p.lower().replace(' ', '_').replace('/', '_')}_status") == "Validated")
            completion_pct = round(validated / len(phases) * 100, 0)

            row = {
                "project_id": f"PQA{project_id:05d}",
                "supplier_id": sid,
                "supplier_name": sup["name"],
                "project_type": proj_type,
                "status": status,
                "creation_date": creation_date.isoformat(),
                "customer_sop_date": customer_sop.isoformat(),
                "supplier_sop_date": ushin_sop.isoformat(),
                "product_family": sup["product_family"],
                "spend_tier": sup["spend_tier"],
                "completion_pct": completion_pct,
                "is_delayed": status == "Delayed" or random.random() < 0.15,
                "project_manager": fake.name(),
                "components": random.randint(1, 8),
            }
            row.update(phase_data)
            rows.append(row)
            project_id += 1

    return pd.DataFrame(rows)


def generate_audits(suppliers: pd.DataFrame) -> pd.DataFrame:
    """Generate audit records — ~2 audits per supplier per year."""
    rows = []
    audit_id = 1
    audit_types = ["System Audit", "Process Audit", "Product Audit",
                   "Pre-Award Audit", "For-Cause Audit", "Re-qualification Audit"]
    audit_type_weights = [0.25, 0.30, 0.15, 0.10, 0.05, 0.15]
    finding_types = ["Observation", "Minor NCR", "Major NCR", "Critical NCR"]
    finding_weights = [0.40, 0.35, 0.20, 0.05]

    for _, sup in suppliers.iterrows():
        sid = sup["supplier_id"]
        arch = ARCHETYPES[sup["archetype"]]
        rng = np.random.RandomState(int(sid[3:]) + SEED + 2000)

        # 1.5-2.5 audits per year × 3 years
        n_audits = int(rng.randint(4, 8))

        for i in range(n_audits):
            audit_offset = int(rng.randint(0, N_MONTHS * 30))
            audit_date = START_DATE + timedelta(days=audit_offset)
            audit_type = random.choices(audit_types, weights=audit_type_weights, k=1)[0]

            # Score correlated with archetype
            score_mean = arch["audit_base"] + arch["audit_trend"] * (audit_offset / 30)
            score = clamp(float(rng.normal(score_mean, arch["audit_noise"])), 30, 100)
            score = round(score, 1)

            # Findings count based on score
            n_findings = max(0, int(rng.poisson(max(0.5, (100 - score) / 15))))
            finding_type = random.choices(finding_types, weights=finding_weights, k=1)[0] if n_findings > 0 else "None"

            is_remote = random.random() < 0.20
            lead_auditor = fake.name()
            duration_days = random.randint(1, 3)
            report_date = (audit_date + timedelta(days=random.randint(5, 15))).isoformat()

            rows.append({
                "audit_id": f"AUD{audit_id:05d}",
                "supplier_id": sid,
                "supplier_name": sup["name"],
                "audit_date": audit_date.isoformat(),
                "audit_type": audit_type,
                "is_remote": is_remote,
                "duration_days": duration_days,
                "lead_auditor": lead_auditor,
                "audit_score": score,
                "n_findings": n_findings,
                "highest_finding_type": finding_type,
                "report_date": report_date,
                "status": random.choices(
                    ["Draft", "Issued", "Closed", "Overdue"],
                    weights=[0.05, 0.15, 0.75, 0.05], k=1)[0],
                "product_family": sup["product_family"],
                "certification_audited": sup["certification"],
                "follow_up_required": n_findings > 0,
                "follow_up_date": (audit_date + timedelta(days=90)).isoformat() if n_findings > 0 else "",
            })
            audit_id += 1

    return pd.DataFrame(rows)


def generate_risk_scores(suppliers: pd.DataFrame, kpis: pd.DataFrame) -> pd.DataFrame:
    """Generate latest risk score per supplier (ML model output simulation)."""
    # Get last 3 months average KPIs per supplier
    latest = kpis[kpis["year_month"] >= kpis["year_month"].max()[:4] + "-01"]
    latest_3m = kpis.groupby("supplier_id").tail(3)
    avg_kpis = latest_3m.groupby("supplier_id").agg({
        "ppm_external": "mean",
        "otd_pct": "mean",
        "audit_score": "mean",
        "scar_count": "mean",
        "risk_label": lambda x: x.mode()[0],
        "risk_label_true": lambda x: x.mode()[0],
    }).reset_index()

    avg_kpis = avg_kpis.merge(
        suppliers[["supplier_id", "product_family", "spend_tier",
                   "annual_spend_eur", "single_source", "strategic_importance",
                   "qualification_status", "archetype"]],
        on="supplier_id"
    )

    # Compute composite risk score 0-100
    avg_kpis["ppm_score"] = avg_kpis["ppm_external"].apply(
        lambda x: clamp(100 - (x / 10), 0, 100))
    avg_kpis["otd_score"] = avg_kpis["otd_pct"].apply(
        lambda x: clamp((x - 80) * 5, 0, 100))
    avg_kpis["audit_score_norm"] = avg_kpis["audit_score"]
    avg_kpis["scar_score"] = avg_kpis["scar_count"].apply(
        lambda x: clamp(100 - x * 20, 0, 100))

    avg_kpis["composite_risk_score"] = (
        avg_kpis["ppm_score"] * 0.30 +
        avg_kpis["otd_score"] * 0.25 +
        avg_kpis["audit_score_norm"] * 0.30 +
        avg_kpis["scar_score"] * 0.15
    ).round(1)

    # Spend-adjusted risk (high spend + high risk = higher priority)
    avg_kpis["spend_risk_priority"] = (
        avg_kpis["annual_spend_eur"] / 1_000_000 *
        (100 - avg_kpis["composite_risk_score"]) / 100
    ).round(2)

    avg_kpis["recommended_action"] = avg_kpis["risk_label"].map({
        "green": "Monitor — next scheduled audit",
        "amber": "Enhanced monitoring + development plan review",
        "red": "Immediate escalation + for-cause audit",
    })

    return avg_kpis[[
        "supplier_id", "ppm_external", "otd_pct", "audit_score",
        "scar_count", "composite_risk_score", "spend_risk_priority",
        "risk_label", "risk_label_true", "recommended_action",
        "product_family", "spend_tier", "annual_spend_eur",
        "single_source", "strategic_importance", "qualification_status"
    ]].rename(columns={
        "ppm_external": "avg_ppm_3m",
        "otd_pct": "avg_otd_3m",
        "audit_score": "avg_audit_score_3m",
        "scar_count": "avg_scar_count_3m",
    })


# ── External events generator ────────────────────────────────────────────────

def generate_external_events(suppliers: pd.DataFrame) -> pd.DataFrame:
    """Generate ESG alerts, sanctions flags, geopolitical and regulatory events."""
    rows = []
    event_id = 1

    event_types = {
        "ESG": [
            "Environmental violation reported by national regulator",
            "Labour rights audit failure — subcontractor facility",
            "CO2 emissions disclosure non-compliance",
            "Waste disposal incident — local authority investigation",
            "Child labour allegation in sub-tier supply chain",
            "Water usage violation — drought-zone restriction",
            "ESG rating downgrade by major rating agency",
        ],
        "Sanctions": [
            "Entity listed on OFAC SDN list",
            "Parent company subject to EU sanctions",
            "Dual-use export control restriction applied",
            "Beneficial owner flagged in sanctions screening",
            "Country-level trade restriction announced",
        ],
        "Geopolitical": [
            "Regional port strike — 2-3 week delay expected",
            "Border closure — customs clearance suspended",
            "Political instability — factory access restricted",
            "Currency devaluation — contract repricing risk",
            "Natural disaster — production facility affected",
            "Energy supply disruption — production capacity reduced",
            "Trade tariff increase — cost impact under review",
        ],
        "Regulatory": [
            "New REACH substance restriction — material review required",
            "RoHS compliance update — product re-certification needed",
            "IATF 16949 certification suspended by CB",
            "AS9100D audit finding — corrective action mandatory",
            "Customer-specific requirement updated — flowdown review needed",
            "Product liability regulation change — design review triggered",
            "Import duty change — total cost of ownership impact",
        ],
        "Financial": [
            "Credit rating downgrade — payment terms review triggered",
            "Insolvency filing by parent company",
            "Acquisition announced — supply continuity review required",
            "Key customer loss — revenue concentration risk",
            "Raw material cost spike — force majeure clause activated",
        ],
    }

    severities = {
        "ESG": ["Medium", "High", "High", "Medium", "Critical", "Medium", "Medium"],
        "Sanctions": ["Critical", "Critical", "High", "High", "High"],
        "Geopolitical": ["Medium", "High", "High", "Medium", "High", "Medium", "Medium"],
        "Regulatory": ["Medium", "High", "Critical", "Medium", "Low", "Medium", "Low"],
        "Financial": ["Medium", "Critical", "High", "Medium", "High"],
    }

    statuses = ["Open", "Under Review", "Mitigated", "Closed", "Escalated"]
    status_weights = [0.20, 0.25, 0.30, 0.15, 0.10]

    # Assign events — higher-risk archetypes get more events
    archetype_event_rate = {
        "stable_green": 0.10,
        "slow_decline": 0.35,
        "improving": 0.30,
        "new_supplier": 0.20,
        "critical_single_source": 0.40,
        "chronic_underperformer": 0.35,
        "recovery": 0.50,
        "high_risk": 0.60,
        "high_spend_stable": 0.15,
    }

    # Country-based geopolitical risk
    high_geo_risk_countries = ["China", "Turkey", "India", "Brazil", "Mexico",
                                "Thailand", "Romania", "Hungary"]

    for _, sup in suppliers.iterrows():
        sid = sup["supplier_id"]
        arch = sup["archetype"]
        country = sup["country"]
        rng = np.random.RandomState(int(sid[3:]) + SEED + 3000)

        base_rate = archetype_event_rate.get(arch, 0.25)
        if country in high_geo_risk_countries:
            base_rate += 0.15
        if sup["single_source"]:
            base_rate += 0.10
        if sup["spend_tier"] == "A":
            base_rate += 0.10  # high-spend suppliers monitored more closely

        n_events = int(rng.poisson(max(0.05, base_rate * 2)))

        for _ in range(n_events):
            event_type = random.choices(
                list(event_types.keys()),
                weights=[0.20, 0.10, 0.25, 0.30, 0.15], k=1)[0]

            events_list = event_types[event_type]
            sev_list = severities[event_type]
            idx = rng.randint(0, len(events_list))
            description = events_list[idx % len(events_list)]
            severity = sev_list[idx % len(sev_list)]

            event_offset = int(rng.randint(0, N_MONTHS * 30))
            event_date = START_DATE + timedelta(days=event_offset)
            status = random.choices(statuses, weights=status_weights, k=1)[0]
            response_due = (event_date + timedelta(days=random.randint(14, 60))).isoformat()
            resolved_date = ""
            if status in ["Mitigated", "Closed"]:
                resolved_date = (event_date + timedelta(
                    days=random.randint(15, 90))).isoformat()

            rows.append({
                "event_id": f"EVT{event_id:05d}",
                "supplier_id": sid,
                "supplier_name": sup["name"],
                "country": country,
                "region": sup["region"],
                "event_type": event_type,
                "severity": severity,
                "description": description,
                "event_date": event_date.isoformat(),
                "status": status,
                "response_due_date": response_due,
                "resolved_date": resolved_date,
                "product_family": sup["product_family"],
                "spend_tier": sup["spend_tier"],
                "annual_spend_eur": sup["annual_spend_eur"],
                "single_source": sup["single_source"],
                "requires_capa": severity in ["High", "Critical"],
                "capa_linked": random.random() < 0.45 if severity in ["High", "Critical"] else False,
                "source": random.choice([
                    "Internal Monitoring", "Risk Intel Feed", "Customer Alert",
                    "News Scan", "Regulator Notice", "Auditor Report", "Supplier Self-Declaration"
                ]),
            })
            event_id += 1

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic supplier portfolio data")
    parser.add_argument("--out", type=str, default="data", help="Output directory")
    parser.add_argument("--suppliers", type=int, default=N_SUPPLIERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--format", choices=["csv", "sqlite", "both"], default="both")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    fake.seed_instance(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Supplier Portfolio Data Generator")
    print(f"Suppliers: {args.suppliers}  |  Months: {N_MONTHS}  |  Seed: {SEED}")
    print(f"{'='*60}\n")

    print("Generating supplier master...")
    suppliers = generate_suppliers(args.suppliers)
    print(f"  ✓ {len(suppliers)} suppliers")

    print("Generating KPI time series...")
    kpis = generate_kpis(suppliers)
    print(f"  ✓ {len(kpis)} KPI records ({N_MONTHS} months × {args.suppliers} suppliers)")

    print("Generating claims...")
    claims = generate_claims(suppliers, kpis)
    print(f"  ✓ {len(claims)} claims")

    print("Generating APQP projects...")
    apqp = generate_apqp_projects(suppliers)
    print(f"  ✓ {len(apqp)} APQP projects")

    print("Generating audit records...")
    audits = generate_audits(suppliers)
    print(f"  ✓ {len(audits)} audit records")

    print("Generating risk scores...")
    risk_scores = generate_risk_scores(suppliers, kpis)
    print(f"  ✓ {len(risk_scores)} risk score records")

    # Risk distribution summary
    dist = risk_scores["risk_label"].value_counts()
    print(f"\n  Risk distribution:")
    for label in ["green", "amber", "red"]:
        n = dist.get(label, 0)
        pct = n / len(risk_scores) * 100
        print(f"    {label:<8} {n:>4} ({pct:.0f}%)")

    print("Generating external events...")
    events = generate_external_events(suppliers)
    print(f"  ✓ {len(events)} external events (ESG, sanctions, geopolitical, regulatory, financial)")

    tables = {
        "suppliers": suppliers,
        "supplier_kpis": kpis,
        "claims": claims,
        "apqp_projects": apqp,
        "audits": audits,
        "risk_scores": risk_scores,
        "external_events": events,
    }

    if args.format in ["csv", "both"]:
        print(f"\nSaving CSV files to {out_dir}/...")
        for name, df in tables.items():
            path = out_dir / f"{name}.csv"
            df.to_csv(path, index=False)
            print(f"  ✓ {name}.csv ({len(df):,} rows, {len(df.columns)} cols)")

    if args.format in ["sqlite", "both"]:
        db_path = out_dir / "supplier_portfolio.db"
        print(f"\nSaving SQLite database to {db_path}...")
        conn = sqlite3.connect(db_path)
        for name, df in tables.items():
            df.to_sql(name, conn, if_exists="replace", index=False)
            print(f"  ✓ {name} ({len(df):,} rows)")
        conn.close()

    # Summary stats
    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Suppliers:        {len(suppliers):>8,}")
    print(f"KPI records:      {len(kpis):>8,}  ({N_MONTHS} months)")
    print(f"Claims:           {len(claims):>8,}")
    print(f"APQP projects:    {len(apqp):>8,}")
    print(f"Audit records:    {len(audits):>8,}")
    print(f"Risk scores:      {len(risk_scores):>8,}")
    print(f"External events:  {len(events):>8,}")
    total = sum(len(df) for df in tables.values())
    print(f"Total rows:       {total:>8,}")
    print(f"\nOutput: {out_dir.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
