"""
ml/train_risk_model.py — SICC Risk Scoring ML Model
Trains RandomForest AND XGBoost, compares on held-out 20%, saves the winner.
Decision criterion: F1-Red (catching bad suppliers is the priority).

Outputs:
    ml/model.pkl               — winning model
    ml/shap_values.pkl         — SHAP payload for all 1,200 suppliers
    ml/feature_names.pkl       — feature list + category encodings
    ml/feature_importances.pkl — sorted DataFrame
    ml/model_metrics.json      — metrics for both models + winner
    ml/training_report.md      — human-readable comparison report

Usage:
    uv run python ml/train_risk_model.py
    uv run python ml/train_risk_model.py --db data/supplier_portfolio.db --out ml/
"""

import argparse
import json
import pickle
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)

# ── Constants ─────────────────────────────────────────────────────────────────

SEED        = 42
TEST_SIZE   = 0.20
LABEL_ORDER = ["green", "amber", "red"]   # green=0, amber=1, red=2

RF_PARAMS = {
    "n_estimators":     300,
    "max_depth":         12,
    "min_samples_leaf":   4,
    "min_samples_split":  8,
    "class_weight":  "balanced",
    "random_state":    SEED,
    "n_jobs":            -1,
}

XGB_PARAMS = {
    "n_estimators":      300,
    "max_depth":           6,
    "learning_rate":    0.05,
    "subsample":        0.80,
    "colsample_bytree": 0.80,
    "min_child_weight":    3,
    "gamma":             0.1,
    "objective":  "multi:softprob",
    "num_class":           3,
    "eval_metric":    "mlogloss",
    "random_state":     SEED,
    "n_jobs":             -1,
}

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train SICC risk scoring models")
    p.add_argument("--db",  default="data/supplier_portfolio.db")
    p.add_argument("--out", default="ml/")
    return p.parse_args()


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(kpis: pd.DataFrame, suppliers: pd.DataFrame):
    """
    Engineer features from the 36-month KPI time series per supplier.
    Returns (features_df, cat_maps).

    Feature groups:
      - Window averages  : 3m / 6m / 12m for 8 core KPIs
      - Volatility       : std dev over 12m for PPM, OTD, audit
      - Trend            : linear slope over full 36m for 4 KPIs
      - Deterioration    : recent 3m vs historical 12m delta
      - Stress peaks     : worst-month values in last 3m
      - Threshold counts : months breaching key thresholds
      - Supplier attrs   : spend tier, strategic importance, qual status,
                           single source, region risk, product family, years active
    """
    kpis = kpis.copy()
    kpis["year_month"] = pd.to_datetime(kpis["year_month"])
    kpis = kpis.sort_values(["supplier_id", "year_month"])

    rows = []
    for sid in kpis["supplier_id"].unique():
        df      = kpis[kpis["supplier_id"] == sid].sort_values("year_month")
        tail_3  = df.tail(3)
        tail_6  = df.tail(6)
        tail_12 = df.tail(12)
        full    = df

        def avg(col, subset):
            return subset[col].mean() if len(subset) > 0 else np.nan

        def std(col, subset):
            return subset[col].std() if len(subset) > 1 else 0.0

        def trend(col):
            if len(full) < 3:
                return 0.0
            x = np.arange(len(full))
            y = full[col].values
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return 0.0
            return float(np.polyfit(x[mask], y[mask], 1)[0])

        feat = {"supplier_id": sid}

        # Window averages
        for col in ["ppm_external", "otd_pct", "audit_score", "scar_count",
                    "cost_of_poor_quality_eur", "ppap_first_time_pass_pct",
                    "ca_closure_rate_pct", "oqd_pct"]:
            feat[f"{col}_3m"]  = avg(col, tail_3)
            feat[f"{col}_6m"]  = avg(col, tail_6)
            feat[f"{col}_12m"] = avg(col, tail_12)

        # Volatility
        for col in ["ppm_external", "otd_pct", "audit_score"]:
            feat[f"{col}_std_12m"] = std(col, tail_12)

        # Trend
        for col in ["ppm_external", "otd_pct", "audit_score", "scar_count"]:
            feat[f"{col}_trend"] = trend(col)

        # Deterioration flags (positive = getting worse)
        feat["ppm_deterioration"]   = avg("ppm_external", tail_3) - avg("ppm_external", df.head(12))
        feat["otd_deterioration"]   = avg("otd_pct",      df.head(12)) - avg("otd_pct",  tail_3)
        feat["audit_deterioration"] = avg("audit_score",  df.head(12)) - avg("audit_score", tail_3)

        # Stress peaks
        feat["ppm_worst_3m"]   = tail_3["ppm_external"].max()
        feat["scar_worst_3m"]  = tail_3["scar_count"].max()
        feat["audit_worst_3m"] = tail_3["audit_score"].min()

        # Threshold breach counters
        feat["months_ppm_above_500"]  = int((full["ppm_external"] > 500).sum())
        feat["months_ppm_above_200"]  = int((full["ppm_external"] > 200).sum())
        feat["months_otd_below_90"]   = int((full["otd_pct"] < 90).sum())
        feat["months_otd_below_95"]   = int((full["otd_pct"] < 95).sum())
        feat["months_audit_below_60"] = int((full["audit_score"] < 60).sum())
        feat["months_audit_below_75"] = int((full["audit_score"] < 75).sum())

        # Target (modal risk label over last 3 months — noisy label)
        feat["risk_label"] = tail_3["risk_label"].mode()[0] if len(tail_3) > 0 else "amber"

        rows.append(feat)

    features_df = pd.DataFrame(rows)

    # Merge supplier attributes
    sup_cols = ["supplier_id", "product_family", "spend_tier", "single_source",
                "strategic_importance", "qualification_status", "years_active",
                "annual_spend_eur", "region"]
    features_df = features_df.merge(suppliers[sup_cols], on="supplier_id", how="left")

    # Encode categoricals
    cat_maps = {}
    spend_tier_map  = {"A": 3, "B": 2, "C": 1}
    strat_imp_map   = {"Critical": 4, "Preferred": 3, "Approved": 2, "Conditional": 1}
    qual_status_map = {"Approved": 4, "Conditionally Approved": 3,
                       "Development": 2, "New": 1, "Suspended": 0}
    region_map      = {"Europe": 2, "Americas": 2, "Asia Pacific": 1,
                       "Middle East & Africa": 1}

    features_df["spend_tier_enc"]    = features_df["spend_tier"].map(spend_tier_map).fillna(1)
    features_df["strat_imp_enc"]     = features_df["strategic_importance"].map(strat_imp_map).fillna(2)
    features_df["qual_status_enc"]   = features_df["qualification_status"].map(qual_status_map).fillna(2)
    features_df["region_risk_enc"]   = features_df["region"].map(region_map).fillna(1)
    features_df["single_source_int"] = features_df["single_source"].astype(int)

    family_dummies = pd.get_dummies(features_df["product_family"], prefix="fam")
    features_df    = pd.concat([features_df, family_dummies], axis=1)

    cat_maps["family_dummies"]       = list(family_dummies.columns)
    cat_maps["spend_tier"]           = spend_tier_map
    cat_maps["strategic_importance"] = strat_imp_map
    cat_maps["qualification_status"] = qual_status_map
    cat_maps["region_risk"]          = region_map

    return features_df, cat_maps


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"supplier_id", "risk_label", "product_family", "spend_tier",
               "strategic_importance", "qualification_status", "region"}
    return [c for c in df.columns if c not in exclude]


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, model_name: str) -> dict:
    y_pred       = model.predict(X_test)
    y_prob       = model.predict_proba(X_test)
    accuracy     = accuracy_score(y_test, y_pred)
    f1_macro     = f1_score(y_test, y_pred, average="macro")
    f1_per_class = f1_score(y_test, y_pred, average=None)
    auc          = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    report       = classification_report(y_test, y_pred, target_names=LABEL_ORDER, output_dict=True)
    report_text  = classification_report(y_test, y_pred, target_names=LABEL_ORDER)

    print(f"\n[SICC ML] ── {model_name} Performance ──────────────────────")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  F1 Macro  : {f1_macro:.4f}")
    print(f"  AUC (OvR) : {auc:.4f}")
    print(f"  F1 green  : {f1_per_class[0]:.4f}")
    print(f"  F1 amber  : {f1_per_class[1]:.4f}")
    print(f"  F1 red    : {f1_per_class[2]:.4f}")
    print(report_text)

    return {
        "model_name":  model_name,
        "accuracy":    round(accuracy, 4),
        "f1_macro":    round(f1_macro, 4),
        "auc_ovr":     round(auc, 4),
        "f1_green":    round(float(f1_per_class[0]), 4),
        "f1_amber":    round(float(f1_per_class[1]), 4),
        "f1_red":      round(float(f1_per_class[2]), 4),
        "report":      report,
        "report_text": report_text,   # kept for markdown report, stripped before JSON save
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train(db_path: str, out_dir: str):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[SICC ML] Loading data from {db_path} ...")
    conn      = sqlite3.connect(db_path)
    kpis      = pd.read_sql("SELECT * FROM supplier_kpis", conn)
    suppliers = pd.read_sql("SELECT * FROM suppliers", conn)
    conn.close()
    print(f"[SICC ML] KPIs: {len(kpis):,} rows | Suppliers: {len(suppliers):,}")

    print("[SICC ML] Engineering features ...")
    features_df, cat_maps = build_features(kpis, suppliers)
    feature_cols = get_feature_cols(features_df)

    X     = features_df[feature_cols].fillna(0)
    y_raw = features_df["risk_label"].values

    le = LabelEncoder()
    le.classes_ = np.array(LABEL_ORDER)
    y = le.transform(y_raw)

    label_dist = dict(zip(*np.unique(y_raw, return_counts=True)))
    print(f"[SICC ML] Features: {len(feature_cols)} | Samples: {len(X)}")
    print(f"[SICC ML] Label distribution: {label_dist}")

    # Train / test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # XGBoost sample weights (manual class balancing)
    class_counts     = np.bincount(y_train)
    class_weights    = len(y_train) / (len(class_counts) * class_counts)
    sample_weights   = np.array([class_weights[yi] for yi in y_train])

    # ── RandomForest ──────────────────────────────────────────────────────────
    print("\n[SICC ML] Training RandomForest ...")
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)
    rf_metrics = evaluate(rf, X_test, y_test, "RandomForest")
    rf_metrics.update({"n_train": int(len(X_train)), "n_test": int(len(X_test)),
                        "n_features": int(len(feature_cols))})

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("\n[SICC ML] Training XGBoost ...")
    xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    xgb_metrics = evaluate(xgb_model, X_test, y_test, "XGBoost")
    xgb_metrics.update({"n_train": int(len(X_train)), "n_test": int(len(X_test)),
                         "n_features": int(len(feature_cols))})

    # ── Model comparison ──────────────────────────────────────────────────────
    print("\n[SICC ML] ── Model Comparison ───────────────────────")
    metrics_to_compare = [
        ("Accuracy",  "accuracy"),
        ("F1 Macro",  "f1_macro"),
        ("AUC (OvR)", "auc_ovr"),
        ("F1 Green",  "f1_green"),
        ("F1 Amber",  "f1_amber"),
        ("F1 Red ★",  "f1_red"),
    ]
    rows_cmp = []
    for label, key in metrics_to_compare:
        rf_val  = rf_metrics[key]
        xgb_val = xgb_metrics[key]
        winner  = "RF" if rf_val >= xgb_val else "XGB"
        rows_cmp.append((label, rf_val, xgb_val, winner))
        print(f"  {label:12s}  RF={rf_val:.4f}  XGB={xgb_val:.4f}  → {winner}")

    # Primary criterion: F1-Red
    if rf_metrics["f1_red"] >= xgb_metrics["f1_red"]:
        winner_name, winner_model, winner_metrics = "RandomForest", rf, rf_metrics
        print(f"\n[SICC ML] ✓ Winner: RandomForest  "
              f"(F1-Red {rf_metrics['f1_red']:.4f} ≥ {xgb_metrics['f1_red']:.4f})")
    else:
        winner_name, winner_model, winner_metrics = "XGBoost", xgb_model, xgb_metrics
        print(f"\n[SICC ML] ✓ Winner: XGBoost  "
              f"(F1-Red {xgb_metrics['f1_red']:.4f} > {rf_metrics['f1_red']:.4f})")

    # ── Feature importance ────────────────────────────────────────────────────
    importances = pd.DataFrame({
        "feature":    feature_cols,
        "importance": winner_model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("\n[SICC ML] ── Top 20 Features ────────────────────────")
    print(importances.head(20).to_string(index=False))

    # ── SHAP values ───────────────────────────────────────────────────────────
    print("\n[SICC ML] Computing SHAP values (full dataset) ...")
    explainer   = shap.TreeExplainer(winner_model)
    shap_values = explainer.shap_values(X)

    shap_values_raw = explainer.shap_values(X)
    # Handle both old (list of arrays) and new (3D array) SHAP output formats
    if isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
        # New format: (n_samples, n_features, n_classes) → convert to list of (n_samples, n_features)
        shap_values = [shap_values_raw[:, :, i] for i in range(shap_values_raw.shape[2])]
    else:
        shap_values = shap_values_raw

    shap_payload = {
        "shap_values":           shap_values,
        "expected_value":        explainer.expected_value,
        "feature_names":         feature_cols,
        "X":                     X,
        "supplier_ids":          features_df["supplier_id"].tolist(),
        "y_pred":                winner_model.predict(X).tolist(),
        "y_pred_proba":          winner_model.predict_proba(X).tolist(),
        "label_encoder_classes": le.classes_.tolist(),
        "label_order":           LABEL_ORDER,
        "winner_name":           winner_name,
    }

    # ── Save artefacts ────────────────────────────────────────────────────────
    # Strip report_text before JSON serialisation
    rf_metrics_json  = {k: v for k, v in rf_metrics.items()  if k != "report_text"}
    xgb_metrics_json = {k: v for k, v in xgb_metrics.items() if k != "report_text"}
    win_metrics_json = {k: v for k, v in winner_metrics.items() if k != "report_text"}

    all_metrics = {
        "winner":          winner_name,
        "run_timestamp":   run_ts,
        "winner_metrics":  win_metrics_json,
        "rf_metrics":      rf_metrics_json,
        "xgb_metrics":     xgb_metrics_json,
        "comparison":      rows_cmp,
    }

    paths = {
        "model":       out_path / "model.pkl",
        "shap":        out_path / "shap_values.pkl",
        "feat":        out_path / "feature_names.pkl",
        "importances": out_path / "feature_importances.pkl",
        "metrics":     out_path / "model_metrics.json",
        "report":      out_path / "training_report.md",
    }

    with open(paths["model"], "wb") as f:
        pickle.dump(winner_model, f, protocol=5)
    with open(paths["shap"], "wb") as f:
        pickle.dump(shap_payload, f, protocol=5)
    with open(paths["feat"], "wb") as f:
        pickle.dump({"feature_names": feature_cols, "cat_maps": cat_maps,
                     "label_encoder_classes": le.classes_.tolist()}, f, protocol=5)
    with open(paths["importances"], "wb") as f:
        pickle.dump(importances, f, protocol=5)
    with open(paths["metrics"], "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Training report (markdown)
    report_md = _build_report(
        run_ts, winner_name, rf_metrics, xgb_metrics,
        rows_cmp, importances, feature_cols, label_dist
    )
    with open(paths["report"], "w") as f:
        f.write(report_md)

    print(f"\n[SICC ML] ✓ model.pkl               → {paths['model']}")
    print(f"[SICC ML] ✓ shap_values.pkl         → {paths['shap']}")
    print(f"[SICC ML] ✓ feature_names.pkl       → {paths['feat']}")
    print(f"[SICC ML] ✓ feature_importances.pkl → {paths['importances']}")
    print(f"[SICC ML] ✓ model_metrics.json      → {paths['metrics']}")
    print(f"[SICC ML] ✓ training_report.md      → {paths['report']}")
    print("\n[SICC ML] Training complete.")

    return winner_model, shap_payload, all_metrics, importances


# ── Report builder ────────────────────────────────────────────────────────────

def _build_report(run_ts, winner_name, rf_m, xgb_m, rows_cmp,
                  importances, feature_cols, label_dist) -> str:
    top10 = importances.head(10)
    total = sum(label_dist.values())

    lines = [
        "# SICC ML Training Report",
        "",
        f"**Run:** {run_ts}  ",
        f"**Winner:** {winner_name}  ",
        f"**Decision criterion:** F1-Red — catching at-risk suppliers is the priority  ",
        "",
        "---",
        "",
        "## Dataset",
        "",
        "| | |",
        "|---|---|",
        f"| Suppliers | 1,200 |",
        f"| Features engineered | {len(feature_cols)} |",
        f"| Train set | {int(total * 0.8)} (80%) |",
        f"| Test set | {int(total * 0.2)} (20%) |",
        f"| Seed | 42 |",
        f"| Label noise | 12% (by design — realistic messiness) |",
        "",
        "**Label distribution (training target):**",
        "",
        "| Label | Count | % |",
        "|---|---|---|",
    ]
    for label, count in sorted(label_dist.items()):
        lines.append(f"| {label} | {count} | {count/total*100:.0f}% |")

    lines += [
        "",
        "---",
        "",
        "## Model Comparison",
        "",
        "| Metric | RandomForest | XGBoost | Winner |",
        "|---|---|---|---|",
    ]
    for label, rf_val, xgb_val, w in rows_cmp:
        lines.append(f"| {label} | {rf_val:.4f} | {xgb_val:.4f} | **{w}** |")

    lines += [
        "",
        f"**Selected model: {winner_name}**  ",
        f"Primary criterion: F1-Red (correctly identifying RED-tier suppliers).",
        "",
        "---",
        "",
        "## RandomForest — Classification Report",
        "",
        "```",
        rf_m.get("report_text", ""),
        "```",
        "",
        "## XGBoost — Classification Report",
        "",
        "```",
        xgb_m.get("report_text", ""),
        "```",
        "",
        "---",
        "",
        "## Top 10 Features (winner model — MDI importance)",
        "",
        "| Rank | Feature | Importance |",
        "|---|---|---|",
    ]
    for i, row in top10.iterrows():
        lines.append(f"| {i+1} | `{row['feature']}` | {row['importance']:.4f} |")

    lines += [
        "",
        "---",
        "",
        "## Model Parameters",
        "",
        "**RandomForest:**",
        "```",
        "\n".join(f"  {k}: {v}" for k, v in RF_PARAMS.items()),
        "```",
        "",
        "**XGBoost:**",
        "```",
        "\n".join(f"  {k}: {v}" for k, v in XGB_PARAMS.items()),
        "```",
        "",
        "---",
        "",
        "## Notes",
        "",
        "- 12% label noise injected at data generation (seed=42) — realistic label uncertainty",
        "- XGBoost class imbalance handled via sample weights (inverse class frequency)",
        "- RandomForest uses `class_weight='balanced'`",
        "- SHAP values precomputed for all 1,200 suppliers via TreeExplainer",
        "- Winning model saved as `ml/model.pkl`, loaded by Streamlit at runtime",
        "- Streamlit degrades gracefully if model not found (falls back to rule-based scores)",
    ]

    return "\n".join(lines)


# ── Monotonicity unit tests ───────────────────────────────────────────────────

def run_monotonicity_tests(model, feature_cols: list[str],
                           model_name: str) -> tuple[int, int, list]:
    """
    Sanity-check that the model responds correctly to KPI changes.
    Perturbs one feature at a time against a stable-green baseline.
    """
    print(f"\n[SICC ML] ── Monotonicity Tests ({model_name}) ──────────")

    baseline = {f: 0.0 for f in feature_cols}
    baseline.update({
        "ppm_external_3m":   50.0,  "ppm_external_6m":   55.0,  "ppm_external_12m":  60.0,
        "otd_pct_3m":        97.0,  "otd_pct_6m":        96.5,  "otd_pct_12m":       96.0,
        "audit_score_3m":    88.0,  "audit_score_6m":    87.0,  "audit_score_12m":   86.0,
        "scar_count_3m":      0.3,  "scar_count_6m":      0.3,  "scar_count_12m":     0.4,
        "spend_tier_enc":     2.0,  "strat_imp_enc":      2.0,  "qual_status_enc":    4.0,
        "region_risk_enc":    2.0,  "single_source_int":  0,    "years_active":       8.0,
    })

    tests = [
        ("PPM spike",       "ppm_external_3m",   50,   900, "red↑"),
        ("OTD collapse",    "otd_pct_3m",         97,    72, "red↑"),
        ("Audit failure",   "audit_score_3m",     88,    45, "red↑"),
        ("SCAR surge",      "scar_count_3m",       0.3,   6, "red↑"),
        ("PPM improvement", "ppm_external_3m",   500,    30, "green↑"),
    ]

    passed = failed = 0
    test_results = []

    for test_name, feature, val_before, val_after, direction in tests:
        if feature not in feature_cols:
            print(f"  SKIP  {test_name} — feature not in model")
            continue

        X_b = pd.DataFrame([{**baseline, feature: val_before}])[feature_cols].fillna(0)
        X_a = pd.DataFrame([{**baseline, feature: val_after}])[feature_cols].fillna(0)

        prob_b = model.predict_proba(X_b)[0]
        prob_a = model.predict_proba(X_a)[0]
        green_b, _, red_b = prob_b
        green_a, _, red_a = prob_a

        ok = (red_a > red_b) if direction == "red↑" else (green_a > green_b)
        passed += ok
        failed += not ok

        print(f"  {'PASS ✓' if ok else 'FAIL ✗'}  {test_name:25s}  {feature}  "
              f"{val_before} → {val_after}  "
              f"[red: {red_b:.2f}→{red_a:.2f}  green: {green_b:.2f}→{green_a:.2f}]")

        test_results.append({
            "test": test_name, "feature": feature,
            "val_before": val_before, "val_after": val_after,
            "direction": direction, "passed": ok,
            "red_before": round(red_b, 3), "red_after": round(red_a, 3),
        })

    print(f"\n  Result: {passed}/{passed+failed} tests passed")
    print("  ✓  All monotonicity checks passed" if not failed
          else "  ⚠  Some monotonicity tests failed — review feature engineering")

    return passed, failed, test_results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    winner_model, shap_payload, all_metrics, importances = train(args.db, args.out)

    feature_cols = shap_payload["feature_names"]
    winner_name  = shap_payload["winner_name"]

    passed, failed, test_results = run_monotonicity_tests(
        winner_model, feature_cols, winner_name
    )

    # Append monotonicity results to training_report.md
    report_path = Path(args.out) / "training_report.md"
    mono_lines  = [
        "",
        "---",
        "",
        "## Monotonicity Tests",
        "",
        f"**Model:** {winner_name}  ",
        f"**Result:** {passed}/{passed+failed} passed  ",
        "",
        "| Test | Feature | Before → After | Red prob change | Pass |",
        "|---|---|---|---|---|",
    ]
    for t in test_results:
        mono_lines.append(
            f"| {t['test']} | `{t['feature']}` | {t['val_before']} → {t['val_after']} "
            f"| {t['red_before']} → {t['red_after']} | {'✓' if t['passed'] else '✗'} |"
        )
    with open(report_path, "a") as f:
        f.write("\n".join(mono_lines))

    print(f"\n[SICC ML] ✓ training_report.md updated with monotonicity results")
