# SICC ML Training Report

**Run:** 2026-05-10 16:58:23  
**Winner:** RandomForest  
**Decision criterion:** F1-Red — catching at-risk suppliers is the priority  

---

## Dataset

| | |
|---|---|
| Suppliers | 1,200 |
| Features engineered | 61 |
| Train set | 960 (80%) |
| Test set | 240 (20%) |
| Seed | 42 |
| Label noise | 12% (by design — realistic messiness) |

**Label distribution (training target):**

| Label | Count | % |
|---|---|---|
| amber | 482 | 40% |
| green | 595 | 50% |
| red | 123 | 10% |

---

## Model Comparison

| Metric | RandomForest | XGBoost | Winner |
|---|---|---|---|
| Accuracy | 0.8833 | 0.8750 | **RF** |
| F1 Macro | 0.8792 | 0.8734 | **RF** |
| AUC (OvR) | 0.9396 | 0.9369 | **RF** |
| F1 Green | 0.9053 | 0.8926 | **RF** |
| F1 Amber | 0.8571 | 0.8526 | **RF** |
| F1 Red ★ | 0.8750 | 0.8750 | **RF** |

**Selected model: RandomForest**  
Primary criterion: F1-Red (correctly identifying RED-tier suppliers).

---

## RandomForest — Classification Report

```
              precision    recall  f1-score   support

       green       0.89      0.92      0.91       119
       amber       0.87      0.84      0.86        96
         red       0.91      0.84      0.88        25

    accuracy                           0.88       240
   macro avg       0.89      0.87      0.88       240
weighted avg       0.88      0.88      0.88       240

```

## XGBoost — Classification Report

```
              precision    recall  f1-score   support

       green       0.88      0.91      0.89       119
       amber       0.86      0.84      0.85        96
         red       0.91      0.84      0.88        25

    accuracy                           0.88       240
   macro avg       0.88      0.86      0.87       240
weighted avg       0.88      0.88      0.87       240

```

---

## Top 10 Features (winner model — MDI importance)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `otd_pct_12m` | 0.0774 |
| 2 | `ppm_worst_3m` | 0.0570 |
| 3 | `oqd_pct_12m` | 0.0551 |
| 4 | `otd_pct_6m` | 0.0542 |
| 5 | `audit_score_12m` | 0.0535 |
| 6 | `months_ppm_above_200` | 0.0417 |
| 7 | `ppm_external_3m` | 0.0401 |
| 8 | `oqd_pct_6m` | 0.0390 |
| 9 | `cost_of_poor_quality_eur_12m` | 0.0377 |
| 10 | `oqd_pct_3m` | 0.0345 |

---

## Model Parameters

**RandomForest:**
```
  n_estimators: 300
  max_depth: 12
  min_samples_leaf: 4
  min_samples_split: 8
  class_weight: balanced
  random_state: 42
  n_jobs: -1
```

**XGBoost:**
```
  n_estimators: 300
  max_depth: 6
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
  min_child_weight: 3
  gamma: 0.1
  objective: multi:softprob
  num_class: 3
  eval_metric: mlogloss
  random_state: 42
  n_jobs: -1
```

---

## Notes

- 12% label noise injected at data generation (seed=42) — realistic label uncertainty
- XGBoost class imbalance handled via sample weights (inverse class frequency)
- RandomForest uses `class_weight='balanced'`
- SHAP values precomputed for all 1,200 suppliers via TreeExplainer
- Winning model saved as `ml/model.pkl`, loaded by Streamlit at runtime
- Streamlit degrades gracefully if model not found (falls back to rule-based scores)
---

## Monotonicity Tests

**Model:** RandomForest  
**Result:** 5/5 passed  

| Test | Feature | Before → After | Red prob change | Pass |
|---|---|---|---|---|
| PPM spike | `ppm_external_3m` | 50 → 900 | 0.269 → 0.34 | ✓ |
| OTD collapse | `otd_pct_3m` | 97 → 72 | 0.269 → 0.269 | ✓ |
| Audit failure | `audit_score_3m` | 88 → 45 | 0.269 → 0.309 | ✓ |
| SCAR surge | `scar_count_3m` | 0.3 → 6 | 0.269 → 0.274 | ✓ |
| PPM improvement | `ppm_external_3m` | 500 → 30 | 0.338 → 0.269 | ✓ |