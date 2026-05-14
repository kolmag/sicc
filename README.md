# ⬡ SICC — Supplier Intelligence Command Center

**App 4 of 4** 

SICC is a production-grade Streamlit application combining ML risk scoring, RAG-powered Q&A, SHAP explainability, and executive decision support across a portfolio of 1,200 suppliers.

---

## What it does

| Capability | Technology |
|---|---|
| Risk scoring with explainability | RandomForest + XGBoost comparison · SHAP waterfall per supplier |
| Semantic Q&A over supplier quality standards | Hybrid BM25 + embedding retrieval · BGE reranker · HyDE |
| Structured grounded answers | OSS-120B generator · OSS-20B groundedness checker · Pydantic output |
| Agentic supplier intake | Supplier dossier → gap analysis → development brief → actions/evidence/exit criteria |
| Agentic early warning alerts | KPI trend drift + events + claims + APQP delays → supplier deterioration watchlist |
| Agentic SCAR/CAPA triage | Claim/manual issue → finding grade → escalation level → containment/evidence/deadlines |
| Agentic APQP launch readiness | APQP gates + PPAP + supplier risk → go / conditional go / hold decision |
| Agentic single-source continuity | Single-source exposure → buffer target → BCP controls → dual-source urgency |
| Agentic audit planning | For-cause triggers → audit type → scope/checklist/evidence plan |
| Shared agent memory | Pydantic-validated SQLite memory + run history + evidence/run-log exports → cross-agent command center |
| Portfolio-level executive overview | Plotly dashboards · ML prediction badges · rule-based composite score |
| APQP/NPI programme governance | 9-phase gate tracker · delay detection |
| Scenario simulation | Outage, cost increase, region disruption impact modelling |
| Embedding space diagnostics | t-SNE 2D + 3D · pre/post reranker similarity chart |

---

## Architecture

```
SICC
├── app.py                        # Streamlit UI — portfolio, agents, Q&A, and simulations
│
├── ml/
│   ├── train_risk_model.py       # RF + XGBoost comparison, SHAP precomputation
│   ├── model.pkl                 # Winning model (RandomForest, F1-Red 0.875, AUC 0.940)
│   └── training_report.md        # Full model comparison report
│
├── scripts/
│   ├── ingest.py                 # KB ingestion: contextual retrieval + typed chunking
│   ├── answer.py                 # RAG pipeline: HyDE → BM25+semantic → RRF → BGE → LLM
│   ├── supplier_intake_agent.py  # Agentic supplier intake → development brief generator
│   ├── supplier_alert_agent.py   # Agentic deterioration alert/watchlist generator
│   ├── scar_capa_agent.py        # Agentic SCAR/CAPA triage and closure governance
│   ├── apqp_readiness_agent.py   # Agentic APQP launch readiness decisioning
│   ├── continuity_agent.py       # Agentic single-source continuity mitigation
│   ├── audit_planning_agent.py   # Agentic for-cause audit planning
│   ├── agent_memory.py           # Validated SQLite memory, run history, and failure tracking
│   └── diagnostics/
│       ├── tsne_viz.py           # Embedding space visualisation (2D + 3D)
│       └── sc_viz.py             # Pre/post reranker similarity chart
│
├── knowledge-base/markdown/      # 16 supplier quality KB documents
│   ├── as9100d_supplier_control_clause_8_4.md
│   ├── iatf_16949_supplier_requirements.md
│   ├── iso_9001_2015_requirements.md
│   ├── ppap_level_requirements.md
│   ├── ppap_submission_checklist.md
│   ├── apqp_phase_gate_guide.md
│   ├── scar_process_escalation.md
│   ├── risk_tier_definitions.md
│   ├── supplier_kpi_definitions.md
│   ├── supplier_qualification_procedure.md
│   ├── for_cause_audit_trigger_criteria.md
│   ├── supplier_development_methodology.md
│   ├── single_source_risk_management.md
│   ├── audit_finding_classification.md
│   ├── external_risk_event_response.md
│   └── corrective_action_closure_requirements.md
│
├── data/
│   └── supplier_portfolio.db     # SQLite — 7 tables, 75,411 rows (generated, not committed)
│
└── generate_supplier_data.py     # Deterministic data generator (seed=42)
```

---

## RAG Pipeline

```
Question
  │
  ├─ Pre-flight: prompt injection check (pattern match, 0ms)
  │
  ▼
HyDE Query Rewriter (OSS-120B via Groq)
  Generates hypothetical document excerpt matching supplier quality vocabulary
  │
  ▼
Hybrid Retrieval
  ├─ Semantic: ChromaDB (text-embedding-3-small, K=20, cosine)
  └─ BM25: rank_bm25 on full collection (K=20)
  → Reciprocal Rank Fusion (k=60)
  │
  ▼
BGE Cross-Encoder Reranker (bge-reranker-v2-m3, Top-7)
  Input: HyDE paragraph — not original question
  GPU on Colab · CPU fallback on M1
  │
  ▼
Chunk Ordering (lost-in-the-middle fix)
  Top chunks at START and END of context window
  │
  ▼
Answer Generator (OSS-120B via Groq)
  Strict grounding — zero outside knowledge · every claim cited [n]
  │
  ▼
Groundedness Checker (OSS-20B via Groq)
  NLI actor/critic · strips unsupported claims · empty answer guard
  │
  ▼
Pydantic Structured Output
  { answer, confidence, action_required, insufficient_evidence, sources }
```

---

## ML Model

**RandomForest** (winner over XGBoost on F1-Red — primary criterion)

| Metric | Score |
|---|---|
| Accuracy | 88.3% |
| AUC (OvR) | 0.940 |
| F1 Macro | 0.879 |
| F1 Red | 0.875 |
| F1 Amber | 0.857 |
| F1 Green | 0.905 |

**61 engineered features** from 36-month KPI time series:
- 3m / 6m / 12m window averages for PPM, OTD, audit score, SCARs, COPQ, OQD, PPAP FTP, CA closure
- Trend (linear slope over 36 months)
- Volatility (std dev over 12 months)
- Deterioration flags (recent 3m vs historical 12m delta)
- Stress peaks (worst-month values)
- Threshold breach counters (months with PPM>500, OTD<90, audit<60 etc.)
- Supplier attributes (spend tier, strategic importance, qualification status, single source, region, product family)

**5/5 monotonicity tests passing** — PPM spike, OTD collapse, audit failure, SCAR surge, PPM improvement all move predicted red probability in the expected direction.

---

## Knowledge Base

**264 chunks** across 16 supplier quality documents. Three chunking strategies by document type:

| Strategy | Documents |
|---|---|
| Clause-level | AS9100D clauses, IATF 16949 clauses, PPAP level requirements |
| Paragraph-level | APQP phase guide, SCAR process, supplier development, qualification procedure, audit criteria |
| Row-level | PPAP submission checklist, KPI definitions |

Each chunk enriched at ingestion with:
- Contextual retrieval context (Haiku, T=0) — Anthropic 2024 pattern
- Summary (1 sentence)
- 3 practitioner queries (domain vocabulary)

---

## Dataset

Generated deterministically (seed=42) with `generate_supplier_data.py`.

| Table | Rows | Description |
|---|---|---|
| `suppliers` | 1,200 | Master — 20 cols, 10 product families, 8 archetypes, 21 countries |
| `supplier_kpis` | 43,200 | 36-month time series — PPM, OTD, audit score, SCAR count, risk label |
| `claims` | 20,886 | 8D-structured, QR/PD/CA/CI phases, chargeback amount |
| `apqp_projects` | 1,431 | Full 9-phase gate per internal APQP schema |
| `audits` | 6,581 | ~5.5 audits per supplier, finding types |
| `risk_scores` | 1,200 | Composite score, 3-tier label, recommended action |
| `external_events` | 913 | ESG, sanctions, geopolitical, regulatory, financial |

**8 supplier archetypes:** stable_green, slow_decline, improving, new_supplier, critical_single_source, chronic_underperformer, recovery, high_risk_high_spend

---

## Stack

```
UI:           Streamlit ≥ 1.32
ML:           scikit-learn, xgboost, shap, pandas, numpy, plotly
RAG:          chromadb, openai (text-embedding-3-small), rank-bm25
LLM:          anthropic (Haiku — ingestion enrichment)
              litellm → groq/openai/gpt-oss-120b (answer + HyDE)
              litellm → groq/openai/gpt-oss-20b (groundedness checker)
Observability: langfuse
Validation:   pydantic
Resilience:   tenacity
Package mgr:  uv
DB:           SQLite (dev)
```

---

## Setup

```bash
# Clone
git clone https://github.com/kolmag/sicc.git
cd sicc

# Install dependencies
uv sync

# Environment variables
cp env.example .env
# Fill in: ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY,
#          LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

# Generate synthetic dataset
uv run python generate_supplier_data.py --out data/

# Train ML model (RF + XGBoost comparison, ~2 min on M1)
uv run python ml/train_risk_model.py

# Ingest knowledge base (~16 min — Haiku contextual retrieval per chunk)
uv run python scripts/ingest.py --reset

# Run the app
uv run streamlit run app.py
```

---

## Diagnostics

```bash
# Embedding space visualisation (2D + 3D t-SNE)
uv run python scripts/diagnostics/tsne_viz.py --color doc_type --out tsne_doc_type.html
uv run python scripts/diagnostics/tsne_viz.py --color risk_domain --out tsne_risk_domain.html

# Pre/post reranker similarity chart
uv run python scripts/diagnostics/sc_viz.py --question "What does PPAP Level 3 require?"
```

---

## Portfolio Context

| App | Description | Key techniques |
|---|---|---|
| App 1: CAPA/8D Expert | 8D problem-solving assistant | RAG, ChromaDB |
| App 2: 8D Expert Workbench | Structured 8D workflow tool | RAG, structured output |
| App 3: Auditor Expert | ISO audit Q&A · 8.03/10 dev eval · MRR 0.903 | RAG, BGE reranker, HyDE, eval framework |
| **App 4: SICC** | **Supplier portfolio intelligence** | **ML + SHAP + hybrid RAG + agentic pipeline** |

SICC fills the ML + agentic gap in the portfolio. Apps 1–3 are all RAG Q&A. SICC demonstrates ML at scale with explainability, hybrid retrieval, and decision support across 1,200 suppliers simultaneously.

---

## What's next (Phases 5–6)

- **Phase 5:** Complete full evaluation run — developer, practitioner, practitioner-blind, adversarial · checkpointed Colab execution · MRR / NDCG / judged scores
- **Phase 6:** `PRODUCTION_ARCHITECTURE.md` · `LESSONS_LEARNED_SICC.md` · pyproject.toml finalisation

---
