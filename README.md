# ⬡ SICC — Supplier Intelligence Command Center

**App 4 of 4** 

SICC is a production-grade supplier portfolio intelligence platform combining ML risk scoring, RAG-powered Q&A, SHAP explainability, and executive decision support across a portfolio of 1,200 suppliers.

It ships with two interfaces that share the same validated AI brain:
- **Next.js web app** — production dashboard with authentication, supplier comparison, streaming chat, CSV export, and Docker Compose deploy
- **Streamlit app** — the validated baseline used during development and evaluation

---

## What it does

| Capability | Technology |
|---|---|
| Risk scoring with explainability | RandomForest + XGBoost comparison · SHAP waterfall per supplier |
| Semantic Q&A over supplier quality standards | Hybrid BM25 + embedding retrieval · BGE reranker · HyDE |
| Structured grounded answers | OSS-120B generator · OSS-20B groundedness checker · Pydantic output |
| Portfolio data Q&A | Intent classifier → structured SQLite queries → tabular results |
| Supplier comparison | Side-by-side metrics, KPI trends, SHAP features (up to 3 suppliers) |
| Agentic supplier intake | Supplier dossier → gap analysis → development brief → actions/evidence/exit criteria |
| Agentic early warning alerts | KPI trend drift + events + claims + APQP delays → supplier deterioration watchlist |
| Agentic SCAR/CAPA triage | Claim/manual issue → finding grade → escalation level → containment/evidence/deadlines |
| Agentic APQP launch readiness | APQP gates + PPAP + supplier risk → go / conditional go / hold decision |
| Agentic single-source continuity | Single-source exposure → buffer target → BCP controls → dual-source urgency |
| Agentic audit planning | For-cause triggers → audit type → scope/checklist/evidence plan |
| Shared agent memory | Pydantic-validated SQLite memory + run history + evidence/run-log exports → cross-agent command center |
| Portfolio-level executive overview | KPI summary cards · ML prediction badges · rule-based composite score |
| APQP/NPI programme governance | 9-phase gate tracker · delay detection |
| Scenario simulation | Outage, cost increase, region disruption impact modelling |
| Embedding space diagnostics | t-SNE 2D + 3D · pre/post reranker similarity chart |

---

## Architecture

```
SICC
├── app.py                        # Streamlit entry point
│
├── web/                          # Next.js production frontend
│   ├── app/
│   │   ├── dashboard/            # Supplier risk dashboard (server component)
│   │   │   ├── [supplierId]/     # Supplier detail: KPIs, SHAP, claims, audits, APQP
│   │   │   └── compare/          # Side-by-side supplier comparison (up to 3)
│   │   ├── chat/                 # Dual-mode chat: KB Q&A (SSE streaming) + portfolio data Q&A
│   │   ├── login/                # Password gate
│   │   └── api/auth/             # Session cookie route handler
│   ├── components/
│   │   ├── SupplierTable.tsx     # Filterable table with sparklines + compare selection
│   │   ├── KpiCharts.tsx         # Recharts line charts (PPM, OTD, Audit, SCARs)
│   │   ├── ShapChart.tsx         # SHAP feature importance bar chart
│   │   ├── FeatureImportanceChart.tsx  # Global mean |SHAP| for RED risk
│   │   ├── ApqpGates.tsx         # 9-phase APQP gate tracker
│   │   ├── ExternalEvents.tsx    # ESG/geopolitical event log
│   │   ├── Sparkline.tsx         # Inline SVG PPM trend sparklines
│   │   └── RiskBadge.tsx         # RED / AMBER / GREEN badge
│   ├── proxy.ts                  # Auth gate (Next.js 16 Proxy)
│   └── Dockerfile
│
├── scripts/
│   ├── api.py                    # FastAPI layer — thin wrapper around validated pipeline
│   │   # Routes: /suppliers, /suppliers/{id}, /suppliers/compare,
│   │   #         /suppliers/sparklines, /chat, /chat/stream, /chat/portfolio,
│   │   #         /model/metrics, /model/feature-importance
│   ├── portfolio_qa.py           # Intent classifier → structured SQLite/pandas queries
│   ├── answer.py                 # RAG pipeline: HyDE → BM25+semantic → RRF → BGE → LLM
│   ├── ingest.py                 # KB ingestion: contextual retrieval + typed chunking
│   ├── generate_supplier_data.py # Deterministic data generator (seed=42)
│   ├── supplier_intake_agent.py
│   ├── supplier_alert_agent.py
│   ├── scar_capa_agent.py
│   ├── apqp_readiness_agent.py
│   ├── continuity_agent.py
│   ├── audit_planning_agent.py
│   ├── agent_memory.py
│   └── diagnostics/
│       ├── tsne_viz.py
│       └── sc_viz.py
│
├── sicc_pages/                   # Streamlit pages (one module per page)
├── utils/                        # Shared helpers
├── ml/
│   ├── train_risk_model.py
│   └── training_report.md
├── knowledge-base/markdown/      # 16 supplier quality KB documents
├── evaluation/
├── data/                         # gitignored — SQLite DB (7 tables, 75,411 rows)
├── docker-compose.yml
└── Dockerfile.api
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

## Portfolio Q&A Pipeline

The chat interface has two modes:

**KB mode** — streams answers from the RAG pipeline above via SSE.

**Portfolio mode** — structured data queries over the SQLite database:

```
Question
  │
  ▼
Intent Classifier (OSS-120B via Groq, T=0)
  → 10 intents: red_risk · single_source · ppm_threshold · otd_performance ·
                metric_ranking · audit_findings · claim_categories ·
                capa_events · geopolitical · apqp_delayed
  │
  ▼
Structured pandas/SQLite Query
  │
  ▼
Tabular result (clickable rows → supplier detail · CSV export)
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

**273 chunks** across 16 supplier quality documents. Three chunking strategies by document type:

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
Web UI:       Next.js 16 · React 19 · TypeScript · Tailwind CSS v4 · shadcn/ui
Charts:       Recharts (KPI trends, SHAP) · inline SVG (sparklines)
Streamlit:    ≥ 1.32 (validated baseline)
API:          FastAPI · uvicorn · SSE streaming
ML:           scikit-learn, xgboost, shap, pandas, numpy, plotly
RAG:          chromadb, openai (text-embedding-3-small), rank-bm25
LLM:          anthropic (Haiku — ingestion enrichment)
              litellm → groq/openai/gpt-oss-120b (answer + HyDE + intent)
              litellm → groq/openai/gpt-oss-20b (groundedness checker)
Observability: langfuse
Validation:   pydantic
Resilience:   tenacity
Package mgr:  uv
DB:           SQLite
Deploy:       Docker Compose
```

---

## Setup

### Option A — Docker Compose (recommended)

```bash
git clone https://github.com/kolmag/sicc.git
cd sicc

cp .env.example .env
# Fill in: GROQ_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY
# Optional: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
# Optional: SICC_PASSWORD (default: sicc2025)

docker compose up --build
```

Open `http://localhost:3000`. The API runs on port 8000.

> **Note:** `data/`, `ml/`, and `chroma_db/` must be pre-populated (see Option B). They are mounted as read-only volumes — rebuild not required when data changes.

### Option B — Local development

```bash
git clone https://github.com/kolmag/sicc.git
cd sicc

# Python environment
uv sync

cp .env.example .env
# Fill in API keys

# Generate synthetic dataset
uv run python scripts/generate_supplier_data.py --out data/

# Train ML model (RF + XGBoost comparison, ~2 min on M1)
uv run python ml/train_risk_model.py

# Ingest knowledge base (~16 min — Haiku contextual retrieval per chunk)
uv run python scripts/ingest.py --reset

# Start the API
uv run sicc-api          # FastAPI on :8000

# Start the web UI (separate terminal)
cd web
npm install
npm run dev              # Next.js on :3000

# Or run the Streamlit baseline
uv run streamlit run app.py
```

---

## Hugging Face Playground

SICC also includes a small playground for practicing with Hugging Face Inference Providers via the OpenAI-compatible router. Hugging Face includes small monthly free credits for experimentation; add a token with Inference Providers permission before running it.

```bash
HUGGINGFACE_API_KEY=hf_...

uv run python scripts/hf_playground.py --challenge supplier-risk
uv run python scripts/hf_playground.py --prompt "Explain PPAP Level 3 in 5 bullets"
uv run python scripts/hf_playground.py \
  --models Qwen/Qwen3-4B-Thinking-2507 openai/gpt-oss-20b
```

Built-in challenges: `supplier-risk`, `scar-triage`, `prompt-doctor`, `red-team`.

---

## Diagnostics

```bash
uv run python scripts/diagnostics/tsne_viz.py --color doc_type --out tsne_doc_type.html
uv run python scripts/diagnostics/tsne_viz.py --color risk_domain --out tsne_risk_domain.html
uv run python scripts/diagnostics/sc_viz.py --question "What does PPAP Level 3 require?"
```

---

## Evaluation

Full evaluation run — 280 questions across 4 sets, judged by Claude Sonnet 4.6.

### Retrieval (developer set)

| Metric | Score |
|---|---|
| MRR | 0.9299 |
| NDCG@7 | 0.941 |

### Answer quality

| Set | Judge composite | Questions answered |
|---|---|---|
| Developer | 0.772 | 52 / 80 |
| Practitioner | 0.664 | 22 / 80 |
| Practitioner blind | 0.583 | 8 / 80 |

### Adversarial robustness

| Test | Result |
|---|---|
| Prompt injection block rate | 100% |
| Out-of-scope → insufficient evidence | 100% |
| Ambiguous query handling | 45.5% → fixed post-eval with broad ambiguous patterns |

Full methodology and gap analysis in [`LESSONS_LEARNED_SICC.md`](LESSONS_LEARNED_SICC.md). Production migration path in [`PRODUCTION_ARCHITECTURE.md`](PRODUCTION_ARCHITECTURE.md).

---

## Portfolio Context

| App | Description | Key techniques |
|---|---|---|
| App 1: CAPA/8D Expert | 8D problem-solving assistant | RAG, ChromaDB |
| App 2: 8D Expert Workbench | Structured 8D workflow tool | RAG, structured output |
| App 3: Auditor Expert | ISO audit Q&A · 8.03/10 dev eval · MRR 0.903 | RAG, BGE reranker, HyDE, eval framework |
| **App 4: SICC** | **Supplier portfolio intelligence** | **ML + SHAP + hybrid RAG + agentic pipeline + Next.js production UI** |

SICC fills the ML + agentic gap in the portfolio. Apps 1–3 are all RAG Q&A. SICC demonstrates ML at scale with explainability, hybrid retrieval, decision support across 1,200 suppliers simultaneously, and a production-ready web interface.
