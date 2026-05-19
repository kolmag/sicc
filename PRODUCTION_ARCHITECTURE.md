# SICC — Production Architecture

This document describes what would need to change to operate SICC as a production internal tool — deployed for SQE, procurement, and supply chain leadership at a manufacturing organisation. It covers component-by-component migration, what's deliberately deferred to Phase 7+, and infrastructure decisions.

---

## Current State vs Production Target

| Component | Current (dev/portfolio) | Production target |
|---|---|---|
| Database | SQLite — 1 file, 75K rows | PostgreSQL (managed) |
| Vector store | ChromaDB local `PersistentClient` | Qdrant Cloud or Weaviate Cloud |
| UI | Streamlit validated baseline + optional FastAPI wrapper | Streamlit Cloud (internal tool) or FastAPI + React |
| LLM — answer + HyDE | Groq OSS-120B (no SLA) | Groq (speed-critical) + Anthropic fallback |
| LLM — groundedness | Groq OSS-20B (no SLA) | Anthropic Claude Haiku (consistent, SLA'd) |
| LLM — KB enrichment | Anthropic Haiku (one-off at ingest) | Same — run once per KB update |
| Secrets | `.env` file | AWS Secrets Manager or Azure Key Vault |
| Auth | None | SSO via SAML/OIDC (Okta, Azure AD) |
| CI/CD | None | GitHub Actions |
| Observability | Langfuse traces | Langfuse Cloud + custom alerting |
| Data source | Synthetic SQLite (seed=42) | Live ERP/MES integration or nightly ETL |
| ML artefacts | Local `.pkl` files | S3 / Azure Blob + versioning |
| Package management | `uv` + `pyproject.toml` | Same — containerised with Docker |

---

## Architecture Diagram — Production

```
Browser (SQE / procurement / leadership)
  │
  ▼
[Auth layer — SSO SAML/OIDC]
  │
  ▼
[Streamlit Cloud / App Server]
  ├── sicc_pages/        — page modules (no change required)
  ├── utils/             — shared helpers (no change required)
  ├── scripts/api.py     — optional thin API wrapper for React/Next.js
  └── scripts/answer.py  — RAG pipeline (LLM clients point to prod)
        │
        ├─► [PostgreSQL] — supplier_portfolio schema
        │     7 tables, live data via nightly ETL from ERP
        │
        ├─► [Qdrant Cloud] — sicc_kb collection
        │     264 chunks (re-ingested on KB update)
        │     embedding: text-embedding-3-small (OpenAI)
        │
        ├─► [Groq API] — OSS-120B (answer + HyDE)
        │   fallback: [Anthropic API] — claude-sonnet-4-x
        │
        ├─► [Anthropic API] — Claude Haiku (groundedness checker)
        │     replaces OSS-20B — more consistent, SLA'd
        │
        └─► [Langfuse Cloud] — traces, spans, evals
              custom dashboards: answer quality, latency, cost
```

The first API slice is intentionally thin: `scripts/api.py` exposes `/health`,
`/chat`, and `/chat/stream` while calling the same `scripts.answer.answer()`
function used by the Streamlit Q&A page. This supports a future Next.js
interface without disturbing the validated Streamlit path.

---

## Component Migration — Detail

### Database: SQLite → PostgreSQL

**What changes:**
- Connection string in `utils/config.py` and `utils/data.py`
- Replace `sqlite3` with `psycopg2` or `asyncpg`
- `load_all_data()` in `utils/data.py` becomes queries against managed Postgres
- Agent memory SQLite (`agent_memory.py`) migrates to the same Postgres instance

**What stays the same:**
- All 7 table schemas are already normalised and portable
- `pandas.read_sql()` works identically against Postgres
- SQLite's deterministic seed-42 data can be loaded to Postgres once as a starting point

**Why not ORM:**
The query surface is read-heavy and well-defined. `pd.read_sql()` with parameterised queries is sufficient and avoids the overhead of an ORM migration on an already working schema.

**Managed options by deployment:**
- AWS: RDS PostgreSQL (Multi-AZ for HA) or Aurora Serverless
- Azure: Azure Database for PostgreSQL Flexible Server
- Self-hosted: Supabase (Postgres + REST API) if the organisation has no cloud DB preference

---

### Vector Store: ChromaDB → Qdrant Cloud

**Why replace ChromaDB:**
- `PersistentClient` is single-node, single-process — not suitable for concurrent users
- The `chroma_noop_telemetry` import workaround creates a hard dependency on module path
- ChromaDB's managed offering (`chromadb.cloud`) is still early; Qdrant Cloud is production-mature

**What changes:**
- `scripts/answer.py`: replace `chromadb.PersistentClient` with `qdrant_client.QdrantClient`
- `scripts/ingest.py`: replace `collection.upsert()` with `qdrant_client.upsert()`
- Re-ingest all 264 chunks to Qdrant — one-time migration (~16 min)
- BM25 index is already separate (built from collection at query time) — no change

**What stays the same:**
- Embedding model (`text-embedding-3-small`) — unchanged
- RRF fusion, BGE reranker, chunk ordering — all model-agnostic
- All chunk metadata schema — payload structure in Qdrant mirrors ChromaDB metadata

**Alternative: Pinecone**
- Better managed experience, easier scaling
- Proprietary metadata filtering (not Chroma-compatible)
- Higher per-query cost at scale
- Qdrant preferred for self-hosted fallback option

---

### LLM: OSS-20B Groundedness Checker → Claude Haiku

**Why replace OSS-20B for groundedness:**
- OSS-20B intermittently returns short non-informative responses ("Correct.") causing false insufficient-evidence classifications (discovered during eval — see `LESSONS_LEARNED_SICC.md`)
- Groq OSS models have no SLA — acceptable for speed-critical generation, not for a correctness gate
- Claude Haiku is deterministic, fast (200ms median), consistent, and SLA'd via Anthropic's API
- Cost: Haiku is cheap enough for the checker role ($0.25/M input tokens)

**What changes in `answer.py`:**
```python
# Current
GROQ_20B = "groq/openai/gpt-oss-20b"
# Production
CHECKER_MODEL = "anthropic/claude-haiku-4-5-20251001"
```

The `check_groundedness()` call switches to the Anthropic client via LiteLLM — one-line change. The prompt is compatible.

**Keep Groq for generation (OSS-120B):**
- Median latency: ~7s for generation (Groq A100)
- Groq's speed advantage is highest value for the answer generation step
- Anthropic Sonnet as fallback if Groq is unavailable (tenacity retry already in place)

---

### Authentication: SSO via SAML/OIDC

**Roles to define:**

| Role | Access | Pages |
|---|---|---|
| SQE | Full read + agent runs | All pages |
| Procurement | Read supplier risk + alerts | Executive Portfolio, Risk Scoring, Early Warning, What-If |
| Leadership | Dashboard read-only | Executive Portfolio, APQP Tracker |
| Admin | All + KB management | All + ingest trigger |

**Streamlit implementation:**
- Streamlit Cloud has native SSO via Google/SAML — simplest path for internal tool
- Custom auth: `streamlit-authenticator` library wraps JWT tokens
- For enterprise: Okta or Azure AD SSO — configure as SAML IdP

**What changes in `app.py`:**
- Add auth check at top of `app.py` before any data loading
- Role check before agent run buttons (SQE-only)
- Audit log: who ran which agent, when (existing agent memory infrastructure can extend this)

---

### CI/CD: GitHub Actions

**Pipeline triggers:**

```yaml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    - uv run pytest tests/           # unit tests (agent memory, pipeline logic)
    - uv run python ml/train_risk_model.py --dry-run  # schema validation
    - uv run python scripts/answer.py --question "What is PPAP Level 3?"  # smoke test

  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    - Build Docker image
    - Push to container registry
    - Deploy to Streamlit Cloud / app server
```

**KB update pipeline (separate, manual trigger):**
```yaml
on:
  workflow_dispatch:
    inputs:
      file: { description: 'KB file to re-ingest', required: false }

jobs:
  ingest:
    - uv run python scripts/ingest.py [--file ${{ inputs.file }}]
    - Verify collection count increased
    - Run retrieval smoke test (5 known questions, MRR check)
```

---

### Data: Synthetic → Live ERP Integration

**What changes:**
- `scripts/generate_supplier_data.py` is retired for production
- `utils/data.py` connects to Postgres populated by nightly ETL from ERP (SAP, Oracle, or custom MES)
- ETL responsibility: data engineering team — outside SICC scope
- SICC requires these tables to be populated externally: `suppliers`, `supplier_kpis`, `claims`, `apqp_projects`, `audits`, `risk_scores`, `external_events`

**ETL contract (what SICC expects):**

| Table | Update frequency | Key requirements |
|---|---|---|
| `suppliers` | On change | `supplier_id` as stable PK |
| `supplier_kpis` | Monthly (per KPI report cycle) | 36-month trailing history required for ML features |
| `claims` | On event | 8D phase tracking required |
| `apqp_projects` | On milestone | 9-phase schema required |
| `audits` | On audit completion | Finding counts and scores required |
| `risk_scores` | After ML model runs | Run after KPI refresh |
| `external_events` | From ESG/sanctions feed | Severity levels required |

**ML model retraining:**
- Trigger: quarterly or when KPI data distribution shifts significantly
- Monitored by: SHAP value drift (most important features changing weight unexpectedly)
- Artefacts: stored in S3 with version tags (`model_v1.2.pkl`) — `utils/ml.py` loads by version

---

### Observability: Langfuse + Alerting

**Already in place:**
- Full trace coverage via `@observe` decorators on all RAG pipeline steps
- Per-step latency, token counts, and model calls logged

**What to add for production:**

| Alert | Threshold | Action |
|---|---|---|
| Answer latency P95 | > 15s | Page on-call SQE + investigate Groq |
| Insufficient evidence rate | > 40% over 1h | KB coverage review |
| Groundedness checker failure rate | > 5% over 1h | Switch to Anthropic checker |
| Groq API errors | > 10% over 5 min | Activate Anthropic fallback |
| Agent run failure | Any | Slack notification to SQE team |

**Custom Langfuse dashboards to build:**
- Daily answer quality score (% high confidence answers)
- Most-asked questions (for KB gap identification)
- Agent run frequency by type (which agents are actually used)
- Cost per session and per agent run

---

### ML Model: Operational Requirements

**Current:**
- Model trained offline (`ml/train_risk_model.py`), artefacts saved to disk
- Re-run manually when needed

**Production:**
- Scheduled retraining: monthly, triggered by Airflow or GitHub Actions cron
- Feature engineering pipeline: `train_risk_model.py` reads from Postgres (not SQLite)
- Artefact versioning: S3 with model registry (MLflow or simple tag-based scheme)
- A/B testing: run new model in shadow mode for 30 days before replacing
- Monotonicity tests: 5 tests currently passing — must pass before any model goes to production

---

## Estimated Migration Effort

| Work item | Effort | Owner |
|---|---|---|
| SQLite → Postgres (schema + ETL contract) | 2 weeks | Backend engineer + data engineering |
| ChromaDB → Qdrant Cloud | 3 days | ML engineer |
| OSS-20B → Haiku (groundedness) | 1 day | ML engineer |
| SSO auth integration | 1 week | Backend engineer |
| CI/CD pipeline | 3 days | DevOps / ML engineer |
| Docker containerisation | 2 days | DevOps |
| Observability alerting | 2 days | ML engineer |
| ML retraining pipeline | 1 week | ML engineer |
| Live ERP integration | 4–8 weeks | Data engineering team |
| **Total (excl. ERP)** | **~5–6 weeks** | 2 engineers |

ERP integration is the longest item and depends entirely on what the organisation's ERP exposes. SICC's internal architecture requires no changes for this — only the ETL and the Postgres schema contract.

---

## What Stays the Same

The following components are already production-quality and require no changes:

- `sicc_pages/` — all 12 page modules
- `utils/` — all 6 shared helper modules
- `scripts/answer.py` — RAG pipeline logic (only client swap)
- `scripts/ingest.py` — KB ingestion pipeline
- `scripts/agent_*.py` — all 6 agents
- `scripts/agent_memory.py` — Pydantic-validated SQLite memory (migrates to Postgres)
- `ml/train_risk_model.py` — training script (replace DB connection only)
- `evaluation/` — eval harness is reusable for production monitoring

---

## Out of Scope for Phase 7

- Multi-tenancy (multiple organisations) — requires tenant isolation in Postgres and separate ChromaDB/Qdrant collections
- Mobile / native app — Streamlit is web-only; React Native would require a full API layer
- Real-time streaming — KPI updates are batch (monthly); streaming would require Kafka and incremental ML
- On-premises deployment — possible (all components have self-hosted options) but requires infrastructure team
