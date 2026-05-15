# SICC — Lessons Learned

Engineering retrospective on building a production-grade supplier intelligence system combining ML risk scoring, hybrid RAG, SHAP explainability, and 6 agentic workflows. Written after the full evaluation run (MRR 0.9299, NDCG 0.941) and post-eval gap analysis.

---

## What Worked Well

### 1. Hybrid Retrieval Outperformed Either Method Alone

BM25 + semantic + RRF fusion was the right call. BM25 handles exact-term lookups (clause numbers, PPAP level numbers, KPI names with specific values like "PPM > 500") that semantic search misses. Semantic search handles paraphrasing and vocabulary variation that BM25 misses. RRF fusion with k=60 produced stable rankings without requiring weight tuning.

Evidence: categories that rely on exact terminology (ISO clause numbers, PPAP element numbers) scored MRR 1.0. If we'd used semantic-only, those would have been weaker.

**Would repeat:** Hybrid retrieval with RRF is the correct default for any domain-specific RAG system.

---

### 2. FINDING: Examples as Retrieval Anchors

The most impactful KB design decision was using labelled `FINDING:` examples in every document. Categories with rich FINDING examples (SCAR=1.0, single_source=0.967, ISO 9001=0.917) vastly outperformed categories with sparse examples (audit=0.454, qualification=0.525).

Why it works: FINDING examples embed domain vocabulary in a narrative context that matches how practitioners ask questions. "A connector pin contamination issue was traced to..." retrieves for questions phrased as "what do I do when I find..." — semantic overlap the plain procedure text doesn't have.

**Would do differently:** Write FINDING examples first for every document, not as an afterthought. The KB documents that scored poorly were written as reference procedures without practitioner-style scenarios.

---

### 3. SHAP Explainability Was Essential for ML Trust

The SHAP waterfall chart per supplier is the feature that makes the ML score credible to a non-technical SQE. Without it, "ML predicts RED" is a black box. With SHAP, you see "PPM slope -11.88/month (pushing GREEN) vs SCAR count (pushing RED)" — actionable.

The SHAP monotonicity tests (5/5 passing) were also unexpectedly useful during development — they caught two feature engineering bugs where the slope calculation was signed incorrectly.

**Would repeat:** SHAP + monotonicity tests before any model goes to production. The tests take 2 minutes to run and catch silent feature bugs.

---

### 4. RandomForest Beat XGBoost on F1-Red (the Only Metric That Mattered)

The model comparison ran both RF and XGBoost against 7 metrics. XGBoost had higher accuracy overall. RandomForest had higher F1-Red (0.875 vs approximately 0.850). F1-Red was the primary selection criterion because missing a RED supplier is far more costly than a false alarm.

Why RF won: the RED class is the smallest class (genuine poor performers are rare). RF's bootstrap sampling naturally handles class imbalance by seeing different subsets per tree. XGBoost's boosting focuses on reducing overall loss, which is dominated by the majority class.

**Next step in production:** Try class-weighted XGBoost (`scale_pos_weight` parameter) before concluding RF is permanently better. The gap may close.

---

### 5. Pydantic Structured Output Prevented Downstream Failures

Every agent and the RAG pipeline returns a Pydantic model. This caught:
- Empty answer strings (validator rejected them before they could render in the UI)
- Missing `confidence` fields (required literal `"high"/"medium"/"low"`)
- Invalid risk level values

Without Pydantic, these would have been silent runtime errors in the Streamlit UI — difficult to debug and invisible to the user.

**Would repeat:** Pydantic output models from the start, not added retrospectively.

---

### 6. Contextual Retrieval Enrichment at Ingest Time

Each chunk is enriched at ingest with: (a) contextual retrieval context summarising where the chunk sits in the document, (b) a 1-sentence summary, (c) 3 practitioner queries. This enriched text is what gets embedded — not just the raw chunk.

The result: chunks are retrievable via multiple vocabulary paths (the chunk's own terms + the enriched queries). The `supplier_kpi_definitions.md` chunks, for example, are retrieved for "what triggers a RED classification" even though the document doesn't use the word "triggers".

**Cost:** ~$2–3 of Haiku tokens for the full 264-chunk collection. Worth it.

---

### 7. Tenacity Retries Were Essential

Groq's API has intermittent rate limits and occasional 500 errors. Without tenacity retries (`stop_after_attempt(3)`, `wait_exponential`), roughly 5–8% of pipeline calls would fail silently during the eval run. With retries, the eval ran clean with 0 judge errors.

**Would repeat:** All LLM calls need retry logic in any production system. The cost of retry latency (a few extra seconds on failure) is always less than the cost of a silent failure.

---

## What Didn't Work / Would Do Differently

### 1. OSS-20B Groundedness Checker Brittleness

The biggest unexpected bug. OSS-20B (via Groq) intermittently returns single-word responses like "Correct." instead of the full cleaned answer. The `< 10 char` empty-answer guard then converted this to `INSUFFICIENT_EVIDENCE_MARKER` — silently making answered questions appear unanswerable.

This bug was masked during development because individual test runs looked fine (the model was usually correct). It was only exposed during the systematic eval run, where a consistent question set showed the non-determinism.

**Root causes identified:**
1. The `"\n\n\n"` stop sequence cut multi-section answers prematurely
2. The system prompt said "Return only the cleaned answer" — OSS-20B interpreted "return" as "verify" and said "Correct." instead of echoing the answer
3. Temperature=0 is not perfectly deterministic on hosted APIs

**Fix applied:** Removed `"\n\n\n"` stop. Updated prompt to explicitly say "always return the full cleaned answer text, never a single word". Added fallback: any response < 80 chars that doesn't explicitly say `INSUFFICIENT_EVIDENCE` is treated as a grounded answer (raw answer returned).

**Lesson:** The groundedness checker is a correctness gate. It must be held to a higher consistency standard than the generator. In production, replace OSS-20B with Claude Haiku — more consistent, SLA'd, cheaper per token than OSS-120B.

---

### 2. KB Written for Reference, Not for Practitioners

The practitioner set returned `insufficient_evidence` for 72.5% of questions. These questions were phrased as operational queries ("a supplier just shipped 400 defective parts — what do I do first?") rather than reference queries ("what is the SCAR containment timeline?").

The KB was written as a quality management reference manual. Practitioners don't ask reference questions — they ask situational questions. The retrieval worked (right chunks surfaced), but the generator had nothing to ground specific operational guidance on.

**Fix applied:** Added practitioner-style FINDING examples to 4 documents. Immediate improvement on targeted tests.

**Lesson:** For a practitioner-facing RAG system, every document needs at least one FINDING example phrased in the vocabulary a practitioner would use when facing that situation. Write the user's question, then write the document section that would answer it — not the other way around.

---

### 3. Ambiguous Query Handling Too Narrow

The initial `AMBIGUOUS_CONTEXT_PATTERNS` list only caught exact short-form questions ("What is the deadline?", "What should I do?"). Broader ambiguous questions with domain terms but no specific referent ("Should we be worried about this supplier?", "Is this acceptable?") slipped through and received high-confidence answers.

The adversarial ambiguous pass rate was 45.5% before the fix. The patterns cover common forms but can't be exhaustive.

**Fix applied:** Added `BROAD_AMBIGUOUS_PATTERNS` for question shapes that indicate missing referent (starts with "should we", "is this", "are we", "what happens next"). These now run the pipeline but cap confidence at `low`.

**Lesson:** Ambiguity detection in RAG is an open problem. The patterns approach scales poorly — a classifier trained on supplier quality queries would be more robust. For production, consider a fast pre-classifier (e.g., a small BERT fine-tuned on domain ambiguous vs specific questions) before the HyDE step.

---

### 4. app.py Grew to 3,874 Lines Before the Split

The monolithic `app.py` was functional but unmaintainable. Every new page feature required navigating ~400 lines of existing code. The refactor to `sicc_pages/` + `utils/` was correct but should have been the initial architecture.

**Lesson:** For any Streamlit app with more than 3 pages, start with the `sicc_pages/` module pattern from day one. The `render(**ctx)` contract is simple and the payoff in maintainability is immediate.

---

### 5. ChromaDB Settings String Created a Fragile Dependency

The telemetry noop requires:
```python
chroma_product_telemetry_impl="scripts.chroma_noop_telemetry.NoopTelemetry"
```
ChromaDB resolves this string via `importlib`, which requires `scripts` to be importable as a package from sys.path. This made `ingest.py` fail with a cryptic `ModuleNotFoundError` when run without the correct sys.path setup (the bug that broke the first re-ingest attempt).

**Lesson:** Any library that resolves class names via string-based importlib creates a hidden runtime dependency. Document these explicitly, add the sys.path setup at the top of every script that uses it, and test from a clean working directory before every release.

---

### 6. Eval Insufficient Evidence Rate Is Misleading

The eval judge only scores questions that returned an answer — `insufficient_evidence` questions are excluded from the composite score. This means the developer composite of 0.772 is calculated over 52/80 questions. The 28 that returned `insufficient_evidence` are a hidden quality signal that isn't in the headline number.

Before the groundedness checker fix, some of those 28 questions had answers in the KB that were being silently dropped. The eval metric didn't surface this — it just showed a lower question count.

**Lesson:** Always report both the judge composite AND the insufficient_evidence rate together. A system with a 0.9 composite on 30% of questions is worse than a system with a 0.75 composite on 80% of questions.

---

### 7. Synthetic Data Is Both a Strength and a Limitation

Deterministic synthetic data (seed=42) was the right call for a portfolio project:
- Eval is reproducible — same questions, same data, same scores across runs
- No data privacy concerns
- The 8 supplier archetypes cover the interesting ML edge cases (slow_decline, recovery, critical_single_source)

The limitation: the synthetic KPIs are smooth and trend-consistent. Real supplier data has seasonal effects, batch quality events, supplier-initiated quality improvements mid-period, and sub-tier disruptions. The ML model may behave differently on real data — specifically, the volatility and threshold-breach features may be noisier.

**For production:** Train on 12–18 months of real historical KPIs from the ERP before deploying the ML model. Run the monotonicity tests on real data — if any fail, the feature engineering needs adjustment.

---

## Key Architectural Decisions

### Why Not Streamlit's Native `pages/` Directory

Streamlit's `pages/` directory creates separate page scripts with independent execution contexts. This means the global sidebar filters (product family, region, spend tier) would need to be duplicated and re-evaluated on every page. Our `sicc_pages/` + `render(**ctx)` pattern passes pre-filtered data into each page, making the filters a single authoritative computation.

### Why HyDE Rather Than Query Expansion

HyDE (Hypothetical Document Embeddings) generates a document excerpt that *would* answer the question, then embeds that excerpt. The advantage over query expansion: the embedding space of "a supplier quality procedure document about PPAP" is much closer to the actual chunk embeddings than the embedding of the original question. This is particularly effective in supplier quality, where question vocabulary ("what score deduction") differs significantly from document vocabulary ("Minor NCR — −2 per finding").

Limitation: HyDE sometimes returns the original question unchanged when the question is phrased as a technical instruction the model doesn't recognise. The fallback to the original question is adequate but loses the vocabulary benefit.

### Why 7 Final Chunks After Reranking

FINAL_K=7 was chosen based on the BGE reranker's typical precision ceiling. Below 5 chunks: important context is frequently missing. Above 10 chunks: the lost-in-the-middle effect becomes significant even with chunk ordering, and the context window fills with lower-relevance material. The chunk ordering fix (top chunks at START and END) makes 7 work better than 10 unordered.

### Why SQLite for Agent Memory

Agent memory is low-volume, append-mostly, single-writer. SQLite with Pydantic validation is zero-config, portable, and sufficient. The schema (`agent_runs`, `agent_memory`) is already normalised for Postgres migration. There's no concurrency requirement in the current deployment model (one SQE at a time per session).

---

## Eval Summary

| Set | Key Metric | Score | Notes |
|---|---|---|---|
| Developer — retrieval | MRR | 0.9299 | 7 categories at MRR 1.0 |
| Developer — retrieval | NDCG@7 | 0.941 | Strong across all categories |
| Developer — answer | Judge composite | 0.772 | On 52/80 answered questions |
| Practitioner — answer | Judge composite | 0.664 | On 22/80 answered questions |
| Practitioner blind — answer | Judge composite | 0.583 | On 8/80; intentionally hard |
| Adversarial — injection | Block rate | 100% | Pattern match pre-flight |
| Adversarial — OOS | Handled | 100% | Groundedness checker + insufficient evidence |
| Adversarial — ambiguous | Handled | 45.5% | Fixed post-eval with broad ambiguous patterns |

**What the scores mean:** Retrieval is production-quality. Answer quality on reference questions (developer set) is good. Practitioner coverage (72.5% insufficient evidence rate) is the primary gap — addressed by adding FINDING examples and fixing the groundedness checker. The adversarial robustness is strong on the hardest cases (injection, OOS).

---

## What This Project Demonstrates

For the portfolio context: SICC is the only app in the 4-app portfolio with:
1. A trained ML model with explainability (SHAP waterfall per supplier)
2. A systematic retrieval evaluation with MRR/NDCG (not just manual spot checks)
3. Six production-grade agentic workflows with validated memory
4. A formal adversarial eval (injection blocking, OOS handling)
5. A gap analysis → fix → retest cycle

The combination of ML + RAG + agents in one coherent application is the differentiator. Apps 1–3 in the portfolio are all RAG Q&A. SICC demonstrates that the same practitioner can operate at the intersection of ML engineering, RAG design, and agentic architecture.
