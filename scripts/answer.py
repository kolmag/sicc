"""
scripts/answer.py — SICC RAG Answer Pipeline
Auditor Expert pattern + hybrid retrieval + chunk ordering fix + Langfuse

Pipeline:
  1. Pre-flight: prompt injection check (0ms)
  2. Query rewriting: HyDE (Groq OSS-120B)
  3. Hybrid retrieval: BM25 + ChromaDB embedding → RRF fusion
  4. BGE reranker (CPU fallback — GPU on Colab)
  5. Chunk ordering: top chunks at START and END (lost-in-the-middle fix)
  6. Answer generation (Groq OSS-120B, strict grounding)
  7. Groundedness checker (Groq OSS-20B, NLI actor/critic)
  8. Pydantic structured output

Usage:
    uv run python scripts/answer.py --question "What does PPAP Level 3 require?"
    uv run python scripts/answer.py --question "..." --family Electronics --risk red
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
import argparse
import re
import time
from pathlib import Path
from typing import Literal, Optional

import chromadb
from chromadb.config import Settings
from openai import OpenAI as OpenAIClient
from dotenv import load_dotenv
from litellm import completion
from pydantic import BaseModel, Field, field_validator
from rank_bm25 import BM25Okapi
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

LANGFUSE_ENABLED = bool(
    os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
)
if LANGFUSE_ENABLED:
    from langfuse import Langfuse, observe
else:
    Langfuse = None

    def observe(*_args, **_kwargs):
        def decorator(func):
            return func
        return decorator

# ── Constants ─────────────────────────────────────────────────────────────────

COLLECTION_NAME  = "sicc_kb"
EMBED_MODEL      = "text-embedding-3-small"
RETRIEVAL_K      = 20       # candidates from ChromaDB
BM25_K           = 20       # candidates from BM25
RRF_K            = 60       # RRF fusion constant
FINAL_K          = 7        # chunks to generator after reranking

GROQ_120B        = "groq/openai/gpt-oss-120b"    # answer + rewrite
GROQ_20B         = "groq/openai/gpt-oss-20b"   # checker (OSS-20B equivalent)
CHROMA_DB_PATH   = "chroma_db"
CHROMA_SETTINGS  = Settings(
    anonymized_telemetry=False,
    chroma_product_telemetry_impl="scripts.chroma_noop_telemetry.NoopTelemetry",
    chroma_telemetry_impl="scripts.chroma_noop_telemetry.NoopTelemetry",
)

# Prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore .{0,20}instructions",           # covers "ignore all previous instructions" etc.
    r"disregard .{0,20}(prompt|instructions)",
    r"you are now",
    r"new (system|persona|role|instructions)",
    r"jailbreak",
    r"bypass (your|the) (filter|restriction|safety)",
    r"act as (if|a|an|though)",
    r"pretend (you are|to be)",
    r"forget .{0,20}instructions",           # covers "forget your previous instructions"
    r"forget everything",
    r"act as dan",                           # DAN jailbreak specifically
    r"dan mode",
    r"unrestricted mode",
    r"no restrictions",
    r"new persona",
]

INSUFFICIENT_EVIDENCE_MARKER = "_FALLBACK_INSUFFICIENT_EVIDENCE"

AMBIGUOUS_CONTEXT_PATTERNS = [
    r"^\s*what\s+is\s+the\s+deadline\s*\??\s*$",
    r"^\s*what\s+is\s+the\s+timeline\s*\??\s*$",
    r"^\s*when\s+is\s+it\s+due\s*\??\s*$",
    r"^\s*what\s+should\s+i\s+do\s*\??\s*$",
    r"^\s*is\s+(it|this|that)\s+(ok|okay|good|bad|acceptable)\s*\??\s*$",
]

# Broader ambiguous patterns — pipeline runs but confidence is forced to "low"
# These are questions with some content but no specific referent or process context.
BROAD_AMBIGUOUS_PATTERNS = [
    r"^\s*is\s+(this|that|it|the\s+supplier)\s+",      # "Is this acceptable?", "Is the supplier ok?"
    r"^\s*should\s+we\s+",                              # "Should we be worried?", "Should we qualify them?"
    r"^\s*are\s+we\s+(ok|okay|compliant|good|fine|safe|at risk)",  # "Are we compliant?"
    r"^\s*how\s+(bad|serious|worried|concerned|risky)\s+is\s+(this|that|it)",  # "How bad is this?"
    r"^\s*what\s+does\s+(this|that|it)\s+mean",         # "What does this mean for us?"
    r"^\s*what\s+happens\s+next",                       # "What happens next?"
    r"^\s*do\s+we\s+need\s+to\s+",                     # "Do we need to escalate?"
    r"^\s*do\s+i\s+need\s+to\s+",                      # "Do I need to do anything?"
    r"^\s*is\s+(it|this)\s+(ok|okay|fine|acceptable|normal|allowed|expected)\s*\??", # "Is it ok?"
    r"^\s*are\s+(they|them|the\s+supplier)\s+",         # "Are they improving?"
    r"^\s*what\s+do\s+i\s+do\s+(now|next|about\s+this)",  # "What do I do now?"
    r"^\s*how\s+do\s+i\s+handle\s+(this|that|it)\s*\??",  # "How do I handle this?"
]

# ── Pydantic output schema ────────────────────────────────────────────────────

class SupplierQAResult(BaseModel):
    answer:               str
    confidence:           Literal["high", "medium", "low"]
    action_required:      bool
    insufficient_evidence: bool
    sources:              list[str]
    retrieved_sources:    list[str] = Field(default_factory=list)
    retrieved_context:    list[dict] = Field(default_factory=list)
    risk_level:           Optional[Literal["high", "medium", "low", "not_applicable"]] = None

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Answer cannot be empty")
        return v.strip()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SICC RAG Q&A")
    p.add_argument("--question", required=True)
    p.add_argument("--family",   default=None, help="Filter by product family")
    p.add_argument("--risk",     default=None, help="Filter by risk_domain")
    p.add_argument("--db",       default=CHROMA_DB_PATH)
    return p.parse_args()


def build_where_filter(risk: Optional[str] = None, family: Optional[str] = None) -> Optional[dict]:
    """Build a Chroma metadata filter from optional CLI/user filters."""
    clauses = []
    if risk:
        clauses.append({"risk_domain": {"$eq": risk}})
    if family:
        clauses.append({"$or": [
            {"commodity": {"$eq": family}},
            {"commodity": {"$eq": "GENERAL"}},
        ]})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


# ── Clients ───────────────────────────────────────────────────────────────────

def build_clients(db_path: str):
    chroma_client = chromadb.PersistentClient(
        path=os.path.abspath(db_path),
        settings=CHROMA_SETTINGS,
    )
    _oai = OpenAIClient(api_key=os.environ["OPENAI_API_KEY"])

    class EmbedFn(chromadb.EmbeddingFunction):
        def __call__(self, input):
            input = [t for t in input if t and t.strip()]
            if not input:
                return []
            response = _oai.embeddings.create(model=EMBED_MODEL, input=input)
            return [d.embedding for d in response.data]

    embed_fn = EmbedFn()
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    langfuse = (
        Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        if LANGFUSE_ENABLED else None
    )
    return collection, langfuse


# ── Step 1: Pre-flight injection check ───────────────────────────────────────

@observe(name="preflight_injection_check")
def preflight_check(question: str) -> bool:
    """Returns True if question is safe, False if injection detected."""
    q_lower = question.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, q_lower):
            return False
    return True


def is_context_missing_question(question: str) -> bool:
    """
    Detect truly bare/underspecified questions that cannot be answered at all.
    Returns True → pipeline short-circuits to insufficient_evidence.
    For questions that are ambiguous but have some content, use is_broad_ambiguous_question().
    """
    q_lower = " ".join(question.lower().split())
    if any(re.search(pattern, q_lower) for pattern in AMBIGUOUS_CONTEXT_PATTERNS):
        return True

    tokens = re.findall(r"[a-z0-9]+", q_lower)
    domain_terms = {
        "ppap", "apqp", "scar", "capa", "8d", "audit", "ncr", "ppm", "otd",
        "gauge", "grr", "msa", "iso", "iatf", "as9100", "red", "amber",
        "green", "supplier", "containment", "qualification", "requalification",
        "esg", "sanctions", "risk", "inspection", "development", "sop",
    }
    vague_heads = {"deadline", "timeline", "due", "score", "good", "bad"}
    has_domain = any(tok in domain_terms for tok in tokens)
    has_vague_head = any(tok in vague_heads for tok in tokens)
    if {"score", "good"}.issubset(tokens) and not has_domain:
        return True
    return len(tokens) <= 5 and has_vague_head and not has_domain


def is_broad_ambiguous_question(question: str) -> bool:
    """
    Detect questions that are ambiguous (no specific referent or process context)
    but have enough content to attempt an answer. Pipeline runs normally but
    confidence is capped at 'low' in the output.
    """
    if is_context_missing_question(question):
        return False
    q_lower = " ".join(question.lower().split())
    return any(re.search(pattern, q_lower) for pattern in BROAD_AMBIGUOUS_PATTERNS)


# ── Step 2: HyDE query rewriting ─────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
@observe(name="hyde_rewriter")
def rewrite_query_hyde(question: str) -> str:
    """
    HyDE: generate a hypothetical document excerpt that would answer the question.
    Passed to BOTH ChromaDB AND BGE reranker (not original question).
    Groq OSS-120B, T=0.
    Retries with a more explicit prompt if model returns empty or too-short response.
    """
    def _call_hyde(prompt_text: str) -> str:
        response = completion(
            model=GROQ_120B,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0,
            max_tokens=200,
            stop=["```"],
        )
        return response.choices[0].message.content.strip()

    # Primary prompt
    primary_prompt = f"""You are an expert supplier quality engineer writing a knowledge base document.

Write a 2-3 sentence excerpt from a supplier quality procedure document that would directly answer this question. Use the vocabulary of a supplier quality manual: PPM, SCAR, OTD, PPAP, APQP, audit findings, risk tier, corrective action, etc.

Question: {question}

Write only the excerpt — no preamble, no explanation:"""

    hyde_text = _call_hyde(primary_prompt)

    # Fix 1: if empty or too short, retry with a more explicit prompt
    if not hyde_text or len(hyde_text) < 20:
        fallback_prompt = f"""Write exactly 2 sentences from a supplier quality manual that answer this question: {question}

Start your response with a relevant term like 'PPAP', 'SCAR', 'audit', 'supplier', 'corrective action', or 'risk tier'. Be specific and use domain vocabulary."""
        hyde_text = _call_hyde(fallback_prompt)

    # Final fallback: use original question
    if not hyde_text or len(hyde_text) < 20:
        hyde_text = question

    return hyde_text


# ── Step 3a: ChromaDB semantic retrieval ─────────────────────────────────────

@observe(name="semantic_retrieval")
def semantic_retrieval(
    collection: chromadb.Collection,
    hyde_text: str,
    question: str,
    k: int = RETRIEVAL_K,
    where_filter: Optional[dict] = None,
) -> list[dict]:
    """
    Retrieve top-k chunks from ChromaDB using HyDE text as query.
    Returns list of {id, document, metadata, distance, rank}.
    """
    if not hyde_text or not hyde_text.strip():
        hyde_text = question  # fallback to original question

    query_kwargs = {
        "query_texts": [hyde_text],
        "n_results":   k,
        "include":     ["documents", "metadatas", "distances"],
    }
    if where_filter:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)

    chunks = []
    for i, (doc_id, doc, meta, dist) in enumerate(zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        chunks.append({
            "id":       doc_id,
            "document": doc,
            "metadata": meta,
            "distance": dist,
            "sem_rank": i,
        })


    return chunks


# ── BM25 index cache (module-level) ──────────────────────────────────────────
# Built once on first call, reused for all subsequent queries.
# Invalidated when collection size changes (new ingestion).

_bm25_cache: dict = {
    "index":      None,
    "ids":        None,
    "docs":       None,
    "metas":      None,
    "collection_size": 0,
}


def _get_bm25_index(collection: chromadb.Collection):
    """Return cached BM25 index, rebuilding only if collection size changed."""
    current_size = collection.count()
    if (_bm25_cache["index"] is None or
            _bm25_cache["collection_size"] != current_size):
        all_results = collection.get(include=["documents", "metadatas"])
        tokenised   = [doc.lower().split() for doc in all_results["documents"]]
        _bm25_cache["index"]           = BM25Okapi(tokenised)
        _bm25_cache["ids"]             = all_results["ids"]
        _bm25_cache["docs"]            = all_results["documents"]
        _bm25_cache["metas"]           = all_results["metadatas"]
        _bm25_cache["collection_size"] = current_size
    return (
        _bm25_cache["index"],
        _bm25_cache["ids"],
        _bm25_cache["docs"],
        _bm25_cache["metas"],
    )


def _metadata_matches_where(meta: dict, where_filter: Optional[dict]) -> bool:
    """Evaluate the simple Chroma filters this pipeline builds for BM25 parity."""
    if not where_filter:
        return True
    if "$and" in where_filter:
        return all(_metadata_matches_where(meta, clause) for clause in where_filter["$and"])
    if "$or" in where_filter:
        return any(_metadata_matches_where(meta, clause) for clause in where_filter["$or"])

    for key, condition in where_filter.items():
        if isinstance(condition, dict):
            if "$eq" in condition and meta.get(key) != condition["$eq"]:
                return False
        elif meta.get(key) != condition:
            return False
    return True


# ── Step 3b: BM25 retrieval ───────────────────────────────────────────────────

@observe(name="bm25_retrieval")
def bm25_retrieval(
    collection: chromadb.Collection,
    question: str,
    k: int = BM25_K,
    where_filter: Optional[dict] = None,
) -> list[dict]:
    """
    BM25 term-based retrieval over all chunks in ChromaDB.
    Handles exact keyword lookups: clause numbers, PPAP levels, KPI names.
    Uses module-level cache — index built once, reused across all queries.
    """
    bm25, all_ids, all_docs, all_metas = _get_bm25_index(collection)

    if not all_ids:
        return []

    query_tokens   = question.lower().split()
    scores         = bm25.get_scores(query_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    chunks = []
    for idx in ranked_indices:
        if len(chunks) >= k:
            break
        if scores[idx] <= 0:
            continue
        if not _metadata_matches_where(all_metas[idx], where_filter):
            continue
        chunks.append({
            "id":         all_ids[idx],
            "document":   all_docs[idx],
            "metadata":   all_metas[idx],
            "bm25_score": scores[idx],
            "bm25_rank":  len(chunks),
        })

    return chunks


# ── Step 3c: Reciprocal Rank Fusion ──────────────────────────────────────────

@observe(name="rrf_fusion")
def reciprocal_rank_fusion(
    sem_chunks: list[dict],
    bm25_chunks: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """
    RRF formula: Score(D) = Σ 1/(k + r_i(D))
    Combines semantic and BM25 rankings into a single ranked list.
    """
    rrf_scores = {}
    chunk_map  = {}

    # Semantic contributions
    for chunk in sem_chunks:
        cid = chunk["id"]
        r   = chunk.get("sem_rank", len(sem_chunks))
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + r)
        chunk_map[cid]  = chunk

    # BM25 contributions
    for chunk in bm25_chunks:
        cid = chunk["id"]
        r   = chunk.get("bm25_rank", len(bm25_chunks))
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + r)
        if cid not in chunk_map:
            chunk_map[cid] = chunk

    # Sort by RRF score descending
    ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    fused = []
    for rank, cid in enumerate(ranked_ids):
        chunk = dict(chunk_map[cid])
        chunk["rrf_score"] = rrf_scores[cid]
        chunk["rrf_rank"]  = rank
        fused.append(chunk)


    return fused


# ── Step 4: BGE reranker (CPU fallback) ──────────────────────────────────────

@observe(name="bge_reranker")
def rerank_chunks(
    chunks: list[dict],
    hyde_text: str,
    top_k: int = FINAL_K,
) -> list[dict]:
    """
    BGE cross-encoder reranker. Input: HyDE text (not original question).
    CPU fallback — GPU runs on Google Colab for benchmark.
    If sentence-transformers not available, returns top_k by RRF score.
    """
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("BAAI/bge-reranker-v2-m3")

        pairs  = [(hyde_text, chunk["metadata"].get("original_text", chunk["document"][:500]))
                  for chunk in chunks[:RETRIEVAL_K]]
        scores = model.predict(pairs)

        ranked = sorted(
            zip(scores, chunks[:RETRIEVAL_K]),
            key=lambda x: x[0],
            reverse=True
        )
        reranked = [chunk for _, chunk in ranked[:top_k]]

    except Exception:
        # CPU fallback — skip reranker, take top_k by RRF score
        reranked = chunks[:top_k]

    return reranked


# ── Step 5: Chunk ordering (lost-in-the-middle fix) ──────────────────────────

def order_chunks_for_context(chunks: list[dict]) -> list[dict]:
    """
    Place top-ranked chunks at START and END of context window.
    Avoids the lost-in-the-middle phenomenon (Liu et al. 2023).
    Pattern: [rank1, rank3, rank5, rank7, rank6, rank4, rank2]
    Top chunk always at position 0; second-ranked at last position.
    """
    if len(chunks) <= 2:
        return chunks

    odds  = chunks[0::2]   # ranks 1, 3, 5, ... → start of context
    evens = chunks[1::2]   # ranks 2, 4, 6, ... → end of context (reversed)
    ordered = odds + evens[::-1]

    # Verify: top-ranked chunk must be at position 0
    assert ordered[0] is chunks[0], "Chunk ordering invariant violated: top chunk not at position 0"

    return ordered


# ── Step 6: Answer generation ─────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
@observe(name="answer_generator")
def generate_answer(
    question: str,
    chunks: list[dict],
) -> str:
    """
    Groq OSS-120B answer generation. Strict grounding — no outside knowledge.
    Every claim cited with [source].
    """
    context_parts = []
    for i, chunk in enumerate(chunks):
        meta   = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        title  = meta.get("headline", "")
        text   = meta.get("original_text", chunk.get("document", ""))[:1600]
        context_parts.append(f"[{i+1}] Source: {source} | {title}\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = """You are a supplier quality expert assistant. Answer questions strictly using the provided SICC context documents.

Rules:
- Use ONLY information from the provided context. Never use outside knowledge.
- Cite every claim with the source number in brackets: [1], [2], etc.
- Read tables, bullet lists, headings, and thresholds carefully. Many SICC answers are in compact tables.
- If the context gives a partial answer, provide the supported partial answer and say what is not specified.
- Say exactly INSUFFICIENT_EVIDENCE only when no retrieved context contains the answer.
- Do not infer defaults, deadlines, score deductions, PPAP levels, capability thresholds, or approval criteria unless they are explicitly stated in the context.
- Be concise and precise. Use supplier quality terminology correctly.
- If the answer requires action, state it clearly with timeline and owner."""

    user_prompt = f"""Context documents:
{context}

Question: {question}

First silently check whether any context excerpt directly addresses the question.
Then answer with citations [n]. If there is no direct support, say only INSUFFICIENT_EVIDENCE."""

    response = completion(
        model=GROQ_120B,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0,
        max_tokens=1000,
        stop=["```"],
    )

    raw_answer = response.choices[0].message.content.strip()

    return raw_answer


# ── Step 7: Groundedness checker ─────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
@observe(name="groundedness_checker")
def check_groundedness(
    question: str,
    answer: str,
    chunks: list[dict],
) -> str:
    """
    Groq OSS-20B NLI actor/critic.
    Strips claims not supported by context.
    Returns cleaned answer or INSUFFICIENT_EVIDENCE_MARKER if everything is stripped.
    """
    if "INSUFFICIENT_EVIDENCE" in answer:
        return INSUFFICIENT_EVIDENCE_MARKER

    context_texts = "\n\n".join(
        chunk.get("metadata", {}).get("original_text", chunk.get("document", ""))[:1200]
        for chunk in chunks
    )

    prompt = f"""You are a groundedness checker. Your job is to verify that every claim in the answer is supported by the provided SICC context, then return the verified answer.

Context:
{context_texts[:6000]}

Question: {question}

Answer to check:
{answer}

Instructions:
1. For each claim in the answer, verify it is explicitly or implicitly supported by the context.
2. Remove claims that clearly contradict the context or introduce facts not present anywhere in the context.
3. Keep all claims that are supported — preserve source citations [n].
4. Numeric thresholds, deadlines, PPAP levels, score deductions, approval criteria, and mandatory actions must be explicitly present in the context. If not, remove them.
5. If the remaining answer no longer directly answers the question, respond exactly INSUFFICIENT_EVIDENCE.
6. If the answer says INSUFFICIENT_EVIDENCE but the context clearly contains a direct answer, return the direct answer with citations instead.
7. IMPORTANT: Always return the full cleaned answer text. Never respond with a single word like "Correct" or "Accurate" — return the complete answer.

Cleaned answer (return the full text):"""

    response = completion(
        model=GROQ_20B,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=800,
        stop=["```"],
    )

    cleaned = response.choices[0].message.content.strip()

    # Fallback: if the checker returned a short or empty response without explicitly
    # saying INSUFFICIENT_EVIDENCE, it's a model failure — return the raw answer.
    # A legitimate "no evidence" response will always contain that marker explicitly.
    if "INSUFFICIENT_EVIDENCE" not in cleaned.upper() and len(cleaned.strip()) < 80:
        cleaned = answer

    # Empty answer guard — only reached if checker explicitly returned INSUFFICIENT_EVIDENCE
    if not cleaned or len(cleaned.strip()) < 10:
        cleaned = INSUFFICIENT_EVIDENCE_MARKER


    return cleaned


# ── Step 8: Structure output ──────────────────────────────────────────────────

@observe(name="structure_output")
def structure_output(
    question: str,
    checked_answer: str,
    chunks: list[dict],
    is_ambiguous: bool = False,
) -> SupplierQAResult:
    """
    Build Pydantic structured output from the checked answer.
    Determines confidence, action_required, sources from answer content.
    """
    normalized_answer = checked_answer.strip().upper()
    context_missing = is_context_missing_question(question)
    insufficient = (
        checked_answer == INSUFFICIENT_EVIDENCE_MARKER
        or "INSUFFICIENT_EVIDENCE" in normalized_answer
        or context_missing
    )
    retrieved_sources = [c["metadata"].get("source", "unknown") for c in chunks]
    retrieved_context = [
        {
            "source": c["metadata"].get("source", "unknown"),
            "text": c["metadata"].get("original_text", "")[:1600],
        }
        for c in chunks[:5]
    ]

    if insufficient:
        return SupplierQAResult(
            answer=(
                "The question is too underspecified to answer from the SICC knowledge base. "
                "Please include the relevant process, requirement, event, score, supplier, or document."
                if context_missing else
                "The knowledge base does not contain sufficient information to answer this question confidently. Please consult the relevant standard or procedure directly."
            ),
            confidence="low",
            action_required=False,
            insufficient_evidence=True,
            sources=[],
            retrieved_sources=retrieved_sources,
            retrieved_context=retrieved_context,
            risk_level="not_applicable",
        )

    # Extract source citations from answer
    cited_indices = [int(m) - 1 for m in re.findall(r"\[(\d+)\]", checked_answer)
                     if m.isdigit() and 0 <= int(m) - 1 < len(chunks)]
    sources = list(dict.fromkeys(
        chunks[i]["metadata"].get("source", "unknown")
        for i in cited_indices
        if i < len(chunks)
    ))
    if not sources:
        sources = [c["metadata"].get("source", "unknown") for c in chunks[:3]]

    # Confidence heuristic
    n_sources  = len(sources)
    has_finding = "FINDING:" in " ".join(
        c["metadata"].get("original_text", "") for c in chunks[:3]
    )
    if is_context_missing_question(question) or is_ambiguous:
        confidence = "low"
    elif n_sources >= 2 and has_finding:
        confidence = "high"
    elif n_sources >= 1:
        confidence = "medium"
    else:
        confidence = "low"

    # Action required heuristic
    action_keywords = [
        "must", "shall", "required", "mandatory", "immediate",
        "within", "escalate", "notify", "suspend", "initiate",
        "scar", "audit", "corrective action",
    ]
    action_required = any(kw in checked_answer.lower() for kw in action_keywords)

    # Risk level heuristic
    risk_level = "not_applicable"
    if any(w in question.lower() for w in ["red", "critical", "immediate", "emergency", "escalate"]):
        risk_level = "high"
    elif any(w in question.lower() for w in ["amber", "monitor", "review", "develop"]):
        risk_level = "medium"
    elif any(w in question.lower() for w in ["green", "standard", "routine"]):
        risk_level = "low"

    return SupplierQAResult(
        answer=checked_answer,
        confidence=confidence,
        action_required=action_required,
        insufficient_evidence=False,
        sources=sources,
        retrieved_sources=retrieved_sources,
        retrieved_context=retrieved_context,
        risk_level=risk_level,
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

@observe(name="sicc_rag_pipeline")
def answer(
    question: str,
    db_path: str = CHROMA_DB_PATH,
    where_filter: Optional[dict] = None,
    session_id: str = "live",
) -> SupplierQAResult:
    """
    Full SICC RAG pipeline. Returns SupplierQAResult.
    """

    # Step 1 — Pre-flight
    if not preflight_check(question):
        return SupplierQAResult(
            answer="I cannot process this request as it appears to contain instructions that could override my safety guidelines.",
            confidence="high",
            action_required=False,
            insufficient_evidence=False,
            sources=[],
            risk_level="not_applicable",
        )

    if is_context_missing_question(question):
        return structure_output(question, INSUFFICIENT_EVIDENCE_MARKER, [])

    ambiguous = is_broad_ambiguous_question(question)

    collection, _ = build_clients(db_path)

    # Step 2 — HyDE rewriting
    hyde_text = rewrite_query_hyde(question)

    # Step 3 — Hybrid retrieval
    sem_chunks  = semantic_retrieval(collection, hyde_text, question, k=RETRIEVAL_K,
                                     where_filter=where_filter)
    bm25_chunks = bm25_retrieval(collection, question, k=BM25_K,
                                 where_filter=where_filter)
    fused       = reciprocal_rank_fusion(sem_chunks, bm25_chunks)

    # Step 4 — Reranking
    reranked = rerank_chunks(fused, hyde_text, top_k=FINAL_K)

    # Step 5 — Chunk ordering
    ordered = order_chunks_for_context(reranked)
    if not ordered:
        return structure_output(question, INSUFFICIENT_EVIDENCE_MARKER, ordered)

    # Step 6 — Answer generation
    raw_answer = generate_answer(question, ordered)

    # Step 7 — Groundedness check
    checked_answer = check_groundedness(question, raw_answer, ordered)

    # Step 8 — Structure output
    result = structure_output(question, checked_answer, ordered, is_ambiguous=ambiguous)

    return result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    where_filter = build_where_filter(risk=args.risk, family=args.family)

    result = answer(
        question=args.question,
        db_path=args.db,
        where_filter=where_filter,
        session_id="cli",
    )

    print("\n" + "═" * 60)
    print(f"ANSWER [{result.confidence.upper()} confidence]")
    print("═" * 60)
    print(result.answer)
    print()
    print(f"Action required : {result.action_required}")
    print(f"Insufficient ev : {result.insufficient_evidence}")
    print(f"Risk level      : {result.risk_level}")
    print(f"Sources         : {', '.join(result.sources)}")
    print("═" * 60)
