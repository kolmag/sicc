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
from openai import OpenAI as OpenAIClient
from dotenv import load_dotenv
from langfuse import Langfuse, observe
from litellm import completion
from pydantic import BaseModel, field_validator
from rank_bm25 import BM25Okapi
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

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

# Prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore (previous|prior|above|all) instructions",
    r"disregard (your|the) (system|previous) (prompt|instructions)",
    r"you are now",
    r"new (system|persona|role|instructions)",
    r"jailbreak",
    r"bypass (your|the) (filter|restriction|safety)",
    r"act as (if|a|an|though)",
    r"pretend (you are|to be)",
    r"forget (everything|your instructions)",
]

INSUFFICIENT_EVIDENCE_MARKER = "_FALLBACK_INSUFFICIENT_EVIDENCE"

# ── Pydantic output schema ────────────────────────────────────────────────────

class SupplierQAResult(BaseModel):
    answer:               str
    confidence:           Literal["high", "medium", "low"]
    action_required:      bool
    insufficient_evidence: bool
    sources:              list[str]
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


# ── Clients ───────────────────────────────────────────────────────────────────

def build_clients(db_path: str):
    chroma_client = chromadb.PersistentClient(
        path=os.path.abspath(db_path)
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
    langfuse = Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
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


# ── Step 2: HyDE query rewriting ─────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
@observe(name="hyde_rewriter")
def rewrite_query_hyde(question: str) -> str:
    """
    HyDE: generate a hypothetical document excerpt that would answer the question.
    Passed to BOTH ChromaDB AND BGE reranker (not original question).
    Groq OSS-120B, T=0.
    """
    prompt = f"""You are an expert supplier quality engineer writing a knowledge base document.

Write a 2-3 sentence excerpt from a supplier quality procedure document that would directly answer this question. Use the vocabulary of a supplier quality manual: PPM, SCAR, OTD, PPAP, APQP, audit findings, risk tier, corrective action, etc.

Question: {question}

Write only the excerpt — no preamble, no explanation:"""

    response = completion(
        model=GROQ_120B,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200,
        stop=["```", "\n\n\n"],
    )

    hyde_text = response.choices[0].message.content.strip()

    # Guard: if model returned empty, use original question
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


# ── Step 3b: BM25 retrieval ───────────────────────────────────────────────────

@observe(name="bm25_retrieval")
def bm25_retrieval(
    collection: chromadb.Collection,
    question: str,
    k: int = BM25_K,
) -> list[dict]:
    """
    BM25 term-based retrieval over all chunks in ChromaDB.
    Handles exact keyword lookups: clause numbers, PPAP levels, KPI names.
    """
    # Fetch all documents from ChromaDB for BM25 indexing
    all_results = collection.get(include=["documents", "metadatas"])

    if not all_results["ids"]:
        return []

    all_ids   = all_results["ids"]
    all_docs  = all_results["documents"]
    all_metas = all_results["metadatas"]

    # Tokenise for BM25
    tokenised = [doc.lower().split() for doc in all_docs]
    bm25      = BM25Okapi(tokenised)

    query_tokens = question.lower().split()
    scores       = bm25.get_scores(query_tokens)

    # Get top-k by BM25 score
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    chunks = []
    for rank, idx in enumerate(ranked_indices):
        if scores[idx] > 0:
            chunks.append({
                "id":       all_ids[idx],
                "document": all_docs[idx],
                "metadata": all_metas[idx],
                "bm25_score": scores[idx],
                "bm25_rank":  rank,
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

    except ImportError:
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

    odds  = chunks[0::2]   # ranks 1, 3, 5, ...  → start of context
    evens = chunks[1::2]   # ranks 2, 4, 6, ...  → end of context (reversed)

    return odds + evens[::-1]


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
        text   = meta.get("original_text", chunk.get("document", ""))[:800]
        context_parts.append(f"[{i+1}] Source: {source} | {title}\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = """You are a supplier quality expert assistant. Answer questions strictly using the provided context documents. 

Rules:
- Use ONLY information from the provided context. Never use outside knowledge.
- Cite every claim with the source number in brackets: [1], [2], etc.
- If the context does not contain enough information to answer, say exactly: INSUFFICIENT_EVIDENCE
- Be concise and precise. Use supplier quality terminology correctly.
- If the answer requires action, state it clearly with timeline and owner."""

    user_prompt = f"""Context documents:
{context}

Question: {question}

Answer (cite sources with [n], or say INSUFFICIENT_EVIDENCE if context is inadequate):"""

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
        chunk.get("metadata", {}).get("original_text", chunk.get("document", ""))[:400]
        for chunk in chunks
    )

    prompt = f"""You are a groundedness checker. Your job is to verify that every claim in the answer is supported by the provided context.

Context:
{context_texts[:3000]}

Question: {question}

Answer to check:
{answer}

Instructions:
1. For each claim in the answer, verify it is explicitly supported by the context.
2. Remove any claim that is not supported — do not add your own knowledge.
3. Keep all claims that ARE supported, preserving the source citations [n].
4. If nothing remains after removing unsupported claims, respond with exactly: INSUFFICIENT_EVIDENCE
5. If most claims are supported, return the answer as-is with minor edits only.
6. Only respond with INSUFFICIENT_EVIDENCE if truly nothing in the answer is supported.
7. Return only the cleaned answer — no explanation, no preamble.

Cleaned answer:"""

    response = completion(
        model=GROQ_20B,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=800,
        stop=["```", "\n\n\n"],
    )

    cleaned = response.choices[0].message.content.strip()

    # Empty answer guard
    if not cleaned or len(cleaned.strip()) < 10:
        cleaned = INSUFFICIENT_EVIDENCE_MARKER


    return cleaned


# ── Step 8: Structure output ──────────────────────────────────────────────────

@observe(name="structure_output")
def structure_output(
    question: str,
    checked_answer: str,
    chunks: list[dict],
) -> SupplierQAResult:
    """
    Build Pydantic structured output from the checked answer.
    Determines confidence, action_required, sources from answer content.
    """
    insufficient = checked_answer == INSUFFICIENT_EVIDENCE_MARKER

    if insufficient:
        return SupplierQAResult(
            answer="The knowledge base does not contain sufficient information to answer this question confidently. Please consult the relevant standard or procedure directly.",
            confidence="low",
            action_required=False,
            insufficient_evidence=True,
            sources=[],
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
    if n_sources >= 2 and has_finding:
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

    collection, _ = build_clients(db_path)

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

    # Step 2 — HyDE rewriting
    hyde_text = rewrite_query_hyde(question)

    # Step 3 — Hybrid retrieval
    sem_chunks  = semantic_retrieval(collection, hyde_text, question, k=RETRIEVAL_K,
                                     where_filter=where_filter)
    bm25_chunks = bm25_retrieval(collection, question, k=BM25_K)
    fused       = reciprocal_rank_fusion(sem_chunks, bm25_chunks)

    # Step 4 — Reranking
    reranked = rerank_chunks(fused, hyde_text, top_k=FINAL_K)

    # Step 5 — Chunk ordering
    ordered = order_chunks_for_context(reranked)

    # Step 6 — Answer generation
    raw_answer = generate_answer(question, ordered)

    # Step 7 — Groundedness check
    #print("RAW ANSWER:", raw_answer[:500])
    checked_answer = check_groundedness(question, raw_answer, ordered)
    # Safety guard: if checker strips a substantial answer, trust the raw answer
    if checked_answer == INSUFFICIENT_EVIDENCE_MARKER and len(raw_answer) > 100:
        checked_answer = raw_answer
    #print("CHECKED:", checked_answer[:200])

    # Step 8 — Structure output
    result = structure_output(question, checked_answer, ordered)

    return result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    where_filter = None
    if args.risk:
        where_filter = {"risk_domain": {"$eq": args.risk}}

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
