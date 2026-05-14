"""
evaluation/eval.py — SICC RAG Evaluation Framework
Imports from scripts/answer.py — no business logic duplication.

Metrics:
  - MRR  (Mean Reciprocal Rank)      — primary retrieval metric
  - NDCG (Normalised Discounted CG)  — ranked retrieval quality
  - Answer quality: correctness, completeness, groundedness (3-dim judge)
  - Adversarial: injection block rate, out-of-scope insufficient_evidence rate

Judge: Claude Sonnet 4.6 via Anthropic Batch API, T=0
       Judge model differs from answer model (no self-scoring)

Usage:
    uv run python evaluation/eval.py --set developer
    uv run python evaluation/eval.py --set all --out evaluation/results/
    uv run python evaluation/eval.py --set developer --limit 10  # smoke test
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import anthropic
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── Path setup — import from scripts/answer.py ────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.answer import (
    CHROMA_DB_PATH,
    INSUFFICIENT_EVIDENCE_MARKER,
    answer,
    build_clients,
    preflight_check,
    reciprocal_rank_fusion,
    rerank_chunks,
    rewrite_query_hyde,
    semantic_retrieval,
    bm25_retrieval,
    order_chunks_for_context,
)

# ── Constants ─────────────────────────────────────────────────────────────────

JUDGE_MODEL   = "claude-sonnet-4-6"   # must differ from answer model
QUESTIONS_DIR = ROOT / "evaluation" / "questions"
RESULTS_DIR   = ROOT / "evaluation" / "results"

SETS = {
    "developer":    QUESTIONS_DIR / "developer.json",
    "practitioner": QUESTIONS_DIR / "practitioner.json",
    "practitioner_blind": QUESTIONS_DIR / "practitioner_blind.json",
    "adversarial":  QUESTIONS_DIR / "adversarial.json",
}

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SICC RAG Evaluation")
    p.add_argument("--set",   default="developer",
                   choices=["developer", "practitioner", "practitioner_blind", "adversarial", "all"])
    p.add_argument("--out",   default=str(RESULTS_DIR))
    p.add_argument("--limit", type=int, default=None,
                   help="Run only first N questions per set (smoke test)")
    p.add_argument("--skip-judge", action="store_true",
                   help="Skip LLM judge — compute retrieval metrics only")
    p.add_argument("--batch", action="store_true",
                   help="Use Anthropic Batch API for judge (async, cheaper)")
    p.add_argument("--category", default=None,
                   help="Run only questions matching this category")
    p.add_argument("--difficulty", default=None,
                   help="Run only questions matching this difficulty")
    p.add_argument("--source", default=None,
                   help="Run only questions matching optional source metadata")
    return p.parse_args()


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def reciprocal_rank(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """
    MRR: 1/rank of first relevant result.
    retrieved_sources: ordered list of source filenames from retrieval
    expected_sources:  list of acceptable source filenames
    Returns 0.0 if no relevant result in top-K.
    """
    for rank, source in enumerate(retrieved_sources, start=1):
        if any(exp in source for exp in expected_sources):
            return 1.0 / rank
    return 0.0


def matched_expected_source(source: str, expected_sources: list[str]) -> str | None:
    """Return the expected source matched by a retrieved source, if any."""
    for exp in expected_sources:
        if exp in source:
            return exp
    return None


def ndcg_at_k(retrieved_sources: list[str], expected_sources: list[str], k: int = 7) -> float:
    """
    NDCG@K: normalised discounted cumulative gain.
    Binary relevance: 1 if source matches expected, 0 otherwise.
    """
    if not expected_sources:
        return 0.0

    def dcg(relevances):
        return sum(rel / np.log2(rank + 2) for rank, rel in enumerate(relevances))

    seen_expected = set()
    relevances = []
    for src in retrieved_sources[:k]:
        match = matched_expected_source(src, expected_sources)
        if match and match not in seen_expected:
            relevances.append(1.0)
            seen_expected.add(match)
        else:
            relevances.append(0.0)
    ideal_relevances = [1.0] * min(len(expected_sources), k)

    actual_dcg = dcg(relevances)
    ideal_dcg  = dcg(ideal_relevances)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def retrieve_for_eval(question: str, db_path: str = CHROMA_DB_PATH) -> list[str]:
    """
    Run retrieval pipeline and return ordered list of source filenames.
    Used to compute MRR and NDCG without running the full answer pipeline.
    """
    collection, _ = build_clients(db_path)

    hyde_text  = rewrite_query_hyde(question)
    sem_chunks = semantic_retrieval(collection, hyde_text, question)
    bm25_chunks = bm25_retrieval(collection, question)
    fused      = reciprocal_rank_fusion(sem_chunks, bm25_chunks)
    reranked   = rerank_chunks(fused, hyde_text)
    ordered    = order_chunks_for_context(reranked)

    return [c["metadata"].get("source", "unknown") for c in ordered]


# ── Judge ─────────────────────────────────────────────────────────────────────

def build_judge_prompt(
    question: str,
    answer_text: str,
    sources: list[str],
    retrieved_context: list[dict] | None = None,
) -> str:
    context_lines = []
    for idx, item in enumerate((retrieved_context or [])[:5], start=1):
        source = item.get("source", "unknown")
        text = " ".join(item.get("text", "").split())[:900]
        if text:
            context_lines.append(f"[{idx}] {source}: {text}")
    context_text = "\n".join(context_lines) if context_lines else "No context excerpts provided."

    return f"""You are an expert supplier quality engineer evaluating a RAG system answer.

Evaluate against the SICC knowledge-base excerpts below, not against your own
memory of public standards. SICC contains internal procedures and thresholds that
may supplement ISO 9001, IATF 16949, AS9100D, PPAP, APQP, and supplier-risk
standards. Do not penalize an answer merely because a requirement is internal
SICC policy rather than public normative text.

Question: {question}

Answer: {answer_text}

Sources cited: {", ".join(sources) if sources else "none"}

Retrieved SICC context excerpts:
{context_text}

Rate the answer on three dimensions, each from 0 to 3:

CORRECTNESS (0-3):
  3 = Fully correct, no factual errors
  2 = Mostly correct, minor inaccuracies
  1 = Partially correct, significant gaps or errors
  0 = Incorrect or misleading

COMPLETENESS (0-3):
  3 = Addresses all key aspects of the question
  2 = Addresses main aspects, minor gaps
  1 = Addresses some aspects, significant gaps
  0 = Does not address the question

GROUNDEDNESS (0-3):
  3 = Every claim is supported by cited sources
  2 = Most claims supported, minor unsupported statements
  1 = Some claims supported, significant unsupported content
  0 = Claims not supported by sources or no sources cited

Do not favor longer responses. A concise accurate answer should score higher than
a verbose answer with unsupported or irrelevant content.

Also provide:
- overall: weighted quality score from 0 to 3, using correctness 40%,
  completeness 35%, groundedness 25%
- key_gap: the single most important gap, or "none"

Respond ONLY with a JSON object — no explanation, no preamble:
{{"correctness": <0-3>, "completeness": <0-3>, "groundedness": <0-3>, "overall": <0-3>, "comment": "<one sentence>", "key_gap": "<one phrase or none>"}}"""


def parse_judge_json(text: str) -> dict:
    """Parse judge JSON robustly even if the model wraps it in fences."""
    text = text.strip().replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    scores = json.loads(match.group(0) if match else text)
    for key in ["correctness", "completeness", "groundedness"]:
        scores[key] = int(scores.get(key, 0))
    if "overall" not in scores:
        scores["overall"] = round(
            scores["correctness"] * 0.40
            + scores["completeness"] * 0.35
            + scores["groundedness"] * 0.25,
            3,
        )
    scores.setdefault("comment", "")
    scores.setdefault("key_gap", "none")
    return scores


def judge_answer_sync(
    client: anthropic.Anthropic,
    question: str,
    answer_text: str,
    sources: list[str],
    retrieved_context: list[dict] | None = None,
) -> dict:
    """Synchronous judge call — used when --batch is not set."""
    prompt = build_judge_prompt(question, answer_text, sources, retrieved_context)
    try:
        response = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=300,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        scores = parse_judge_json(text)
        scores["judge_model"] = JUDGE_MODEL
        return scores
    except Exception as e:
        return {
            "correctness": 0, "completeness": 0, "groundedness": 0,
            "overall": 0, "comment": f"Judge error: {e}",
            "key_gap": "judge_failed", "judge_error": str(e),
            "judge_model": JUDGE_MODEL,
        }


def judge_batch_submit(
    client: anthropic.Anthropic,
    eval_results: list[dict],
) -> str:
    """Submit all judge requests as a single Anthropic Batch API call. Returns batch_id."""
    requests = []
    for r in eval_results:
        if r.get("skipped") or r.get("blocked"):
            continue
        answer_text = r.get("answer", "")
        if not answer_text or r.get("insufficient_evidence"):
            continue
        prompt = build_judge_prompt(
            r["question"], answer_text, r.get("sources", []), r.get("retrieved_context", [])
        )
        requests.append({
            "custom_id": r["id"],
            "params": {
                "model": JUDGE_MODEL,
                "max_tokens": 300,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
        })

    if not requests:
        return None

    batch = client.beta.messages.batches.create(requests=requests)
    print(f"[eval] Batch submitted: {batch.id} ({len(requests)} requests)")
    return batch.id


def judge_batch_wait(client: anthropic.Anthropic, batch_id: str) -> dict:
    """Poll batch until complete. Returns {custom_id: scores} dict."""
    print(f"[eval] Waiting for batch {batch_id}...")
    while True:
        batch = client.beta.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        print(f"[eval] Batch status: {status} "
              f"({batch.request_counts.succeeded} succeeded, "
              f"{batch.request_counts.errored} errored)")
        if status == "ended":
            break
        time.sleep(30)

    scores_map = {}
    for result in client.beta.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            try:
                text = result.result.message.content[0].text.strip()
                scores = parse_judge_json(text)
                scores["judge_model"] = JUDGE_MODEL
                scores_map[result.custom_id] = scores
            except Exception as e:
                scores_map[result.custom_id] = {
                    "correctness": 0, "completeness": 0, "groundedness": 0,
                    "overall": 0, "comment": f"Parse error: {e}",
                    "key_gap": "judge_parse_failed", "judge_error": str(e),
                    "judge_model": JUDGE_MODEL,
                }
        else:
            error_text = str(getattr(result.result, "error", result.result))
            scores_map[result.custom_id] = {
                "correctness": 0, "completeness": 0, "groundedness": 0,
                "overall": 0, "comment": f"Batch judge error: {error_text}",
                "key_gap": "judge_batch_failed", "judge_error": error_text,
                "judge_model": JUDGE_MODEL,
            }
    return scores_map


def should_judge(row: dict) -> bool:
    """Rows eligible for judge scoring."""
    return bool(row.get("answer")) and not (
        row.get("blocked") or row.get("error") or row.get("insufficient_evidence")
    )


def apply_batch_scores(results: list[dict], scores_map: dict) -> None:
    """Attach batch scores and make missing judge rows explicit."""
    for row in results:
        if row.get("id") in scores_map:
            row.update(scores_map[row["id"]])
        elif should_judge(row):
            row.update({
                "correctness": 0,
                "completeness": 0,
                "groundedness": 0,
                "overall": 0,
                "comment": "Missing judge result from batch",
                "key_gap": "judge_missing",
                "judge_error": "missing_from_batch_results",
                "judge_model": JUDGE_MODEL,
            })


# ── Developer set evaluation ──────────────────────────────────────────────────

def eval_developer(
    questions: list[dict],
    client: anthropic.Anthropic,
    skip_judge: bool,
    use_batch: bool,
    limit: int = None,
) -> list[dict]:
    """
    Run developer set. Computes MRR, NDCG, and judge scores.
    expected_sources is mandatory for this set.
    """
    if limit:
        questions = questions[:limit]

    results = []
    print(f"\n[eval] Developer set: {len(questions)} questions")

    for q in tqdm(questions, desc="Developer"):
        start = time.perf_counter()
        row = {
            "id":               q["id"],
            "question":         q["question"],
            "expected_sources": q["expected_sources"],
            "category":         q.get("category", ""),
            "difficulty":       q.get("difficulty", "medium"),
            "source":           q.get("source", "developer"),
        }

        # Check injection (should not trigger on developer questions)
        if not preflight_check(q["question"]):
            row["blocked"]  = True
            row["rr"]       = 0.0
            row["ndcg"]     = 0.0
            row["latency_s"] = round(time.perf_counter() - start, 3)
            results.append(row)
            continue

        try:
            result = answer(q["question"], session_id="eval_developer")
            retrieved = result.retrieved_sources

            row["retrieved_sources"] = retrieved
            row["rr"]   = reciprocal_rank(retrieved, q["expected_sources"])
            row["ndcg"] = ndcg_at_k(retrieved, q["expected_sources"])
            row["answer"]               = result.answer
            row["confidence"]           = result.confidence
            row["action_required"]      = result.action_required
            row["insufficient_evidence"] = result.insufficient_evidence
            row["sources"]              = result.sources
            row["retrieved_context"]    = result.retrieved_context

            # Judge scores
            if not skip_judge and not result.insufficient_evidence:
                if not use_batch:
                    scores = judge_answer_sync(
                        client, q["question"], result.answer, result.sources,
                        result.retrieved_context,
                    )
                    row.update(scores)

        except Exception as e:
            row["error"] = str(e)
            row["rr"]    = 0.0
            row["ndcg"]  = 0.0

        row["latency_s"] = round(time.perf_counter() - start, 3)
        results.append(row)
        time.sleep(0.5)   # rate limit headroom

    # Batch judge
    if use_batch and not skip_judge:
        batch_id   = judge_batch_submit(client, results)
        if batch_id:
            scores_map = judge_batch_wait(client, batch_id)
            apply_batch_scores(results, scores_map)

    return results


# ── Practitioner set evaluation ───────────────────────────────────────────────

def eval_practitioner(
    questions: list[dict],
    client: anthropic.Anthropic,
    skip_judge: bool,
    use_batch: bool,
    limit: int = None,
) -> list[dict]:
    """
    Run practitioner set. No expected_sources — judge quality only.
    """
    if limit:
        questions = questions[:limit]

    results = []
    print(f"\n[eval] Practitioner set: {len(questions)} questions")

    for q in tqdm(questions, desc="Practitioner"):
        start = time.perf_counter()
        row = {
            "id":       q["id"],
            "question": q["question"],
            "category": q.get("category", ""),
            "difficulty": q.get("difficulty", "medium"),
            "source": q.get("source", "practitioner"),
        }

        if not preflight_check(q["question"]):
            row["blocked"] = True
            row["latency_s"] = round(time.perf_counter() - start, 3)
            results.append(row)
            continue

        try:
            result = answer(q["question"], session_id="eval_practitioner")
            row["answer"]               = result.answer
            row["confidence"]           = result.confidence
            row["action_required"]      = result.action_required
            row["insufficient_evidence"] = result.insufficient_evidence
            row["sources"]              = result.sources
            row["retrieved_sources"]    = result.retrieved_sources
            row["retrieved_context"]    = result.retrieved_context

            if not skip_judge and not result.insufficient_evidence:
                if not use_batch:
                    scores = judge_answer_sync(
                        client, q["question"], result.answer, result.sources,
                        result.retrieved_context,
                    )
                    row.update(scores)

        except Exception as e:
            row["error"] = str(e)

        row["latency_s"] = round(time.perf_counter() - start, 3)
        results.append(row)
        time.sleep(0.5)

    if use_batch and not skip_judge:
        batch_id = judge_batch_submit(client, results)
        if batch_id:
            scores_map = judge_batch_wait(client, batch_id)
            apply_batch_scores(results, scores_map)

    return results


# ── Adversarial set evaluation ────────────────────────────────────────────────

def eval_adversarial(
    questions: list[dict],
    limit: int = None,
) -> list[dict]:
    """
    Run adversarial set. Measures:
    - Injection block rate (expected_type == injection → must be blocked)
    - Out-of-scope insufficient_evidence rate
    - Ambiguous question handling
    No judge needed — pass/fail based on expected_result.
    """
    if limit:
        buckets: dict[str, list[dict]] = {}
        for q in questions:
            buckets.setdefault(q["expected_type"], []).append(q)

        questions = []
        while len(questions) < limit and any(buckets.values()):
            for expected_type in list(buckets.keys()):
                if buckets[expected_type] and len(questions) < limit:
                    questions.append(buckets[expected_type].pop(0))

    results = []
    print(f"\n[eval] Adversarial set: {len(questions)} questions")

    for q in tqdm(questions, desc="Adversarial"):
        start = time.perf_counter()
        row = {
            "id":              q["id"],
            "question":        q["question"],
            "expected_type":   q["expected_type"],
            "expected_result": q["expected_result"],
            "category":        q.get("category", ""),
            "source":          q.get("source", "adversarial"),
        }

        # Injection check
        is_blocked = not preflight_check(q["question"])
        row["blocked"] = is_blocked

        if q["expected_type"] == "injection":
            row["pass"] = is_blocked
            row["note"] = "Blocked by pre-flight" if is_blocked else "FAIL: injection not blocked"
            row["latency_s"] = round(time.perf_counter() - start, 3)
            results.append(row)
            continue

        if is_blocked:
            # Non-injection question was blocked — that's a false positive
            row["pass"] = False
            row["note"] = "False positive block on non-injection question"
            row["latency_s"] = round(time.perf_counter() - start, 3)
            results.append(row)
            continue

        try:
            result = answer(q["question"], session_id="eval_adversarial")
            row["answer"]               = result.answer
            row["confidence"]           = result.confidence
            row["insufficient_evidence"] = result.insufficient_evidence
            row["sources"]              = result.sources
            row["retrieved_sources"]    = result.retrieved_sources
            row["retrieved_context"]    = result.retrieved_context

            # Pass criteria by expected_result
            if q["expected_result"] == "insufficient_evidence":
                row["pass"] = result.insufficient_evidence
                row["note"] = ("Correctly returned insufficient_evidence"
                               if result.insufficient_evidence
                               else f"FAIL: returned answer instead of insufficient_evidence (conf={result.confidence})")

            elif q["expected_result"] == "low_confidence_or_general":
                row["pass"] = result.confidence in ("low", "medium") or result.insufficient_evidence
                row["note"] = (f"Returned confidence={result.confidence} — acceptable"
                               if row["pass"]
                               else f"FAIL: returned high confidence for ambiguous question")

            elif q["expected_result"] == "partial_answer":
                # Should return something but not high confidence
                row["pass"] = not result.insufficient_evidence
                row["note"] = ("Returned partial answer" if row["pass"]
                               else "FAIL: returned insufficient_evidence for question with partial KB coverage")

            else:
                row["pass"] = True
                row["note"] = "No specific pass criteria"

        except Exception as e:
            row["error"] = str(e)
            row["pass"]  = False

        row["latency_s"] = round(time.perf_counter() - start, 3)
        results.append(row)
        time.sleep(0.3)

    return results


# ── Metrics summary ───────────────────────────────────────────────────────────

def compute_metrics(
    dev_results:   list[dict],
    prac_results:  list[dict],
    blind_results: list[dict],
    adv_results:   list[dict],
) -> dict:
    """Aggregate all metrics into a summary dict."""
    metrics = {}

    def composite(row: dict) -> float:
        if "overall" in row:
            return float(row["overall"]) / 3.0
        return (
            row["correctness"] * 0.40
            + row["completeness"] * 0.35
            + row["groundedness"] * 0.25
        ) / 3.0

    def latency_stats(rows: list[dict]) -> dict:
        values = [float(r["latency_s"]) for r in rows if "latency_s" in r]
        if not values:
            return {"median": None, "p95": None, "mean": None}
        arr = np.array(values, dtype=float)
        return {
            "median": round(float(np.median(arr)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "mean": round(float(np.mean(arr)), 3),
        }

    def bootstrap_mean(values: list[float], n: int = 100, seed: int = 42) -> dict:
        if not values:
            return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
        rng = np.random.default_rng(seed)
        arr = np.array(values, dtype=float)
        samples = [float(rng.choice(arr, size=len(arr), replace=True).mean())
                   for _ in range(n)]
        return {
            "mean": round(float(arr.mean()), 4),
            "std": round(float(np.std(samples, ddof=1)), 4) if len(samples) > 1 else 0.0,
            "ci95_low": round(float(np.percentile(samples, 2.5)), 4),
            "ci95_high": round(float(np.percentile(samples, 97.5)), 4),
        }

    def category_metric(rows: list[dict], value_key: str) -> dict:
        categories = {}
        for row in rows:
            cat = row.get("category", "unknown")
            categories.setdefault(cat, []).append(row[value_key])
        return {
            cat: round(float(np.mean(values)), 4)
            for cat, values in sorted(categories.items())
        }

    def category_judge(rows: list[dict]) -> dict:
        categories = {}
        for row in rows:
            if "correctness" not in row:
                continue
            cat = row.get("category", "unknown")
            categories.setdefault(cat, []).append(composite(row))
        return {
            cat: round(float(np.mean(values)), 3)
            for cat, values in sorted(categories.items())
        }

    def source_judge(rows: list[dict]) -> dict:
        sources = {}
        for row in rows:
            if "correctness" not in row:
                continue
            source = row.get("source", "unknown")
            sources.setdefault(source, []).append(composite(row))
        return {
            source: {"composite": round(float(np.mean(values)), 3), "n": len(values)}
            for source, values in sorted(sources.items())
        }

    def score_distribution(rows: list[dict]) -> dict:
        scores = [composite(row) for row in rows if "correctness" in row]
        if not scores:
            return {}
        return {
            "pass_rate_ge_0_67": round(float(np.mean([s >= 0.67 for s in scores])), 4),
            "excellent_ge_0_89": round(float(np.mean([s >= 0.89 for s in scores])), 4),
            "weak_lt_0_67": sum(1 for s in scores if s < 0.67),
            "n": len(scores),
        }

    def failure_rows(rows: list[dict], limit: int = 10) -> list[dict]:
        failures = []
        for row in rows:
            if row.get("error"):
                reason = f"error: {row['error']}"
            elif row.get("judge_error"):
                reason = f"judge_error: {row['judge_error']}"
            elif row.get("blocked"):
                reason = "blocked"
            elif row.get("rr") == 0:
                reason = "rr=0"
            elif "correctness" in row and composite(row) < 0.67:
                reason = f"judge={composite(row):.3f}"
            else:
                continue
            failures.append({
                "id": row.get("id", ""),
                "category": row.get("category", ""),
                "reason": reason,
                "question": row.get("question", "")[:120],
                "key_gap": row.get("key_gap", ""),
            })
        return failures[:limit]

    def add_practitioner_metrics(prefix: str, results: list[dict]) -> None:
        if not results:
            return

        valid = [r for r in results if not r.get("blocked") and not r.get("error")]
        judged = [r for r in valid if "correctness" in r]

        metrics[f"{prefix}_n"] = len(results)
        metrics[f"{prefix}_latency"] = latency_stats(valid)
        metrics[f"{prefix}_insufficient"] = sum(
            1 for r in valid if r.get("insufficient_evidence")
        )
        metrics[f"{prefix}_judge_errors"] = sum(1 for r in results if r.get("judge_error"))

        if judged:
            metrics[f"{prefix}_judge_n"] = len(judged)
            metrics[f"{prefix}_correctness_avg"] = round(float(np.mean([
                r["correctness"] for r in judged
            ])), 3)
            metrics[f"{prefix}_completeness_avg"] = round(float(np.mean([
                r["completeness"] for r in judged
            ])), 3)
            metrics[f"{prefix}_groundedness_avg"] = round(float(np.mean([
                r["groundedness"] for r in judged
            ])), 3)
            metrics[f"{prefix}_composite_score"] = round(float(np.mean([
                composite(r) for r in judged
            ])), 3)
            metrics[f"{prefix}_overall_avg"] = round(float(np.mean([
                r.get("overall", composite(r) * 3) for r in judged
            ])), 3)
            metrics[f"{prefix}_composite_by_category"] = category_judge(judged)
            metrics[f"{prefix}_composite_by_source"] = source_judge(judged)
            metrics[f"{prefix}_score_distribution"] = score_distribution(judged)

        metrics[f"{prefix}_failures"] = failure_rows(results)

    # ── Developer metrics ─────────────────────────────────────────────────────
    if dev_results:
        valid      = [r for r in dev_results if "rr" in r and not r.get("blocked")]
        rr_scores  = [r["rr"] for r in valid]
        ndcg_scores= [r["ndcg"] for r in valid]

        metrics["dev_n"]            = len(dev_results)
        metrics["dev_latency"]      = latency_stats(valid)
        metrics["dev_mrr"]          = round(float(np.mean(rr_scores)), 4) if rr_scores else 0
        metrics["dev_ndcg"]         = round(float(np.mean(ndcg_scores)), 4) if ndcg_scores else 0
        metrics["dev_rr_at_1"]      = round(float(np.mean([1 if r["rr"] == 1.0 else 0 for r in valid])), 4)
        metrics["dev_insufficient"] = sum(1 for r in valid if r.get("insufficient_evidence"))
        metrics["dev_mrr_bootstrap"] = bootstrap_mean(rr_scores)
        metrics["dev_ndcg_bootstrap"] = bootstrap_mean(ndcg_scores)
        metrics["dev_failures"] = failure_rows(dev_results)
        metrics["dev_judge_errors"] = sum(1 for r in dev_results if r.get("judge_error"))

        # Judge scores (if run)
        judged = [r for r in valid if "correctness" in r]
        if judged:
            metrics["dev_judge_n"]           = len(judged)
            metrics["dev_correctness_avg"]   = round(float(np.mean([r["correctness"]   for r in judged])), 3)
            metrics["dev_completeness_avg"]  = round(float(np.mean([r["completeness"]  for r in judged])), 3)
            metrics["dev_groundedness_avg"]  = round(float(np.mean([r["groundedness"]  for r in judged])), 3)
            metrics["dev_composite_score"]   = round(float(np.mean([
                composite(r) for r in judged
            ])), 3)
            metrics["dev_overall_avg"] = round(float(np.mean([
                r.get("overall", composite(r) * 3) for r in judged
            ])), 3)
            metrics["dev_composite_by_category"] = category_judge(judged)
            metrics["dev_composite_by_source"] = source_judge(judged)
            metrics["dev_score_distribution"] = score_distribution(judged)

        # Per-category retrieval metrics
        metrics["dev_mrr_by_category"] = category_metric(valid, "rr")
        metrics["dev_ndcg_by_category"] = category_metric(valid, "ndcg")

    # ── Practitioner metrics ──────────────────────────────────────────────────
    add_practitioner_metrics("prac", prac_results)
    add_practitioner_metrics("blind", blind_results)

    # ── Adversarial metrics ───────────────────────────────────────────────────
    if adv_results:
        def pass_rate(rows: list[dict]) -> float | None:
            if not rows:
                return None
            return round(sum(1 for r in rows if r.get("pass", False)) / len(rows), 4)

        injection = [r for r in adv_results if r["expected_type"] == "injection"]
        oos       = [r for r in adv_results if r["expected_type"] == "out_of_scope"]
        ambiguous = [r for r in adv_results if r["expected_type"] == "ambiguous"]

        metrics["adv_n"]                         = len(adv_results)
        metrics["adv_latency"]                   = latency_stats(adv_results)
        metrics["adv_injection_n"]               = len(injection)
        metrics["adv_oos_n"]                     = len(oos)
        metrics["adv_ambiguous_n"]               = len(ambiguous)
        metrics["adv_injection_block_rate"]       = pass_rate(injection)
        metrics["adv_oos_pass_rate"]             = pass_rate(oos)
        metrics["adv_ambiguous_pass_rate"]       = pass_rate(ambiguous)
        metrics["adv_overall_pass_rate"]         = pass_rate(adv_results)

    return metrics


# ── Report builder ────────────────────────────────────────────────────────────

def build_report(metrics: dict, run_ts: str, sets_run: list[str]) -> str:
    def format_rate(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value * 100:.1f}%"

    def practitioner_section(prefix: str, title: str) -> list[str]:
        if f"{prefix}_composite_score" not in metrics:
            return []
        latency = metrics.get(f"{prefix}_latency", {})
        dist = metrics.get(f"{prefix}_score_distribution", {})

        section = [
            f"## {title} — Answer Quality (Judge)",
            "",
            f"| Dimension | Score (0–3) | Normalised (0–1) |",
            f"|---|---|---|",
            f"| Correctness | {metrics[f'{prefix}_correctness_avg']} | {metrics[f'{prefix}_correctness_avg']/3:.3f} |",
            f"| Completeness | {metrics[f'{prefix}_completeness_avg']} | {metrics[f'{prefix}_completeness_avg']/3:.3f} |",
            f"| Groundedness | {metrics[f'{prefix}_groundedness_avg']} | {metrics[f'{prefix}_groundedness_avg']/3:.3f} |",
            f"| **Composite** | **{metrics[f'{prefix}_composite_score']*3:.3f}** | **{metrics[f'{prefix}_composite_score']:.3f}** |",
            f"| Weighted overall avg | {metrics.get(f'{prefix}_overall_avg', '-')} |  |",
            f"| Questions judged | {metrics[f'{prefix}_judge_n']} | — |",
            f"| Judge errors | {metrics.get(f'{prefix}_judge_errors', 0)} |  |",
            f"| Insufficient evidence | {metrics[f'{prefix}_insufficient']} |  |",
            f"| Median latency | {latency.get('median', 'n/a')}s |  |",
            f"| P95 latency | {latency.get('p95', 'n/a')}s |  |",
            f"| Pass rate (≥0.67) | {format_rate(dist.get('pass_rate_ge_0_67'))} |  |",
            "",
        ]
        if metrics.get(f"{prefix}_composite_by_category"):
            section += [f"## {title} — Judge Composite by Category", ""]
            section += ["| Category | Composite |", "|---|---|"]
            for cat, score in sorted(metrics[f"{prefix}_composite_by_category"].items(),
                                     key=lambda x: -x[1]):
                section.append(f"| {cat} | {score} |")
            section.append("")
        if metrics.get(f"{prefix}_composite_by_source"):
            section += [f"## {title} — Judge Composite by Source", ""]
            section += ["| Source | Composite | N |", "|---|---:|---:|"]
            for source, stats in sorted(metrics[f"{prefix}_composite_by_source"].items(),
                                        key=lambda x: -x[1]["composite"]):
                section.append(f"| {source} | {stats['composite']} | {stats['n']} |")
            section.append("")
        if metrics.get(f"{prefix}_failures"):
            section += [f"## {title} — Review Queue", ""]
            section += ["| ID | Category | Reason | Key Gap | Question |", "|---|---|---|---|---|"]
            for row in metrics[f"{prefix}_failures"]:
                section.append(
                    f"| {row['id']} | {row['category']} | {row['reason']} | {row.get('key_gap', '')} | {row['question']} |"
                )
            section.append("")
        return section

    lines = [
        "# SICC RAG Evaluation Report",
        "",
        f"**Run:** {run_ts}  ",
        f"**Sets:** {', '.join(sets_run)}  ",
        f"**Judge:** {JUDGE_MODEL} · T=0  ",
        f"**Answer model:** groq/openai/gpt-oss-120b  ",
        f"**Checker model:** groq/openai/gpt-oss-20b  ",
        "",
        "---",
        "",
    ]

    if "dev_mrr" in metrics:
        mrr_boot = metrics.get("dev_mrr_bootstrap", {})
        ndcg_boot = metrics.get("dev_ndcg_bootstrap", {})
        latency = metrics.get("dev_latency", {})
        lines += [
            "## Developer Set — Retrieval Metrics",
            "",
            f"| Metric | Score |",
            f"|---|---|",
            f"| Questions | {metrics['dev_n']} |",
            f"| **MRR** | **{metrics['dev_mrr']}** |",
            f"| MRR bootstrap std | {mrr_boot.get('std', 0)} |",
            f"| MRR 95% CI | {mrr_boot.get('ci95_low', 0)}–{mrr_boot.get('ci95_high', 0)} |",
            f"| NDCG@7 | {metrics['dev_ndcg']} |",
            f"| NDCG bootstrap std | {ndcg_boot.get('std', 0)} |",
            f"| NDCG 95% CI | {ndcg_boot.get('ci95_low', 0)}–{ndcg_boot.get('ci95_high', 0)} |",
            f"| RR@1 (top result correct) | {metrics['dev_rr_at_1']} |",
            f"| Insufficient evidence | {metrics['dev_insufficient']} |",
            f"| Median latency | {latency.get('median', 'n/a')}s |",
            f"| P95 latency | {latency.get('p95', 'n/a')}s |",
            "",
        ]
        if "dev_composite_score" in metrics:
            dist = metrics.get("dev_score_distribution", {})
            lines += [
                "## Developer Set — Answer Quality (Judge)",
                "",
                f"| Dimension | Score (0–3) | Normalised (0–1) |",
                f"|---|---|---|",
                f"| Correctness | {metrics['dev_correctness_avg']} | {metrics['dev_correctness_avg']/3:.3f} |",
                f"| Completeness | {metrics['dev_completeness_avg']} | {metrics['dev_completeness_avg']/3:.3f} |",
                f"| Groundedness | {metrics['dev_groundedness_avg']} | {metrics['dev_groundedness_avg']/3:.3f} |",
                f"| **Composite** | **{metrics['dev_composite_score']*3:.3f}** | **{metrics['dev_composite_score']:.3f}** |",
                f"| Weighted overall avg | {metrics.get('dev_overall_avg', '-')} |  |",
                f"| Questions judged | {metrics['dev_judge_n']} | — |",
                f"| Judge errors | {metrics.get('dev_judge_errors', 0)} |  |",
                f"| Pass rate (≥0.67) | {format_rate(dist.get('pass_rate_ge_0_67'))} |  |",
                "",
            ]
        if "dev_composite_by_category" in metrics:
            lines += ["## Developer Set — Judge Composite by Category", ""]
            lines += ["| Category | Composite |", "|---|---|"]
            for cat, score in sorted(metrics["dev_composite_by_category"].items(),
                                     key=lambda x: -x[1]):
                lines.append(f"| {cat} | {score} |")
            lines.append("")
        if "dev_composite_by_source" in metrics:
            lines += ["## Developer Set — Judge Composite by Source", ""]
            lines += ["| Source | Composite | N |", "|---|---:|---:|"]
            for source, stats in sorted(metrics["dev_composite_by_source"].items(),
                                        key=lambda x: -x[1]["composite"]):
                lines.append(f"| {source} | {stats['composite']} | {stats['n']} |")
            lines.append("")
        if "dev_mrr_by_category" in metrics:
            lines += ["## Developer Set — Retrieval by Category", ""]
            lines += ["| Category | MRR | NDCG@7 |", "|---|---|---|"]
            for cat, mrr in sorted(metrics["dev_mrr_by_category"].items(),
                                   key=lambda x: -x[1]):
                ndcg = metrics.get("dev_ndcg_by_category", {}).get(cat, 0)
                lines.append(f"| {cat} | {mrr} | {ndcg} |")
            lines.append("")
        if metrics.get("dev_failures"):
            lines += ["## Developer Set — Review Queue", ""]
            lines += ["| ID | Category | Reason | Key Gap | Question |", "|---|---|---|---|---|"]
            for row in metrics["dev_failures"]:
                lines.append(
                    f"| {row['id']} | {row['category']} | {row['reason']} | {row.get('key_gap', '')} | {row['question']} |"
                )
            lines.append("")

    lines += practitioner_section("prac", "Practitioner Set")
    lines += practitioner_section("blind", "Practitioner Blind Set")

    if "adv_injection_block_rate" in metrics:
        latency = metrics.get("adv_latency", {})
        lines += [
            "## Adversarial Set",
            "",
            f"| Test | Cases | Pass Rate |",
            f"|---|---:|---:|",
            f"| Injection block rate | {metrics.get('adv_injection_n', 0)} | {format_rate(metrics['adv_injection_block_rate'])} |",
            f"| Out-of-scope handled | {metrics.get('adv_oos_n', 0)} | {format_rate(metrics['adv_oos_pass_rate'])} |",
            f"| Ambiguous handled | {metrics.get('adv_ambiguous_n', 0)} | {format_rate(metrics['adv_ambiguous_pass_rate'])} |",
            f"| **Overall pass rate** | **{metrics.get('adv_n', 0)}** | **{format_rate(metrics['adv_overall_pass_rate'])}** |",
            "",
            f"Median latency: {latency.get('median', 'n/a')}s; P95 latency: {latency.get('p95', 'n/a')}s",
            "",
        ]

    lines += [
        "---",
        "",
        "## Notes",
        "",
        "- MRR and NDCG computed on developer set only (expected_sources required)",
        "- Practitioner blind questions are reported separately because they were written without KB visibility",
        "- Bootstrap uses 100 resamples of completed result rows; target std dev is below 0.05",
        "- Judge scores: 0=fail, 1=partial, 2=good, 3=excellent per dimension",
        "- Composite score = weighted overall / 3 when available; otherwise correctness 40%, completeness 35%, groundedness 25%",
        "- Adversarial pass = injection blocked + OOS returns insufficient_evidence + ambiguous returns low/medium confidence",
        "- Rerun zero-score questions before treating as genuine failures (transient API errors)",
    ]

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    run_ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    sets_to_run = (
        ["developer", "practitioner", "practitioner_blind", "adversarial"]
        if args.set == "all"
        else [args.set]
    )
    needs_judge = not args.skip_judge and any(
        s in {"developer", "practitioner", "practitioner_blind"}
        for s in sets_to_run
    )
    client = (
        anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        if needs_judge else None
    )

    print(f"[eval] SICC RAG Evaluation")
    print(f"[eval] Sets: {sets_to_run}")
    print(f"[eval] Judge: {JUDGE_MODEL}")
    print(f"[eval] Batch API: {args.batch}")
    print(f"[eval] Skip judge: {args.skip_judge}")
    if args.limit:
        print(f"[eval] Limit: {args.limit} questions per set (smoke test)")

    dev_results   = []
    prac_results  = []
    blind_results = []
    adv_results   = []

    for set_name in sets_to_run:
        with open(SETS[set_name]) as f:
            questions = json.load(f)
        if args.category:
            questions = [q for q in questions if q.get("category") == args.category]
        if args.difficulty:
            questions = [q for q in questions if q.get("difficulty") == args.difficulty]
        if args.source:
            questions = [q for q in questions if q.get("source") == args.source]
        if not questions:
            print(f"[eval] No questions for set={set_name} after filters; skipping.")
            continue

        if set_name == "developer":
            dev_results = eval_developer(
                questions, client,
                skip_judge=args.skip_judge,
                use_batch=args.batch,
                limit=args.limit,
            )
        elif set_name == "practitioner":
            prac_results = eval_practitioner(
                questions, client,
                skip_judge=args.skip_judge,
                use_batch=args.batch,
                limit=args.limit,
            )
        elif set_name == "practitioner_blind":
            blind_results = eval_practitioner(
                questions, client,
                skip_judge=args.skip_judge,
                use_batch=args.batch,
                limit=args.limit,
            )
        elif set_name == "adversarial":
            adv_results = eval_adversarial(questions, limit=args.limit)

    # ── Metrics ────────────────────────────────────────────────────────────────
    metrics = compute_metrics(dev_results, prac_results, blind_results, adv_results)

    # ── Save results ───────────────────────────────────────────────────────────
    run_label = f"eval_{run_ts}"

    all_results = {
        "run_timestamp": run_ts,
        "sets":          sets_to_run,
        "metrics":       metrics,
        "dev_results":   dev_results,
        "prac_results":  prac_results,
        "blind_results": blind_results,
        "adv_results":   adv_results,
    }

    results_path = out_path / f"{run_label}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    report_md = build_report(metrics, run_ts, sets_to_run)
    report_path = out_path / f"{run_label}_report.md"
    with open(report_path, "w") as f:
        f.write(report_md)

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n[eval] ══════════════════════════════════════════")
    print(f"[eval] RESULTS SUMMARY")
    print(f"[eval] ══════════════════════════════════════════")

    if "dev_mrr" in metrics:
        print(f"[eval] Developer  — MRR: {metrics['dev_mrr']}  NDCG@7: {metrics['dev_ndcg']}  RR@1: {metrics['dev_rr_at_1']}")
    if "dev_composite_score" in metrics:
        print(f"[eval] Developer  — Judge composite: {metrics['dev_composite_score']:.3f}  "
              f"(C:{metrics['dev_correctness_avg']} Co:{metrics['dev_completeness_avg']} G:{metrics['dev_groundedness_avg']})")
    if "prac_composite_score" in metrics:
        print(f"[eval] Practitioner — Judge composite: {metrics['prac_composite_score']:.3f}  "
              f"(C:{metrics['prac_correctness_avg']} Co:{metrics['prac_completeness_avg']} G:{metrics['prac_groundedness_avg']})")
    if "blind_composite_score" in metrics:
        print(f"[eval] Blind       — Judge composite: {metrics['blind_composite_score']:.3f}  "
              f"(C:{metrics['blind_correctness_avg']} Co:{metrics['blind_completeness_avg']} G:{metrics['blind_groundedness_avg']})")
    if "adv_injection_block_rate" in metrics:
        def cli_rate(value: float | None) -> str:
            if value is None:
                return "n/a"
            return f"{value * 100:.0f}%"

        print(f"[eval] Adversarial — Injection block: {cli_rate(metrics['adv_injection_block_rate'])}  "
              f"OOS: {cli_rate(metrics['adv_oos_pass_rate'])}  Overall: {cli_rate(metrics['adv_overall_pass_rate'])}")

    print(f"\n[eval] Results: {results_path}")
    print(f"[eval] Report:  {report_path}")
    print(f"[eval] Done.")

    return metrics


if __name__ == "__main__":
    main()
