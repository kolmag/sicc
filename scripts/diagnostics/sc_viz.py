"""
scripts/diagnostics/sc_viz.py — SICC Similarity Chart
Pre/post BGE reranker cosine similarity for a given query.
Shows how much the reranker changes the retrieval ordering.

Usage:
    uv run python scripts/diagnostics/sc_viz.py --question "What does PPAP Level 3 require?"
    uv run python scripts/diagnostics/sc_viz.py --question "..." --out sc_viz.html
"""

import argparse
import os

import chromadb
import numpy as np
import plotly.graph_objects as go
from openai import OpenAI as OpenAIClient
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "sicc_kb"
EMBED_MODEL     = "text-embedding-3-small"
CHROMA_DB_PATH  = "chroma_db"
TOP_K           = 15


def parse_args():
    p = argparse.ArgumentParser(description="Similarity chart pre/post reranker")
    p.add_argument("--question", required=True)
    p.add_argument("--db",       default=CHROMA_DB_PATH)
    p.add_argument("--out",      default="sc_viz.html")
    p.add_argument("--top-k",    type=int, default=TOP_K)
    return p.parse_args()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def run_sc_viz(question: str, db_path: str, out_path: str, top_k: int):
    from litellm import completion
    from openai import OpenAI

    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    chroma_client = chromadb.PersistentClient(path=os.path.abspath(db_path))
    _oai = OpenAIClient(api_key=os.environ["OPENAI_API_KEY"])

    class EmbedFn(chromadb.EmbeddingFunction):
        def __call__(self, input):
            input = [t for t in input if t and t.strip()]
            if not input:
                return []
            response = _oai.embeddings.create(model=EMBED_MODEL, input=input)
            return [d.embedding for d in response.data]

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=EmbedFn(),
        metadata={"hnsw:space": "cosine"},
    )

    print(f"[sc_viz] Question: {question}")

    # Generate HyDE
    print("[sc_viz] Generating HyDE...")
    hyde_response = completion(
        model="groq/openai/gpt-oss-120b",
        messages=[{"role": "user", "content":
            f"Write a 2-3 sentence excerpt from a supplier quality procedure that answers: {question}\nExcerpt only:"}],
        temperature=0, max_tokens=150,
    )
    hyde_text = hyde_response.choices[0].message.content.strip()
    print(f"[sc_viz] HyDE: {hyde_text[:100]}...")

    # Embed HyDE
    hyde_embed = openai_client.embeddings.create(
        model=EMBED_MODEL, input=hyde_text
    ).data[0].embedding
    hyde_vec = np.array(hyde_embed)

    # Semantic retrieval (top_k × 2 for reranker input)
    results = collection.query(
        query_texts=[hyde_text],
        n_results=top_k * 2,
        include=["documents", "metadatas", "embeddings", "distances"],
    )

    ids       = results["ids"][0]
    metas     = results["metadatas"][0]
    embeds    = np.array(results["embeddings"][0])
    distances = results["distances"][0]

    # Pre-reranker cosine similarities (HyDE vs chunk embedding)
    pre_scores = [cosine_similarity(hyde_vec, embeds[i]) for i in range(len(ids))]
    labels     = [f"{m.get('source','?')} | {m.get('headline','')[:40]}" for m in metas]

    # BGE reranker (with CPU fallback)
    post_scores = pre_scores[:]  # default = same as pre if reranker unavailable
    reranker_used = "cosine (no reranker)"

    try:
        from sentence_transformers import CrossEncoder
        print("[sc_viz] Running BGE reranker...")
        reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
        chunk_texts = [m.get("original_text", d)[:500]
                       for m, d in zip(metas, results["documents"][0])]
        pairs = [(hyde_text, t) for t in chunk_texts]
        post_scores = reranker.predict(pairs).tolist()
        reranker_used = "bge-reranker-v2-m3"
        print(f"[sc_viz] BGE reranker scores computed.")
    except ImportError:
        print("[sc_viz] sentence-transformers not available — showing cosine only.")

    # Sort by pre-score for display
    order = np.argsort(pre_scores)[::-1][:top_k]

    pre_sorted  = [pre_scores[i]  for i in order]
    post_sorted = [post_scores[i] for i in order]
    labs_sorted = [labels[i]      for i in order]

    # Rank change arrows
    pre_ranks  = {ids[i]: r for r, i in enumerate(np.argsort(pre_scores)[::-1])}
    post_ranks = {ids[i]: r for r, i in enumerate(np.argsort(post_scores)[::-1])}
    rank_deltas = [pre_ranks[ids[i]] - post_ranks[ids[i]] for i in order]

    # Plot
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Pre-reranker (cosine)",
        y=labs_sorted,
        x=pre_sorted,
        orientation="h",
        marker_color="#3b82f6",
        opacity=0.75,
    ))

    fig.add_trace(go.Bar(
        name=f"Post-reranker ({reranker_used})",
        y=labs_sorted,
        x=post_sorted,
        orientation="h",
        marker_color="#f87171",
        opacity=0.75,
    ))

    # Rank delta annotations
    for i, (lab, delta) in enumerate(zip(labs_sorted, rank_deltas)):
        if delta != 0:
            arrow = f"↑{abs(delta)}" if delta > 0 else f"↓{abs(delta)}"
            color = "#34d399" if delta > 0 else "#fb923c"
            fig.add_annotation(
                x=max(pre_sorted[i], post_sorted[i]) + 0.01,
                y=lab,
                text=f"<b>{arrow}</b>",
                showarrow=False,
                font=dict(color=color, size=10),
                xanchor="left",
            )

    fig.update_layout(
        title=f"Similarity Chart — Pre/Post Reranker<br><sup>{question[:80]}</sup>",
        xaxis_title="Score",
        yaxis_title="",
        barmode="group",
        height=max(500, top_k * 45),
        template="plotly_dark",
        paper_bgcolor="#0f1923",
        plot_bgcolor="#0f1923",
        font=dict(color="#94a3b8", size=11),
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=10, r=120, t=80, b=30),
    )

    fig.write_html(out_path)
    print(f"[sc_viz] ✓ Saved to {out_path}")
    print(f"[sc_viz] Open in browser: file://{os.path.abspath(out_path)}")

    # Console summary
    print(f"\n[sc_viz] Top-{top_k} similarity scores:")
    print(f"  {'Rank':>4}  {'Pre':>6}  {'Post':>6}  {'Delta':>6}  Label")
    for i, (lab, pre, post, delta) in enumerate(
            zip(labs_sorted, pre_sorted, post_sorted, rank_deltas)):
        arrow = f"+{delta}" if delta > 0 else str(delta)
        print(f"  {i+1:>4}  {pre:>6.3f}  {post:>6.3f}  {arrow:>6}  {lab[:60]}")


if __name__ == "__main__":
    args = parse_args()
    run_sc_viz(
        question=args.question,
        db_path=args.db,
        out_path=args.out,
        top_k=args.top_k,
    )
