"""
scripts/diagnostics/tsne_viz.py — SICC Embedding Space Visualisation
Produces one HTML file with both 2D and 3D t-SNE plots side by side.
Chunks coloured by doc_type or risk_domain.

Usage:
    uv run python scripts/diagnostics/tsne_viz.py
    uv run python scripts/diagnostics/tsne_viz.py --color risk_domain --out tsne_risk.html
"""

import argparse
import os

import chromadb
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
from openai import OpenAI as OpenAIClient
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE

load_dotenv()

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"]     = "False"

COLLECTION_NAME = "sicc_kb"
EMBED_MODEL     = "text-embedding-3-small"
CHROMA_DB_PATH  = "chroma_db"

PALETTE = [
    "#3b82f6", "#f87171", "#34d399", "#fb923c", "#c084fc",
    "#60a5fa", "#fbbf24", "#f472b6", "#a3e635", "#38bdf8",
]


def parse_args():
    p = argparse.ArgumentParser(description="t-SNE embedding space visualisation")
    p.add_argument("--db",         default=CHROMA_DB_PATH)
    p.add_argument("--color",      default="doc_type",
                   choices=["doc_type", "risk_domain", "doc_category", "source"],
                   help="Metadata field to colour by")
    p.add_argument("--out",        default="tsne_viz.html")
    p.add_argument("--perplexity", type=int, default=30)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def run_tsne(db_path: str, color_field: str, out_path: str,
             perplexity: int, seed: int):

    _oai = OpenAIClient(api_key=os.environ["OPENAI_API_KEY"])

    class EmbedFn(chromadb.EmbeddingFunction):
        def __call__(self, input):
            input = [t for t in input if t and t.strip()]
            if not input:
                return []
            response = _oai.embeddings.create(model=EMBED_MODEL, input=input)
            return [d.embedding for d in response.data]

    chroma_client = chromadb.PersistentClient(path=os.path.abspath(db_path))
    collection    = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=EmbedFn(),
        metadata={"hnsw:space": "cosine"},
    )

    print(f"[tsne] Fetching {collection.count()} chunks from ChromaDB...")
    results    = collection.get(include=["embeddings", "metadatas", "documents"])

    if not results["ids"]:
        print("[tsne] Collection is empty — run ingest.py first.")
        return

    embeddings = np.array(results["embeddings"])
    metadatas  = results["metadatas"]
    ids        = results["ids"]

    # ── t-SNE 2D ──────────────────────────────────────────────────────────────
    print(f"[tsne] Running t-SNE 2D on {len(embeddings)} vectors "
          f"(perplexity={perplexity}, seed={seed})...")
    coords_2d = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(embeddings) - 1),
        random_state=seed,
        max_iter=1000,
        metric="cosine",
    ).fit_transform(embeddings)

    # ── t-SNE 3D ──────────────────────────────────────────────────────────────
    print(f"[tsne] Running t-SNE 3D...")
    coords_3d = TSNE(
        n_components=3,
        perplexity=min(perplexity, len(embeddings) - 1),
        random_state=seed,
        max_iter=1000,
        metric="cosine",
    ).fit_transform(embeddings)

    # ── Metadata ───────────────────────────────────────────────────────────────
    color_vals  = [m.get(color_field, "unknown") for m in metadatas]
    sources     = [m.get("source", "unknown") for m in metadatas]
    headlines   = [m.get("headline", "")[:60] for m in metadatas]
    summaries   = [m.get("summary", "")[:100] for m in metadatas]

    unique_vals = sorted(set(color_vals))
    color_map   = {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(unique_vals)}

    hover_text = [
        f"<b>{ids[i]}</b><br>"
        f"Source: {sources[i]}<br>"
        f"Headline: {headlines[i]}<br>"
        f"Summary: {summaries[i]}<br>"
        f"{color_field}: {color_vals[i]}"
        for i in range(len(ids))
    ]

    # ── Build figure: 2D left, 3D right ───────────────────────────────────────
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter"}, {"type": "scatter3d"}]],
        subplot_titles=(
            f"2D t-SNE — {color_field}",
            f"3D t-SNE — {color_field}",
        ),
        horizontal_spacing=0.05,
    )

    for val in unique_vals:
        mask = [i for i, v in enumerate(color_vals) if v == val]
        col  = color_map[val]

        fig.add_trace(go.Scatter(
            x=[coords_2d[i, 0] for i in mask],
            y=[coords_2d[i, 1] for i in mask],
            mode="markers",
            name=val,
            marker=dict(size=7, color=col, opacity=0.8),
            text=[hover_text[i] for i in mask],
            hoverinfo="text",
            legendgroup=val,
            showlegend=True,
        ), row=1, col=1)

        fig.add_trace(go.Scatter3d(
            x=[coords_3d[i, 0] for i in mask],
            y=[coords_3d[i, 1] for i in mask],
            z=[coords_3d[i, 2] for i in mask],
            mode="markers",
            name=val,
            marker=dict(size=4, color=col, opacity=0.8),
            text=[hover_text[i] for i in mask],
            hoverinfo="text",
            legendgroup=val,
            showlegend=False,
        ), row=1, col=2)

    fig.update_layout(
        title=dict(
            text=f"SICC KB Embedding Space · {len(embeddings)} chunks · {color_field}",
            font=dict(size=14, color="#f1f5f9"),
        ),
        paper_bgcolor="#0f1923",
        plot_bgcolor="#0f1923",
        font=dict(color="#94a3b8", family="DM Sans"),
        legend=dict(
            bgcolor="#131c2e",
            bordercolor="#1e2d45",
            borderwidth=1,
            font=dict(size=11),
            title=dict(text=color_field, font=dict(size=11, color="#64748b")),
        ),
        width=1400,
        height=680,
        margin=dict(l=20, r=20, t=80, b=20),
    )

    fig.update_xaxes(gridcolor="#1e2d45", linecolor="#1e2d45",
                     zerolinecolor="#1e2d45", title_text="dim 1", row=1, col=1)
    fig.update_yaxes(gridcolor="#1e2d45", linecolor="#1e2d45",
                     zerolinecolor="#1e2d45", title_text="dim 2", row=1, col=1)
    fig.update_scenes(
        xaxis=dict(backgroundcolor="#0f1923", gridcolor="#1e2d45",
                   showbackground=True, title_text="dim 1"),
        yaxis=dict(backgroundcolor="#0f1923", gridcolor="#1e2d45",
                   showbackground=True, title_text="dim 2"),
        zaxis=dict(backgroundcolor="#0f1923", gridcolor="#1e2d45",
                   showbackground=True, title_text="dim 3"),
        bgcolor="#0f1923",
    )

    fig.write_html(out_path)
    print(f"[tsne] ✓ Saved to {out_path}")
    print(f"[tsne] Open: file://{os.path.abspath(out_path)}")
    print(f"[tsne] Chunks: {len(embeddings)} | Categories: {len(unique_vals)}")
    for val, cnt in sorted(
        {v: color_vals.count(v) for v in unique_vals}.items(), key=lambda x: -x[1]
    ):
        print(f"  {val:40s} {cnt:4d} chunks")


if __name__ == "__main__":
    args = parse_args()
    run_tsne(
        db_path=args.db,
        color_field=args.color,
        out_path=args.out,
        perplexity=args.perplexity,
        seed=args.seed,
    )
