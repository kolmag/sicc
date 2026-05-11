"""
scripts/ingest.py — SICC RAG Knowledge Base Ingestion
Auditor Expert pattern + contextual retrieval + document-type-specific chunking

Pipeline per document:
  1. Parse metadata header
  2. Split by doc_type strategy (clause / paragraph / row-level)
  3. Generate headline + practitioner queries per chunk (Haiku, T=0)
  4. Generate contextual retrieval context (Haiku, T=0) — Anthropic 2024 pattern
  5. Build embed_text: context + headline + queries + original_text
  6. Embed with text-embedding-3-small
  7. Store in ChromaDB with full metadata

Usage:
    uv run python scripts/ingest.py
    uv run python scripts/ingest.py --kb knowledge-base/markdown --db chroma_db --reset
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
import argparse
import re
import time
from pathlib import Path

import anthropic
import chromadb
from openai import OpenAI as OpenAIClient
from dotenv import load_dotenv
from langfuse import Langfuse, observe
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

CHUNK_SIZE        = 400          # tokens — BGE 512-token limit × safety margin
OVERLAP_TOKENS    = 40           # paragraph chunking overlap
SYNTHETIC_QUERIES = 3            # practitioner queries per chunk
HAIKU_MODEL       = "claude-haiku-4-5-20251001"
EMBED_MODEL       = "text-embedding-3-small"
COLLECTION_NAME   = "sicc_kb"
README_EXCLUDE    = {"README.md", "readme.md"}

# Document type → chunking strategy
CHUNKING_STRATEGY = {
    "AS9100D_clause":    "clause",
    "IATF_clause":       "clause",
    "PPAP_requirement":  "clause",
    "APQP_guide":        "paragraph",
    "SQE_procedure":     "paragraph",
    "scorecard":         "row",
}

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Ingest SICC KB into ChromaDB")
    p.add_argument("--kb",    default="knowledge-base/markdown",
                   help="Path to KB markdown directory")
    p.add_argument("--db",    default="chroma_db",
                   help="Path to ChromaDB directory")
    p.add_argument("--reset", action="store_true",
                   help="Delete and recreate the collection before ingesting")
    p.add_argument("--dry-run", action="store_true",
                   help="Parse and chunk only — do not embed or store")
    return p.parse_args()


# ── Clients ───────────────────────────────────────────────────────────────────

def build_clients(db_path: str):
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    chroma_client = chromadb.PersistentClient(
        path=os.path.abspath(db_path)
    )
    _oai = OpenAIClient(api_key=os.environ["OPENAI_API_KEY"])

    class EmbedFn(chromadb.EmbeddingFunction):
        def __call__(self, input):
            response = _oai.embeddings.create(model=EMBED_MODEL, input=input)
            return [d.embedding for d in response.data]

    embed_fn = EmbedFn()

    langfuse = Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )

    return anthropic_client, chroma_client, embed_fn, langfuse


# ── Metadata parsing ──────────────────────────────────────────────────────────

def parse_metadata_header(content: str) -> dict:
    """
    Extract metadata from HTML comment block at top of markdown:
    <!-- key: value -->
    Returns dict with defaults if keys are missing.
    """
    defaults = {
        "doc_category": "procedure",
        "doc_type":     "SQE_procedure",
        "supplier":     "GLOBAL",
        "commodity":    "GENERAL",
        "risk_domain":  "quality",
    }

    comment_match = re.search(r"<!--(.*?)-->", content, re.DOTALL)
    if not comment_match:
        return defaults

    meta = dict(defaults)
    for line in comment_match.group(1).strip().splitlines():
        line = line.strip()
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip()

    return meta


def strip_metadata_comment(content: str) -> str:
    """Remove the metadata comment block from the document content."""
    return re.sub(r"<!--.*?-->", "", content, count=1, flags=re.DOTALL).strip()


# ── Chunking strategies ───────────────────────────────────────────────────────

def chunk_by_clause(content: str) -> list[dict]:
    """
    Clause-level chunking for standards (AS9100D, IATF, PPAP).
    Splits on ## headings — each clause = one chunk.
    """
    chunks = []
    sections = re.split(r"\n(?=## )", content)

    for section in sections:
        section = section.strip()
        if not section or len(section) < 50:
            continue

        # Extract headline from first ## line
        lines = section.splitlines()
        headline = lines[0].lstrip("#").strip() if lines else "Unknown clause"
        body = "\n".join(lines[1:]).strip()

        if body:
            chunks.append({"headline": headline, "original_text": body})
        elif section:
            chunks.append({"headline": headline, "original_text": section})

    return chunks


def chunk_by_paragraph(content: str, chunk_size: int = CHUNK_SIZE) -> list[dict]:
    """
    Paragraph-level chunking for procedures and guides.
    Splits on double newlines, respects ## section headers as chunk anchors.
    Merges short paragraphs up to chunk_size tokens.
    """
    chunks = []
    current_headline = "General"
    current_text = []
    current_tokens = 0

    # Rough token estimate: 1 token ≈ 4 chars
    def token_estimate(text: str) -> int:
        return len(text) // 4

    paragraphs = re.split(r"\n\n+", content)

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # New section header — flush current chunk, update headline
        if para.startswith("## ") or para.startswith("### "):
            if current_text:
                chunks.append({
                    "headline":      current_headline,
                    "original_text": "\n\n".join(current_text),
                })
                current_text = []
                current_tokens = 0
            current_headline = para.lstrip("#").strip()
            continue

        para_tokens = token_estimate(para)

        # If adding this paragraph exceeds chunk_size, flush first
        if current_tokens + para_tokens > chunk_size and current_text:
            chunks.append({
                "headline":      current_headline,
                "original_text": "\n\n".join(current_text),
            })
            current_text = []
            current_tokens = 0

        current_text.append(para)
        current_tokens += para_tokens

    # Flush remainder
    if current_text:
        chunks.append({
            "headline":      current_headline,
            "original_text": "\n\n".join(current_text),
        })

    return chunks


def chunk_by_row(content: str) -> list[dict]:
    """
    Row-level chunking for scorecards and checklists (PPAP checklist, KPI definitions).
    Each ## section + its table rows = one chunk.
    Non-table sections fall back to paragraph chunking.
    """
    chunks = []
    sections = re.split(r"\n(?=## )", content)

    for section in sections:
        section = section.strip()
        if not section or len(section) < 30:
            continue

        lines = section.splitlines()
        headline = lines[0].lstrip("#").strip() if lines else "Unknown"
        body = "\n".join(lines[1:]).strip()

        if not body:
            continue

        # If section contains a table, split into header + data rows
        table_lines = [l for l in body.splitlines() if l.strip().startswith("|")]
        if len(table_lines) >= 3:
            # Keep header + separator + each data row as separate chunks
            header_rows = []
            data_rows   = []
            in_header   = True

            for line in body.splitlines():
                if line.strip().startswith("|"):
                    if in_header and re.match(r"\|[-| ]+\|", line):
                        in_header = False
                        header_rows.append(line)
                    elif in_header:
                        header_rows.append(line)
                    else:
                        data_rows.append(line)
                else:
                    # Non-table prose in this section — batch with header
                    header_rows.append(line)

            header_text = "\n".join(header_rows)

            if data_rows:
                # Group every 3 rows into one chunk (avoids single-row micro-chunks)
                for i in range(0, len(data_rows), 3):
                    row_group = data_rows[i:i+3]
                    chunk_text = header_text + "\n" + "\n".join(row_group)
                    chunks.append({
                        "headline":      f"{headline} (rows {i+1}–{i+len(row_group)})",
                        "original_text": chunk_text,
                    })
            else:
                chunks.append({"headline": headline, "original_text": body})
        else:
            # No table — treat as a paragraph chunk
            chunks.append({"headline": headline, "original_text": body})

    return chunks


def chunk_document(content: str, doc_type: str) -> list[dict]:
    """Route to the correct chunking strategy based on doc_type."""
    strategy = CHUNKING_STRATEGY.get(doc_type, "paragraph")

    if strategy == "clause":
        return chunk_by_clause(content)
    elif strategy == "row":
        return chunk_by_row(content)
    else:
        return chunk_by_paragraph(content)


# ── LLM enrichment ────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_enrichment(
    client: anthropic.Anthropic,
    headline: str,
    original_text: str,
    doc_title: str,
    doc_type: str,
) -> dict:
    """
    Generate summary + practitioner_queries for a chunk.
    Uses Claude Haiku, T=0. Returns {summary, practitioner_queries}.
    """
    prompt = f"""You are an expert supplier quality engineer indexing a knowledge base chunk.

Document: {doc_title}
Document type: {doc_type}
Chunk headline: {headline}

Chunk content:
{original_text[:1500]}

Generate:
1. A 1-sentence summary of what this chunk covers (max 30 words).
2. Exactly {SYNTHETIC_QUERIES} practitioner questions that this chunk would answer. Write them as a quality engineer or procurement manager would ask them — use domain vocabulary (PPM, SCAR, OTD, PPAP, etc.).

Respond in this exact format — no other text:
SUMMARY: <one sentence>
Q1: <question>
Q2: <question>
Q3: <question>"""

    response = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=300,
        temperature=0,
        stop_sequences=["```"],
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    summary = ""
    queries = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("SUMMARY:"):
            summary = line[8:].strip()
        elif re.match(r"Q\d:", line):
            queries.append(re.sub(r"Q\d:\s*", "", line).strip())

    return {
        "summary":              summary or headline,
        "practitioner_queries": queries[:SYNTHETIC_QUERIES],
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_contextual_retrieval(
    client: anthropic.Anthropic,
    whole_document: str,
    chunk_content: str,
) -> str:
    """
    Anthropic contextual retrieval pattern (2024).
    Generates 50-100 token context situating the chunk within the document.
    Prepended to chunk before embedding.
    """
    prompt = f"""<document>{whole_document[:4000]}</document>

Here is the chunk we want to situate:
<chunk>{chunk_content[:600]}</chunk>

Please give a short succinct context to situate this chunk within the overall document for retrieval purposes. Answer only with the succinct context and nothing else."""

    response = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=120,
        temperature=0,
        stop_sequences=["```"],
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text.strip()


# ── Embed text builder ────────────────────────────────────────────────────────

def build_embed_text(
    context: str,
    headline: str,
    summary: str,
    practitioner_queries: list[str],
    original_text: str,
) -> str:
    """
    Combine all enrichment into the text that gets embedded.
    Pattern: context + headline + summary + queries + original_text
    Same pattern as Auditor Expert — proven in benchmark.
    """
    queries_text = "\n".join(f"- {q}" for q in practitioner_queries)
    return (
        f"{context}\n\n"
        f"### {headline}\n\n"
        f"{summary}\n\n"
        f"Practitioner questions this answers:\n{queries_text}\n\n"
        f"---\n{original_text}"
    )


# ── Main ingestion ────────────────────────────────────────────────────────────

@observe(name="ingest_document")
def ingest_document(
    filepath: Path,
    collection: chromadb.Collection,
    anthropic_client: anthropic.Anthropic,
    dry_run: bool = False,
) -> int:
    """Ingest one markdown document. Returns number of chunks ingested."""

    if filepath.name in README_EXCLUDE:
        print(f"  SKIP {filepath.name} (excluded)")
        return 0

    raw_content = filepath.read_text(encoding="utf-8")
    metadata    = parse_metadata_header(raw_content)
    content     = strip_metadata_comment(raw_content)

    # Document title from first # heading
    title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
    doc_title   = title_match.group(1).strip() if title_match else filepath.stem

    doc_type    = metadata.get("doc_type", "SQE_procedure")
    chunks      = chunk_document(content, doc_type)


    if not chunks:
        print(f"  WARN {filepath.name} — no chunks produced")
        return 0

    ids        = []
    documents  = []
    metadatas  = []

    for i, chunk in enumerate(chunks):
        headline      = chunk["headline"]
        original_text = chunk["original_text"]

        if len(original_text.strip()) < 30:
            continue

        if dry_run:
            print(f"    [dry] chunk {i:03d} | {headline[:60]}")
            continue

        # Contextual retrieval context
        context = generate_contextual_retrieval(
            anthropic_client, content, original_text
        )

        # Enrichment: summary + practitioner queries
        enrichment = generate_enrichment(
            anthropic_client, headline, original_text, doc_title, doc_type
        )

        # Build embed_text
        embed_text = build_embed_text(
            context=context,
            headline=headline,
            summary=enrichment["summary"],
            practitioner_queries=enrichment["practitioner_queries"],
            original_text=original_text,
        )

        chunk_id = f"{filepath.stem}_{i:03d}"

        ids.append(chunk_id)
        documents.append(embed_text)
        metadatas.append({
            "source":               filepath.name,
            "doc_title":            doc_title,
            "doc_category":         metadata.get("doc_category", "procedure"),
            "doc_type":             doc_type,
            "supplier":             metadata.get("supplier", "GLOBAL"),
            "commodity":            metadata.get("commodity", "GENERAL"),
            "risk_domain":          metadata.get("risk_domain", "quality"),
            "chunk_index":          i,
            "headline":             headline,
            "summary":              enrichment["summary"],
            "original_text":        original_text,
            "context":              context,
            "practitioner_queries": " | ".join(enrichment["practitioner_queries"]),
        })

        # Small delay to avoid Haiku rate limits
        time.sleep(0.3)

    if not dry_run and ids:
        # Upsert in batches of 50
        batch_size = 50
        for start in range(0, len(ids), batch_size):
            collection.upsert(
                ids=ids[start:start+batch_size],
                documents=documents[start:start+batch_size],
                metadatas=metadatas[start:start+batch_size],
            )

    return len(ids)


@observe(name="ingest_all")
def ingest_all(kb_path: str, db_path: str, reset: bool, dry_run: bool):
    """Ingest all markdown documents in the KB directory."""

    anthropic_client, chroma_client, embed_fn, langfuse = build_clients(db_path)

    # Collection setup
    if reset:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
            print(f"[ingest] Deleted existing collection: {COLLECTION_NAME}")
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    kb_dir   = Path(kb_path)
    md_files = sorted([f for f in kb_dir.glob("*.md") if f.name not in README_EXCLUDE])

    if not md_files:
        print(f"[ingest] No markdown files found in {kb_path}")
        return

    print(f"[ingest] Found {len(md_files)} documents in {kb_path}")
    print(f"[ingest] ChromaDB: {os.path.abspath(db_path)}")
    print(f"[ingest] Collection: {COLLECTION_NAME}")
    print(f"[ingest] Mode: {'DRY RUN' if dry_run else 'FULL INGEST'}")
    print(f"[ingest] Contextual retrieval: ON (Haiku, T=0)")
    print(f"[ingest] Synthetic queries: {SYNTHETIC_QUERIES} per chunk")
    print()

    total_chunks = 0

    for filepath in tqdm(md_files, desc="Ingesting documents"):
        print(f"\n→ {filepath.name}")
        n = ingest_document(
            filepath=filepath,
            collection=collection,
            anthropic_client=anthropic_client,
            dry_run=dry_run,
        )
        total_chunks += n
        print(f"  ✓ {n} chunks ingested")

    print(f"\n[ingest] ══════════════════════════════════════")
    print(f"[ingest] Documents processed : {len(md_files)}")
    print(f"[ingest] Total chunks stored : {total_chunks}")
    print(f"[ingest] Collection size     : {collection.count()} chunks")
    print(f"[ingest] ChromaDB location   : {os.path.abspath(db_path)}")
    print(f"[ingest] Done.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    ingest_all(
        kb_path=args.kb,
        db_path=args.db,
        reset=args.reset,
        dry_run=args.dry_run,
    )
