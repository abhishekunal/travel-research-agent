"""
scripts/index_corpus.py
=======================
One-time script to build the RAG vector database from corpus/*.md files.

What it does:
  1. Reads every markdown file in corpus/
  2. Parses the YAML frontmatter at the top of each file (city, country)
  3. Splits each file on `##` headers — one chunk per top-level section
  4. Embeds each chunk with OpenAI text-embedding-3-small
  5. Stores chunks + metadata in a Chroma persistent collection
  6. Persists the collection to vector_db/

The vector_db/ folder is committed to the repo so Streamlit Cloud doesn't
re-index on every cold start. See plans/WEEKEND_4_PLAN.md.

When to run:
  - Once initially, after setting OPENAI_API_KEY in .env.
  - Whenever you add, edit, or remove a corpus/*.md file.
  - The script deletes and recreates the collection on each run, so repeat
    runs produce a clean state (not a merge).

Cost (as of 2026 Q4):
  text-embedding-3-small is $0.02 / 1M tokens. The 8-city corpus is well
  under 100k tokens, so a full re-index costs around $0.001 per run.

Run from the project root:
  python scripts/index_corpus.py
"""

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import MarkdownHeaderTextSplitter
import tiktoken


# ── Config ────────────────────────────────────────────────────────────
# All paths are resolved relative to the project root (parent of scripts/)
# so the script can be run from anywhere, not only from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "corpus"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"

COLLECTION_NAME = "travel_knowledge"
EMBEDDING_MODEL = "text-embedding-3-small"
PRICE_PER_1M_TOKENS = 0.02  # text-embedding-3-small pricing, used for the estimate only

# Split only on H2 (##), not H3 (###).
# One chunk per top-level section ("Understand", "Get around", etc.).
# This gives chunks of ~100–400 tokens, which is the sweet spot for
# embedding-based retrieval — small enough to be focused, large enough
# to carry useful context on its own.
HEADERS_TO_SPLIT_ON = [("##", "section")]


# ── Helpers ───────────────────────────────────────────────────────────
def parse_frontmatter(text: str) -> tuple[dict, str]:
    """
    Split a markdown file into (frontmatter_dict, body_text).

    Expects YAML-style frontmatter at the very top:
        ---
        city: Tokyo
        country: Japan
        ---
        # Tokyo
        ...

    Returns ({}, text) unchanged if no frontmatter is found. We parse this
    ourselves rather than pulling in PyYAML — the format is simple enough,
    and avoiding a dependency keeps requirements.txt lean.
    """
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
    if not match:
        return {}, text

    raw_meta, body = match.groups()
    meta = {}
    for line in raw_meta.strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, body


def chunk_file(path: Path, splitter: MarkdownHeaderTextSplitter) -> list[dict]:
    """
    Read a corpus markdown file, split it into chunks, attach metadata.

    Returns a list of dicts: {"text": str, "metadata": dict}.

    Each chunk is prefixed with "# {city}\n## {section}\n\n" so that when
    retrieval pulls a chunk in isolation, the LLM still sees which city
    and section it came from. Without this prefix a chunk like
    "Metro single ride: ¥170–320" could be about any city.
    """
    full = path.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(full)

    city = frontmatter.get("city", path.stem.replace("-", " ").title())
    country = frontmatter.get("country", "Unknown")

    # MarkdownHeaderTextSplitter returns LangChain Document objects.
    # Each has .page_content (the chunk text) and .metadata (header values).
    docs = splitter.split_text(body)

    chunks = []
    for doc in docs:
        # Skip the pre-first-H2 fragment the splitter emits for each file.
        # It only contains the "# {city}" H1 title with no body content,
        # so retrieving it is noise. Skipping is signaled by missing
        # section metadata.
        if "section" not in doc.metadata:
            continue

        section = doc.metadata["section"]
        chunk_text = f"# {city}\n## {section}\n\n{doc.page_content}"
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "city": city,
                "country": country,
                "section": section,
                "source_file": path.name,
            }
        })
    return chunks


def estimate_cost(texts: list[str]) -> tuple[int, float]:
    """Return (total_tokens, estimated_cost_usd) for a list of texts."""
    enc = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    total_tokens = sum(len(enc.encode(t)) for t in texts)
    cost_usd = (total_tokens / 1_000_000) * PRICE_PER_1M_TOKENS
    return total_tokens, cost_usd


# ── Main ──────────────────────────────────────────────────────────────
def main() -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: OPENAI_API_KEY not found. "
            "Add it to .env in the project root."
        )

    if not CORPUS_DIR.exists():
        sys.exit(f"ERROR: corpus folder not found at {CORPUS_DIR}")

    corpus_files = sorted(CORPUS_DIR.glob("*.md"))
    if not corpus_files:
        sys.exit(f"ERROR: no .md files in {CORPUS_DIR}")

    print(f"Found {len(corpus_files)} corpus files in {CORPUS_DIR.name}/")

    # Chunk every file.
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON
    )
    all_chunks: list[dict] = []
    for path in corpus_files:
        chunks = chunk_file(path, splitter)
        print(f"  {path.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Cost estimate before calling OpenAI.
    # The first run also downloads tiktoken's encoding file (~2 MB) from
    # Azure storage — may take a few seconds on first execution.
    total_tokens, cost_usd = estimate_cost([c["text"] for c in all_chunks])
    print(f"Total tokens:   {total_tokens:,}")
    print(f"Estimated cost: ${cost_usd:.4f}")

    # Build the Chroma collection.
    print(f"\nBuilding Chroma collection at {VECTOR_DB_DIR}/")
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

    # Chroma lets us attach an embedding function to the collection — then
    # calling .add() with raw text handles the embedding automatically.
    # Same function is attached at query time so queries are embedded with
    # the same model.
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBEDDING_MODEL,
    )

    # Delete-and-recreate so re-running produces a clean state rather than
    # accumulating stale chunks from previous runs.
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass  # Collection didn't exist yet — fine.

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
    )

    # One add() call = one batched request to OpenAI's embeddings endpoint.
    # For 80ish chunks the batch is tiny; no need to chunk the chunks.
    print(f"  Embedding and inserting {len(all_chunks)} chunks...")
    collection.add(
        ids=[f"chunk_{i:03d}" for i in range(len(all_chunks))],
        documents=[c["text"] for c in all_chunks],
        metadatas=[c["metadata"] for c in all_chunks],
    )

    print(
        f"\n✓ Done. Collection '{COLLECTION_NAME}' has "
        f"{collection.count()} chunks."
    )
    print(f"✓ Persisted to {VECTOR_DB_DIR}/")


if __name__ == "__main__":
    main()
