"""
scripts/test_retrieval.py
=========================
Verify that RAG retrieval returns the right chunks for sample questions.

Run this after scripts/index_corpus.py. Read-only — doesn't modify the DB.

For each test query, prints the top-3 most-similar chunks with:
  - city / section (from metadata)
  - distance score (lower = closer semantic match)
  - first ~200 characters of the chunk body

What to look for in the output:
  - The RIGHT CITY appears in the top-3 for city-specific queries
    (e.g., "cherry blossoms" → Tokyo, not LA)
  - The RIGHT SECTION appears for topic-specific queries
    (e.g., "how to get around" → the Get around section)
  - Semantic matches work even without word overlap
    (e.g., "pink trees" → Tokyo / Best Time to Visit, which says "cherry blossoms")

If results look off, the fix is usually in corpus content (a topic
is thin or missing in a city file) or chunking (sections too big or
too small). It is rarely in the retriever itself.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
COLLECTION_NAME = "travel_knowledge"
EMBEDDING_MODEL = "text-embedding-3-small"

# A handful of representative queries chosen to probe different retrieval
# capabilities. Edit this list freely to test cases that matter to you.
TEST_QUERIES = [
    # City + topic (easy case)
    "how to get around Paris",
    # Semantic match: user phrase does NOT appear verbatim in the corpus.
    # "Pink trees" should still find Tokyo's cherry blossoms section.
    "when do the pink trees bloom in Japan",
    # Section-specific: safety advice
    "pickpocketing warning",
    # Food culture
    "deep dish pizza culture",
    # Logistics / costs
    "how much to budget per day in Barcelona",
    # Day trips (relatively new section — good test that it's chunked right)
    "weekend getaway near Bangkok",
    # Etiquette (abstract concept)
    "dining etiquette and tipping customs",
]


def main() -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY not found. Add it to .env.")

    if not VECTOR_DB_DIR.exists():
        sys.exit(
            f"ERROR: {VECTOR_DB_DIR} not found. "
            "Run scripts/index_corpus.py first."
        )

    # anonymized_telemetry=False silences the "Failed to send telemetry event"
    # warnings. Chromadb 0.6.3 has a known regression in its telemetry
    # client — the warnings are cosmetic, but noise obscures real output.
    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    # Reuse the same embedding function used during indexing. If these two
    # ever mismatched, queries would compute coordinates in a different
    # space than the stored chunks, and all results would be garbage.
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBEDDING_MODEL,
    )

    try:
        collection = client.get_collection(
            COLLECTION_NAME, embedding_function=ef
        )
    except Exception as e:
        sys.exit(
            f"ERROR: couldn't load collection '{COLLECTION_NAME}'. "
            f"Did indexing complete? Underlying error: {e}"
        )

    print(f"Collection '{COLLECTION_NAME}' loaded with "
          f"{collection.count()} chunks.\n")

    for query in TEST_QUERIES:
        print("─" * 72)
        print(f"Q: {query}")

        # query_texts: chromadb embeds these with the collection's embedding
        # function, then returns the N nearest chunks.
        results = collection.query(query_texts=[query], n_results=3)

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
            # Strip the "# City\n## Section\n\n" prefix we added at index
            # time so the printed snippet is just the body.
            body = doc.split("\n\n", 1)[1] if "\n\n" in doc else doc
            snippet = body[:200].replace("\n", " ").strip()
            print(
                f"  {i}. [{meta['city']} / {meta['section']}] "
                f"(distance: {dist:.3f})"
            )
            print(f"     {snippet}...")
        print()


if __name__ == "__main__":
    main()
