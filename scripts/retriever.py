"""
retriever.py
────────────
Two retrieval strategies for the benchmark:

  Strategy A — RawRetriever
    Embed query as-is → cosine search → top-K chunks

  Strategy B — QueryExpandedRetriever
    Pass query through QueryReWriter (wraps MockGenerativeModel)
    → embed expanded query → cosine search → top-K chunks

Both expose a .retrieve(query, top_k) → List[RetrievalResult] interface
so the benchmark runner can treat them identically.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from langchain.schema import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Wraps a retrieved chunk with strategy metadata attached."""
    document:      Document
    score:         float
    chunk_id:      str
    rewritten_query: Optional[str] = None   # populated by Strategy B only
    latency_ms:    float = 0.0


# ---------------------------------------------------------------------------
# QueryReWriter  (Strategy B helper)
# ---------------------------------------------------------------------------

class QueryReWriter:
    """
    Wraps MockGenerativeModel (or a real LLM) to rewrite / expand queries.

    Usage
    -----
    rewriter = QueryReWriter(gen_model)
    expanded = rewriter.rewrite("How does the system handle peak load?")
    # → "system load balancing peak traffic handling capacity scaling"
    """

    def __init__(self, gen_model):
        """
        gen_model: any object with .generate_content(prompt) -> response.text
        Compatible with MockGenerativeModel and vertexai.GenerativeModel.
        """
        self._model = gen_model

    def rewrite(self, query: str) -> str:
        """
        Expand the query into a richer keyword/semantic string.
        Returns the raw expansion string.
        """
        prompt = (
            f"Rewrite the following search query into a list of relevant keywords "
            f"and related concepts that would help retrieve relevant technical documents. "
            f"Return ONLY the expanded query, no explanation.\n\n"
            f"Query: {query}"
        )
        response = self._model.generate_content(prompt)
        expanded = response.text.strip()
        logger.debug(f"QueryReWriter: '{query[:50]}' → '{expanded[:60]}'")
        return expanded


# ---------------------------------------------------------------------------
# Strategy A — Raw vector retrieval
# ---------------------------------------------------------------------------

class RawRetriever:
    """
    Embeds the query as-is and does a cosine similarity search.

    Usage
    -----
    retriever = RawRetriever(vector_store)
    results = retriever.retrieve("How does the system handle peak load?", top_k=3)
    """

    def __init__(self, vector_store):
        """vector_store: VectorStore instance (from vector_store.py)"""
        self._vs = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        t0 = time.perf_counter()

        scored_docs = self._vs.query_with_scores(query, top_k=top_k)

        latency_ms = (time.perf_counter() - t0) * 1000
        results = [
            RetrievalResult(
                document=doc,
                score=float(score),
                chunk_id=str(doc.metadata.get("chunk_id", i)),
                latency_ms=latency_ms / len(scored_docs) if scored_docs else 0,
            )
            for i, (doc, score) in enumerate(scored_docs)
        ]
        logger.info(
            f"RawRetriever: '{query[:50]}' → {len(results)} results "
            f"in {latency_ms:.1f}ms"
        )
        return results


# ---------------------------------------------------------------------------
# Strategy B — Query-expanded retrieval
# ---------------------------------------------------------------------------

class QueryExpandedRetriever:
    """
    Rewrites the query via QueryReWriter then does cosine similarity search.

    Usage
    -----
    retriever = QueryExpandedRetriever(vector_store, gen_model)
    results = retriever.retrieve("How does the system handle peak load?", top_k=3)
    # results[i].rewritten_query shows what the model expanded to
    """

    def __init__(self, vector_store, gen_model):
        self._vs      = vector_store
        self._rewriter = QueryReWriter(gen_model)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        t0 = time.perf_counter()

        expanded_query = self._rewriter.rewrite(query)
        scored_docs    = self._vs.query_with_scores(expanded_query, top_k=top_k)

        latency_ms = (time.perf_counter() - t0) * 1000
        results = [
            RetrievalResult(
                document=doc,
                score=float(score),
                chunk_id=str(doc.metadata.get("chunk_id", i)),
                rewritten_query=expanded_query,
                latency_ms=latency_ms / len(scored_docs) if scored_docs else 0,
            )
            for i, (doc, score) in enumerate(scored_docs)
        ]
        logger.info(
            f"QueryExpandedRetriever: '{query[:50]}' → expanded='{expanded_query[:50]}' "
            f"→ {len(results)} results in {latency_ms:.1f}ms"
        )
        return results


# --------------------------------------------------------------------------- #
#  Smoke test                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")

    from langchain.schema import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    from vector_store import VectorStore
    from mocks import MockGenerativeModel

    # Build a minimal vector store with fake docs
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = VectorStore(backend="chromadb", db_location="./db_test_ret", embeddings=emb)
    vs.add([
        Document(page_content="The load balancer distributes traffic across multiple servers to prevent overload.", metadata={"chunk_id": 0}),
        Document(page_content="Auto-scaling adjusts server capacity based on real-time traffic metrics.", metadata={"chunk_id": 1}),
        Document(page_content="Circuit breakers prevent cascading failures in distributed systems.", metadata={"chunk_id": 2}),
        Document(page_content="ACID transactions ensure data consistency across database operations.", metadata={"chunk_id": 3}),
    ])

    query = "How does the system handle peak load?"

    # Strategy A
    raw = RawRetriever(vs)
    a_results = raw.retrieve(query, top_k=2)
    print(f"\nStrategy A (raw) — top 2:")
    for r in a_results:
        print(f"  chunk_id={r.chunk_id}  score={r.score:.4f}  |  {r.document.page_content[:60]}")

    # Strategy B
    gen = MockGenerativeModel()
    exp = QueryExpandedRetriever(vs, gen)
    b_results = exp.retrieve(query, top_k=2)
    print(f"\nStrategy B (expanded) — top 2:")
    print(f"  rewritten: '{b_results[0].rewritten_query}'")
    for r in b_results:
        print(f"  chunk_id={r.chunk_id}  score={r.score:.4f}  |  {r.document.page_content[:60]}")

    print("\nRetriever smoke test: OK")