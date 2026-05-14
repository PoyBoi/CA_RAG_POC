"""
retriever.py
────────────
Two retrieval strategies for the benchmark:

  Strategy A — RawRetriever
    Embed query as-is → cosine search → top-K chunks

  Strategy B — QueryExpandedRetriever
    Pass query through QueryReWriter (wraps any LLM)
    → embed expanded query → cosine search → top-K chunks

Both expose a .retrieve(query, top_k) → List[RetrievalResult] interface
so the benchmark runner can treat them identically.
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import List, Optional

from langchain.schema import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Wraps a retrieved chunk with strategy metadata attached."""
    document:        Document
    score:           float
    chunk_id:        str
    rewritten_query: Optional[str] = None   # populated by Strategy B only
    latency_ms:      float = 0.0


# ---------------------------------------------------------------------------
# QueryReWriter  (Strategy B helper)
# ---------------------------------------------------------------------------

class QueryReWriter:
    """
    Wraps any LLM to rewrite / expand queries.

    Handles two interfaces automatically:
      • MockGenerativeModel / Vertex AI  →  .generate_content(prompt).text
      • ChatOllama / LangChain LLMs      →  .invoke(prompt).content

    Usage
    -----
    # With real Ollama LLM (from CoreDependancies):
    rewriter = QueryReWriter(core.llm)

    # With mock (tests / dry-runs):
    rewriter = QueryReWriter(MockGenerativeModel())
    """

    PROMPT_TEMPLATE = (
        "Rewrite the following search query into 8-12 relevant keywords and related "
        "concepts that would help retrieve technical documents. "
        "Return ONLY the keywords as a single comma-separated line. "
        "No explanation, no preamble, no bullet points. Maximum 20 words.\n\n"
        "Query: {query}"
    )

    def __init__(self, llm):
        self._llm = llm

    def rewrite(self, query: str) -> str:
        """Expand the query into a richer keyword/semantic string."""
        prompt = self.PROMPT_TEMPLATE.format(query=query)

        # Duck-type: try LangChain interface (.invoke) first, fall back to Vertex/mock
        if hasattr(self._llm, "invoke"):
            response = self._llm.invoke(prompt)
            # ChatOllama returns AIMessage; .content is the string
            expanded = getattr(response, "content", str(response))
        elif hasattr(self._llm, "generate_content"):
            response = self._llm.generate_content(prompt)
            expanded = response.text
        else:
            raise TypeError(
                f"QueryReWriter: unsupported LLM type '{type(self._llm).__name__}'. "
                "Expected .invoke() (LangChain) or .generate_content() (Vertex/mock)."
            )

        # DeepSeek-R1 wraps reasoning in <think>...</think> — strip before using
        expanded = re.sub(r"<think>.*?</think>", "", expanded, flags=re.DOTALL).strip()

        logger.debug(f"QueryReWriter: '{query[:50]}' -> '{expanded[:80]}'")
        return expanded


# ---------------------------------------------------------------------------
# Strategy A — Raw vector retrieval
# ---------------------------------------------------------------------------

class RawRetriever:
    """Embeds the query as-is and does a cosine similarity search."""

    def __init__(self, vector_store):
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
        logger.info(f"RawRetriever: '{query[:50]}' -> {len(results)} results in {latency_ms:.1f}ms")
        return results


# ---------------------------------------------------------------------------
# Strategy B — Query-expanded retrieval
# ---------------------------------------------------------------------------

class QueryExpandedRetriever:
    """
    Rewrites the query via QueryReWriter then does cosine similarity search.

    Pass core.llm for real LLM expansion, or MockGenerativeModel() for tests.
    """

    def __init__(self, vector_store, llm):
        self._vs       = vector_store
        self._rewriter = QueryReWriter(llm)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        t0 = time.perf_counter()
        expanded_query = self._rewriter.rewrite(query)
        scored_docs    = self._vs.query_with_scores(expanded_query, top_k=top_k)
        latency_ms     = (time.perf_counter() - t0) * 1000

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
            f"QueryExpandedRetriever: '{query[:50]}'"
            f" -> expanded='{expanded_query[:60]}'"
            f" -> {len(results)} results in {latency_ms:.1f}ms"
        )
        return results