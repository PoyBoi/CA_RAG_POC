"""
benchmark_runner.py
───────────────────
Runs Strategy A (raw) vs Strategy B (query-expanded) on a fixed query set,
produces benchmark_results.json and prints a comparison table.

Output format
─────────────
benchmark_results.json:
  {
    "queries": [
      {
        "input_query": "...",
        "strategy_a": { "top_k": [...], "latency_ms": ... },
        "strategy_b": { "top_k": [...], "rewritten_query": "...", "latency_ms": ... },
        "diff": { "overlap_ids": [...], "only_in_a": [...], "only_in_b": [...] }
      },
      ...
    ]
  }

Console table:
  | Input Query | Strategy A (Raw Vector) | Strategy B (Query Expanded) |
  |-------------|-------------------------|------------------------------|
  | ...         | chunk IDs + scores      | chunk IDs + scores + rewrite |
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Benchmark query set (assignment requires ≥ 3 complex queries)
BENCHMARK_QUERIES = [
    "How does the system handle peak load?",
    "What are the failure recovery mechanisms?",
    "How is data consistency maintained?",
    # TODO: add more domain-specific queries once dataset is finalised
]

TOP_K = 3


class BenchmarkRunner:
    """
    Usage
    -----
    runner = BenchmarkRunner(
        raw_retriever=RawRetriever(vs),
        expanded_retriever=QueryExpandedRetriever(vs, gen_model),
        queries=BENCHMARK_QUERIES,
        top_k=3,
    )
    results = runner.run()
    runner.save("benchmark_results.json")
    runner.print_table()
    """

    def __init__(
        self,
        raw_retriever,
        expanded_retriever,
        queries:    List[str] = None,
        top_k:      int       = TOP_K,
        output_dir: str       = ".",
    ):
        self.raw      = raw_retriever
        self.expanded = expanded_retriever
        self.queries  = queries or BENCHMARK_QUERIES
        self.top_k    = top_k
        self.output_dir = Path(output_dir)
        self._results: Dict[str, Any] = {}

    # ---------------------------------------------------------------------- #
    #  Core                                                                    #
    # ---------------------------------------------------------------------- #

    def run(self) -> Dict[str, Any]:
        """Run both strategies on all queries and collect results."""
        query_results = []

        for query in self.queries:
            logger.info(f"Benchmarking: '{query}'")

            a_hits = self.raw.retrieve(query, top_k=self.top_k)
            b_hits = self.expanded.retrieve(query, top_k=self.top_k)

            a_ids = [r.chunk_id for r in a_hits]
            b_ids = [r.chunk_id for r in b_hits]

            entry = {
                "input_query": query,
                "strategy_a": {
                    "top_k": [
                        {
                            "chunk_id": r.chunk_id,
                            "score":    round(r.score, 4),
                            "snippet":  r.document.page_content[:120],
                        }
                        for r in a_hits
                    ],
                    "latency_ms": round(sum(r.latency_ms for r in a_hits), 2),
                },
                "strategy_b": {
                    "rewritten_query": b_hits[0].rewritten_query if b_hits else "",
                    "top_k": [
                        {
                            "chunk_id": r.chunk_id,
                            "score":    round(r.score, 4),
                            "snippet":  r.document.page_content[:120],
                        }
                        for r in b_hits
                    ],
                    "latency_ms": round(sum(r.latency_ms for r in b_hits), 2),
                },
                "diff": {
                    "overlap_ids": list(set(a_ids) & set(b_ids)),
                    "only_in_a":   list(set(a_ids) - set(b_ids)),
                    "only_in_b":   list(set(b_ids) - set(a_ids)),
                },
            }
            query_results.append(entry)

        self._results = {"queries": query_results}
        return self._results

    # ---------------------------------------------------------------------- #
    #  Output                                                                  #
    # ---------------------------------------------------------------------- #

    def save(self, filename: str = "benchmark_results.json"):
        out = self.output_dir / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(self._results, f, indent=2)
        logger.info(f"Benchmark: saved to '{out}'")
        return out

    def print_table(self):
        """Print a readable comparison table to stdout."""
        if not self._results:
            print("No results yet — run run() first")
            return

        col_w = 35

        header = (
            f"{'Input Query':<{col_w}} | "
            f"{'Strategy A (Raw Vector)':<{col_w}} | "
            f"{'Strategy B (Query Expanded)':<{col_w}}"
        )
        sep = "-" * len(header)
        print(f"\n{sep}")
        print(header)
        print(sep)

        for entry in self._results["queries"]:
            a_summary = ", ".join(
                f"{r['chunk_id']}({r['score']})"
                for r in entry["strategy_a"]["top_k"]
            )
            b_summary = (
                f"rewrite='{entry['strategy_b']['rewritten_query'][:20]}...' | "
                + ", ".join(
                    f"{r['chunk_id']}({r['score']})"
                    for r in entry["strategy_b"]["top_k"]
                )
            )
            print(
                f"{entry['input_query']:<{col_w}} | "
                f"{a_summary:<{col_w}} | "
                f"{b_summary:<{col_w}}"
            )

        print(sep)

    def generate_markdown(self) -> str:
        """
        Generate a Markdown table for retrieval_benchmark.md.
        TODO: also include commentary on observed A vs B differences.
        """
        if not self._results:
            return "No results."

        lines = [
            "| Input Query | Strategy A (Raw Vector) | Strategy B (Query Expanded) |",
            "|---|---|---|",
        ]
        for entry in self._results["queries"]:
            a_str = "; ".join(
                f"chunk {r['chunk_id']} ({r['score']})"
                for r in entry["strategy_a"]["top_k"]
            )
            b_str = (
                f"*rewrite:* {entry['strategy_b']['rewritten_query'][:60]}... "
                + "; ".join(
                    f"chunk {r['chunk_id']} ({r['score']})"
                    for r in entry["strategy_b"]["top_k"]
                )
            )
            lines.append(f"| {entry['input_query']} | {a_str} | {b_str} |")

        return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Smoke test                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")

    from langchain.schema import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    from vector_store import VectorStore
    from mocks import MockGenerativeModel
    from retriever import RawRetriever, QueryExpandedRetriever

    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = VectorStore(backend="chromadb", db_location="./db_test_bench", embeddings=emb)
    vs.add([
        Document(page_content="Load balancers distribute traffic to prevent overload and ensure uptime.", metadata={"chunk_id": 0}),
        Document(page_content="Auto-scaling adjusts capacity based on real-time metrics and thresholds.", metadata={"chunk_id": 1}),
        Document(page_content="Circuit breakers stop cascading failures in microservice architectures.", metadata={"chunk_id": 2}),
        Document(page_content="ACID compliance ensures consistent database state after each transaction.", metadata={"chunk_id": 3}),
        Document(page_content="Retry logic with exponential backoff handles transient network failures.", metadata={"chunk_id": 4}),
    ])

    gen   = MockGenerativeModel()
    raw   = RawRetriever(vs)
    expnd = QueryExpandedRetriever(vs, gen)

    runner = BenchmarkRunner(raw_retriever=raw, expanded_retriever=expnd, top_k=3)
    results = runner.run()
    runner.print_table()
    out = runner.save("benchmark_results.json")
    print(f"\nSaved: {out}")
    print("\nMarkdown:\n")
    print(runner.generate_markdown())
    print("\nBenchmarkRunner smoke test: OK")