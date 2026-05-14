"""
benchmark_runner.py
───────────────────
Runs Strategy A (raw) vs Strategy B (query-expanded) on a fixed query set,
produces benchmark_results.json and prints a comparison table.

Usage
-----
# With real LLM (Ollama must be running, DB must be ingested):
python scripts/benchmark_runner.py --llm deepseek

# With mock LLM (no Ollama needed — for CI / quick checks):
python scripts/benchmark_runner.py --stub-llm

Flags
-----
--llm       llama | qwen | deepseek  (default: llama)
--stub-llm  Use mock LLM instead of Ollama
--top-k     Number of chunks to retrieve per query  (default: 3)
--db        Path to the ChromaDB directory  (default: ./db)
--out       Output JSON filename  (default: benchmark_results.json)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Benchmark query set (assignment requires >= 3 complex queries)
# ---------------------------------------------------------------------------
BENCHMARK_QUERIES = [
    "What are the main approaches to machine learning?",
    "How does artificial intelligence pose risks to society?",
    "What is the difference between narrow AI and artificial general intelligence?",
    "How have neural networks contributed to the development of AI?",
    "What ethical concerns are associated with artificial intelligence?",
]

TOP_K = 3


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    def __init__(self, raw_retriever, expanded_retriever, queries=None, top_k=TOP_K, output_dir="."):
        self.raw      = raw_retriever
        self.expanded = expanded_retriever
        self.queries  = queries or BENCHMARK_QUERIES
        self.top_k    = top_k
        self.output_dir = Path(output_dir)
        self._results: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
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

    def save(self, filename: str = "benchmark_results.json"):
        out = self.output_dir / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(self._results, f, indent=2)
        logger.info(f"Benchmark saved -> '{out}'")
        return out

    def print_table(self):
        if not self._results:
            print("No results — run run() first")
            return

        sep = "=" * 110
        print(f"\n{sep}")
        print(f"{'QUERY':<38} | {'STRATEGY A  (raw vector)':<33} | {'STRATEGY B  (query expanded)'}")
        print(sep)

        for entry in self._results["queries"]:
            a_ids = ", ".join(
                f"chunk {r['chunk_id']} ({r['score']})"
                for r in entry["strategy_a"]["top_k"]
            )
            rewrite_short = (entry["strategy_b"]["rewritten_query"] or "")[:40]
            b_ids = ", ".join(
                f"chunk {r['chunk_id']} ({r['score']})"
                for r in entry["strategy_b"]["top_k"]
            )
            print(f"{entry['input_query']:<38} | {a_ids:<33} | rewrite: '{rewrite_short}...'")
            print(f"{'':38} | {'':33} | {b_ids}")
            overlap = entry["diff"]["overlap_ids"]
            only_b  = entry["diff"]["only_in_b"]
            print(f"{'':38} | latency {entry['strategy_a']['latency_ms']:.1f}ms"
                  f"{'':14} | latency {entry['strategy_b']['latency_ms']:.1f}ms"
                  f"  overlap={overlap}  new_in_B={only_b}")
            print("-" * 110)

    def generate_markdown(self) -> str:
        if not self._results:
            return "No results."
        lines = [
            "| Input Query | Strategy A (Raw Vector) | Strategy B (Query Expanded) |",
            "|---|---|---|",
        ]
        for entry in self._results["queries"]:
            a_str = "; ".join(
                f"chunk {r['chunk_id']} (score {r['score']})"
                for r in entry["strategy_a"]["top_k"]
            )
            b_str = (
                f"*rewrite:* {(entry['strategy_b']['rewritten_query'] or '')[:80]} | "
                + "; ".join(
                    f"chunk {r['chunk_id']} (score {r['score']})"
                    for r in entry["strategy_b"]["top_k"]
                )
            )
            lines.append(f"| {entry['input_query']} | {a_str} | {b_str} |")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="RAG Benchmark Runner")
    p.add_argument("--llm",      default="llama", choices=["llama", "qwen", "deepseek"])
    p.add_argument("--stub-llm", action="store_true", help="Use mock LLM (no Ollama)")
    p.add_argument("--top-k",    type=int, default=TOP_K)
    p.add_argument("--db",       default="./db",                   help="ChromaDB directory")
    p.add_argument("--out",      default="benchmark_results.json", help="Output JSON filename")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    args = parse_args()

    # ── 1. Boot core dependencies (loads real ChromaDB from --db) ──────────
    from core_block import CoreDependancies
    core = CoreDependancies(
        llm_model=args.llm,
        vdb="chromadb",
        db_location=args.db,
        ingest_data=False,       # don't re-ingest — use existing DB
        use_stub_llm=args.stub_llm,
    )

    # ── 2. Check the store actually has documents ──────────────────────────
    try:
        count = core.vectorstore._collection.count()
    except Exception:
        count = -1

    if count == 0:
        logger.error(
            "ChromaDB is empty — run the pipeline with --ingest first:\n"
            "  python scripts/pipeline.py --ingest --llm deepseek"
        )
        sys.exit(1)

    logger.info(f"ChromaDB loaded: {count} vectors")

    # ── 3. Wrap vectorstore in VectorStore for query_with_scores ──────────
    from vector_store import VectorStore
    vs = VectorStore(
        backend="chromadb",
        db_location=args.db,
        embeddings=core.embeddings,
    )

    # ── 4. Build retrievers ────────────────────────────────────────────────
    from retriever import RawRetriever, QueryExpandedRetriever

    raw_retriever      = RawRetriever(vs)
    llm_for_rewrite    = core.llm      # real ChatOllama (or stub)
    expanded_retriever = QueryExpandedRetriever(vs, llm_for_rewrite)

    # ── 5. Run benchmark ───────────────────────────────────────────────────
    runner = BenchmarkRunner(
        raw_retriever=raw_retriever,
        expanded_retriever=expanded_retriever,
        top_k=args.top_k,
    )

    print("\nRunning benchmark — Strategy B will call the LLM once per query...")
    results = runner.run()

    # ── 6. Output ──────────────────────────────────────────────────────────
    runner.print_table()
    out_path = runner.save(args.out)

    md = runner.generate_markdown()
    print("\n--- Markdown (paste into retrieval_benchmark.md) ---\n")
    print(md)
    print(f"\nJSON saved to: {out_path}")


if __name__ == "__main__":
    main()