"""
pipeline.py
───────────
End-to-end RAG pipeline runner.

Wires together:
  DataIngestor → VectorStore → CoreDependencies (LLM + Retriever) → Query loop

Usage
-----
python pipeline.py --ingest --llm llama --vdb chromadb

Flags
-----
--ingest      Re-run ingestion (default: skip if manifest up to date)
--llm         llama | qwen | deepseek (default: llama)
--vdb         chromadb | faiss (default: chromadb)
--stub-llm    Use mock LLM (no Ollama required — for testing)
--query       Run a single query and exit (non-interactive)
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
DATA_DIR    = "./data"
DB_LOCATION = "./db"


def parse_args():
    p = argparse.ArgumentParser(description="RAG Pipeline")
    p.add_argument("--ingest",    action="store_true", help="Run ingestion")
    p.add_argument("--llm",       default="llama",     choices=["llama", "qwen", "deepseek"])
    p.add_argument("--vdb",       default="chromadb",  choices=["chromadb", "faiss"])
    p.add_argument("--stub-llm",  action="store_true", help="Use stub LLM (no Ollama)")
    p.add_argument("--query",     type=str,            help="Single query mode")
    p.add_argument("--top-k",     type=int, default=5, help="Number of chunks to retrieve")
    return p.parse_args()


def ingest(core, data_dir: str):
    """Load, chunk, and add documents to the vector store."""
    from data_ingestor import DataIngestor

    logger.info(f"Ingesting from '{data_dir}'")
    ingestor = DataIngestor(data_dir=data_dir)

    # TODO: switch to incremental once load_incremental() is implemented
    # new_docs, updated_manifest = ingestor.load_incremental(core.manifest)
    chunks = ingestor.load_and_chunk()

    if not chunks:
        logger.warning("Ingestion: no chunks produced — check data_dir")
        return []

    from vector_store import VectorStore
    # VectorStore is already initialised in core._init_vdb; re-use it
    # For now, add directly via the langchain store
    core.vectorstore.add_documents(chunks)
    logger.info(f"Ingestion: {len(chunks)} chunks added to vector store")

    core._init_retriever(chunks)
    return chunks


def run_query(core, query: str, top_k: int = 5) -> str:
    """RAG: retrieve relevant chunks then generate an answer with the LLM."""
    from langchain_core.prompts import ChatPromptTemplate

    if core.retriever is None:
        raise RuntimeError("Retriever not initialised — run ingestion first")

    # Retrieve
    docs = core.retriever.invoke(query)[:top_k]
    context = "\n\n".join(d.page_content for d in docs)

    # Augment + generate
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Use the following context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    chain = prompt | core.llm
    response = chain.invoke({"context": context, "question": query})
    return response.content


def interactive_loop(core, top_k: int):
    print("\nRAG Pipeline ready. Type 'quit' to exit.\n")
    while True:
        try:
            query = input("Query > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        try:
            answer = run_query(core, query, top_k=top_k)
            print(f"\nAnswer:\n{answer}\n")
        except Exception as e:
            logger.error(f"Query failed: {e}")


def main():
    args = parse_args()

    from core_block import CoreDependancies
    core = CoreDependancies(
        llm_model=args.llm,
        vdb=args.vdb,
        db_location=DB_LOCATION,
        ingest_data=args.ingest,
        use_stub_llm=args.stub_llm,
    )

    chunks = []
    if args.ingest:
        chunks = ingest(core, DATA_DIR)
    else:
        # No ingestion — just initialise retriever without BM25
        if core.vectorstore is not None:
            core._init_retriever()
        else:
            logger.warning("VectorStore is empty — run with --ingest first")

    status = core.health_check()
    logger.info(f"Pipeline status: {status}")

    if args.query:
        answer = run_query(core, args.query, top_k=args.top_k)
        print(f"\nAnswer:\n{answer}")
    else:
        interactive_loop(core, top_k=args.top_k)


if __name__ == "__main__":
    main()