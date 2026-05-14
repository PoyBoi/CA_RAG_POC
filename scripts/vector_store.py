"""
vector_store.py
───────────────
Thin wrapper around ChromaDB (primary) with a FAISS fallback path.

Why cosine similarity?
  • Embeddings are L2-normalised (all-MiniLM-L6-v2 + normalize_embeddings=True)
  • For unit vectors: cosine_similarity = dot_product — fast and exact
  • Magnitude-invariant: long vs short texts don't skew results
  • Industry standard for semantic search; Euclidean on normalised vecs gives
    the same ranking but cosine is more interpretable (bounded [-1, 1])
  (expand in decisions.md)

Vertex AI migration path:
  Replace ChromaDB with Vertex AI Vector Search (Matching Engine):
    1. Create an Index with cosine distance metric
    2. Upsert via index.upsert_datapoints()
    3. Deploy to an IndexEndpoint
    4. Query via endpoint.find_neighbors()
  The add() / query() interface here is designed to be swappable.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain.schema import Document

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Unified wrapper: ChromaDB (default) or FAISS.

    Usage
    -----
    vs = VectorStore(backend="chromadb", db_location="./db", embeddings=emb)
    vs.add(chunks)
    results = vs.query("how does RAG work?", top_k=5)
    """

    def __init__(
        self,
        backend:     str    = "chromadb",   # "chromadb" | "faiss"
        db_location: str    = "./db",
        embeddings:  object = None,          # HuggingFaceEmbeddings or compatible
        collection:  str    = "rag_collection",
    ):
        self.backend     = backend
        self.db_location = Path(db_location)
        self.embeddings  = embeddings
        self.collection  = collection
        self._store      = None             # underlying LangChain vectorstore

        self.db_location.mkdir(parents=True, exist_ok=True)
        self._load_or_create()

    # ---------------------------------------------------------------------- #
    #  Lifecycle                                                               #
    # ---------------------------------------------------------------------- #

    def _load_or_create(self):
        if self.backend == "chromadb":
            self._init_chroma()
        elif self.backend == "faiss":
            self._init_faiss()
        else:
            raise ValueError(f"Unknown backend '{self.backend}'")

    def _init_chroma(self):
        from langchain_chroma import Chroma
        self._store = Chroma(
            collection_name=self.collection,
            embedding_function=self.embeddings,
            persist_directory=str(self.db_location),
        )
        logger.info(f"VectorStore: ChromaDB ready at '{self.db_location}'")

    def _init_faiss(self):
        # TODO: implement
        # from langchain_community.vectorstores import FAISS
        # faiss_path = self.db_location / "faiss_index"
        # if faiss_path.exists():
        #     self._store = FAISS.load_local(...)
        # else:
        #     self._store = None   # created on first add()
        raise NotImplementedError("FAISS backend not yet implemented in VectorStore")

    # ---------------------------------------------------------------------- #
    #  Core API                                                                #
    # ---------------------------------------------------------------------- #

    def add(
        self,
        chunks:    List[Document],
        ids:       Optional[List[str]] = None,
    ) -> None:
        """
        Add chunked Documents to the vector store.
        Embeddings are computed internally by the embedding_function.
        """
        if not chunks:
            logger.warning("VectorStore.add: empty chunks list, nothing to do")
            return

        if self.backend == "chromadb":
            self._store.add_documents(documents=chunks, ids=ids)
        elif self.backend == "faiss":
            # TODO: handle FAISS first-time creation (from_documents) vs add
            raise NotImplementedError
        
        logger.info(f"VectorStore: added {len(chunks)} chunks")

    def query(
        self,
        query_text: str,
        top_k:      int  = 5,
        filter:     Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Embed query_text and return top_k most similar Documents.
        Uses cosine similarity (see module docstring for rationale).
        """
        if self._store is None:
            raise RuntimeError("VectorStore: store is None — add documents first")

        kwargs = {"k": top_k}
        if filter:
            kwargs["filter"] = filter

        results = self._store.similarity_search(query_text, **kwargs)
        logger.debug(f"VectorStore.query: returned {len(results)} docs for '{query_text[:60]}'")
        return results

    def query_with_scores(
        self,
        query_text: str,
        top_k:      int = 5,
    ) -> List[tuple]:
        """
        Like query() but returns (Document, score) tuples, deduplicated by chunk_id.
        Fetches top_k * 3 candidates then deduplicates so the final list has
        at most top_k unique chunks even when the ensemble returns repeats.
        """
        if self._store is None:
            raise RuntimeError("VectorStore: store is None — add documents first")

        # Over-fetch so deduplication does not shrink results below top_k
        raw = self._store.similarity_search_with_score(query_text, k=top_k * 3)

        seen, deduped = set(), []
        for doc, score in raw:
            key = doc.metadata.get("chunk_id", doc.page_content[:80])
            if key not in seen:
                seen.add(key)
                deduped.append((doc, score))
            if len(deduped) == top_k:
                break

        return deduped

    def delete(self, ids: List[str]) -> None:
        """Remove specific documents by ID."""
        # TODO: implement per-backend
        raise NotImplementedError

    def count(self) -> int:
        """Return the number of stored vectors."""
        if self.backend == "chromadb":
            return self._store._collection.count()
        raise NotImplementedError(f"count() not implemented for {self.backend}")

    def as_langchain_retriever(self, search_type: str = "similarity", k: int = 5):
        """Expose as a LangChain-compatible retriever for chain integration."""
        return self._store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k},
        )


# --------------------------------------------------------------------------- #
#  Smoke test                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")

    from langchain_huggingface import HuggingFaceEmbeddings

    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

    vs = VectorStore(backend="chromadb", db_location="./db_test_vs", embeddings=emb)

    # Add some fake docs
    test_docs = [
        Document(page_content="RAG combines retrieval and generation.", metadata={"source": "test", "chunk_id": 0}),
        Document(page_content="ChromaDB stores dense vector embeddings.", metadata={"source": "test", "chunk_id": 1}),
        Document(page_content="BM25 is a keyword-based retrieval algorithm.", metadata={"source": "test", "chunk_id": 2}),
    ]
    vs.add(test_docs)
    print(f"Stored: {vs.count()} vectors")

    # Query
    results = vs.query("what is RAG?", top_k=2)
    print(f"\nTop-2 results for 'what is RAG?':")
    for i, doc in enumerate(results):
        print(f"  [{i}] {doc.page_content}")

    # With scores
    scored = vs.query_with_scores("vector database", top_k=2)
    print(f"\nWith scores:")
    for doc, score in scored:
        print(f"  score={score:.4f}  |  {doc.page_content[:60]}")

    print("\nVectorStore smoke test: OK")