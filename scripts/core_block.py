import os, sys, logging, json, torch
from pathlib import Path
from typing import List, Dict, Optional, Literal

from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model aliases → Ollama model tags
# ---------------------------------------------------------------------------
MODEL_MAP = {
    "llama":    "llama3.2-vision:11b",
    "qwen":     "qwen2.5:7b",
    "deepseek": "deepseek-r1:14b",
}

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MANIFEST_FILE           = "ingest_manifest.json"


class CoreDependancies:
    """
    Bootstraps all shared infrastructure for the RAG pipeline:
      • Embedding model  (HuggingFace sentence-transformers, GPU-aware)
      • Vector store     (ChromaDB  or  FAISS)
      • LLM              (Ollama via LangChain  or  stub for dry-runs)
      • Ingest manifest  (incremental ingestion — skip unchanged docs)
      • Ensemble retriever (BM25 + dense vector + MMR)  ← call after ingest

    Boot order: embeddings → vdb → llm → manifest
    The retriever is NOT initialised here; call `_init_retriever(docs)` after
    you've loaded / ingested documents.
    """

    def __init__(
        self,
        llm_model:    Literal["llama", "qwen", "deepseek"] = "llama",
        vdb:          Literal["chromadb", "faiss"]          = "chromadb",
        db_location:  str  = "./db",
        ingest_data:  bool = False,
        use_stub_llm: bool = False,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.llm_model    = llm_model
        self.vdb          = vdb
        self.db_location  = db_location
        self.ingest_data  = ingest_data
        self.use_stub_llm = use_stub_llm
        self.embedding_model_name = embedding_model

        # Slots — populated by _init_* methods
        self.llm:         object = None
        self.vectorstore: object = None
        self.embeddings:  object = None
        self.retriever:   object = None
        self.manifest:    Dict[str, str] = {}   # { doc_id: content_hash }

        # Ordered boot sequence
        self._init_embedding()
        self._init_vdb()
        self._init_llm()
        self._init_db()

    # ---------------------------------------------------------------------- #
    #  Embedding model                                                         #
    # ---------------------------------------------------------------------- #

    def _init_embedding(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Embeddings: loading '{self.embedding_model_name}' on {device}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},   # cosine-ready unit vectors
        )
        logger.info(f"Embeddings: ready ({device})")

    # ---------------------------------------------------------------------- #
    #  LLM                                                                     #
    # ---------------------------------------------------------------------- #

    def _init_llm(self):
        if self.use_stub_llm:
            from unittest.mock import MagicMock
            self.llm = MagicMock()
            self.llm.invoke.return_value.content = "[STUB] LLM response"
            logger.info("LLM: stub loaded")
            return

        from langchain_ollama import ChatOllama
        model_name = MODEL_MAP.get(self.llm_model)
        if not model_name:
            raise ValueError(
                f"Unknown llm_model '{self.llm_model}'. "
                f"Valid choices: {list(MODEL_MAP)}"
            )

        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1,   # low temp — factual RAG answers, not creative
            num_ctx=4096,
        )
        logger.info(f"LLM: Ollama/{model_name} ready")

    # ---------------------------------------------------------------------- #
    #  Vector store                                                            #
    # ---------------------------------------------------------------------- #

    def _init_vdb(self):
        if self.vdb == "chromadb":
            self._init_chroma()
        elif self.vdb == "faiss":
            self._init_faiss()
        else:
            raise ValueError(
                f"Unknown vdb '{self.vdb}'. Valid choices: chromadb, faiss"
            )

    def _init_chroma(self):
        from langchain_chroma import Chroma
        self.vectorstore = Chroma(
            collection_name="rag_collection",
            embedding_function=self.embeddings,
            persist_directory=self.db_location,
        )
        logger.info(f"VDB: ChromaDB at '{self.db_location}'")

    def _init_faiss(self):
        from langchain_community.vectorstores import FAISS

        faiss_index_path = Path(self.db_location) / "faiss_index"
        if faiss_index_path.exists():
            self.vectorstore = FAISS.load_local(
                str(faiss_index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info(f"VDB: FAISS loaded from '{faiss_index_path}'")
        else:
            self.vectorstore = None
            logger.warning(
                "VDB: FAISS index not found — will be created on first ingest"
            )

    def save_faiss(self):
        """Persist FAISS index to disk.  Call after any ingestion."""
        if self.vdb != "faiss" or self.vectorstore is None:
            logger.warning("save_faiss: nothing to save (not FAISS or store is None)")
            return
        faiss_index_path = Path(self.db_location) / "faiss_index"
        self.vectorstore.save_local(str(faiss_index_path))
        logger.info(f"VDB: FAISS saved to '{faiss_index_path}'")

    # ---------------------------------------------------------------------- #
    #  Ingest manifest (incremental loader)                                   #
    # ---------------------------------------------------------------------- #

    def _init_db(self):
        """
        Load the ingest manifest — a JSON mapping doc_id → content_hash.
        Allows incremental ingestion: docs whose hash hasn't changed are skipped.
        """
        Path(self.db_location).mkdir(parents=True, exist_ok=True)
        manifest_path = Path(self.db_location) / MANIFEST_FILE

        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                self.manifest = json.load(f)
            logger.info(f"Manifest: {len(self.manifest)} tracked docs loaded")
        else:
            self.manifest = {}
            logger.info("Manifest: none found — starting fresh")

        if self.ingest_data:
            logger.info("ingest_data=True — new/changed docs will be processed next ingest")

    def _save_manifest(self):
        manifest_path = Path(self.db_location) / MANIFEST_FILE
        with open(manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)
        logger.debug(f"Manifest: saved ({len(self.manifest)} docs)")

    def is_doc_changed(self, doc_id: str, content_hash: str) -> bool:
        """Return True if the doc is new or its content has changed."""
        return self.manifest.get(doc_id) != content_hash

    def mark_doc_ingested(self, doc_id: str, content_hash: str):
        """Record that a doc has been ingested at this content hash."""
        self.manifest[doc_id] = content_hash
        self._save_manifest()

    # ---------------------------------------------------------------------- #
    #  Ensemble retriever                                                      #
    # ---------------------------------------------------------------------- #

    def _init_retriever(self, docs=None):
        """
        Build an EnsembleRetriever:
          • BM25  (keyword)      weight 0.3
          • Dense vector         weight 0.4
          • MMR   (diversity)    weight 0.3

        Falls back to plain vector retriever if no docs are supplied.
        Call this *after* ingestion so BM25 has documents to index.
        """
        if self.vectorstore is None:
            raise RuntimeError(
                "_init_retriever: vectorstore is None — ingest documents first"
            )

        if not docs:
            logger.warning(
                "Retriever: no docs provided — falling back to vector-only retriever"
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
            return

        from langchain.retrievers import EnsembleRetriever
        from langchain_community.retrievers import BM25Retriever

        bm25   = BM25Retriever.from_documents(docs, k=5)
        vector = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )
        semantic = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7},
        )

        self.retriever = EnsembleRetriever(
            retrievers=[bm25, vector, semantic],
            weights=[0.3, 0.4, 0.3],
        )
        logger.info("Retriever: EnsembleRetriever (BM25 + dense + MMR) ready")

    # ---------------------------------------------------------------------- #
    #  Health check                                                            #
    # ---------------------------------------------------------------------- #

    def health_check(self) -> Dict[str, str]:
        """
        Quick sanity check — returns a status dict so callers can gate on it.
        Does NOT run inference; just confirms each component is loaded.
        """
        status = {}

        status["embeddings"]  = "ok" if self.embeddings  is not None else "MISSING"
        status["vectorstore"] = "ok" if self.vectorstore is not None else "MISSING"
        status["llm"]         = "ok" if self.llm         is not None else "MISSING"
        status["retriever"]   = "ok" if self.retriever   is not None else "not initialised (call _init_retriever after ingest)"
        status["manifest_docs"] = str(len(self.manifest))

        for k, v in status.items():
            icon = "✓" if v == "ok" else "✗"
            logger.info(f"  [{icon}] {k}: {v}")

        return status


# --------------------------------------------------------------------------- #
#  Smoke test                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s | %(name)s | %(message)s",
    )

    print("\n=== CoreDependancies smoke test ===\n")

    # Use stub LLM so the test runs without Ollama, and a temp db dir
    core = CoreDependancies(
        llm_model="llama",
        vdb="chromadb",
        db_location="./db_test",
        ingest_data=True,
        use_stub_llm=True,
    )

    status = core.health_check()
    print("\nHealth check:", status)

    # Verify stub LLM responds
    response = core.llm.invoke("Hello")
    print(f"\nStub LLM response: {response.content}")

    # Verify embeddings produce vectors
    test_texts = ["RAG pipeline test", "vector database embedding"]
    vecs = core.embeddings.embed_documents(test_texts)
    print(f"\nEmbedding dim: {len(vecs[0])}  (expected 384 for all-MiniLM-L6-v2)")
    assert len(vecs) == 2 and len(vecs[0]) == 384, "Embedding shape mismatch!"

    # Verify manifest incremental logic
    core.mark_doc_ingested("doc_001", "hash_abc")
    assert not core.is_doc_changed("doc_001", "hash_abc"), "Should not be changed"
    assert     core.is_doc_changed("doc_001", "hash_xyz"), "Should be changed"
    print("\nManifest incremental logic: OK")

    print("\n=== All checks passed ===\n")