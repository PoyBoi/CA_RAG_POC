import os, pickle, sys, logging, re, asyncio, torch, json
from pathlib import Path
from typing import List, Dict, Optional, Literal
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

MODEL_MAP = {
    "llama":    "llama3.2",
    "qwen":     "qwen2.5:7b",
    "deepseek": "deepseek-r1:14b",
}

MANIFEST_FILE = "ingest_manifest.json"  # tracks which docs have been ingested


class init_core_dependancies:
    def __init__(
        self,
        llm_model: Literal["llama", "qwen", "deepseek"],
        vdb: Literal["chromadb", "faiss"],
        db_location: str = "./db",
        ingest_data: bool = False,
        use_stub_llm: bool = False,
    ):
        self.llm_model    = llm_model
        self.vdb          = vdb
        self.db_location  = db_location
        self.ingest_data  = ingest_data
        self.use_stub_llm = use_stub_llm

        # slots — populated by init methods below
        self.llm          = None
        self.vectorstore  = None
        self.embeddings   = None
        self.retriever    = None
        self.manifest: Dict[str, str] = {}   # { doc_id : content_hash }

        # boot order matters — embeddings before vdb, db check before retriever
        self._init_embedding()
        self._init_vdb()
        self._init_llm()
        self._init_db()
        # _init_retriever() is NOT called here — needs docs loaded first
        # call it manually after ingestion: instance._init_retriever(docs)

    # ------------------------------------------------------------------ #

    def _init_llm(self):
        if self.use_stub_llm:
            # stub for dry-runs / mocking (assignment requirement)
            from unittest.mock import MagicMock
            self.llm = MagicMock()
            self.llm.invoke.return_value.content = "[STUB] LLM response"
            logger.info("LLM: stub loaded")
            return

        from langchain_ollama import ChatOllama
        model_name = MODEL_MAP.get(self.llm_model)
        if not model_name:
            raise ValueError(f"Unknown llm_model '{self.llm_model}'. Choose from: {list(MODEL_MAP)}")

        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1,        # low temp for factual RAG answers
            num_ctx=4096,           # context window
        )
        logger.info(f"LLM: Ollama/{model_name} loaded")

    # ------------------------------------------------------------------ #

    def _init_vdb(self):
        if self.vdb == "chromadb":
            from langchain_chroma import Chroma
            self.vectorstore = Chroma(
                collection_name="rag_collection",
                embedding_function=self.embeddings,
                persist_directory=self.db_location,
            )
            logger.info(f"VDB: ChromaDB at '{self.db_location}'")

        elif self.vdb == "faiss":
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
                # empty — will be populated during ingestion
                self.vectorstore = None
                logger.warning("VDB: FAISS index not found, will be created on first ingest")
        else:
            raise ValueError(f"Unknown vdb '{self.vdb}'. Choose from: chromadb, faiss")

    # ------------------------------------------------------------------ #

    def _init_embedding(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # cosine-ready
        )
        logger.info(f"Embeddings: {embedding_model} loaded")

    # ------------------------------------------------------------------ #

    def _init_retriever(self, docs=None):
        from langchain.retrievers import EnsembleRetriever
        from langchain_community.retrievers import BM25Retriever

        if docs is None or len(docs) == 0:
            logger.warning("Retriever: no docs passed, skipping ensemble — vectorstore-only fallback")
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
            return

        bm25 = BM25Retriever.from_documents(docs, k=5)
        vector = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5})
        semantic = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7})

        self.retriever = EnsembleRetriever(
            retrievers=[bm25, vector, semantic],
            weights=[0.3, 0.4, 0.3],
        )
        logger.info("Retriever: EnsembleRetriever (BM25 + vector + MMR semantic) ready")

    # ------------------------------------------------------------------ #

    def _init_db(self):
        """
        Loads the ingest manifest — a JSON file that maps doc_id → content hash.
        Used for incremental ingestion: skip docs that haven't changed.
        """
        manifest_path = Path(self.db_location) / MANIFEST_FILE
        Path(self.db_location).mkdir(parents=True, exist_ok=True)

        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                self.manifest = json.load(f)
            logger.info(f"Manifest: loaded {len(self.manifest)} tracked docs")
        else:
            self.manifest = {}
            logger.info("Manifest: none found, starting fresh")

        if self.ingest_data:
            logger.info("ingest_data=True — new/changed docs will be processed on next ingest call")

    def _save_manifest(self):
        manifest_path = Path(self.db_location) / MANIFEST_FILE
        with open(manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)