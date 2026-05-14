"""
data_ingestor.py
────────────────
Responsible for:
  1. Loading raw text / parquet / txt files from disk
  2. Chunking with fixed-size + overlap (documented below)
  3. Incremental ingestion — skip docs whose content hash hasn't changed
  4. Returning clean List[Document] with metadata for downstream use

Chunking strategy (decision):
  • 256 tokens / 50-token overlap
  • Rationale: keeps chunks small enough to fit cleanly into context while
    the overlap prevents answer spans from being split across chunk boundaries.
    Trade-off: more chunks → higher retrieval cost, but better precision.

TODO:
  [ ] Replace hardcoded dataset path with config / env var
  [ ] Add PDF loader (pypdf or pdfplumber)
  [ ] Explore semantic chunking (split on sentence boundaries) as alternative
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunking constants  (document this choice in decisions.md)
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 512   # larger chunks for Wikipedia-style long-form prose
CHUNK_OVERLAP = 100   # bigger overlap to avoid splitting key explanatory sentences



# ---------------------------------------------------------------------------
# Hardcoded fallback dataset  (used when ./data is empty or missing)
# Satisfies the assignment requirement of 5-10 technical paragraphs.
# ---------------------------------------------------------------------------
FALLBACK_PARAGRAPHS = [
    ("para_001", "Artificial intelligence (AI) is the simulation of human intelligence in machines. "
     "AI systems are designed to perform tasks such as learning, reasoning, problem-solving, "
     "perception, and language understanding. The field was founded on the assumption that "
     "human intelligence can be precisely described and simulated by a machine."),

    ("para_002", "Machine learning is a subset of AI that enables systems to learn and improve "
     "from experience without being explicitly programmed. It focuses on developing computer "
     "programs that can access data and use it to learn for themselves. Supervised learning, "
     "unsupervised learning, and reinforcement learning are its three main paradigms."),

    ("para_003", "Deep learning uses multi-layered artificial neural networks to model and process "
     "complex patterns in data. It has achieved breakthrough results in image recognition, "
     "natural language processing, and speech recognition. The depth of the network allows "
     "it to learn hierarchical representations of raw input data automatically."),

    ("para_004", "A Retrieval-Augmented Generation (RAG) pipeline combines a retrieval system "
     "with a generative language model. The retrieval component fetches relevant documents "
     "from a vector database, and the generation component uses those documents as context "
     "to produce accurate, grounded responses. This reduces hallucination in language models."),

    ("para_005", "Vector databases store high-dimensional embeddings and enable efficient "
     "similarity search using distance metrics such as cosine similarity or Euclidean distance. "
     "ChromaDB, FAISS, and Pinecone are popular options. Cosine similarity is preferred for "
     "semantic search because it is magnitude-invariant and works well with normalised embeddings."),

    ("para_006", "Query expansion is a technique to improve information retrieval by reformulating "
     "the original query with additional relevant terms. A language model can rewrite a short "
     "query into a richer set of keywords that better match the vocabulary used in the target "
     "documents, improving recall without sacrificing precision."),

    ("para_007", "Embeddings are dense vector representations of text that capture semantic meaning. "
     "The sentence-transformers library produces embeddings by fine-tuning BERT-based models "
     "on sentence-pair tasks. The all-MiniLM-L6-v2 model produces 384-dimensional embeddings "
     "and is widely used for semantic search due to its balance of speed and quality."),

    ("para_008", "Ethical concerns in AI include algorithmic bias, lack of transparency, and risks "
     "to privacy and autonomy. Bias can enter AI systems through skewed training data or flawed "
     "objective functions. Explainability research aims to make model decisions interpretable "
     "so that humans can audit and correct them."),

    ("para_009", "Artificial General Intelligence (AGI) refers to a hypothetical AI system capable "
     "of performing any intellectual task that a human can. Unlike narrow AI, which is optimised "
     "for a specific domain, AGI would generalise across tasks without retraining. Most researchers "
     "consider AGI to be decades away, if achievable at all."),

    ("para_010", "Natural language processing (NLP) is a branch of AI concerned with the interaction "
     "between computers and human language. Tasks include sentiment analysis, named entity recognition, "
     "machine translation, and question answering. Large language models such as GPT and BERT have "
     "significantly advanced the state of the art across all NLP benchmarks."),
]

class DataIngestor:
    """
    Load, chunk, and prepare documents for vector store ingestion.

    Usage
    -----
    ingestor = DataIngestor(data_dir="./data", chunk_size=256, chunk_overlap=50)
    docs = ingestor.load_and_chunk()          # full reload
    new_docs = ingestor.load_incremental(manifest)  # only changed docs
    """

    def __init__(
        self,
        data_dir:      str = "./data",
        chunk_size:    int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.data_dir      = Path(data_dir)
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,   # captures char offset in metadata
        )

    # ---------------------------------------------------------------------- #
    #  Public API                                                              #
    # ---------------------------------------------------------------------- #

    def load_and_chunk(self) -> List[Document]:
        """
        Load ALL supported files in data_dir and return chunked Documents.
        Falls back to the hardcoded FALLBACK_PARAGRAPHS dataset if data_dir
        is empty or does not exist — satisfies the assignment requirement of
        having a working dataset out of the box.
        """
        raw_docs = self._load_all()

        if not raw_docs:
            logger.warning(
                f"DataIngestor: no files found in '{self.data_dir}' "
                "— using hardcoded fallback dataset"
            )
            raw_docs = [
                Document(
                    page_content=text,
                    metadata={"source": "fallback", "doc_id": doc_id},
                )
                for doc_id, text in FALLBACK_PARAGRAPHS
            ]

        chunks = self._chunk(raw_docs)
        logger.info(f"DataIngestor: {len(raw_docs)} source docs → {len(chunks)} chunks")
        return chunks

    def load_incremental(
        self,
        manifest: Dict[str, str],
    ) -> Tuple[List[Document], Dict[str, str]]:
        """
        Only process files whose content hash differs from the manifest.
        Returns (new_chunks, updated_manifest).
        """
        # TODO: implement incremental logic
        # Pseudocode:
        #   for each file in data_dir:
        #       h = _hash_file(file)
        #       if manifest.get(file.stem) == h:
        #           continue                          # unchanged, skip
        #       raw = _load_file(file)
        #       chunks = _chunk([raw])
        #       new_chunks.extend(chunks)
        #       updated_manifest[file.stem] = h
        #   return new_chunks, updated_manifest
        raise NotImplementedError

    # ---------------------------------------------------------------------- #
    #  Internals                                                               #
    # ---------------------------------------------------------------------- #

    def _load_all(self) -> List[Document]:
        """
        Discover and load all supported files in data_dir.
        Currently supports: .txt, .parquet
        """
        docs = []
        for path in sorted(self.data_dir.glob("**/*")):
            if path.suffix == ".txt":
                docs.append(self._load_txt(path))
            elif path.suffix == ".parquet":
                docs.extend(self._load_parquet(path))
            # TODO: add .pdf via pypdf/pdfplumber
        return docs

    def _load_txt(self, path: Path) -> Document:
        """Load a plain-text file as a single Document, cleaning markup first."""
        text = self._clean_wikitext(path.read_text(encoding="utf-8"))
        return Document(
            page_content=text,
            metadata={"source": str(path), "doc_id": path.stem},
        )

    def _load_parquet(self, path: Path) -> List[Document]:
        """
        Load a parquet file.
        Expects a 'text' column; all other columns become metadata.
        TODO: make text column name configurable
        """
        # TODO: implement
        # import pandas as pd
        # df = pd.read_parquet(path)
        # assert "text" in df.columns, f"Parquet {path} missing 'text' column"
        # return [
        #     Document(
        #         page_content=row["text"],
        #         metadata={
        #             "source": str(path),
        #             "doc_id": f"{path.stem}_{i}",
        #             **{k: v for k, v in row.items() if k != "text"},
        #         },
        #     )
        #     for i, row in df.iterrows()
        # ]
        raise NotImplementedError

    @staticmethod
    def _clean_wikitext(text: str) -> str:
        """
        Strip Wikipedia markup before chunking so it does not pollute embeddings.
        Removes: {{templates}}, [[File:...]], [[Link|Label]]->Label,
        <ref> blocks, HTML tags, ==headers==, bare URLs, excess whitespace.
        """
        import re
        # Remove {{templates}} - loop handles nested cases
        for _ in range(3):
            text = re.sub(r"\{\{[^{}]*\}\}", "", text)
        # Remove [[File:...]] / [[Image:...]] embeds
        text = re.sub(r"\[\[(File|Image):[^\]]*\]\]", "", text, flags=re.IGNORECASE)
        # [[Link|Label]] -> Label,  [[Link]] -> Link
        text = re.sub(r"\[\[(?:[^|\]]+\|)?([^\]]+)\]\]", r"\1", text)
        # <ref>...</ref> blocks (multiline)
        text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
        text = re.sub(r"<ref[^/]*/?>", "", text)
        # Remaining HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # ==Section Headers== -> plain text
        text = re.sub(r"={2,}\s*(.+?)\s*={2,}", r"\1", text)
        # Bare URLs
        text = re.sub(r"https?://\S+", "", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
    def _load_txt(self, path: Path) -> Document:
        """Load a plain-text file as a single Document, cleaning markup first."""
        text = self._clean_wikitext(path.read_text(encoding="utf-8"))
        return Document(
            page_content=text,
            metadata={"source": str(path), "doc_id": path.stem},
        )

    def _load_parquet(self, path: Path) -> List[Document]:
        """
        Load a parquet file.
        Expects a 'text' column; all other columns become metadata.
        TODO: make text column name configurable
        """
        # TODO: implement
        # import pandas as pd
        # df = pd.read_parquet(path)
        # assert "text" in df.columns, f"Parquet {path} missing 'text' column"
        # return [
        #     Document(
        #         page_content=row["text"],
        #         metadata={
        #             "source": str(path),
        #             "doc_id": f"{path.stem}_{i}",
        #             **{k: v for k, v in row.items() if k != "text"},
        #         },
        #     )
        #     for i, row in df.iterrows()
        # ]
        raise NotImplementedError

    @staticmethod
    def _clean_wikitext(text: str) -> str:
        """
        Strip Wikipedia markup before chunking so it does not pollute embeddings.
        Removes: citations {{...}}, file embeds [[File:...]], wikilinks [[X|Y]] -> Y,
        ref tags, section headers ==X==, bare URLs, and excess whitespace.
        """
        import re
        # Remove nested {{templates}} - iterate twice for nested cases
        for _ in range(3):
            text = re.sub(r"\{\{[^{}]*\}\}", "", text)
        # Remove [[File:...]] and [[Image:...]] embeds
        text = re.sub(r"\[\[(File|Image):[^\]]*\]\]", "", text, flags=re.IGNORECASE)
        # Convert [[Link|Label]] -> Label, [[Link]] -> Link
        text = re.sub(r"\[\[(?:[^|\]]+\|)?([^\]]+)\]\]", r"", text)
        # Remove <ref>...</ref> blocks (including multiline)
        text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
        text = re.sub(r"<ref[^/]*/?>", "", text)   # self-closing <ref ... />
        # Remove remaining HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Flatten ==Section Headers== to plain text
        text = re.sub(r"={2,}\s*(.+?)\s*={2,}", r"", text)
        # Remove bare URLs
        text = re.sub(r"https?://\S+", "", text)
        # Collapse excess whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _chunk(self, docs: List[Document]) -> List[Document]:
        """
        Split each Document using the configured RecursiveCharacterTextSplitter.
        Injects chunk_id into metadata.
        """
        chunks = self.splitter.split_documents(docs)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
        return chunks

    @staticmethod
    def _hash_file(path: Path) -> str:
        """MD5 hash of file contents — used for incremental change detection."""
        return hashlib.md5(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
#  Smoke test                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")

    # Create a tiny in-memory test without needing real data files
    from langchain.schema import Document

    ingestor = DataIngestor(chunk_size=100, chunk_overlap=20)

    # Manually exercise the splitter on a fake doc
    fake_doc = Document(
        page_content="RAG pipelines combine retrieval with generation. " * 20,
        metadata={"source": "test", "doc_id": "fake_001"},
    )
    chunks = ingestor._chunk([fake_doc])
    print(f"Input doc chars : {len(fake_doc.page_content)}")
    print(f"Output chunks   : {len(chunks)}")
    for i, c in enumerate(chunks[:3]):
        print(f"  chunk[{i}] len={len(c.page_content)}  meta={c.metadata}")
    print("\nDataIngestor smoke test: OK")