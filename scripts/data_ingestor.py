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
CHUNK_SIZE    = 256   # tokens / chars depending on splitter
CHUNK_OVERLAP = 50


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
        Does not consult the manifest — full reload.
        """
        raw_docs = self._load_all()
        chunks   = self._chunk(raw_docs)
        logger.info(f"DataIngestor: {len(raw_docs)} files → {len(chunks)} chunks")
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
        """Load a plain-text file as a single Document."""
        text = path.read_text(encoding="utf-8")
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