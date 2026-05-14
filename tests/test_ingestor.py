"""
tests/test_ingestor.py
────────────────────
Covers:
  • Chunking output shape and metadata fields
  • Overlap sanity check
  • Hash-based incremental detection
"""

import pytest
from langchain.schema import Document
from ..scripts.data_ingestor import DataIngestor


# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture
def ingestor():
    return DataIngestor(chunk_size=100, chunk_overlap=20)


@pytest.fixture
def single_doc():
    return Document(
        page_content="RAG pipelines combine dense retrieval with language generation. " * 15,
        metadata={"source": "test_source", "doc_id": "doc_001"},
    )


# --------------------------------------------------------------------------- #
#  Tests                                                                       #
# --------------------------------------------------------------------------- #

class TestChunking:
    def test_produces_multiple_chunks(self, ingestor, single_doc):
        chunks = ingestor._chunk([single_doc])
        assert len(chunks) > 1, "Long document should produce multiple chunks"

    def test_each_chunk_has_chunk_id(self, ingestor, single_doc):
        chunks = ingestor._chunk([single_doc])
        for chunk in chunks:
            assert "chunk_id" in chunk.metadata, "chunk_id missing from metadata"

    def test_chunk_ids_are_sequential(self, ingestor, single_doc):
        chunks = ingestor._chunk([single_doc])
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert ids == list(range(len(chunks))), "chunk_ids should be 0-indexed sequential"

    def test_source_metadata_preserved(self, ingestor, single_doc):
        chunks = ingestor._chunk([single_doc])
        for chunk in chunks:
            assert chunk.metadata.get("source") == "test_source"

    def test_chunk_length_within_bounds(self, ingestor, single_doc):
        chunks = ingestor._chunk([single_doc])
        for chunk in chunks:
            # Allow slight overshoot from RecursiveCharacterTextSplitter
            assert len(chunk.page_content) <= ingestor.chunk_size * 1.5

    def test_empty_doc_list_returns_empty(self, ingestor):
        chunks = ingestor._chunk([])
        assert chunks == []

    def test_overlap_creates_shared_content(self, ingestor, single_doc):
        """
        Adjacent chunks should share some content due to overlap.
        Not a strict text equality check — just verify boundaries aren't perfectly clean.
        """
        chunks = ingestor._chunk([single_doc])
        if len(chunks) >= 2:
            # The end of chunk[0] and the start of chunk[1] should overlap
            end_of_first   = chunks[0].page_content[-ingestor.chunk_overlap:]
            start_of_second = chunks[1].page_content[:ingestor.chunk_overlap]
            # At least some chars should be shared
            common = set(end_of_first.split()) & set(start_of_second.split())
            assert len(common) > 0, "Expected some content overlap between adjacent chunks"


class TestIncrementalHashing:
    def test_same_content_same_hash(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Hello world")
        h1 = DataIngestor._hash_file(f)
        h2 = DataIngestor._hash_file(f)
        assert h1 == h2

    def test_changed_content_different_hash(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("version 1")
        h1 = DataIngestor._hash_file(f)
        f.write_text("version 2 — different content")
        h2 = DataIngestor._hash_file(f)
        assert h1 != h2

    def test_hash_is_string(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("test")
        assert isinstance(DataIngestor._hash_file(f), str)