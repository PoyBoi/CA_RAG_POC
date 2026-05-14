"""
tests/test_vector_store.py
──────────────────────────
Covers:
  • add() stores documents
  • query() returns top-K results
  • query_with_scores() includes float scores
  • count() reflects added docs
  • Retriever returns LangChain-compatible object
"""

import pytest
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from ..scripts.vector_store import VectorStore


@pytest.fixture(scope="module")
def emb():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        encode_kwargs={"normalize_embeddings": True},
    )


@pytest.fixture
def vs(tmp_path, emb):
    store = VectorStore(backend="chromadb", db_location=str(tmp_path / "db"), embeddings=emb)
    return store


@pytest.fixture
def populated_vs(vs):
    docs = [
        Document(page_content="Load balancers route traffic to healthy servers.", metadata={"chunk_id": 0}),
        Document(page_content="ACID ensures consistent database transactions.", metadata={"chunk_id": 1}),
        Document(page_content="BM25 is a keyword retrieval scoring algorithm.", metadata={"chunk_id": 2}),
    ]
    vs.add(docs)
    return vs


class TestVectorStoreAdd:
    def test_add_increases_count(self, vs):
        before = vs.count()
        vs.add([Document(page_content="test doc", metadata={"chunk_id": 99})])
        assert vs.count() == before + 1

    def test_add_empty_list_no_error(self, vs):
        vs.add([])   # should not raise

    def test_add_multiple(self, vs):
        docs = [Document(page_content=f"doc {i}", metadata={"chunk_id": i}) for i in range(5)]
        before = vs.count()
        vs.add(docs)
        assert vs.count() == before + 5


class TestVectorStoreQuery:
    def test_query_returns_list(self, populated_vs):
        results = populated_vs.query("load balancing", top_k=2)
        assert isinstance(results, list)

    def test_query_returns_top_k(self, populated_vs):
        results = populated_vs.query("database transactions", top_k=2)
        assert len(results) == 2

    def test_query_results_are_documents(self, populated_vs):
        results = populated_vs.query("traffic", top_k=1)
        assert isinstance(results[0], Document)

    def test_query_with_scores_returns_tuples(self, populated_vs):
        results = populated_vs.query_with_scores("BM25", top_k=2)
        assert isinstance(results, list)
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)

    def test_top_result_relevant(self, populated_vs):
        results = populated_vs.query("BM25 keyword retrieval", top_k=1)
        assert "BM25" in results[0].page_content or "keyword" in results[0].page_content.lower()


class TestVectorStoreRetriever:
    def test_as_langchain_retriever(self, populated_vs):
        retriever = populated_vs.as_langchain_retriever(k=2)
        assert retriever is not None
        # Should have invoke/get_relevant_documents method
        assert hasattr(retriever, "get_relevant_documents") or hasattr(retriever, "invoke")