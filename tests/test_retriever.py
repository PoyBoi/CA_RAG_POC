"""
tests/test_retriever.py
────────────────────────
Covers:
  • Both strategies return top-K results
  • RetrievalResult has required fields
  • Strategy B populates rewritten_query
  • QueryReWriter produces non-empty expansion
"""

import pytest
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

from mocks import MockTextEmbeddingModel, MockGenerativeModel
from vector_store import VectorStore
from retriever import RawRetriever, QueryExpandedRetriever, QueryReWriter, RetrievalResult

TOP_K = 3

# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def vector_store(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("db_ret_test")
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = VectorStore(backend="chromadb", db_location=str(tmp), embeddings=emb)
    vs.add([
        Document(page_content="Load balancers distribute traffic across servers.", metadata={"chunk_id": 0}),
        Document(page_content="Auto-scaling adjusts capacity based on traffic.", metadata={"chunk_id": 1}),
        Document(page_content="Circuit breakers prevent cascading failures.", metadata={"chunk_id": 2}),
        Document(page_content="ACID transactions ensure data consistency.", metadata={"chunk_id": 3}),
        Document(page_content="Retry logic handles transient network errors.", metadata={"chunk_id": 4}),
    ])
    return vs


@pytest.fixture(scope="module")
def gen_model():
    return MockGenerativeModel()


@pytest.fixture(scope="module")
def raw_retriever(vector_store):
    return RawRetriever(vector_store)


@pytest.fixture(scope="module")
def expanded_retriever(vector_store, gen_model):
    return QueryExpandedRetriever(vector_store, gen_model)


# --------------------------------------------------------------------------- #
#  QueryReWriter                                                                #
# --------------------------------------------------------------------------- #

class TestQueryReWriter:
    def test_rewrite_returns_string(self, gen_model):
        rw = QueryReWriter(gen_model)
        result = rw.rewrite("How does the system handle peak load?")
        assert isinstance(result, str)

    def test_rewrite_non_empty(self, gen_model):
        rw = QueryReWriter(gen_model)
        result = rw.rewrite("What are failure recovery mechanisms?")
        assert len(result.strip()) > 0

    def test_rewrite_different_from_input(self, gen_model):
        rw = QueryReWriter(gen_model)
        query = "How does the system handle peak load?"
        result = rw.rewrite(query)
        # Expansion should differ from raw query
        assert result.lower() != query.lower()


# --------------------------------------------------------------------------- #
#  Strategy A — RawRetriever                                                   #
# --------------------------------------------------------------------------- #

class TestRawRetriever:
    def test_returns_list(self, raw_retriever):
        results = raw_retriever.retrieve("peak load", top_k=TOP_K)
        assert isinstance(results, list)

    def test_returns_top_k(self, raw_retriever):
        results = raw_retriever.retrieve("data consistency", top_k=TOP_K)
        assert len(results) == TOP_K

    def test_each_result_is_retrieval_result(self, raw_retriever):
        results = raw_retriever.retrieve("failure", top_k=2)
        for r in results:
            assert isinstance(r, RetrievalResult)

    def test_result_has_document(self, raw_retriever):
        results = raw_retriever.retrieve("load", top_k=1)
        assert results[0].document is not None
        assert len(results[0].document.page_content) > 0

    def test_result_has_score(self, raw_retriever):
        results = raw_retriever.retrieve("load", top_k=1)
        assert isinstance(results[0].score, float)

    def test_result_has_chunk_id(self, raw_retriever):
        results = raw_retriever.retrieve("load", top_k=1)
        assert results[0].chunk_id is not None

    def test_no_rewritten_query(self, raw_retriever):
        results = raw_retriever.retrieve("load", top_k=1)
        assert results[0].rewritten_query is None

    def test_top_1_is_most_relevant(self, raw_retriever):
        """The top result for 'load balancer' should reference load/balancing."""
        results = raw_retriever.retrieve("load balancer traffic", top_k=3)
        top_content = results[0].document.page_content.lower()
        assert any(kw in top_content for kw in ["load", "traffic", "balanc", "server"])


# --------------------------------------------------------------------------- #
#  Strategy B — QueryExpandedRetriever                                         #
# --------------------------------------------------------------------------- #

class TestQueryExpandedRetriever:
    def test_returns_list(self, expanded_retriever):
        results = expanded_retriever.retrieve("peak load", top_k=TOP_K)
        assert isinstance(results, list)

    def test_returns_top_k(self, expanded_retriever):
        results = expanded_retriever.retrieve("data consistency", top_k=TOP_K)
        assert len(results) == TOP_K

    def test_each_result_is_retrieval_result(self, expanded_retriever):
        results = expanded_retriever.retrieve("failure", top_k=2)
        for r in results:
            assert isinstance(r, RetrievalResult)

    def test_rewritten_query_populated(self, expanded_retriever):
        results = expanded_retriever.retrieve("How does the system handle peak load?", top_k=1)
        assert results[0].rewritten_query is not None
        assert isinstance(results[0].rewritten_query, str)
        assert len(results[0].rewritten_query.strip()) > 0

    def test_rewritten_query_same_across_results(self, expanded_retriever):
        """All results from one call should share the same rewritten query."""
        results = expanded_retriever.retrieve("peak load", top_k=3)
        queries = {r.rewritten_query for r in results}
        assert len(queries) == 1, "All results in one call should use the same rewritten query"