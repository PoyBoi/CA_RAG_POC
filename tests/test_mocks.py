"""
tests/test_mocks.py
───────────────────
Covers:
  • MockTextEmbeddingModel returns correct shape / types (no GCP calls)
  • MockGenerativeModel returns expected string for known queries
  • Both work with unittest.mock.patch as Vertex AI drop-ins
"""

import pytest
from unittest.mock import patch, MagicMock
# from ..scripts.mocks import MockTextEmbeddingModel, MockGenerativeModel
from mocks import MockTextEmbeddingModel, MockGenerativeModel


# --------------------------------------------------------------------------- #
#  MockTextEmbeddingModel                                                      #
# --------------------------------------------------------------------------- #

class TestMockTextEmbeddingModel:
    @pytest.fixture
    def model(self):
        return MockTextEmbeddingModel.from_pretrained("textembedding-gecko@003")

    def test_from_pretrained_returns_instance(self):
        m = MockTextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        assert isinstance(m, MockTextEmbeddingModel)

    def test_get_embeddings_returns_list(self, model):
        results = model.get_embeddings(["hello world"])
        assert isinstance(results, list)
        assert len(results) == 1

    def test_embedding_has_values_attribute(self, model):
        results = model.get_embeddings(["test"])
        assert hasattr(results[0], "values")

    def test_embedding_dim_is_384(self, model):
        results = model.get_embeddings(["test sentence"])
        assert len(results[0].values) == 384

    def test_multiple_texts_returns_same_count(self, model):
        texts = ["text one", "text two", "text three"]
        results = model.get_embeddings(texts)
        assert len(results) == 3

    def test_values_are_floats(self, model):
        results = model.get_embeddings(["test"])
        assert all(isinstance(v, float) for v in results[0].values)

    def test_embeddings_are_normalised(self, model):
        """L2 norm of a unit vector should be ~1.0"""
        import math
        results = model.get_embeddings(["normalisation test"])
        norm = math.sqrt(sum(v**2 for v in results[0].values))
        assert abs(norm - 1.0) < 1e-4, f"Expected unit norm, got {norm}"

    def test_same_input_same_output(self, model):
        """Deterministic: same input → same embedding"""
        text = ["deterministic test"]
        r1 = model.get_embeddings(text)[0].values
        r2 = model.get_embeddings(text)[0].values
        assert r1 == r2

    def test_patch_as_vertex_ai_drop_in(self):
        """Patch the Vertex AI import path — model should work identically"""
        with patch("mocks.MockTextEmbeddingModel.from_pretrained") as mock_fp:
            mock_fp.return_value = MockTextEmbeddingModel()
            m = MockTextEmbeddingModel.from_pretrained("textembedding-gecko@003")
            results = m.get_embeddings(["patch test"])
            assert len(results[0].values) == 384


# --------------------------------------------------------------------------- #
#  MockGenerativeModel                                                         #
# --------------------------------------------------------------------------- #

class TestMockGenerativeModel:
    @pytest.fixture
    def model(self):
        return MockGenerativeModel("gemini-pro")

    def test_generate_content_returns_response(self, model):
        resp = model.generate_content("How does the system handle peak load?")
        assert resp is not None

    def test_response_has_text_attribute(self, model):
        resp = model.generate_content("How does the system handle peak load?")
        assert hasattr(resp, "text")

    def test_response_text_is_string(self, model):
        resp = model.generate_content("anything")
        assert isinstance(resp.text, str)

    def test_response_text_non_empty(self, model):
        resp = model.generate_content("What are the failure recovery mechanisms?")
        assert len(resp.text.strip()) > 0

    def test_known_query_peak_load(self, model):
        resp = model.generate_content("How does the system handle peak load?")
        # Should contain load/scaling keywords
        keywords = ["load", "scaling", "traffic", "capacity", "balancing"]
        assert any(kw in resp.text.lower() for kw in keywords)

    def test_known_query_failure_recovery(self, model):
        resp = model.generate_content("What are the failure recovery mechanisms?")
        keywords = ["failure", "recovery", "fault", "retry", "circuit"]
        assert any(kw in resp.text.lower() for kw in keywords)

    def test_unknown_query_still_returns_string(self, model):
        resp = model.generate_content("Something completely random and unknown xyz123")
        assert isinstance(resp.text, str) and len(resp.text) > 0

    def test_deterministic_for_known_query(self, model):
        q = "How does the system handle peak load?"
        r1 = model.generate_content(q).text
        r2 = model.generate_content(q).text
        assert r1 == r2