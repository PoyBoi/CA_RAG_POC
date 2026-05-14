"""
mocks.py
────────
Drop-in mocks for Vertex AI models — no GCP credentials needed.

  MockTextEmbeddingModel   → mirrors vertexai.language_models.TextEmbeddingModel
  MockGenerativeModel      → mirrors vertexai.generativeai.GenerativeModel

Both are deterministic: same input always produces the same output.
Under the hood MockTextEmbeddingModel uses real sentence-transformers so
vector similarity tests still make semantic sense.

Usage in tests
--------------
from unittest.mock import patch
with patch("vertexai.language_models.TextEmbeddingModel", MockTextEmbeddingModel):
    ...
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared result stubs  (mirror Vertex AI response shapes)
# ---------------------------------------------------------------------------

class _FakeTextEmbedding:
    """Mirrors vertexai.language_models.TextEmbedding  (has .values)"""
    def __init__(self, values: List[float]):
        self.values = values


class _FakeGenerateContentResponse:
    """Mirrors vertexai.generativeai.types.GenerateContentResponse  (has .text)"""
    def __init__(self, text: str):
        self.text = text


# ---------------------------------------------------------------------------
# MockTextEmbeddingModel
# ---------------------------------------------------------------------------

class MockTextEmbeddingModel:
    """
    Mirrors: vertexai.language_models.TextEmbeddingModel

    Uses sentence-transformers under the hood so embeddings are real and
    comparable — just without hitting GCP.

    Interface:
        model = MockTextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        results = model.get_embeddings(["text a", "text b"])
        vec: List[float] = results[0].values
    """

    # Map of known Vertex model IDs to local HF equivalents
    _MODEL_ALIAS = {
        "textembedding-gecko@003": "sentence-transformers/all-MiniLM-L6-v2",
        "textembedding-gecko":     "sentence-transformers/all-MiniLM-L6-v2",
    }

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        logger.info(f"MockTextEmbeddingModel: using '{model_name}' (local)")

    @classmethod
    def from_pretrained(cls, model_id: str) -> "MockTextEmbeddingModel":
        """Mirror of TextEmbeddingModel.from_pretrained()."""
        local_name = cls._MODEL_ALIAS.get(model_id, "sentence-transformers/all-MiniLM-L6-v2")
        logger.info(f"MockTextEmbeddingModel.from_pretrained('{model_id}') → '{local_name}'")
        return cls(local_name)

    def get_embeddings(self, texts: List[str]) -> List[_FakeTextEmbedding]:
        """Same signature as TextEmbeddingModel.get_embeddings()."""
        vecs = self._model.encode(texts, normalize_embeddings=True).tolist()
        return [_FakeTextEmbedding(v) for v in vecs]


# ---------------------------------------------------------------------------
# MockGenerativeModel
# ---------------------------------------------------------------------------

# Hardcoded query expansion map (deterministic)
_EXPANSION_MAP = {
    "how does the system handle peak load?":
        "system load balancing peak traffic handling capacity scaling auto-scaling",

    "what are the failure recovery mechanisms?":
        "failure recovery fault tolerance retry logic circuit breaker error handling fallback",

    "how is data consistency maintained?":
        "data consistency ACID transactions eventual consistency distributed sync replication",
}

_DEFAULT_EXPANSION = "expanded query terms related to {query}"


class MockGenerativeModel:
    """
    Mirrors: vertexai.generativeai.GenerativeModel

    Returns hardcoded query expansions for known inputs; falls back to a
    template string for unknown queries.

    Interface:
        model = MockGenerativeModel("gemini-pro")
        response = model.generate_content("How does the system handle peak load?")
        print(response.text)  # → "system load balancing peak traffic..."
    """

    def __init__(self, model_name: str = "gemini-pro"):
        self.model_name = model_name
        logger.info(f"MockGenerativeModel: stub for '{model_name}'")

    def generate_content(self, prompt: str) -> _FakeGenerateContentResponse:
        """
        Deterministic expansion lookup.
        Falls back to a generic template so tests never get a None back.
        """
        key = prompt.strip().lower()
        expansion = _EXPANSION_MAP.get(
            key,
            _DEFAULT_EXPANSION.format(query=key),
        )
        logger.debug(f"MockGenerativeModel: '{prompt[:60]}' → '{expansion[:60]}'")
        return _FakeGenerateContentResponse(text=expansion)

    # TODO: add generate_content_async() mirror when needed


# --------------------------------------------------------------------------- #
#  Smoke test                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")

    # TextEmbeddingModel mock
    emb_model = MockTextEmbeddingModel.from_pretrained("textembedding-gecko@003")
    texts = ["How does RAG work?", "Vector similarity search"]
    results = emb_model.get_embeddings(texts)

    print(f"\nMockTextEmbeddingModel")
    for i, r in enumerate(results):
        print(f"  [{i}] dim={len(r.values)}  first3={[round(v,4) for v in r.values[:3]]}")
    assert len(results) == 2 and len(results[0].values) == 384

    # GenerativeModel mock
    gen_model = MockGenerativeModel("gemini-pro")
    queries = [
        "How does the system handle peak load?",
        "What are the failure recovery mechanisms?",
        "Something completely new",
    ]
    print(f"\nMockGenerativeModel")
    for q in queries:
        resp = gen_model.generate_content(q)
        print(f"  Q: {q[:50]}")
        print(f"  A: {resp.text}\n")
    assert isinstance(gen_model.generate_content(queries[0]).text, str)

    print("Mocks smoke test: OK")