"""
embedding_model.py
──────────────────
Wraps sentence-transformers behind the same interface as
vertexai.language_models.TextEmbeddingModel.get_embeddings()
so that swapping to textembedding-gecko later is a one-line change.

Interface contract:
  model = EmbeddingModel()
  results: List[TextEmbedding] = model.get_embeddings(["text a", "text b"])
  vec: List[float] = results[0].values

Migration path to Vertex AI (documented in decisions.md):
  Replace SentenceTransformerEmbeddingModel with VertexTextEmbeddingModel,
  which calls TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
  — interface stays identical.
"""

import logging
from typing import List, Protocol

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Shared result type  (mirrors vertexai.language_models.TextEmbedding)
# ---------------------------------------------------------------------------

class TextEmbedding:
    """Thin wrapper so `.values` always gives you List[float]."""

    def __init__(self, values: List[float]):
        self.values = values

    def __repr__(self):
        return f"TextEmbedding(dim={len(self.values)}, first3={self.values[:3]})"


# ---------------------------------------------------------------------------
# Protocol — every embedding model must satisfy this
# ---------------------------------------------------------------------------

class TextEmbeddingModelProtocol(Protocol):
    def get_embeddings(self, texts: List[str]) -> List[TextEmbedding]: ...
    def embed(self, texts: List[str]) -> List[List[float]]: ...


# ---------------------------------------------------------------------------
# Local sentence-transformers implementation
# ---------------------------------------------------------------------------

class SentenceTransformerEmbeddingModel:
    """
    Wraps sentence-transformers to match the TextEmbeddingModel interface.
    Embeddings are L2-normalised (cosine-ready).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        logger.info(f"EmbeddingModel: '{model_name}' on {device}")

    # Mirror of vertexai interface
    def get_embeddings(self, texts: List[str]) -> List[TextEmbedding]:
        vecs = self._model.encode(texts, normalize_embeddings=True).tolist()
        return [TextEmbedding(v) for v in vecs]

    # Convenience method used internally
    def embed(self, texts: List[str]) -> List[List[float]]:
        return [e.values for e in self.get_embeddings(texts)]


# ---------------------------------------------------------------------------
# Vertex AI stub  (swap in when running on GCP)
# ---------------------------------------------------------------------------

class VertexTextEmbeddingModel:
    """
    TODO: implement when deploying to GCP.

    from vertexai.language_models import TextEmbeddingModel
    self._model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")

    get_embeddings() and embed() interface stays identical.
    """

    def get_embeddings(self, texts: List[str]) -> List[TextEmbedding]:
        raise NotImplementedError("VertexTextEmbeddingModel not yet wired up")

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
#  Smoke test                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")

    model = SentenceTransformerEmbeddingModel()

    texts = [
        "How does a RAG pipeline work?",
        "Vector databases store dense embeddings for similarity search.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    results = model.get_embeddings(texts)
    print(f"\nEmbedded {len(results)} texts")
    for i, r in enumerate(results):
        print(f"  [{i}] dim={len(r.values)}  first3={[round(v,4) for v in r.values[:3]]}")

    # Verify embed() shorthand
    vecs = model.embed(texts[:2])
    assert len(vecs) == 2 and len(vecs[0]) == 384
    print("\nEmbeddingModel smoke test: OK")