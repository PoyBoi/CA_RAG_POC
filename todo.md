# Senior GenAI Assessment — RAG Pipeline Todos

---

## 1. Project Structure
- [ ] Modular layout with separate files for: Ingestion, Embedding, Vector Store, Retrieval, Mocks, Benchmark Runner
- [ ] `tests/` directory with separate test files per module
- [ ] `docs/` directory for the decisions writeup
- [ ] `retrieval_benchmark.md` at repo root

---

## 2. Data Ingestor
- [ ] `DataIngestor` class
  - [ ] Load raw text (5–10 technical paragraphs — hardcoded dataset or `.txt` file)
  - [ ] Chunk the text (fixed-size with overlap, e.g. 256 tokens / 50 overlap — **document this choice**)
  - [ ] Return clean `List[Document]` with metadata (`chunk_id`, `source`)

---

## 3. Embedding Model
- [ ] Wrap `sentence-transformers` (`all-MiniLM-L6-v2`) behind a class interface
  that mirrors `TextEmbeddingModel.get_embeddings()` signature
- [ ] Method: `embed(texts: List[str]) -> List[List[float]]`

---

## 4. Vector Store
- [ ] `VectorStore` class wrapping **ChromaDB**
  - [ ] `add(chunks, embeddings, metadata)`
  - [ ] `query(embedding, top_k) -> List[Document]`
  - [ ] Use **cosine similarity** (document the why in decisions writeup)

---

## 5. Mocks
- [ ] `MockTextEmbeddingModel` — mirrors `vertexai.language_models.TextEmbeddingModel`
  - Returns deterministic fake embeddings (sentence-transformers under the hood is fine)
- [ ] `MockGenerativeModel` — mirrors `vertexai.generativeai.GenerativeModel`
  - `generate_content(prompt)` returns a hardcoded query expansion string
  - Example: `"How does the system handle peak load?"` → `"system load balancing peak traffic handling capacity scaling"`

---

## 6. Retrieval
- [ ] **Strategy A — Raw Retriever**
  - Embed query as-is → cosine search → return top-K chunks
- [ ] **Strategy B — Query Expanded Retriever**
  - [ ] `QueryReWriter` class — wraps `MockGenerativeModel`, rewrites the input query
  - Pass query to `QueryReWriter` → get expanded query → embed → cosine search → return top-K chunks

---

## 7. Benchmark Runner
- [ ] Run both strategies on **at least 3 complex queries**
  (use the doc's example + 2 more, e.g. *"What are the failure recovery mechanisms?"* / *"How is data consistency maintained?"*)
- [ ] Output `benchmark_results.json` — structured diff of Strategy A vs B per query
- [ ] Print/render the comparison table:

| Input Query | Result A (Raw Vector) | Result B (Query Expanded) |
|---|---|---|
| How does the system handle peak load? | Top 3 chunk IDs + scores | Top 3 chunk IDs + scores + rewritten query shown |

---

## 8. Tests
- [ ] Verify chunking output shape and metadata
- [ ] Verify top-K results are returned for both strategies
- [ ] Verify `MockTextEmbeddingModel` and `MockGenerativeModel` return expected shapes/types without hitting GCP
- [ ] Use `unittest.mock` / `pytest-mock` to patch `vertexai.*` imports

---

## 9. Documentation (`decisions.md`)
- [ ] **Similarity metric choice** — why cosine over Euclidean
  (unit-normalised embeddings, magnitude-invariant, standard for semantic search)
- [ ] **Vertex AI migration path** — how you'd swap out each layer:
  - `sentence-transformers` → `textembedding-gecko` via `TextEmbeddingModel.get_embeddings()`
  - `ChromaDB` → Vertex AI Vector Search (Matching Engine): index creation, upsert via `IndexEndpoint`, deploy + query flow

---

## 10. `retrieval_benchmark.md`
- [ ] Copy the benchmark table output here
- [ ] Brief commentary on observed differences between Strategy A and B per query

---

## 11. Local LLM — Full Generation Pipeline (Beyond Assignment Scope)
- [ ] Integrate **Ollama** with LangChain for actual response generation
  - [ ] Augmentation step — inject retrieved chunks into a prompt template before sending to LLM
  - [ ] Full RAG loop: Query → Retrieve → Augment → Generate → Response

---

## 12. Advanced Retrieval (Beyond Assignment Scope)
- [ ] **Ensemble Retriever** — combine BM25 (keyword) + semantic (embedding) + vector search
  - Use `langchain.retrievers.EnsembleRetriever` with configurable weights
- [ ] **Agent-based Query ReWriter** — replace the mock with a real LLM-backed rewriter via Ollama

---

## 13. Productionisation (Beyond Assignment Scope)
- [ ] **VdB Optimisation** — tune ChromaDB HNSW params (`ef_construction`, `M`, `ef_search`)
- [ ] **Token management** — track prompt token counts, enforce context window limits
- [ ] **Inference timing** — log retrieval latency + generation latency per query
- [ ] **Guard-rails** — input validation, empty-result handling, max chunk fallback
- [ ] **Edge-case handling** — no results found, query too short, duplicate chunks
- [ ] **Docker** — containerise the full pipeline with `requirements.txt` pinned

---

## Execution Order

**Assignment core (submit this):**
`DataIngestor` → `EmbeddingModel` → `VectorStore` → `Mocks` → `RawRetriever` → `QueryExpandedRetriever` → `BenchmarkRunner` → Tests → Docs → `retrieval_benchmark.md`

**Portfolio layer (build on top):**
`Ollama + LangChain` → `Augmentation / Prompt Template` → `BM25 + Ensemble Retriever` → `Agent ReWriter` → `VdB Tuning` → `Guard-rails + Edge Cases` → `Inference Logging` → `Docker`