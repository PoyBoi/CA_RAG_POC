# Architecture & Design Decisions

## 1. Similarity Metric: Cosine vs Euclidean

**Choice: Cosine similarity**

All embeddings are L2-normalised at encode time (`normalize_embeddings=True` in both
`HuggingFaceEmbeddings` and `SentenceTransformer`).  For unit vectors the cosine
similarity equals the dot product, so the distance computation is fast.

Key reasons:
- **Magnitude-invariant** — a short paragraph and a long paragraph about the same
  topic will be ranked equally; Euclidean distance penalises longer texts.
- **Bounded and interpretable** — values in [-1, 1]; a score of 0.9+ reliably
  signals high semantic overlap.
- **Industry standard** — all major semantic search benchmarks (BEIR, MTEB) report
  cosine similarity; ChromaDB and FAISS both support it natively.
- **Already the default** in ChromaDB (`cosine` distance space) and in
  FAISS when vectors are normalised before insertion.

Trade-off: Euclidean distance can be better when magnitude carries semantic meaning
(e.g. TF-IDF weighted vectors), but for dense transformer embeddings this is rarely
the case.

---

## 2. Chunking Strategy: Fixed-size with overlap

**Choice: 256 tokens / 50-token overlap via `RecursiveCharacterTextSplitter`**

- **256 tokens** fits comfortably within the embedding model's max sequence length
  (all-MiniLM-L6-v2 supports 512 tokens) while keeping chunks focused.
- **50-token overlap** prevents answer spans from being split across chunk
  boundaries — a question whose answer straddles the boundary of two chunks
  will still be captured in at least one.
- `RecursiveCharacterTextSplitter` tries sentence → paragraph → word boundaries
  before hard-cutting, which preserves semantic coherence better than naive
  character splitting.

Trade-off: smaller chunks increase retrieval cost and may lose inter-sentence
context.  A future iteration could explore semantic chunking (splitting on sentence
boundaries detected by a sentence segmenter) or hierarchical chunking (parent +
child documents).

---

## 3. Retrieval Strategies

### Strategy A — Raw Vector Retrieval
Embed query as-is → cosine similarity → top-K.
Simple baseline; fast; sensitive to vocabulary mismatch.

### Strategy B — Query-Expanded Retrieval
Rewrite query with `MockGenerativeModel` (or a real LLM) → embed expanded
query → cosine similarity → top-K.
Better recall for sparse or vague queries; adds LLM latency.

### Ensemble Retriever (portfolio layer)
`EnsembleRetriever` with:
- BM25 (weight 0.3) — exact keyword match, good for named entities
- Dense vector (weight 0.4) — semantic similarity
- MMR dense (weight 0.3) — diversity via Maximal Marginal Relevance (`lambda_mult=0.7`)

Weights are empirical starting points; tune via the benchmark runner.

---

## 4. Vertex AI Migration Path

| Component | Local (current) | Vertex AI (migration) |
|---|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | `TextEmbeddingModel.from_pretrained("textembedding-gecko@003")` |
| Vector store | ChromaDB | Vertex AI Vector Search (Matching Engine) |
| Generative model | Ollama / MockGenerativeModel | `GenerativeModel("gemini-pro")` |

**Embedding swap:** Replace `SentenceTransformerEmbeddingModel` with
`VertexTextEmbeddingModel` in `embedding_model.py`.  Interface is identical
(`.get_embeddings()` → `.values`).

**Vector store swap (Matching Engine):**
1. Create an Index with `distanceMeasureType: COSINE_DISTANCE`
2. Upsert datapoints via `index.upsert_datapoints(datapoints=[...])`
3. Deploy to an `IndexEndpoint`
4. Query via `endpoint.find_neighbors(deployed_index_id=..., queries=[...], num_neighbors=k)`

The `VectorStore.add()` / `VectorStore.query()` interface in `vector_store.py` is
designed so only the backend implementation changes — callers are unaffected.

---

## 5. Incremental Ingestion

The `DataIngestor` computes an MD5 hash of each source file and compares it
against the manifest (`ingest_manifest.json`).  Only new or changed files are
chunked and re-embedded.  This keeps re-ingestion fast for large corpora.

---

## 6. LLM Configuration

- `temperature=0.1` — factual RAG answers; low diversity preferred
- `num_ctx=4096` — sufficient for retrieved chunks + question; increase to 8192
  for longer documents
- Model selected via `MODEL_MAP` alias; swap by changing the alias

---

## TODO (expand before submission)

- [ ] Data analysis methodology notes (5 analyses + graph descriptions)
- [ ] Data cleaning steps performed on the dataset
- [ ] Benchmark observations: Strategy A vs B per query
- [ ] HNSW tuning rationale (`ef_construction`, `M`, `ef_search`)