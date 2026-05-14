# Design Decisions

These are the choices I made building this pipeline and why. It's not a spec, but more of a running log of what I decided and what I'd do differently at scale.

---

## Similarity metric

I went with cosine similarity throughout. The short version: all embeddings are L2-normalised before storage, and for unit vectors cosine similarity is just a dot product, so it's fast and the math is clean. More importantly it's magnitude-invariant, which matters here because Wikipedia chunks vary a lot in length. A 40-word sentence and a 200-word paragraph on the same topic should rank equally against a query. Euclidean distance would penalise the longer one just for being longer, which isn't what you want for semantic search.

Euclidean *does* make more sense when magnitude carries meaning, TF-IDF vectors being the classic example, but dense transformer embeddings aren't that. Cosine is the right call.

---

## Chunking

I started at 256 chars with 50-char overlap, which is the standard default you see everywhere. It worked badly on this dataset. Wikipedia writes in long, dense paragraphs and 256 chars was cutting mid-sentence constantly, producing chunks that were grammatically incomplete and embedding poorly as a result.

Moved to 512 chars / 100-char overlap. The overlap is the more important number, it's there to make sure an answer that happens to sit right on a chunk boundary still gets captured fully in at least one chunk. `RecursiveCharacterTextSplitter` is good here because it tries to split on sentence and paragraph boundaries before resorting to hard character cuts, so you get cleaner chunks than a naive splitter would give you.

The honest trade-off: bigger chunks mean each one covers more topics, which dilutes the embedding signal slightly. If I were doing this properly I'd look at semantic chunking, actually detecting sentence boundaries and grouping by topic rather than by character count. But for this use case, 512/100 was the right pragmatic move.

---

## The two retrieval strategies

Strategy A is the baseline: embed the query as-is, cosine search, return top-K. Simple, fast (sub-10ms), works well when the user's vocabulary matches the document's.

Strategy B adds a query rewriting step before the search. The LLM takes the original query and expands it into a richer set of keywords and related terms that are more likely to appear verbatim in the source documents. The idea is to bridge the vocabulary gap, if a user asks "how does AI go wrong in society" but the document says "unintended consequences and risks", the rewrite can surface that connection.

Based on running the benchmark, it's genuinely useful for vague or abstractly-phrased queries, and not useful (sometimes actively harmful) for queries that are already precise. For the neural networks question, the rewrite generalised too aggressively and ended up returning the Wikipedia intro definition instead of the actual neural network history section. The original query was already specific enough, the rewrite lost information rather than adding it.

The other real cost is latency. Running deepseek-r1:14b locally takes about 15 seconds per rewrite, which makes Strategy B ~1200x slower than A in this setup. That gap disappears with a hosted model (Gemini Flash runs in ~500ms), but it's worth being honest about in a local context.

---

## Moving this to Vertex AI

The local stack, sentence-transformers, ChromaDB, Ollama, was the right choice for building and iterating quickly. Here's how each piece swaps out for production on GCP.

### Embedding model

Locally I'm using `all-MiniLM-L6-v2` from sentence-transformers. The Vertex equivalent would be something via `TextEmbeddingModel` (we would need to iterate through and find the best fit for us, minimising the inference/quality tradeoff):

```python
import vertexai
from vertexai.language_models import TextEmbeddingModel

vertexai.init(project="your-project", location="us-central1")
model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
results = model.get_embeddings(["your text here"])
embeddings = [r.values for r in results]
```

The swap point in this codebase is `embedding_model.py`, there's already a `VertexTextEmbeddingModel` stub class there waiting to be filled in. The interface is identical to the local version (`.get_embeddings()` → `.values`), so nothing else in the pipeline changes.

One thing to watch: gecko@003 produces 768-dimensional vectors vs 384 for all-MiniLM. The vector index is created with a fixed dimension, so you'd need to rebuild it from scratch when switching, you can't migrate in place.

### Vector store

ChromaDB runs in-process which is fine for development but obviously doesn't scale. On Vertex AI the equivalent is Vector Search (previously Matching Engine), which is a managed ANN service. The migration is four steps:

```python
from google.cloud import aiplatform

aiplatform.init(project="your-project", location="us-central1")

# 1. Creating the index
index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="rag-index",
    dimensions=768,
    approximate_neighbors_count=10,
    distance_measure_type="COSINE_DISTANCE",
    index_update_method="STREAM_UPDATE",  # lets you upsert without full rebuild
)

# 2. Upserting chunks (equivalent to VectorStore.add())
datapoints = [
    aiplatform.MatchingEngineIndex.Datapoint(
        datapoint_id=str(chunk.metadata["chunk_id"]),
        feature_vector=embedding,
    )
    for chunk, embedding in zip(chunks, embeddings)
]
index.upsert_datapoints(datapoints=datapoints)

# 3. Deploying to an endpoint, one-time setup
endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="rag-endpoint",
    public_endpoint_enabled=True,
)
endpoint.deploy_index(index=index, deployed_index_id="rag_deployed")

# 4. Querying (with it's equivalent in VectorStore.query())
query_vector = embedding_model.embed([query_text])[0]
response = endpoint.find_neighbors(
    deployed_index_id="rag_deployed",
    queries=[query_vector],
    num_neighbors=top_k,
)
for neighbor in response[0]:
    print(neighbor.id, neighbor.distance)
```

The `VectorStore` class in `vector_store.py` is structured so only the backend methods change, `add()` and `query()` present the same interface to everything above them, so `pipeline.py`, the benchmark runner, and the retrievers don't need to be touched.

### Query rewriting LLM

Locally, this is Ollama running deepseek-r1:14b. The Vertex equivalent would be Gemini:

```python
from vertexai.generative_models import GenerativeModel

model = GenerativeModel("gemini-1.5-flash")
response = model.generate_content(prompt)
expanded_query = response.text
```

`QueryReWriter` in `retriever.py` already handles both interfaces, it checks for `.invoke()` (LangChain/Ollama) and `.generate_content()` (Vertex/mock) and calls whichever one exists. So you just pass a `GenerativeModel` instance to `QueryExpandedRetriever` instead of `ChatOllama` and it works.

Flash is the right model choice here, rewrites don't need the quality ceiling of Pro, saving tokens and cost overhead, and the latency difference is substantial (~500ms vs ~2s).

### Why the mocks matter for this migration

The mock classes (`MockTextEmbeddingModel`, `MockGenerativeModel`) were built to mirror the Vertex API signatures exactly, `.from_pretrained()`, `.get_embeddings()`, `.generate_content()`. The whole test suite runs against those mocks (with no GCP credentials), and because the real Vertex classes present the same interface, the tests stay valid after migration without modification. That's the main practical value of having written them that way rather than just returning hardcoded strings.

---

## Incremental ingestion

The ingest manifest (`ingest_manifest.json`) tracks an MD5 hash of each source file. On re-ingestion, any file whose hash matches the stored value is skipped. This means you can add new documents to `./data` and re-run ingestion without re-embedding everything, only the new or changed files get processed. For a small Wikipedia article this doesn't matter much, but at corpus scale it's the difference between a 30-second operation and a 3-hour one.

---

## Data cleaning

The Wikipedia source has a lot of markup that doesn't carry semantic meaning, citation templates like `{{Sfnp|Russell|2021}}`, file embeds like `[[File:diagram.png|thumb|caption]]`, wikilinks like `[[deep learning|Deep Learning]]`. If you leave that in, the embedder wastes capacity encoding markup syntax instead of content, and you end up with chunks that are essentially just reference lists.

The `_clean_wikitext()` method in `data_ingestor.py` strips all of that before chunking. The main cases it handles: citation/template blocks, file embeds, converting wikilinks to their display text, `<ref>` citation blocks, section headers, and bare URLs.

There's still a residual problem, chunks that are *entirely* reference entries (a URL and nothing else) survive the cleaner because the URL is valid text, just useless for retrieval. A minimum content-length filter post-cleaning (drop anything under ~100 chars) would catch those. It's in the backlog.