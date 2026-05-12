Deliverables:
- [ ] Local LLM Model
- [ ] RAG
    - [ ] Ingestion pipeline for raw data
        - [ ] Make a Class that handles the ingestion (do the things below)
        - Optional:
            - [ ] Cleaning that data up
            - [ ] Formatting that data
            - [ ] Doing basic "data" analysis on the dataset
    - [ ] Generate Embeddings
        - use `sentence-transformers` (or any other local library) to emulate `vertexAI`'s local embedding model's behaviour
    - [ ] VdB
        - `FAISS` / `ChromaDB` (prefer ChromaDB)
        - Optional:
            - [ ] Optimise the VdB
    - [ ] Mocking
        - Mock the `vertexai.language_models.TextEmbeddingModel` and `GenerativeModel` for the query expansion phase.
    - [ ] Retrieval
        - [ ] Raw Vector Search
            - Traditional embedding-based similarity search
            - Optional:
                - [ ] Use ensemble retreiver also as an option
                    - BM25, sematic, vector
        - [ ] AI-Enhanced Retrieval
            - Using a (mocked) model to rewrite/expand the user query into a better embedding-friendly format before searching
            - Optional:
                - [ ] `QueryReWriter` Class
                - [ ] Use a agent based `query re-writer` OR a small trained & fine-tuned model that trims down sentences based on their importance, basically a `transformer-based_sentence-trimmer`
    - [ ] Productivisation:
        - Check out:
            - scaling
            - edge-cases
            - token management
            - log inference timing (eval and observability)
            - guard-rails

- [ ] Submission:
    - [ ] `Pytest Suite` (mocking the GCP SDK)
    - [ ] A `retrieval_benchmark.md` file inside the repo showing the output of the "Strategy A vs Strategy B" comparison.
    - [ ] Documentation:
        - [ ] Explain how you would migrate this to `Vertex AI Vector Search` (`Matching Engine`) in production
        - [ ] Explain the choice of `similarity metric` (Cosine vs. Euclidean)
            - [ ] The Benchmarking Test:
                - A `.json` of the diff of the two different methods that we deployed for atleast `5` complex queries
                - Table content:

| Input Query | Result A (Raw Vector) | Result B (QueryReWriter) |
|---|---|---|
| How does the system handle peak load? | Top 3 chunks retrieved via direct embedding | Top 3 chunks retrieved after the "Query Expansion" model rewrites the input |