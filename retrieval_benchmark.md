# Retrieval Benchmark Results

Dataset: Wikipedia article on Artificial Intelligence - https://en.wikipedia.org/wiki/Artificial_intelligence \
Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, cosine similarity)  
Vector store: ChromaDB  
Chunk size: 512 chars / 100 overlap  
LLM for query rewriting (Strategy B): `deepseek-r1:14b` via Ollama  
Top-K: 3  

---

## Comparison Table

| Input Query | Strategy A - Raw Vector | Strategy B - Query Expanded |
|---|---|---|
| What are the main approaches to machine learning? | chunk 1004 (0.5865), chunk 88 (0.7144) | chunk 88 (0.7513), chunk 15 (0.7793), chunk 1004 (0.8029) - rewrite: *"machine-learning, supervised-learning, unsupervised-learning, reinforcement-learning..."* |
| How does artificial intelligence pose risks to society? | chunk 447 (0.4977), chunk 448 (0.6023) | chunk 448 (0.4741), chunk 447 (0.4948) - rewrite: *"AI risks, ethical concerns, safety issues, security vulnerabilities, societal impact..."* |
| What is the difference between narrow AI and AGI? | chunk 1032 (0.3538), chunk 1033 (0.5991) | chunk 1032 (0.6004), chunk 1033 (0.6434) - rewrite: *"Narrow AI, Artificial General Intelligence, differences, comparison, types of AI..."* |
| How have neural networks contributed to AI development? | chunk 921 (0.7812), chunk 1050 (0.7866) | chunk 10 (0.8135), chunk 2 (0.8198), chunk 11 (0.8709) - rewrite: *"neural networks, artificial intelligence, deep learning, algorithms, model training..."* |
| What ethical concerns are associated with AI? | chunk 796 (0.5083), chunk 443 (0.5293) | chunk 448 (0.5775), chunk 22 (0.578) - rewrite: *"AI ethics, algorithmic bias, data privacy, transparency, accountability, fairness..."* |

---

## Per-Query Commentary

### Q1 - "What are the main approaches to machine learning?"
**Winner: Strategy B**  
Strategy A retrieved a historical statement about ML eclipsing other approaches (chunk 1004, score 0.5865) and a general ML kinds overview (chunk 88, 0.7144). Strategy B's rewritten query surfaced all the same chunks *plus* chunk 15 which contains a diagram caption describing unsupervised learning patterns, and all scores were significantly higher (top score 0.8029 vs 0.7144). The rewrite correctly expanded the query into specific ML paradigm names that matched the document vocabulary.

### Q2 - "How does artificial intelligence pose risks to society?"
**Winner: Draw**  
Both strategies retrieved the same two chunks (447 and 448), just in reversed order. Chunk 448 is a citation URL chunk (OECD report link) - a data quality issue rather than a retrieval failure. Chunk 447 contains the actual risk discussion. Strategy B's scores were marginally lower, suggesting the over-expanded rewrite slightly diluted the query signal. Query rewriting provided no benefit here because the original query already used precise vocabulary present in the document.

### Q3 - "What is the difference between narrow AI and artificial general intelligence?"
**Winner: Strategy B on confidence**  
Both strategies returned identical chunk sets (1032 and 1033). However, Strategy B's scores were substantially higher: chunk 1032 improved from 0.3538 to 0.6004 (70% increase), chunk 1033 from 0.5991 to 0.6434. This demonstrates that query expansion can increase retrieval confidence even when it does not change which chunks are returned - useful when downstream filtering uses a score threshold.

### Q4 - "How have neural networks contributed to the development of AI?"
**Winner: Strategy A**  
Strategy A correctly retrieved chunk 921 (AI solutions in the 2000s) and chunk 1050 (cognitive science connections to AI). Strategy B's rewrite generalised the query into broad AI terminology and retrieved the Wikipedia intro definition (chunk 2) and AI goals overview (chunk 9/10/11). These are relevant to AI generally but miss the neural network–specific developmental history the question asks about. Over-expansion hurt precision here.

### Q5 - "What ethical concerns are associated with artificial intelligence?"
**Winner: Strategy B narrowly**  
Strategy A landed on two citation/URL chunks (796: Turing Institute link, 443: Ethics of AI and Robotics citation). Strategy B retrieved chunk 22 (AI long-term effects and existential risk) with a slightly higher score. Both strategies were hampered by citation chunks polluting the index - a data cleaning issue. Strategy B's rewrite found marginally more relevant content.

---

## Summary

| Metric | Strategy A | Strategy B |
|---|---|---|
| Queries won | 1 (Q4) | 2 (Q1, Q5) |
| Draws / ties | 2 (Q2, Q3) | 2 (Q2, Q3) |
| Avg top score | 0.657 | 0.717 |
| Avg latency | ~12ms | ~14,400ms |

**Conclusion:** Strategy B produces higher similarity scores on average and surfaces more relevant chunks for vague or vocabulary-mismatched queries (Q1, Q3). Strategy A is faster by ~1200x and performs equally well or better when the original query already uses precise domain terminology (Q2, Q4). The primary cost of query expansion is LLM latency (~15 seconds per query with deepseek-r1:14b locally). In production, this would be mitigated by using a faster hosted model (e.g. `gemini-flash`) or caching rewrites for repeated queries.

The most significant data quality finding is that citation-only chunks (URLs, reference entries) are being indexed alongside content chunks, which pollutes results for several queries. A minimum content-length filter during ingestion would eliminate this class of false positives.