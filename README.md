# Context-Aware RAG Assignment

Showcase of implemention of a local RAG pipeline with demonstrated ingestion of raw textual data, generation of embeddings, and semantic searching

# How to run ?

1. Clone the repository
2. CD to the location of the folder
3. Install the dependancies:
```
pip install -r requirements.txt
```
- Optional (if you want to run Ollama, please run this):
```
curl -fsSL https://ollama.com/install.sh | sh
```
4. In the root directory, to do a stub-test for the core file, run:
```
python scripts/core_block.py
```
5. In the same directory, to do a test for the data ingestor, run:
```
python scripts/data_ingestor.py
```
6. To run with LLM (or stub LLM), follow the following commands:
    - Stub LLM:
    ```
    python scripts/pipeline.py --ingest --stub-llm
    ```
    - Llama 3.2:
    ```
    python scripts/pipeline.py --ingest --llm llama
    ```
    - Deepseek R1:
    ```
    python scripts/pipeline.py --ingest --llm deepseek 
    ```
    - Qwen 2.5:
    ```
    python scripts/pipeline.py --ingest --llm qwen
    ```