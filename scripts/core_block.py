import os, pickle, sys, logging, re, asyncio, torch, ollama, langchain

from typing import List, Dict, Optional, Literal

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from dotenv import load_dotenv
load_dotenv()

class init_core_dependancies:
    def __init__(
        self,
        llm_model: Literal["llama", "qwen"],
        vdb: Literal["chromadb", "faiss"],
        db_location: str = "",
        ingest_data: bool = False,

        # args for basic pass-through (dry run)
        use_stub_llm: bool = False,
    ):
        self.llm_model = llm_model
        self.vdb = vdb
        self.db_location = db_location
        self.ingest_data = ingest_data
        # call the init functions here that will be defined later on
    
    def _init_llm(self):
        from langchain_ollama import ChatOllama
        # load LLM from ollama, passing it onto langchain
        self.llm_model
        # ollama run deepseek-r1:14b

    def _init_vdb(self):
        # load chroma alongside it's location and all that
        from langchain_chroma import Chroma
        self.vdb
    
    def _init_embedding(self, embedding_model: str = None):
        # choose a valid, small and locally viable embedding model to init here
        from sentence_transformers import SentenceTransformer
        sentences = ["This is an example sentence", "Each sentence is converted"]

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(sentences)
        print(embeddings)

    def _init_retreiver(self):
        from langchain.retrievers import EnsembleRetriever
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.retrievers import BM25Retriever
        from langchain.retrievers import EnsembleRetriever
        from sentence_transformers import CrossEncoder
    
    def _init_db(self):
        self.db_location
        # init the db and check if it needs to be read
        # basically using the incremental logic here, mark which rows are new and all that and store that in a file
        # Shows which files need to be worked on, IF they need to be worked on using `ingest_data` (bool)