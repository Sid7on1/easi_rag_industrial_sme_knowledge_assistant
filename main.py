import logging
import os
import sys
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from pydantic import BaseModel
from langchain import LLMChain, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInference
from langchain.indexes import HuggingFaceIndex
from langchain.llms import AI21
from langchain.text_splitter import TokenTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from uvicorn import run
from click import Command, Group
from rich import print
from rich.logging import RichHandler
from rich.console import Console

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[RichHandler(rich_tracebacks=True)],
)

# Initialize console
console = Console()

# Define constants
RAG_MODEL_NAME = "t5-base"
INDEX_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 512
MAX_CONTEXT_LENGTH = 2048

# Define data models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

# Define exception classes
class RAGException(Exception):
    pass

# Define helper classes and utilities
class RAGSystem:
    def __init__(self, model_name: str, index_model_name: str, embedding_model_name: str):
        self.model_name = model_name
        self.index_model_name = index_model_name
        self.embedding_model_name = embedding_model_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.index_model = SentenceTransformer(index_model_name)
        self.embedding_model = HuggingFaceInference(embedding_model_name)
        self.index = HuggingFaceIndex(self.embedding_model)
        self.chain = RetrievalQA(
            self.model,
            self.tokenizer,
            self.index,
            return_source_documents=True,
        )

    def handle_query(self, query: str) -> str:
        try:
            response = self.chain({"query": query})
            return response["answer"]
        except Exception as e:
            logging.error(f"Error handling query: {e}")
            raise RAGException("Error handling query")

# Define main class
class RAGApplication:
    def __init__(self):
        self.rag_system = None

    def initialize_rag_system(self):
        self.rag_system = RAGSystem(
            RAG_MODEL_NAME, INDEX_MODEL_NAME, EMBEDDING_MODEL_NAME
        )

    def run_cli_mode(self):
        self.initialize_rag_system()
        while True:
            query = input("Enter query: ")
            try:
                answer = self.rag_system.handle_query(query)
                print(f"Answer: {answer}")
            except RAGException as e:
                print(f"Error: {e}")

    def run_api_server(self):
        self.initialize_rag_system()
        app = FastAPI()

        @app.post("/query", response_model=QueryResponse)
        async def handle_query(request: QueryRequest):
            try:
                answer = self.rag_system.handle_query(request.query)
                return QueryResponse(answer=answer)
            except RAGException as e:
                return JSONResponse(content={"error": str(e)}, status_code=400)

        run(app, host="0.0.0.0", port=8000)

# Define CLI commands
class RAGCLI:
    def __init__(self):
        self.app = RAGApplication()

    def cli_mode(self):
        self.app.run_cli_mode()

    def api_server(self):
        self.app.run_api_server()

# Define main function
def main():
    cli = RAGCLI()
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        cli.api_server()
    else:
        cli.cli_mode()

if __name__ == "__main__":
    main()