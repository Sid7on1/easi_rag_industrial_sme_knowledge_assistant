import logging
import os
import re
from typing import List, Tuple
from langchain import LLMChain
from langchain.chains import HuggingFacePipeline
from langchain.llms import AI21
from python_docx import Document
from openpyxl import load_workbook
from tiktoken import get_encoding
from pandas import DataFrame, read_excel
from PIL import Image
from pytesseract import image_to_string
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from pydantic import BaseModel
from enum import Enum
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MAX_CHUNK_SIZE = 512
MIN_CHUNK_SIZE = 128
THREAD_POOL_SIZE = 5
IMAGE_TEXT_EXTRACTION_THRESHOLD = 0.5

# Configuration
class DocumentProcessorConfig(BaseModel):
    max_chunk_size: int = MAX_CHUNK_SIZE
    min_chunk_size: int = MIN_CHUNK_SIZE
    thread_pool_size: int = THREAD_POOL_SIZE
    image_text_extraction_threshold: float = IMAGE_TEXT_EXTRACTION_THRESHOLD

# Exception classes
class DocumentProcessorException(Exception):
    pass

class InvalidDocumentException(DocumentProcessorException):
    pass

class ChunkingException(DocumentProcessorException):
    pass

# Data structures/models
class DocumentChunk(BaseModel):
    text: str
    chunk_id: int

class PreprocessedDocument(BaseModel):
    chunks: List[DocumentChunk]

# Helper classes and utilities
class TextExtractor:
    def __init__(self, config: DocumentProcessorConfig):
        self.config = config

    def extract_text_from_image(self, image_path: str) -> str:
        try:
            image = Image.open(image_path)
            text = image_to_string(image)
            return text
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return ""

class Chunker:
    def __init__(self, config: DocumentProcessorConfig):
        self.config = config

    def apply_hierarchical_chunking(self, text: str) -> List[DocumentChunk]:
        chunks = []
        chunk_id = 0
        while len(text) > self.config.max_chunk_size:
            chunk = text[:self.config.max_chunk_size]
            chunks.append(DocumentChunk(text=chunk, chunk_id=chunk_id))
            text = text[self.config.max_chunk_size:]
            chunk_id += 1
        if len(text) > self.config.min_chunk_size:
            chunks.append(DocumentChunk(text=text, chunk_id=chunk_id))
        return chunks

# Main class
class DocumentProcessor:
    def __init__(self, config: DocumentProcessorConfig):
        self.config = config
        self.text_extractor = TextExtractor(config)
        self.chunker = Chunker(config)
        self.llm_chain = LLMChain(llm=AI21(), prompt="")

    def load_documents(self, document_paths: List[str]) -> List[PreprocessedDocument]:
        preprocessed_documents = []
        with ThreadPoolExecutor(max_workers=self.config.thread_pool_size) as executor:
            futures = []
            for document_path in document_paths:
                futures.append(executor.submit(self.preprocess_document, document_path))
            for future in futures:
                preprocessed_document = future.result()
                preprocessed_documents.append(preprocessed_document)
        return preprocessed_documents

    def preprocess_document(self, document_path: str) -> PreprocessedDocument:
        try:
            if document_path.endswith(".docx"):
                return self.preprocess_word_document(document_path)
            elif document_path.endswith(".xlsx") or document_path.endswith(".xls"):
                return self.preprocess_excel_document(document_path)
            elif document_path.endswith(".jpg") or document_path.endswith(".png"):
                return self.preprocess_image_document(document_path)
            else:
                raise InvalidDocumentException(f"Unsupported document type: {document_path}")
        except Exception as e:
            logger.error(f"Error preprocessing document: {e}")
            raise

    def preprocess_word_document(self, document_path: str) -> PreprocessedDocument:
        document = Document(document_path)
        text = ""
        for paragraph in document.paragraphs:
            text += paragraph.text + " "
        chunks = self.chunker.apply_hierarchical_chunking(text)
        return PreprocessedDocument(chunks=chunks)

    def preprocess_excel_document(self, document_path: str) -> PreprocessedDocument:
        workbook = load_workbook(document_path)
        text = ""
        for sheet in workbook.worksheets:
            for row in sheet.rows:
                for cell in row:
                    text += str(cell.value) + " "
        chunks = self.chunker.apply_hierarchical_chunking(text)
        return PreprocessedDocument(chunks=chunks)

    def preprocess_image_document(self, document_path: str) -> PreprocessedDocument:
        text = self.text_extractor.extract_text_from_image(document_path)
        chunks = self.chunker.apply_hierarchical_chunking(text)
        return PreprocessedDocument(chunks=chunks)

    def create_chunks(self, text: str) -> List[DocumentChunk]:
        return self.chunker.apply_hierarchical_chunking(text)

    def extract_text_from_images(self, image_paths: List[str]) -> List[str]:
        texts = []
        with ThreadPoolExecutor(max_workers=self.config.thread_pool_size) as executor:
            futures = []
            for image_path in image_paths:
                futures.append(executor.submit(self.text_extractor.extract_text_from_image, image_path))
            for future in futures:
                text = future.result()
                texts.append(text)
        return texts

# Unit test compatibility
def test_document_processor():
    config = DocumentProcessorConfig()
    document_processor = DocumentProcessor(config)
    document_paths = ["test.docx", "test.xlsx", "test.jpg"]
    preprocessed_documents = document_processor.load_documents(document_paths)
    for preprocessed_document in preprocessed_documents:
        print(preprocessed_document)

if __name__ == "__main__":
    test_document_processor()