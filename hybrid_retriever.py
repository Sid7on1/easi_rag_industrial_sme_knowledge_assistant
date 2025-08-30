import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from scipy import spatial
import faiss
from typing import List, Dict
from pydantic import BaseModel
from config import Config

class DenseRetriever(BaseModel):
    """Dense retriever class using Sentence Transformers."""
    model: SentenceTransformer

    def encode_dense_vectors(self, sentences: List[str]) -> np.ndarray:
        """Encode a list of sentences into dense vectors."""
        try:
            vectors = self.model.encode(sentences)
            return vectors
        except Exception as e:
            logging.error(f"Error encoding dense vectors: {e}")
            raise

class SparseRetriever(BaseModel):
    """Sparse retriever class using BM25."""
    index: BM25Okapi

    def encode_sparse_vectors(self, sentences: List[str]) -> np.ndarray:
        """Encode a list of sentences into sparse vectors."""
        try:
            vectors = self.index.get_vectors(sentences)
            return vectors
        except Exception as e:
            logging.error(f"Error encoding sparse vectors: {e}")
            raise

class HybridRetriever(BaseModel):
    """Hybrid retriever class combining dense and sparse retrieval."""
    dense_retriever: DenseRetriever
    sparse_retriever: SparseRetriever
    faiss_index: faiss.IndexFlatL2

    def encode_dense_vectors(self, sentences: List[str]) -> np.ndarray:
        """Encode a list of sentences into dense vectors."""
        return self.dense_retriever.encode_dense_vectors(sentences)

    def encode_sparse_vectors(self, sentences: List[str]) -> np.ndarray:
        """Encode a list of sentences into sparse vectors."""
        return self.sparse_retriever.encode_sparse_vectors(sentences)

    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Perform hybrid search using dense and sparse retrieval."""
        try:
            # Encode query into dense and sparse vectors
            dense_query = self.dense_retriever.encode_dense_vectors([query])
            sparse_query = self.sparse_retriever.encode_sparse_vectors([query])

            # Perform dense search
            dense_results = self.dense_retriever.model.encode(["query"])
            dense_distances = spatial.distance.cdist(dense_query, dense_results, "cosine")
            dense_indices = np.argsort(dense_distances, axis=1)[:top_k]

            # Perform sparse search
            sparse_results = self.sparse_retriever.index.get_vectors(["query"])
            sparse_distances = np.linalg.norm(sparse_query - sparse_results, axis=1)
            sparse_indices = np.argsort(sparse_distances)[:top_k]

            # Combine results using reranking
            results = self.rerank_results(dense_indices, sparse_indices, top_k)

            return results
        except Exception as e:
            logging.error(f"Error performing hybrid search: {e}")
            raise

    def rerank_results(self, dense_indices: np.ndarray, sparse_indices: np.ndarray, top_k: int) -> List[Dict]:
        """Rerank results using a combination of dense and sparse retrieval."""
        try:
            # Get dense and sparse results
            dense_results = self.dense_retriever.model.encode(["query"])
            sparse_results = self.sparse_retriever.index.get_vectors(["query"])

            # Calculate similarities
            dense_similarities = 1 - spatial.distance.cdist(dense_results, dense_results, "cosine")
            sparse_similarities = np.dot(sparse_results, sparse_results.T)

            # Rerank results
            reranked_indices = []
            for i in range(top_k):
                dense_index = dense_indices[i]
                sparse_index = sparse_indices[i]
                reranked_index = np.argmax(dense_similarities[dense_index] * sparse_similarities[sparse_index])
                reranked_indices.append(reranked_index)

            return reranked_indices
        except Exception as e:
            logging.error(f"Error reranking results: {e}")
            raise

    def combine_retrievals(self, dense_results: List[Dict], sparse_results: List[Dict]) -> List[Dict]:
        """Combine dense and sparse retrieval results."""
        try:
            # Combine results using a weighted average
            combined_results = []
            for dense_result, sparse_result in zip(dense_results, sparse_results):
                combined_result = {
                    "dense": dense_result,
                    "sparse": sparse_result,
                    "score": (dense_result["score"] + sparse_result["score"]) / 2
                }
                combined_results.append(combined_result)

            return combined_results
        except Exception as e:
            logging.error(f"Error combining retrievals: {e}")
            raise

def main():
    # Load configuration
    config = Config()

    # Initialize retrievers
    dense_retriever = DenseRetriever(model=SentenceTransformer("all-MiniLM-L6-v2"))
    sparse_retriever = SparseRetriever(index=BM25Okapi.from_documents(["query"] * 100))
    faiss_index = faiss.IndexFlatL2(128)
    hybrid_retriever = HybridRetriever(dense_retriever=dense_retriever, sparse_retriever=sparse_retriever, faiss_index=faiss_index)

    # Perform hybrid search
    query = "What is the meaning of life?"
    results = hybrid_retriever.hybrid_search(query, top_k=10)

    # Print results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()