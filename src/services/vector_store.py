"""
FAISS-based vector store for similarity search.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import faiss

from ..models import DocumentChunk


class VectorStore:
    """FAISS-based vector storage and similarity search."""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.chunk_metadata: List[Dict[str, Any]] = []
    
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[DocumentChunk]):
        """Add embeddings and associated chunks to the store."""
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        
        self.index.add(embeddings)
        self.chunk_metadata.extend([chunk.to_dict() for chunk in chunks])
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar chunks."""
        if self.index.ntotal == 0:
            return []
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Valid index
                results.append((self.chunk_metadata[idx], float(score)))
        
        return results
    
    def get_total_chunks(self) -> int:
        """Get total number of chunks in the store."""
        return self.index.ntotal