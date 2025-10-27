"""
OpenAI embedding service for text vectorization.
"""

from typing import List
import numpy as np
from openai import OpenAI

from ..core.config import Config


class EmbeddingService:
    """Handles text embedding operations using OpenAI."""
    
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        response = self.client.embeddings.create(
            model=Config.EMBEDDING_MODEL,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype=np.float32)
    
    def embed_single_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.embed_texts([text])[0]
    
    @staticmethod
    def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return embeddings / norms