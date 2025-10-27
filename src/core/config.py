"""
Configuration settings for the PDF RAG system.
"""

import os
from typing import Optional


class Config:
    """Application configuration."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    CHAT_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536
    
    # Document Processing
    DEFAULT_CHUNK_SIZE: int = 2000
    DEFAULT_OVERLAP: int = 350
    MIN_CHUNK_SIZE: int = 50
    
    # Vector Search
    DEFAULT_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Server Configuration
    DEFAULT_PORT: int = 7860
    DEFAULT_HOST: str = "0.0.0.0"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration settings."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        return True