"""
Core data models for the PDF RAG system.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document with metadata."""
    text: str
    page_num: int
    chunk_id: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary format."""
        return {
            "text": self.text,
            "page_num": self.page_num,
            "chunk_id": self.chunk_id,
            **self.metadata
        }