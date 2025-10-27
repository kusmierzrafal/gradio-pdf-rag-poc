"""
Document processing service for PDF text extraction and chunking.
"""

import uuid
from typing import List, Tuple

import fitz  # PyMuPDF

from ..core.config import Config
from ..models import DocumentChunk


class DocumentProcessor:
    """Handles PDF document processing and text chunking."""

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> List[Tuple[str, int]]:
        """Extract text from PDF with page numbers."""
        pages = []
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                if text.strip():
                    pages.append((text, page_num))
        return pages

    @classmethod
    def create_chunks(
        cls, 
        pages: List[Tuple[str, int]], 
        chunk_size: int = None, 
        overlap: int = None
    ) -> List[DocumentChunk]:
        """Split pages into overlapping chunks."""
        chunk_size = chunk_size or Config.DEFAULT_CHUNK_SIZE
        overlap = overlap or Config.DEFAULT_OVERLAP
        
        chunks = []
        for text, page_num in pages:
            page_chunks = cls._split_text_into_chunks(text, chunk_size, overlap)
            for chunk_text in page_chunks:
                if len(chunk_text.strip()) >= Config.MIN_CHUNK_SIZE:
                    chunk = DocumentChunk(
                        text=chunk_text,
                        page_num=page_num,
                        chunk_id=str(uuid.uuid4())[:8]
                    )
                    chunks.append(chunk)
        return chunks

    @staticmethod
    def _split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:
                    chunk = text[start:break_point + 1]
                    end = break_point + 1
            
            chunks.append(chunk)
            start = max(start + 1, end - overlap)
            
            if end >= len(text):
                break
        
        return chunks