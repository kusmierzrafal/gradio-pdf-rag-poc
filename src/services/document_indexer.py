"""
Document indexing orchestrator that combines all processing services.
"""

import time
from typing import Tuple, Dict, Any

from ..core.config import Config
from ..services.document_processor import DocumentProcessor
from ..services.embedding_service import EmbeddingService
from ..services.vector_store import VectorStore


class DocumentIndexer:
    """Orchestrates the complete document indexing pipeline."""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
    
    def create_index(self, pdf_path: str, chunk_size: int = None, overlap: int = None) -> Tuple[Dict[str, Any], str]:
        """Create a complete searchable index from a PDF document."""
        try:
            start_time = time.time()
            
            # Extract text from PDF
            pages = self.document_processor.extract_text_from_pdf(pdf_path)
            if not pages:
                return {}, "No text found in PDF"
            
            # Create chunks
            chunks = self.document_processor.create_chunks(pages, chunk_size, overlap)
            if not chunks:
                return {}, "No chunks created from PDF"
            
            # Generate embeddings
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = self.embedding_service.embed_texts(chunk_texts)
            normalized_embeddings = self.embedding_service.normalize_embeddings(embeddings)
            
            # Create vector store
            vector_store = VectorStore(Config.EMBEDDING_DIMENSIONS)
            vector_store.add_embeddings(normalized_embeddings, chunks)
            
            # Prepare state
            state = {
                "vector_store": vector_store,
                "chunks": [chunk.to_dict() for chunk in chunks],
                "total_pages": len(pages),
                "total_chunks": len(chunks)
            }
            
            elapsed_time = time.time() - start_time
            message = f"Successfully indexed {len(chunks)} chunks from {len(pages)} pages in {elapsed_time:.2f}s"
            
            return state, message
            
        except Exception as e:
            return {}, f"Error processing PDF: {str(e)}"