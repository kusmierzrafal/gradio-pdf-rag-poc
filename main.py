#!/usr/bin/env python3
"""
PDF RAG System - Main Application Entry Point

A professional document analysis and structured data extraction system
using retrieval-augmented generation (RAG) with OpenAI embeddings and FAISS vector search.
"""

import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import Config
from src.ui import PDFRagInterface


def main():
    """Main entry point for the PDF RAG application."""
    try:
        # Validate configuration
        Config.validate()
        
        # Create and launch interface
        interface = PDFRagInterface()
        app = interface.create_interface()
        
        # Get server configuration
        port = int(os.getenv("PORT", Config.DEFAULT_PORT))
        host = os.getenv("HOST", Config.DEFAULT_HOST)
        
        print(f"🚀 Starting PDF RAG System on {host}:{port}")
        print(f"📊 Using OpenAI model: {Config.CHAT_MODEL}")
        print(f"🔍 Embedding model: {Config.EMBEDDING_MODEL}")
        
        app.launch(server_name=host, server_port=port)
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("💡 Please set your OPENAI_API_KEY environment variable")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()