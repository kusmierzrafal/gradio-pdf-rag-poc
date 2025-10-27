"""
Question answering system using hybrid semantic and keyword search.
"""

import re
from typing import List, Tuple, Dict, Any
from openai import OpenAI

from ..core.config import Config
from ..services.embedding_service import EmbeddingService
from ..services.vector_store import VectorStore


class QuestionAnsweringSystem:
    """Handles document question answering using hybrid semantic and keyword search."""
    
    KEYWORD_PATTERNS = {
        'regon': [r'\bregon\b', r'\b\d{9}\b'],
        'nip': [r'\bnip\b', r'\b\d{10}\b'],
        'adres': [r'\badres\b', r'\bulica\b', r'\bul\.\b', r'wrocław', r'warszawa'],
        'nazwa': [r'\bnazwa\b', r'\bspółka\b', r'\bfirma\b', r'archicom', r'projekt'],
        'telefon': [r'\btelefon\b', r'\btel\b', r'\+\d+', r'\bphone\b'],
        'email': [r'\bemail\b', r'\be-mail\b', r'@\w+\.\w+'],
        'strona': [r'\bstrona\b', r'\bwww\b', r'internetowa', r'website'],
        'kondygnacji': [r'\bkondygnacji\b', r'\bkondygnacja\b', r'\bpiętr\b', r'\bfloor\b'],
        'lokali': [r'\blokali\b', r'\blokale\b', r'\bmieszkan\b', r'w budynku'],
        'termin': [r'\btermin\b', r'\bdata\b', r'\brozpocz\b', r'\bzakończ\b', r'\brobót\b']
    }

    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.embedding_service = EmbeddingService()

    @staticmethod
    def _build_context(search_results: List[Tuple[Dict[str, Any], float]]) -> str:
        """Format search results into context string."""
        context_parts = []
        for i, (metadata, score) in enumerate(search_results, 1):
            header = f"[Source {i} | page {metadata['page_num']} | score {score:.3f}]"
            context_parts.extend([header, metadata["text"].strip(), ""])
        return "\n".join(context_parts)

    def answer_question(self, state: Dict[str, Any], question: str, top_k: int = 5, temperature: float = 0.0) -> str:
        """Answer a question using the indexed document."""
        if not state or "vector_store" not in state:
            return "Please index a PDF first."
        
        vector_store: VectorStore = state["vector_store"]
        chunks = state["chunks"]
        
        # Semantic search
        query_embedding = self.embedding_service.embed_texts([question])
        query_embedding = self.embedding_service.normalize_embeddings(query_embedding)
        semantic_results = vector_store.similarity_search(query_embedding, k=top_k)
        
        # Keyword-based search for specific terms
        keyword_results = self._perform_keyword_search(question, chunks)
        
        # Combine and deduplicate results
        final_results = self._combine_search_results(semantic_results, keyword_results, top_k)
        
        # Generate answer using LLM
        context = self._build_context(final_results)
        return self._generate_answer(question, context, temperature)

    def _perform_keyword_search(self, question: str, chunks: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float]]:
        """Search for chunks using keyword patterns."""
        question_lower = question.lower()
        relevant_patterns = []
        
        for keyword, patterns in self.KEYWORD_PATTERNS.items():
            if keyword in question_lower:
                relevant_patterns.extend(patterns)
        
        keyword_results = []
        if relevant_patterns:
            for chunk in chunks:
                for pattern in relevant_patterns:
                    if re.search(pattern, chunk["text"], re.I):
                        score = len(re.findall(pattern, chunk["text"], re.I)) * 0.8
                        keyword_results.append((chunk, score))
                        break
        
        return keyword_results

    @staticmethod
    def _combine_search_results(
        semantic_results: List[Tuple[Dict[str, Any], float]],
        keyword_results: List[Tuple[Dict[str, Any], float]],
        max_results: int
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Combine and deduplicate semantic and keyword search results."""
        combined_results = []
        seen_chunks = set()
        
        # Add semantic results first
        for metadata, score in semantic_results:
            chunk_key = f"{metadata['page_num']}_{metadata['chunk_id']}"
            if chunk_key not in seen_chunks:
                combined_results.append((metadata, score))
                seen_chunks.add(chunk_key)
        
        # Add keyword results if not already included
        for metadata, score in keyword_results:
            chunk_key = f"{metadata['page_num']}_{metadata['chunk_id']}"
            if chunk_key not in seen_chunks:
                combined_results.append((metadata, score))
                seen_chunks.add(chunk_key)
        
        # Sort by score and limit results
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:max(max_results, 8)]

    def _generate_answer(self, question: str, context: str, temperature: float) -> str:
        """Generate answer using OpenAI GPT."""
        system_prompt = (
            "You are a precise assistant. Answer the user's question ONLY using the provided PDF context. "
            "If the context is insufficient, say you don't know. Be concise. Do not fabricate details.\n"
            "For Polish queries like 'Podaj X' (Give/Provide X), extract the exact value you find in the context.\n"
            "For questions about numbers, dates, or specific values, provide the exact text from the document.\n"
            "Use bullet points for lists. Avoid repeating the question."
        )
        
        user_prompt = f"Question:\n{question}\n\nContext:\n{context}"
        
        response = self.client.chat.completions.create(
            model=Config.CHAT_MODEL,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        
        return response.choices[0].message.content.strip()