"""
Data extraction system for structured information retrieval from documents.
"""

import re
import json
from typing import List, Dict, Any
from openai import OpenAI

from ..core.config import Config


class DataExtractionSystem:
    """Handles structured data extraction from documents."""
    
    EXTRACTION_KEYWORDS = [
        # Polish terms
        "regon", "nip", "adres", "nazwa", "telefon", "email", "strona", 
        "kondygnacji", "lokali", "budynku", "mieszkania", "deweloper",
        "spółka", "firma", "powierzchnia", "cena", "liczba",
        # English terms
        "company", "address", "phone", "website", "floor", "apartment",
        "building", "developer", "price", "area", "number"
    ]

    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)

    @staticmethod
    def _parse_extraction_schema(schema_text: str) -> List[str]:
        """Parse schema to extract field names."""
        try:
            maybe_json = json.loads(schema_text)
            if isinstance(maybe_json, dict):
                return list(maybe_json.keys())
            elif isinstance(maybe_json, list):
                return [str(x) for x in maybe_json]
        except Exception:
            pass
        
        # Parse as comma/line separated
        raw = [w.strip() for w in re.split(r"[,;\n]+", schema_text) if w.strip()]
        return raw if raw else [
            "CompanyName", "Address", "Website", "ContactName", "ContactTitle",
            "Email", "Phone", "ProductsOrServices", "PricingModel",
            "KeyDates", "ContractLength", "PaymentTerms", "Notes"
        ]

    @classmethod
    def _select_relevant_chunks(cls, chunks: List[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
        """Select most relevant chunks for extraction."""
        # Build comprehensive keyword list
        all_keywords = cls.EXTRACTION_KEYWORDS.copy()
        for key in keys:
            all_keywords.extend([
                key, key.lower(), key.replace("_", " "), key.replace("_", "")
            ])
        
        # Create keyword pattern
        keyword_pattern = re.compile(r'\b(' + '|'.join(re.escape(kw) for kw in set(all_keywords)) + r')\b', re.I)
        
        selected = []
        
        # Always include first 5 chunks (basic info)
        selected.extend(chunks[:5])
        
        # Add keyword-rich chunks
        keyword_chunks = [chunk for chunk in chunks if keyword_pattern.search(chunk["text"])]
        keyword_chunks.sort(key=lambda c: len(keyword_pattern.findall(c["text"])), reverse=True)
        
        for chunk in keyword_chunks[:10]:
            if chunk not in selected:
                selected.append(chunk)
        
        # Sample from different document sections
        total_chunks = len(chunks)
        if total_chunks > 20:
            sample_indices = [
                total_chunks // 4, total_chunks // 2, 
                3 * total_chunks // 4, total_chunks - 5
            ]
            for idx in sample_indices:
                if 0 <= idx < total_chunks:
                    chunk = chunks[idx]
                    if chunk not in selected:
                        selected.append(chunk)
        
        return selected[:20]

    def extract_structured_data(self, state: Dict[str, Any], schema_text: str) -> str:
        """Extract structured data from the indexed document."""
        if not state or "chunks" not in state:
            return json.dumps({"error": "Please index a PDF first."}, ensure_ascii=False)
        
        chunks = state["chunks"]
        keys = self._parse_extraction_schema(schema_text)
        selected_chunks = self._select_relevant_chunks(chunks, keys)
        
        # Build context
        context_parts = []
        for i, chunk in enumerate(selected_chunks, 1):
            context_parts.extend([
                f"[Chunk {i} | page {chunk['page_num']}]",
                chunk["text"].strip()
            ])
        context = "\n".join(context_parts)
        
        # Extract using LLM
        system_prompt = (
            "You are a precise data extraction assistant. Extract the requested fields as JSON from the provided document context. "
            "Rules:\n"
            "1. Use ONLY information explicitly present in the context\n"
            "2. If a field is not found, set its value to an empty string\n"
            "3. Do not invent or assume any values\n"
            "4. For numbers, extract the exact digits you see\n"
            "5. Search through ALL chunks carefully"
        )
        
        user_prompt = (
            f"Document context (multiple chunks from different pages):\n{context}\n\n"
            f"Extract these fields as JSON:\n{json.dumps(keys, ensure_ascii=False)}\n\n"
            "Pay attention to Polish terms like:\n"
            "- REGON (registration number)\n"
            "- NIP (tax ID)\n"
            "- adres (address)\n"
            "- nazwa (company name)\n"
            "- telefon (phone)\n"
            "- strona internetowa (website)\n"
            "- liczba kondygnacji (number of floors)\n"
            "- liczba lokali w budynku (number of units in building)"
        )
        
        response = self.client.chat.completions.create(
            model=Config.CHAT_MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return response.choices[0].message.content.strip()