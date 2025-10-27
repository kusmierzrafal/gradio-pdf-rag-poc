"""
Gradio user interface for the PDF RAG system.
"""

import os
import json
import gradio as gr

from ..core.config import Config


class PDFRagInterface:
    """Gradio interface for PDF RAG system."""
    
    def __init__(self):
        # Import here to avoid circular imports
        from ..services.document_indexer import DocumentIndexer
        from ..services.question_answering import QuestionAnsweringSystem  
        from ..services.data_extraction import DataExtractionSystem
        
        self.document_indexer = DocumentIndexer()
        self.qa_system = QuestionAnsweringSystem()
        self.extraction_system = DataExtractionSystem()
    
    def create_interface(self) -> gr.Blocks:
        """Create and configure the Gradio interface."""
        with gr.Blocks(title="PDF RAG PoC - Vendor Prospect") as demo:
            gr.Markdown("# 📄 PDF RAG PoC — Vendor Prospect\nUpload a PDF, then ask questions or extract a structured JSON. Answers include page-cited source snippets.")

            with gr.Row():
                pdf = gr.File(label="Upload PDF", file_types=[".pdf"])
                chunk_sz = gr.Slider(600, 3000, value=Config.DEFAULT_CHUNK_SIZE, step=100, label="Chunk size (chars)")
                overlap = gr.Slider(0, 600, value=Config.DEFAULT_OVERLAP, step=50, label="Overlap (chars)")
            
            build_btn = gr.Button("🔧 Build Index")
            status = gr.Markdown("")
            state = gr.State(value=None)

            build_btn.click(
                self._on_build_index, 
                inputs=[pdf, chunk_sz, overlap], 
                outputs=[status, state]
            )

            with gr.Tabs():
                with gr.Tab("Ask the PDF"):
                    question = gr.Textbox(
                        label="Your question", 
                        placeholder="e.g., What products does the vendor offer and at what price?"
                    )
                    topk = gr.Slider(1, 10, value=Config.DEFAULT_TOP_K, step=1, label="Top-K chunks")
                    temp = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Creativity (temperature)")
                    
                    ask_btn = gr.Button("💬 Ask")
                    answer = gr.Markdown(label="Answer")
                    
                    ask_btn.click(
                        self.qa_system.answer_question, 
                        inputs=[state, question, topk, temp], 
                        outputs=[answer]
                    )

                with gr.Tab("Extract JSON"):
                    schema_box = gr.Textbox(
                        label="Schema (comma-separated keys OR JSON template)", 
                        placeholder="", 
                        lines=3,
                        value=""
                    )
                    
                    extract_btn = gr.Button("🧩 Extract JSON", variant="primary")
                    
                    with gr.Row():
                        with gr.Column():
                            json_out = gr.JSON(
                                label="📄 Extracted Data", 
                                show_label=True,
                                container=True,
                                height=400
                            )

                    extract_btn.click(
                        self._extract_and_parse, 
                        inputs=[state, schema_box], 
                        outputs=[json_out]
                    )

            gr.Markdown("— Built with OpenAI, FAISS, and PyMuPDF. Set your `OPENAI_API_KEY` before running.")
        
        return demo
    
    def _on_build_index(self, pdf_file, chunk_size: int, overlap: int):
        """Handle PDF indexing."""
        if not pdf_file:
            return gr.update(value="Please upload a PDF first."), None
        
        state, message = self.document_indexer.create_index(pdf_file.name, int(chunk_size), int(overlap))
        
        if state:
            meta = f"**Indexed file:** `{os.path.basename(pdf_file.name)}`<br>**Chunks:** {len(state['chunks'])}"
            return gr.update(value=f"✅ {message}<br>{meta}"), state
        else:
            return gr.update(value=f"❌ {message}"), None
    
    def _extract_and_parse(self, state, schema: str):
        """Handle data extraction and JSON parsing."""
        txt = self.extraction_system.extract_structured_data(state, schema or "")
        try:
            obj = json.loads(txt)
        except Exception:
            # Fallback: show raw text if JSON parsing fails
            obj = {"_raw": txt, "_error": "Failed to parse JSON"}
        return obj