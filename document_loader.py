from typing import Iterator, Union
from pathlib import Path
import io
import streamlit as st
from PIL import Image

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from docling.document_converter import DocumentConverter

class ArsipyDocumentLoader(BaseLoader):
    """Unified document loader supporting both images and PDFs from uploads"""
    
    def __init__(self, uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> None:
        self._file = uploaded_file
        self._converter = DocumentConverter()
        
    def lazy_load(self) -> Iterator[LCDocument]:
        # Convert uploaded file to BytesIO
        bytes_data = self._file.getvalue()
        file_stream = io.BytesIO(bytes_data)
        
        if self._file.type == "application/pdf":
            try:
                # Handle PDFs with Docling
                converted_doc = self._converter.convert_from_bytes(file_stream, "pdf")
                if hasattr(converted_doc, 'document'):
                    doc_content = converted_doc.document
                else:
                    doc_content = converted_doc
                
                text = doc_content.export_to_markdown(strict_text=True) if hasattr(doc_content, 'export_to_markdown') else str(doc_content)
                yield LCDocument(page_content=text)
            except Exception as e:
                raise ValueError(f"Error processing PDF: {str(e)}")
            
        elif self._file.type in ["image/png", "image/jpeg", "image/jpg"]:
            # Handle images using existing ImageAnalyzer
            from image_analyzer import ImageAnalyzer
            analyzer = ImageAnalyzer()
            with Image.open(file_stream) as img:
                result = analyzer.analyze_hybrid(img)
                yield LCDocument(page_content=result["text"])
        else:
            raise ValueError(f"Unsupported file type: {self._file.type}")
