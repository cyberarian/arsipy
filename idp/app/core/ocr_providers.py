import streamlit as st
from mistralai import Mistral
import google.generativeai as genai
from ..utils.file_utils import prepare_file_for_mistral
from ..utils.pdf_utils import render_pdf_pages

class OCRProvider:
    @staticmethod
    def get_provider(name):
        providers = {
            "Mistral": MistralOCR,
            "Google": GoogleOCR,
            "Tesseract": TesseractOCR,
            "PyMuPDF": PyMuPDFOCR,
            "PyPDF2": PyPDF2OCR
        }
        return providers.get(name)

class MistralOCR:
    def __init__(self, api_key):
        self.client = Mistral(api_key=api_key)
    
    def process(self, file_bytes, file_name):
        # ... Mistral specific processing code ...

# Similar classes for other providers