import streamlit as st

class OCRInterface:
    @staticmethod
    def render_provider_selector():
        provider_options = [
            "Mistral", "Google", "Tesseract", "PyMuPDF", "PyPDF2"
        ]
        return st.selectbox(
            "Select OCR Provider",
            options=provider_options,
            help="Choose the OCR provider"
        )
    
    @staticmethod
    def render_results_tabs(results, quality_metrics):
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Text", "Images", "Quality", "Comparison", "Debug"
        ])
        # ... tab content rendering code ...
