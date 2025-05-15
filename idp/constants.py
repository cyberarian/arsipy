# d:\pandasai\idp\constants.py
import os

# --- Core Directories ---
PROCESSED_DIR = "processed_docs"
ASSETS_DIR = "images_dir"

# --- OCR Quality Metrics Thresholds ---
# Example thresholds (adjust based on testing)
OCR_METRICS = {
    "text_quality": {
        "good": 0.85, # Score >= 0.85 is considered good
        "medium": 0.60 # Score >= 0.60 is considered medium
        # Scores below medium are considered low
    },
    # Add other metric definitions here if needed
}

# --- OCR Provider Performance Characteristics (from ref) ---
OCR_PERFORMANCE_METRICS = {
    "providers": {
        "Mistral": {
            "ideal_for": ["Complex documents", "Tables", "Mixed layouts"],
            "strengths": ["Structure preservation", "Layout understanding", "Image extraction (via specific API)"], # Adjusted strength note
            "base_conf": 0.85 # Example confidence
        },
        "Google": {
            "ideal_for": ["Images", "Handwriting", "Screenshots", "Multi-language"],
            "strengths": ["Visual understanding", "Multiple languages", "Context awareness"],
            "base_conf": 0.80 # Example confidence
        },
        "Tesseract": {
            "ideal_for": ["Simple documents", "Clear text", "Basic layouts", "Offline"],
            "strengths": ["Speed", "Offline processing", "Language support"],
            "base_conf": 0.60 # Example confidence
        },
        "Groq": {
            "ideal_for": ["High-quality OCR", "Fast API response", "Technical documents"],
            "strengths": ["High accuracy (model dependent)", "Fast processing", "Technical text"],
            "base_conf": 0.88 # Example confidence
        },
        # --- Direct Text Extractors (Not strictly OCR, but related) ---
        "PyMuPDF": {
            "ideal_for": ["Clean PDFs", "Digital documents", "Fast text extraction"],
            "strengths": ["Very Fast", "Layout preservation (text only)", "PDF handling"],
            "base_conf": 0.95 # Confidence in extracting existing text
        },
        "PyPDF2": {
            "ideal_for": ["Basic PDFs", "Simple text extraction"],
            "strengths": ["Simple processing", "Memory efficient", "Basic extraction"],
            "base_conf": 0.70 # Confidence in extracting existing text
        }
    },
    "weights": { # Example weights for a combined quality score (if implemented)
        "confidence": 0.4,
        "structure": 0.3,
        "format": 0.3
    }
}

# --- Manifest File ---
# Make sure PROCESSED_DIR is defined *before* being used here
PROCESSED_DOCS_MANIFEST_PATH = os.path.join(PROCESSED_DIR, "processed_manifest.json")

# --- Other Constants ---
# Example: Default chunk size if not specified elsewhere
# DEFAULT_CHUNK_SIZE = 500
# DEFAULT_CHUNK_OVERLAP = 50

# IMPORTANT: Avoid importing from other local modules like utils.py, tabs/*, etc. here
# to prevent circular dependencies. Standard library imports (like os) are fine.
