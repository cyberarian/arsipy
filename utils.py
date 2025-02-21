import easyocr
import pytesseract
import cv2
import numpy as np
from PIL import Image
from textblob import TextBlob

def process_image(image_path):
    """Process image using both EasyOCR and Tesseract"""
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])
    
    # Read image using both methods
    easy_ocr_result = reader.readtext(image_path)
    
    # Process with Tesseract
    image = Image.open(image_path)
    tesseract_text = pytesseract.image_to_string(image)
    
    return {
        'easyocr': ' '.join([text[1] for text in easy_ocr_result]),
        'tesseract': tesseract_text
    }

def preprocess_text(text):
    """Clean and preprocess extracted text"""
    blob = TextBlob(text)
    # Basic spelling correction
    corrected = str(blob.correct())
    # Remove extra whitespace
    cleaned = ' '.join(corrected.split())
    return cleaned
