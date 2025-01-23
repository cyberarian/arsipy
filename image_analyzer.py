import streamlit as st
import google.generativeai as genai
from groq import Groq
from huggingface_hub import InferenceClient
from PIL import Image
import numpy as np
import cv2
import pytesseract
import os
import io
import base64
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure Tesseract path
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # windows
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract' # linux

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        # Initialize API clients
        self.groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp")
        self.huggingface_client = InferenceClient()

    def enhance_image(self, image):
        """Advanced image preprocessing pipeline"""
        img = np.array(image)
        
        # Resize if needed
        max_dimension = 2000
        height, width = img.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            img = cv2.resize(img, None, fx=scale, fy=scale)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Deskew
        coords = np.column_stack(np.where(binary > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        (h, w) = binary.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(binary, M, (w, h), 
                                flags=cv2.INTER_CUBIC, 
                                borderMode=cv2.BORDER_REPLICATE)
        return rotated

class TextProcessor:
    @staticmethod
    def format_text(text):
        """Format and structure extracted text"""
        if not text:
            return ""
        
        # Split into lines and clean
        lines = text.splitlines()
        lines = [line.strip() for line in lines if line.strip()]
        
        # Detect paragraphs
        formatted_lines = []
        current_paragraph = []
        
        for line in lines:
            if not line[-1] in '.!?':
                current_paragraph.append(line)
            else:
                current_paragraph.append(line)
                formatted_lines.append(' '.join(current_paragraph))
                current_paragraph = []
        
        if current_paragraph:
            formatted_lines.append(' '.join(current_paragraph))
        
        return '\n\n'.join(formatted_lines)

    @staticmethod
    def calculate_metrics(text):
        """Calculate text metrics"""
        words = text.split()
        lines = text.splitlines()
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        return {
            "word_count": len(words),
            "line_count": len(lines),
            "paragraph_count": len(paragraphs),
            "char_count": len(text)
        }

class ImageAnalyzer:
    def __init__(self):
        self.model_manager = ModelManager()
        self.text_processor = TextProcessor()

    def analyze_hybrid(self, image_file):
        """Enhanced analysis using OCR + Gemini"""
        try:
            # Initial OCR
            image = Image.open(image_file)
            enhanced_image = self.model_manager.enhance_image(image)
            
            # OCR with custom config
            custom_config = r'--oem 3 --psm 3 -l eng'
            ocr_text = pytesseract.image_to_string(enhanced_image, config=custom_config)
            
            # Get OCR confidence
            ocr_data = pytesseract.image_to_data(
                enhanced_image,
                output_type=pytesseract.Output.DICT,
                config=custom_config
            )
            confidences = [int(conf) for conf in ocr_data['conf'] if conf != '-1']
            ocr_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Enhance with Gemini
            prompt = f"""
            Enhance and format this OCR text. Fix any errors and maintain structure:
            {ocr_text}
            
            Format guidelines:
            1. Preserve paragraphs
            2. Fix obvious OCR errors
            3. Maintain any lists or sections
            4. Keep original formatting
            5. Ensure proper punctuation
            
            Return enhanced text only.
            """
            
            gemini_response = self.model_manager.gemini_model.generate_content([
                prompt,
                image
            ])
            
            enhanced_text = self.text_processor.format_text(gemini_response.text)
            metrics = self.text_processor.calculate_metrics(enhanced_text)
            
            return {
                "text": enhanced_text,
                "original_ocr": ocr_text,
                "confidence": max(ocr_confidence / 100.0, 0.85),
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Hybrid analysis error: {str(e)}")
            return {
                "text": "",
                "original_ocr": "",
                "confidence": 0.0,
                "metrics": {
                    "word_count": 0,
                    "line_count": 0,
                    "paragraph_count": 0,
                    "char_count": 0
                }
            }

def image_analyzer_main():
       
    st.markdown("""
        ### Memperkenalkan Sistem Ekstraksi Teks Canggih Kami

        Sistem canggih ini memanfaatkan kombinasi yang kuat antara pytesseract, pustaka Python yang menyediakan antarmuka ke mesin OCR Tesseract, dan Multimodal AI dari Google Gemini 2.0 Flash, sinergi ini memungkinkan sistem kami memberikan akurasi dan presisi ekstraksi teks yang optimal.

        #### Memulai dengan Unggah Gambar

        Untuk memulai proses analisis, cukup unggah gambar yang berisi teks yang ingin Anda ekstrak. Sistem kami kemudian akan memanfaatkan kemampuan canggih dari pytesseract dan Google Gemini:

        - Mengenali teks dengan tepat: Mengidentifikasi dan mengekstrak teks secara akurat dari gambar yang diunggah, termasuk jenis huruf, tata letak, dan bahasa.

        - Meningkatkan kualitas teks: Menerapkan penyempurnaan berbasis AI untuk menyempurnakan teks yang diekstrak, mengoreksi kesalahan, dan meningkatkan keterbacaan secara keseluruhan.

        - Menghasilkan output berkualitas tinggi: Memberikan Anda output teks yang bersih, terformat, dan mudah dibaca, siap untuk pemrosesan atau analisis lebih lanjut.

        Unggah Gambar Anda Sekarang
        
        Klik tombol unggah untuk memulai proses analisis. Sistem kami akan mengekstrak teks dengan cepat dan efisien dari gambar Anda, memanfaatkan kekuatan gabungan pytesseract dan Google Gemini untuk memberikan hasil yang optimal.
              
    """)
    
    uploaded_file = st.file_uploader(
        "Upload document",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file:
        analyzer = ImageAnalyzer()
        
        with st.spinner('Analyzing document...'):
            result = analyzer.analyze_hybrid(uploaded_file)
            
            if result["text"]:
                # Display metrics
                st.subheader("Document Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Confidence Score", f"{result['confidence']:.2%}")
                    st.metric("Word Count", result['metrics']['word_count'])
                
                with col2:
                    st.metric("Paragraphs", result['metrics']['paragraph_count'])
                    st.metric("Characters", result['metrics']['char_count'])
                
                # Display enhanced text
                st.subheader("Enhanced Text")
                st.write(result["text"])
                
                # Show original OCR
                with st.expander("View Original OCR Text"):
                    st.text(result["original_ocr"])
            else:
                st.error("No text could be extracted from the image")

if __name__ == "__main__":
    image_analyzer_main()