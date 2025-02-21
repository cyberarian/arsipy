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
import math
import re
from typing import Dict, List, Tuple

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # windows
#pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract' # linux

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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def gemini_enhance_text(self, image, ocr_text, confidence, layout_info=None):
        """Enhanced Gemini processing with layout preservation"""
        try:
            # Skip layout analysis if layout_info is not provided
            if layout_info:
                # First, analyze the document structure
                analysis_prompt = """
                Analyze this document's layout and structure. Identify:
                1. Document type and format
                2. Section headers and their hierarchy
                3. Paragraph styles and indentation
                4. Special formatting (lists, tables, quotes)
                5. Font variations and emphasis
                6. Any unique formatting elements
                """
                
                layout_analysis = self.gemini_model.generate_content([analysis_prompt, image])
                layout_context = f"""
                Document Analysis:
                {layout_analysis.text}

                Layout Information:
                {layout_info}
                """
            else:
                layout_context = "No layout information available."

            # Enhanced OCR prompt
            enhancement_prompt = f"""
            Task: Enhance OCR text while preserving formatting and structure.

            {layout_context}

            Original OCR Text (Confidence: {confidence:.2%}):
            {ocr_text}

            Instructions:
            1. Maintain exact formatting:
               - Keep all line breaks and spacing
               - Preserve paragraph indentation
               - Match original text alignment
               - Retain special characters and symbols
               
            2. Preserve structural elements:
               - Headers and subheaders
               - List formatting and bullets
               - Table structures
               - Block quotes
               - Footnotes
               
            3. Fix while preserving:
               - Correct OCR errors
               - Fix spacing issues
               - Maintain original hyphenation
               - Keep original line wrapping

            Return the text with original formatting intact.
            """
            
            response = self.gemini_model.generate_content([enhancement_prompt, image])
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini enhancement failed: {str(e)}")
            return ocr_text

    def _extract_layout_info(self, image: np.ndarray) -> Dict:
        """Extract detailed layout information from image"""
        h, w = image.shape[:2]
        layout_info = {
            'paragraphs': [],
            'headers': [],
            'lists': [],
            'tables': [],
            'spacing': {}
        }
        
        # Get document structure using Tesseract
        custom_config = '--psm 3 -c preserve_interword_spaces=1'
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=custom_config)
        
        # Analyze line spacing
        line_heights = []
        current_block = []
        
        for i in range(len(data['text'])):
            if data['text'][i].strip():
                current_block.append({
                    'text': data['text'][i],
                    'left': data['left'][i],
                    'top': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i],
                    'conf': data['conf'][i]
                })
            elif current_block:
                self._analyze_block(current_block, layout_info)
                current_block = []
        
        if current_block:
            self._analyze_block(current_block, layout_info)
        
        return layout_info

    def _analyze_block(self, block: List[Dict], layout_info: Dict) -> None:
        """Analyze text block for formatting patterns"""
        if not block:
            return
            
        # Calculate indentation and alignment
        left_margins = [item['left'] for item in block]
        indentation = min(left_margins)
        
        # Detect if block is a header
        if len(block) == 1 and block[0]['height'] > 20:  # Adjust threshold as needed
            layout_info['headers'].append({
                'text': block[0]['text'],
                'position': block[0]['top'],
                'size': block[0]['height']
            })
            return
            
        # Detect lists
        first_text = block[0]['text']
        if re.match(r'^[\d•\-*]\s', first_text):
            layout_info['lists'].append({
                'marker': first_text[0],
                'indentation': indentation,
                'items': [item['text'] for item in block]
            })
            return
            
        # Detect paragraphs
        layout_info['paragraphs'].append({
            'indentation': indentation,
            'line_count': len(block),
            'avg_line_height': sum(item['height'] for item in block) / len(block)
        })

    def _detect_tables(self, image: np.ndarray) -> List[Dict]:
        """Detect table structures in the image"""
        # Find horizontal and vertical lines
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Detect lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
        
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine lines
        table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)
        
        # Find contours
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tables = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 100 and h > 100:  # Filter small boxes
                tables.append({
                    'x': x, 'y': y,
                    'width': w, 'height': h
                })
        
        return tables

    def gemini_enhance_text(self, image, ocr_text, confidence):
        """Use Gemini to improve OCR results"""
        try:
            prompt = f"""
            Task: Enhance and correct OCR text while preserving formatting and structure.

            Original OCR Text (Confidence: {confidence:.2%}):
            {ocr_text}

            Instructions:
            1. Fix OCR errors while maintaining original meaning
            2. Preserve document structure (paragraphs, lists, tables)
            3. Correct spelling and grammar
            4. Maintain formatting (indentation, spacing)
            5. Keep original technical terms intact
            6. Tag uncertain corrections with [?]

            Please return the enhanced text with original structure preserved.
            """

            response = self.gemini_model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            logger.error(f"Gemini enhancement failed: {str(e)}")
            return ocr_text

    def enhance_image(self, image):
        """Enhanced image preprocessing pipeline with advanced techniques"""
        img = np.array(image)
        
        # Convert to grayscale if image is RGB/RGBA
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Resize if needed
        max_dimension = 2000
        height, width = img.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            img = cv2.resize(img, None, fx=scale, fy=scale)
            height, width = img.shape[:2]
        
        # Multi-scale processing with consistent sizes
        scales = [0.5, 1.0, 1.5]
        processed_images = []
        base_height, base_width = height, width
        
        for scale in scales:
            # Calculate dimensions for this scale
            scaled_height = int(base_height * scale)
            scaled_width = int(base_width * scale)
            
            # Scale image
            scaled = cv2.resize(img, (scaled_width, scaled_height))
            
            # Enhanced contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(scaled)
            
            # Advanced denoising
            denoised = cv2.fastNlMeansDenoising(enhanced)
            
            # Binarization with Otsu's method
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Resize back to original size for combining
            resized = cv2.resize(binary, (width, height))
            processed_images.append(resized)
        
        # Combine results from different scales
        try:
            # Convert to float32 for weighted addition
            img1 = processed_images[0].astype(np.float32)
            img2 = processed_images[1].astype(np.float32)
            img3 = processed_images[2].astype(np.float32)
            
            # Combine images with weights
            temp = cv2.addWeighted(img1, 0.3, img2, 0.4, 0)
            result = cv2.addWeighted(temp, 1.0, img3, 0.3, 0)
            
            # Convert back to uint8
            result = result.astype(np.uint8)
            
            return result
            
        except Exception as e:
            logger.error(f"Error combining processed images: {str(e)}")
            # Fallback to single-scale processing
            return processed_images[1]  # Return the 1.0 scale result

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

    @staticmethod
    def analyze_structure(text):
        """Analyze document structure"""
        structure = {
            "paragraphs": [],
            "tables": [],
            "lists": [],
            "headers": []
        }
        
        lines = text.split('\n')
        current_block = []
        
        for i, line in enumerate(lines):
            # Detect headers
            if line.strip() and (line.isupper() or line.strip().endswith(':')):
                structure["headers"].append(line.strip())
            
            # Detect tables (basic detection)
            if '|' in line and '-|-' in line:
                table_start = i
                table_lines = []
                while i < len(lines) and '|' in lines[i]:
                    table_lines.append(lines[i])
                    i += 1
                structure["tables"].append('\n'.join(table_lines))
            
            # Detect lists
            if line.strip().startswith(('- ', '* ', '• ', '1.', '2.')):
                structure["lists"].append(line.strip())
            
            # Group paragraphs
            if line.strip():
                current_block.append(line)
            elif current_block:
                structure["paragraphs"].append(' '.join(current_block))
                current_block = []
        
        return structure

class ImageAnalyzer:
    def __init__(self):
        self.model_manager = ModelManager()
        self.text_processor = TextProcessor()

    def analyze_hybrid(self, image_file):
        """Enhanced hybrid analysis with layout preservation"""
        try:
            image = Image.open(image_file)
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Get image array for processing
            img_array = np.array(image)
            
            # Extract layout information
            layout_info = self.model_manager._extract_layout_info(img_array)
            
            # Enhance image
            enhanced_image = self.model_manager.enhance_image(image)
            
            # Detect tables
            tables = self.model_manager._detect_tables(img_array)
            layout_info['tables'] = tables
            
            # Multiple OCR passes with layout preservation
            best_result = self._perform_multi_pass_ocr(enhanced_image, layout_info)
            
            # Enhance with Gemini (remove layout_info parameter)
            enhanced_text = self.model_manager.gemini_enhance_text(
                image,
                best_result['text'],
                best_result['confidence']
            )
            
            # Process and format the enhanced text
            final_text = self.text_processor.format_text(enhanced_text)  # Use format_text instead of format_text_with_layout
            
            # Ensure confidence is properly normalized
            normalized_confidence = min(1.0, best_result['confidence'])
            
            return {
                "text": final_text,
                "original_ocr": best_result['text'],
                "confidence": normalized_confidence,  # This will be between 0 and 1
                "metrics": self.text_processor.calculate_metrics(final_text),
                "structure": self.text_processor.analyze_structure(final_text)
            }
            
        except Exception as e:
            logger.error(f"Hybrid analysis error: {str(e)}")
            return {
                "text": "",
                "original_ocr": "",
                "confidence": 0.0,
                "metrics": {"word_count": 0, "line_count": 0, "paragraph_count": 0, "char_count": 0},
                "structure": {}
            }

    def _perform_multi_pass_ocr(self, image: np.ndarray, layout_info: Dict) -> Dict:
        """Perform multiple OCR passes with different configurations"""
        best_result = {"text": "", "confidence": 0.0}
        
        configs = [
            '--psm 3 -c preserve_interword_spaces=1',  # Auto page segmentation
            '--psm 4 -c preserve_interword_spaces=1',  # Column detection
            '--psm 6 -c preserve_interword_spaces=1',  # Uniform block of text
        ]
        
        for config in configs:
            try:
                text = pytesseract.image_to_string(image, config=config)
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=config)
                
                # Calculate confidence
                confidences = [int(conf) for conf in data['conf'] if conf != '-1']
                if confidences:
                    # Normalize confidence to 0-1 range
                    confidence = sum(confidences) / (len(confidences) * 100.0)  # Division by 100 to normalize
                    
                    # Additional quality metrics
                    layout_score = self._evaluate_layout_match(text, layout_info)
                    
                    # Combine scores and ensure final score is between 0 and 1
                    final_score = min(1.0, (confidence * 0.7) + (layout_score * 0.3))
                    
                    if final_score > best_result['confidence']:
                        best_result = {
                            "text": text,
                            "confidence": final_score
                        }
                        
            except Exception as e:
                logger.warning(f"OCR error with config {config}: {str(e)}")
                
        return best_result

    def _evaluate_layout_match(self, text: str, layout_info: Dict) -> float:
        """Evaluate how well the OCR text matches the expected layout"""
        score = 0.0
        total_weights = 0.0
        
        # Check paragraph structure
        if layout_info['paragraphs']:
            weight = 0.4
            total_weights += weight
            para_count = len(re.split(r'\n\s*\n', text))
            score += weight * min(para_count / len(layout_info['paragraphs']), 1.0)
        
        # Check list items
        if layout_info['lists']:
            weight = 0.3
            total_weights += weight
            list_markers = sum(1 for line in text.split('\n') if re.match(r'^[\d•\-*]\s', line))
            score += weight * min(list_markers / sum(len(lst['items']) for lst in layout_info['lists']), 1.0)
        
        # Check headers
        if layout_info['headers']:
            weight = 0.3
            total_weights += weight
            header_matches = sum(1 for header in layout_info['headers'] if header['text'].strip() in text)
            score += weight * min(header_matches / len(layout_info['headers']), 1.0)
        
        return score / total_weights if total_weights > 0 else 0.0

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
                
            else:
                st.error("No text could be extracted from the image")

if __name__ == "__main__":
    image_analyzer_main()