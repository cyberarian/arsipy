import os
import tempfile
import logging
from typing import List, Optional
from datetime import datetime
import pytesseract
import fitz  # PyMuPDF
from PIL import Image
import io
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import traceback
import numpy as np
import cv2
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class UnifiedDocumentProcessor:
    """Handle both vector storage and CRUD operations"""
    
    def __init__(self, vectorstore, gemini_api_key: str):
        self.vectorstore = vectorstore
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        logger.setLevel(logging.DEBUG)  # Set to DEBUG for more verbose logging
        self.min_text_threshold = 10  # Minimum characters to consider a page as text-based
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel("gemini-pro-vision")

    def extract_text(self, file) -> str:
        """Extract text with OCR fallback for scanned PDFs"""
        try:
            logger.debug(f"Starting text extraction for {file.name} ({file.type})")
            # Store original file position
            original_position = file.tell()
            file_content = file.read()
            file.seek(original_position)  # Reset file position
            
            if file.type == 'application/pdf':
                logger.debug("Processing PDF file")
                pdf = fitz.open(stream=file_content, filetype="pdf")
                text = ""
                try:
                    for page_num, page in enumerate(pdf):
                        # Try normal text extraction first
                        page_text = page.get_text()
                        
                        # If minimal text found, assume it's a scanned page
                        if len(page_text.strip()) < self.min_text_threshold:
                            logger.info(f"Page {page_num + 1} appears to be scanned, using OCR")
                            # Convert PDF page to image
                            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            
                            # Enhance image for OCR
                            img = self._enhance_image_for_ocr(img)
                            
                            # Extract text using OCR
                            page_text = pytesseract.image_to_string(img, lang='eng+ind')
                            
                        text += page_text + "\n"
                        logger.debug(f"Page {page_num + 1}: Extracted {len(page_text)} characters")
                finally:
                    pdf.close()
                
            elif file.type.startswith('image/'):
                logger.debug("Processing image file")
                image = Image.open(io.BytesIO(file_content))
                text = pytesseract.image_to_string(image, lang='eng+ind')
                
            elif file.type == 'text/plain':
                logger.debug("Processing text file")
                text = file_content.decode('utf-8')
            
            else:
                raise ValueError(f"Unsupported file type: {file.type}")
            
            # Validate extracted text
            if not text.strip():
                logger.warning(f"Empty text extracted from {file.name}")
                return ""
            
            logger.info(f"Successfully extracted {len(text)} characters from {file.name}")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from {file.name}: {str(e)}")
            logger.error(traceback.format_exc())
            return ""

    def _enhance_image_for_ocr(self, image):
        """Enhance image quality for better OCR results"""
        try:
            # Convert PIL Image to OpenCV format
            img = np.array(image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Noise reduction
            denoised = cv2.fastNlMeansDenoising(binary)
            
            # Increase contrast
            enhanced = cv2.convertScaleAbs(denoised, alpha=1.5, beta=0)
            
            # Convert back to PIL Image
            return Image.fromarray(enhanced)
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {str(e)}, using original image")
            return image

    def process_document(self, file) -> dict:
        """Process document with debugging output"""
        try:
            logger.info(f"Starting to process {file.name}")
            
            # Extract text
            text = self.extract_text(file)
            if not text:
                return {'success': False, 'error': 'No text could be extracted'}

            # Create chunks
            chunks = self.text_splitter.split_text(text)
            if not chunks:
                return {'success': False, 'error': 'Text splitting produced no chunks'}

            # Extract metadata
            lines = text.split('\n')
            metadata = {
                'title': next((line for line in lines if line.strip()), file.name),
                'file_title': file.name,
                'description': ' '.join(lines[1:4]) if len(lines) > 1 else '',
                'source': file.name,
                'file_type': file.type,
                'processed_at': datetime.now().isoformat(),
                'total_chars': len(text),
                'chunk_count': len(chunks)
            }

            # Create documents for vectorstore
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = metadata.copy()
                doc_metadata['chunk_index'] = i
                documents.append(Document(
                    page_content=chunk,
                    metadata=doc_metadata
                ))

            # Add to vectorstore
            self.vectorstore.add_documents(documents)
            
            return {
                'success': True,
                'metadata': metadata,
                'document_count': len(documents),
                'total_chars': len(text)
            }

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}

    def process_multiple(self, files: List) -> List[dict]:
        """Process multiple documents"""
        results = []
        for file in files:
            result = self.process_document(file)
            results.append({
                'filename': file.name,
                'success': result['success'],
                'error': result.get('error', None)
            })
        return results

    def process_document(self, file_content: bytes, file_type: str) -> dict:
        """
        Process document and extract text using appropriate method
        """
        try:
            # First try direct Gemini processing
            result = self._try_gemini_direct(file_content, file_type)
            
            # If Gemini couldn't extract text, fall back to OCR
            if not result["text"].strip():
                result = self._process_with_ocr(file_content, file_type)
            
            return result
            
        except Exception as e:
            logger.error(f"Document processing error: {str(e)}")
            return {
                "text": "",
                "confidence": 0.0,
                "method": "failed",
                "error": str(e)
            }

    def _try_gemini_direct(self, file_content: bytes, file_type: str) -> dict:
        """Attempt direct text extraction using Gemini"""
        try:
            if file_type == "application/pdf":
                # Convert first page of PDF to image
                pdf = fitz.open(stream=file_content, filetype="pdf")
                first_page = pdf[0]
                pix = first_page.get_pixmap()
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
            else:
                # For images, use directly
                image = Image.open(io.BytesIO(file_content))
            
            # Try Gemini vision analysis
            prompt = """
            Please extract and analyze the text from this document.
            1. Extract all visible text
            2. Preserve the document structure and formatting
            3. Identify key sections and headers
            4. Note any tables or lists
            
            Format the output as clean, structured text.
            """
            
            response = self.gemini_model.generate_content([prompt, image])
            
            return {
                "text": response.text,
                "confidence": 0.95,  # Gemini direct processing typically has high confidence
                "method": "gemini_direct"
            }
            
        except Exception as e:
            logger.warning(f"Gemini direct processing failed: {str(e)}")
            return {"text": "", "confidence": 0.0, "method": "gemini_failed"}

    def _process_with_ocr(self, file_content: bytes, file_type: str) -> dict:
        """Process document using OCR and enhance with Gemini"""
        try:
            # Convert document to image if PDF
            if file_type == "application/pdf":
                pdf = fitz.open(stream=file_content, filetype="pdf")
                first_page = pdf[0]
                pix = first_page.get_pixmap()
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
            else:
                image = Image.open(io.BytesIO(file_content))
            
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Perform OCR with multiple PSM modes
            ocr_text, confidence = self._perform_ocr(processed_image)
            
            # Enhance OCR results with Gemini
            enhanced_text = self._enhance_with_gemini(image, ocr_text, confidence)
            
            return {
                "text": enhanced_text,
                "confidence": confidence,
                "method": "ocr_gemini",
                "original_ocr": ocr_text
            }
            
        except Exception as e:
            logger.error(f"OCR processing error: {str(e)}")
            return {
                "text": "",
                "confidence": 0.0,
                "method": "ocr_failed",
                "error": str(e)
            }

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for better OCR results"""
        img = np.array(image)
        
        # Convert to grayscale if needed
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        img = cv2.fastNlMeansDenoising(img)
        
        return img

    def _perform_ocr(self, image: np.ndarray) -> tuple[str, float]:
        """Perform OCR with multiple PSM modes"""
        best_text = ""
        best_confidence = 0.0
        
        for psm in [3, 4, 6]:  # Try different page segmentation modes
            config = f'--oem 3 --psm {psm}'
            try:
                text = pytesseract.image_to_string(image, config=config)
                data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
                
                # Calculate confidence
                confidences = [int(conf) for conf in data['conf'] if conf != '-1']
                if confidences:
                    confidence = sum(confidences) / len(confidences)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_text = text
                        
            except Exception as e:
                logger.warning(f"OCR error with PSM {psm}: {str(e)}")
                
        return best_text, best_confidence / 100.0

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _enhance_with_gemini(self, image: Image.Image, ocr_text: str, confidence: float) -> str:
        """Enhance OCR results using Gemini"""
        prompt = f"""
        Analyze and enhance this OCR output. The original text was extracted with {confidence:.2%} confidence.

        Original OCR Text:
        {ocr_text}

        Please:
        1. Correct any obvious OCR errors
        2. Properly format paragraphs and sections
        3. Identify and structure:
           - Headers and titles
           - Lists and bullet points
           - Tables (if any)
           - Key information fields
        4. Mark uncertain corrections with [?]
        
        Return the enhanced text while preserving the original structure.
        """
        
        try:
            response = self.gemini_model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            logger.error(f"Gemini enhancement failed: {str(e)}")
            return ocr_text

# Add alias for backward compatibility
DocumentProcessor = UnifiedDocumentProcessor