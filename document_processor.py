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
import json

logger = logging.getLogger(__name__)

class UnifiedDocumentProcessor:
    """Handle both vector storage and CRUD operations"""
    
    def __init__(self, vectorstore):
        """Initialize document processor
        
        Args:
            vectorstore: Vector store instance for document storage
        """
        self.vectorstore = vectorstore
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        logger.setLevel(logging.DEBUG)
        self.min_text_threshold = 10  # Minimum characters to consider a page as text-based
        self.module_titles = self._load_module_titles()

    def _load_module_titles(self) -> dict:
        """Load module titles from reference file"""
        try:
            ref_path = os.path.join(os.path.dirname(__file__), 'references', 'module_titles.json')
            with open(ref_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load module titles: {str(e)}")
            return {}

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

    def process_document(self, file, metadata=None) -> dict:
        """
        Process document and create Document objects
        
        Args:
            file: File object to process
            metadata: Optional dict containing document metadata
            
        Returns:
            dict: Processing results
        """
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

            # Get title from module_titles, metadata, or filename
            source_title = file.name
            if file.name in self.module_titles:
                ref_data = self.module_titles[file.name]
                source_title = f"{ref_data['judul']}, oleh {ref_data['pengajar']}"
            elif metadata and isinstance(metadata, dict):
                if metadata.get('judul') and metadata.get('pengajar'):
                    source_title = f"{metadata['judul']}, oleh {metadata['pengajar']}"

            # Create documents with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'source': str(source_title),
                        'title': str(source_title),
                        'file_type': str(file.type),
                        'chunk_index': str(i),
                        'original_filename': str(file.name)
                    }
                )
                documents.append(doc)

            # Add to vectorstore
            try:
                # Add to vectorstore
                self.vectorstore.add_documents(documents)
            except Exception as e:
                # Catch potential embedding errors (like 429 quota issues)
                error_str = str(e).lower()
                if "429" in error_str and "quota" in error_str:
                    error_msg = "API Quota Exceeded during document processing. Please enable billing on your Google Cloud project to continue."
                    logger.error(f"{error_msg} - Details: {e}")
                    raise Exception(error_msg) from e # Re-raise with a user-friendly message
                raise e
            
            return {
                'success': True,
                'metadata': {
                    'source': source_title,
                    'title': source_title,
                    'document_count': len(documents)
                },
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