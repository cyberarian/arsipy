# d:\pandasai\arsipy\document_processor.py
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
import json # Ensure json is imported

logger = logging.getLogger(__name__)

# ==============================================================================
# TESSERACT PATH CONFIGURATION (Keep as is)
# ==============================================================================
try:
    tesseract_executable_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_executable_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_executable_path
        logger.info(f"Pytesseract command path set to: {tesseract_executable_path}")
    else:
        logger.error(f"Tesseract executable NOT FOUND at specified path: {tesseract_executable_path}")
        logger.error("OCR functionality will likely fail. Please verify the path.")
except Exception as config_err:
    logger.error(f"Error configuring Pytesseract path: {config_err}")
# ==============================================================================

class UnifiedDocumentProcessor:
    """Handle both vector storage and CRUD operations"""

    def __init__(self, vectorstore):
        """Initialize document processor"""
        self.vectorstore = vectorstore # This should be the Langchain Chroma object
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        logger.setLevel(logging.DEBUG)
        self.min_text_threshold = 10
        # Define path here for reuse
        self.module_titles_path = os.path.join(os.path.dirname(__file__), 'references', 'module_titles.json')
        self.module_titles = self._load_module_titles()

    def _load_module_titles(self) -> dict:
        """Load module titles from reference file"""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.module_titles_path), exist_ok=True)
            if os.path.exists(self.module_titles_path):
                with open(self.module_titles_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Module titles file not found at {self.module_titles_path}, creating an empty one.")
                # Create an empty file if it doesn't exist
                with open(self.module_titles_path, 'w', encoding='utf-8') as f:
                    json.dump({}, f)
                return {}
        except Exception as e:
            logger.warning(f"Could not load or create module titles file: {str(e)}")
            return {}

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ NEW METHOD: Update and Save Module Titles ++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _update_and_save_module_title(self, filename: str, title: str, author: str):
        """Updates the module titles dictionary and saves it to the JSON file."""
        if not filename or not title or not author:
            logger.warning("Skipping module title update due to missing filename, title, or author.")
            return

        try:
            # Load the latest data just before writing to minimize race conditions (though unlikely here)
            current_titles = self._load_module_titles() # Reload to get latest

            # Update the entry
            current_titles[filename] = {
                "judul": title,
                "pengajar": author
            }
            self.module_titles = current_titles # Update the in-memory version too

            # Save back to the file
            with open(self.module_titles_path, 'w', encoding='utf-8') as f:
                json.dump(current_titles, f, ensure_ascii=False, indent=4)
            logger.info(f"Updated and saved module title for: {filename}")

        except Exception as e:
            logger.error(f"Failed to update or save module titles file at {self.module_titles_path}: {str(e)}")
            logger.error(traceback.format_exc())
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ END OF NEW METHOD ++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def extract_text(self, file) -> str:
        """Extract text with OCR fallback for scanned PDFs"""
        # (Keep existing implementation - No changes needed here)
        try:
            logger.debug(f"Starting text extraction for {file.name} ({file.type})")
            original_position = file.tell()
            file_content = file.read()
            file.seek(original_position)

            if file.type == 'application/pdf':
                logger.debug("Processing PDF file")
                pdf = fitz.open(stream=file_content, filetype="pdf")
                text = ""
                try:
                    for page_num, page in enumerate(pdf):
                        page_text = page.get_text()
                        if len(page_text.strip()) < self.min_text_threshold:
                            logger.info(f"Page {page_num + 1} appears to be scanned, using OCR")
                            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            img = self._enhance_image_for_ocr(img)
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

            if not text.strip():
                logger.warning(f"Empty text extracted from {file.name}")
                return ""
            logger.info(f"Successfully extracted {len(text)} characters from {file.name}")
            return text
        except pytesseract.TesseractNotFoundError:
             logger.error("Pytesseract TesseractNotFoundError: Tesseract executable not found or path is incorrect.")
             logger.error(f"Current pytesseract.tesseract_cmd: {pytesseract.pytesseract.tesseract_cmd}")
             logger.error(traceback.format_exc())
             return ""
        except Exception as e:
            logger.error(f"Error extracting text from {file.name}: {str(e)}")
            logger.error(traceback.format_exc())
            return ""

    def _enhance_image_for_ocr(self, image):
        """Enhance image quality for better OCR results"""
        # (Keep existing implementation - No changes needed here)
        try:
            img = np.array(image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            denoised = cv2.fastNlMeansDenoising(binary)
            enhanced = cv2.convertScaleAbs(denoised, alpha=1.5, beta=0)
            return Image.fromarray(enhanced)
        except Exception as e:
            logger.warning(f"Image enhancement failed: {str(e)}, using original image")
            return image

    def process_document(self, file, metadata=None) -> dict:
        """
        Process document, create Document objects, add to vectorstore,
        and update module titles JSON if metadata is provided.

        Args:
            file: File object to process
            metadata: Optional dict containing document metadata (expects 'judul', 'pengajar')

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

            # --- Determine Source Title ---
            # Prioritize provided metadata, then module_titles.json, then filename
            source_title = file.name # Default
            provided_title = None
            provided_author = None

            if metadata and isinstance(metadata, dict):
                provided_title = metadata.get('judul')
                provided_author = metadata.get('pengajar')
                if provided_title and provided_author:
                    source_title = f"{provided_title}, oleh {provided_author}"
                    logger.info(f"Using provided metadata for source title: '{source_title}'")
                else:
                    logger.debug("Provided metadata missing 'judul' or 'pengajar', checking module titles.")
                    # Fall through to check module_titles

            # If not set by provided metadata, check the loaded module titles
            if source_title == file.name and file.name in self.module_titles:
                 ref_data = self.module_titles[file.name]
                 source_title = f"{ref_data['judul']}, oleh {ref_data['pengajar']}"
                 logger.info(f"Using module_titles.json for source title: '{source_title}'")

            # --- Create documents with metadata ---
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'source': str(source_title), # Use the determined title
                        'title': str(source_title),  # Use the determined title
                        'file_type': str(file.type),
                        'chunk_index': str(i),
                        'original_filename': str(file.name) # Keep original filename separate
                    }
                )
                documents.append(doc)

            # --- Add to vectorstore ---
            self.vectorstore.add_documents(documents)
            logger.info(f"Added {len(documents)} chunks for '{file.name}' to vectorstore.")

            # --- Update module_titles.json IF metadata was provided ---
            if provided_title and provided_author:
                self._update_and_save_module_title(file.name, provided_title, provided_author)
            # --- End Update ---

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
            logger.error(f"Error processing document {file.name}: {str(e)}")
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}

    def process_multiple(self, files: List) -> List[dict]:
        """Process multiple documents"""
        # (Keep existing implementation - No changes needed here)
        results = []
        for file in files:
            # Note: The replacement logic will be handled in app.py *before* calling process_document
            result = self.process_document(file)
            results.append({
                'filename': file.name,
                'success': result['success'],
                'error': result.get('error', None)
            })
        return results

    def delete_document_by_filename(self, filename: str) -> bool:
        """
        Deletes all document chunks associated with a specific original filename
        from the Chroma vector store.

        Args:
            filename (str): The original filename stored in the metadata.

        Returns:
            bool: True if deletion was attempted (even if no docs found), False on error.
        """
        # (Keep existing implementation - No changes needed here)
        if not self.vectorstore or not hasattr(self.vectorstore, '_collection'):
            logger.error("Cannot delete document: Vectorstore or underlying collection not available.")
            return False

        logger.info(f"Attempting to delete document chunks for filename: {filename}")
        try:
            collection = self.vectorstore._collection
            # Check current count before deleting (optional, for logging)
            # count_before = collection.count(where={"original_filename": filename})
            # logger.debug(f"Chunks found for {filename} before deletion: {count_before}")

            collection.delete(where={"original_filename": filename})

            # Check count after deleting (optional, for logging - may not be accurate immediately depending on Chroma version/setup)
            # count_after = collection.count(where={"original_filename": filename})
            # logger.debug(f"Chunks found for {filename} after deletion attempt: {count_after}")

            logger.info(f"Deletion command executed for filename: {filename}. Verification recommended.")

            # Optional: Persist changes immediately if needed
            # try:
            #     self.vectorstore.persist()
            #     logger.info("Vectorstore changes persisted after deletion.")
            # except Exception as persist_err:
            #     logger.error(f"Error persisting vectorstore after deletion: {persist_err}")

            return True # Indicate deletion command was sent

        except Exception as e:
            logger.error(f"Error deleting document chunks for filename {filename}: {e}")
            logger.error(traceback.format_exc())
            return False
