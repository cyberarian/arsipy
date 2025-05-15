# d:\pandasai\idp\ocr_providers.py
import streamlit as st
import io
import os
import sys
import base64 # Needed for image encoding
import time
import re # Import regex for process_ocr_response example
import json # Needed for table formatting in prompts and parsing

# --- Import necessary components from utils.py FIRST ---
from utils import fitz, pdfplumber, Groq as GroqClient, Image as PILImage # Renamed to avoid conflict with groq module

# --- Define extract_tables_from_pdf_bytes EARLIER ---
def extract_tables_from_pdf_bytes(pdf_bytes, filename: str) -> list[list[list[str]]]:
    """
    Extracts tables from a PDF file represented by bytes using pdfplumber.

    Args:
        pdf_bytes: Bytes of the PDF file.
        filename: Original name of the file (for logging).

    Returns:
        A list of tables. Each table is a list of rows, and each row is a list of cell strings.
        Returns an empty list if pdfplumber is not available or an error occurs.
    """
    if not pdfplumber: # pdfplumber is imported from utils
        st.warning("pdfplumber library not available, skipping table extraction.")
        return []

    extracted_tables_data = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            st.info(f"Found {len(pdf.pages)} pages in '{filename}' for table extraction.")
            for i, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                if page_tables:
                    st.info(f"Page {i+1}: Found {len(page_tables)} table(s).")
                    for table_idx, table_content_raw in enumerate(page_tables):
                        cleaned_table = []
                        if table_content_raw: # Ensure table_content_raw is not None
                            for row in table_content_raw:
                                if row: # Ensure row is not None
                                    cleaned_table.append([(str(cell) if cell is not None else "") for cell in row])
                                else:
                                    cleaned_table.append([]) # Append empty list for None row
                        if cleaned_table:
                            extracted_tables_data.append(cleaned_table)
    except Exception as e:
        st.error(f"Error extracting tables from '{filename}' using pdfplumber: {e}")
        return []
    return extracted_tables_data

def package_ocr_result(
    original_filename: str,
    provider: str,
    extracted_text: str | None,
    metrics_data: dict | None, # Renamed from quality_data
    processing_timestamp: float # e.g., from time.time()
) -> dict | None:
    """Packages OCR results, including keywords, into a dictionary suitable for a DataFrame row."""
    if not extracted_text:
        st.warning("Cannot package result: No extracted text provided.")
        return None # Cannot package if no text was extracted

    keywords = extract_keywords(extracted_text)

    record = {
        "filename": original_filename,
        "processed_at": processing_timestamp, # Store as Unix timestamp
        "provider": provider,
        # "quality_score": metrics_data.get("score") if metrics_data else None, # REMOVED quality score
        # "quality_metrics": quality_data.get("metrics") if metrics_data else None, # Often too verbose for main DF view
        "keywords": keywords,
        # Consider adding a unique ID here if needed: 'record_id': str(uuid.uuid4())
    }
    # Optionally add the full text back if desired, or store it separately
    # record["extracted_text"] = extracted_text
    st.info(f"Result packaged for {original_filename} (Keywords: {keywords[:3]}...)")
    return record

# --- Keyword Extraction Library ---
try:
    import yake
except ImportError:
    st.warning("YAKE library not found for keyword extraction. Please install: pip install yake. Falling back to basic method.")
    yake = None # Placeholder if not installed

# Ensure necessary imports are present
try:
    from mistralai import Mistral
except (ImportError, AttributeError): # Added AttributeError for cases like client class name changes
    st.warning("MistralAI library not found or client class incorrect. Please install/check: pip install mistralai")
    Mistral = None # Placeholder

try:
    import google.generativeai as genai # For Google Gemini
except ImportError:
    st.warning("Google Generative AI library not found. Please install it: pip install google-generativeai")
    genai = None # Placeholder

try:
    import pytesseract
    from PIL import Image
    # Check if PILImage from utils is available, prefer that if so
    if PILImage is None and Image is not None: # If utils.PILImage is None but local PIL.Image is fine
        PILImage = Image # Use local PIL.Image as fallback
    elif Image is None and PILImage is not None: # If local PIL.Image failed but utils.PILImage is fine
        Image = PILImage # This case is less likely if utils.py is imported first
except ImportError: # If Pillow itself is not installed
    st.warning("Pytesseract or Pillow library not found. Please install them: pip install pytesseract Pillow")
    pytesseract = None
    # PILImage will be None from utils import if Pillow isn't installed

try:
    import PyPDF2
except ImportError:
    st.warning("PyPDF2 library not found. Please install it: pip install pypdf2")
    PyPDF2 = None # Placeholder

try:
    import markdownify
except ImportError:
    st.warning("Markdownify library not found. Please install: pip install markdownify. HTML to Markdown conversion will fail.")
    markdownify = None

# fitz, pdfplumber, GroqClient, PILImage are already imported from utils at the top

# --- Import evaluation function (assuming ocr_evaluation.py exists) ---
try:
    from ocr_evaluation import evaluate_ocr_quality
except ImportError:
    st.warning("ocr_evaluation.py not found or evaluate_ocr_quality function missing. Quality calculation disabled.")
    evaluate_ocr_quality = None # Placeholder

# --- Gemini Specific Configurations (for mdr_tab.py) ---
GEMINI_SAFETY_SETTINGS_NONE = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

GEMINI_GENERATION_CONFIG_JSON = {
    "temperature": 0.1, # Low temperature for more deterministic output
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json", # Crucial for structured output
}

@st.cache_resource
def get_gemini_client(api_key_name="GEMINI_API_KEY"):
    """Initializes and returns a Google Gemini client."""
    if not genai:
        st.error("Google Generative AI library (google-generativeai) is not available.")
        return None
    api_key = st.secrets.get(api_key_name)
    if not api_key:
        st.error(f"{api_key_name} not found in st.secrets.toml.")
        return None
    genai.configure(api_key=api_key)
    # The model itself will be fetched in the calling function, e.g., genai.GenerativeModel(...)
    return genai # Return the configured genai module

# --- Define OCR Helper Functions WITHIN this file ---

def prepare_file_for_mistral(file_bytes, filename):
    """
    Prepares file data (bytes) into the format required by the Mistral API.
    This is a placeholder - implementation depends on Mistral API specifics.
    It might just return the bytes and filename, or require specific encoding/wrapping.
    Assuming the client's `files.create` method handles bytes directly.
    """
    # Example: Assuming Mistral client's upload method handles bytes directly
    # No special preparation might be needed, just return inputs.
    # If it required base64:
    # encoded_content = base64.b64encode(file_bytes).decode('utf-8')
    # return encoded_content, filename
    # print(f"Debug: Preparing '{filename}' for Mistral.") # Keep if needed for debugging
    return file_bytes, filename # Placeholder: assumes client handles bytes

def render_pdf_pages(pdf_bytes) -> list:
    """Renders PDF pages as PIL Images using PyMuPDF (fitz from utils)."""
    images = []
    if not fitz:
        st.error("PyMuPDF (fitz) is required to render PDF pages.")
        return images
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            # Increase resolution (dpi) for better OCR quality
            pix = page.get_pixmap(dpi=300)
            if pix.alpha: # Handle transparency if present
                 img = PILImage.frombytes("RGBA", [pix.width, pix.height], pix.samples)
            else:
                 img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        pdf_document.close()
    except Exception as e:
        st.error(f"Error rendering PDF pages: {e}")
        # Optionally return partially rendered images if needed
        # return images
    return images

# --- Re-introduce process_ocr_response (adapted from ref) ---
def process_ocr_response(response_data, base_filename):
    """
    Processes the raw response from the Mistral OCR API into plain text.
    Adjust based on the actual Mistral OCR API response structure.
    This version assumes a structure like the reference file implies.
    """
    # print(f"Debug: Processing Mistral OCR response for '{base_filename}'. Raw data: {response_data}") # Debugging
    if isinstance(response_data, dict):
        all_content = []
        # --- Adapt based on ACTUAL response structure from client.ocr.process ---
        # Example: Assuming pages with markdown content
        for page_idx, page in enumerate(response_data.get('pages', [])):
            # Limit pages if needed (e.g., first 5)
            # if page_idx >= 5:
            #     all_content.append("\n\n---\n[Page Break]\n---\n\n*Note: Document truncated.*")
            #     break

            page_content = page.get('markdown', page.get('text')) # Look for markdown or text
            if page_content:
                all_content.append(page_content)
            else:
                all_content.append(f"[Page {page_idx+1}: No text content found in response]")
        # --- End Adaptation ---

        if all_content:
            return "\n\n---\n[Page Break]\n---\n\n".join(all_content)
        else:
            st.warning(f"Could not find expected text/markdown field in Mistral OCR response for {base_filename}.")
            return json.dumps(response_data, indent=2) # Return raw structure as fallback

    elif isinstance(response_data, str):
        return response_data # If response is already string
    else:
        st.warning("Unexpected Mistral OCR response format. Cannot extract text.")
        return None

# --- Helper Function to Convert JSON Table Data to Markdown ---
def json_to_markdown_table(table_data: list[dict]) -> str:
    """Converts a list of dictionaries (parsed JSON) into a Markdown table string."""
    if not table_data or not isinstance(table_data, list) or not isinstance(table_data[0], dict):
        return "[Error: Invalid table data format for Markdown conversion]"

    # Extract headers from the keys of the first dictionary
    headers = list(table_data[0].keys())
    if not headers:
        return "[Error: Table data has no headers]"

    # Create the Markdown header row
    md_header = "| " + " | ".join(headers) + " |"
    # Create the separator row
    md_separator = "| " + " | ".join(["---"] * len(headers)) + " |"

    # Create the data rows
    md_rows = []
    for row_dict in table_data:
        row_values = [str(row_dict.get(header, "")) for header in headers] # Get value or empty string
        md_rows.append("| " + " | ".join(row_values) + " |")

    return "\n".join([md_header, md_separator] + md_rows)

# --- Helper Function to Replace JSON Blocks with Markdown Tables ---
def replace_json_table_blocks(text: str) -> str:
    """Finds ```json [...] ``` blocks and replaces them with Markdown tables."""
    def replacer(match):
        json_string = match.group(1).strip()
        try:
            table_data = json.loads(json_string)
            return json_to_markdown_table(table_data)
        except json.JSONDecodeError as e:
            return f"[Error parsing table JSON: {e}]\n```json\n{json_string}\n```" # Keep original block on error
        except Exception as e:
            return f"[Error converting table data: {e}]\n```json\n{json_string}\n```"

    # Regex to find ```json ... ``` blocks (non-greedy)
    return re.sub(r"```json\s*([\s\S]*?)\s*```", replacer, text)

# --- Use st.secrets for API Keys ---
@st.cache_resource
def get_ocr_client(provider):
    """Gets the client/library instance for the selected OCR provider."""
    try:
        if provider == "Mistral":
            if not Mistral: return None # Check if import failed
            api_key = st.secrets.get("MISTRAL_API_KEY")
            if not api_key:
                st.error("MISTRAL_API_KEY not found in st.secrets.")
                return None
            # Use the correct client class name from your import
            return Mistral(api_key=api_key)
        elif provider == "Google":
            if not genai: return None
            api_key = st.secrets.get("GEMINI_API_KEY")
            if not api_key:
                st.error("GEMINI_API_KEY not found in st.secrets.")
                return None
            genai.configure(api_key=api_key)
            return genai.GenerativeModel('gemini-1.5-flash-latest') # Updated to a common flash model
        elif provider == "Tesseract":
            if not pytesseract or not Image: return None # Check needed libraries
            # Check for Tesseract path in secrets or environment, then default
            tesseract_path_secret = st.secrets.get("TESSERACT_PATH")
            tesseract_path_env = os.environ.get('TESSERACT_PATH')
            default_win_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

            tesseract_cmd = None
            if tesseract_path_secret:
                tesseract_cmd = tesseract_path_secret
            elif tesseract_path_env:
                tesseract_cmd = tesseract_path_env
            elif sys.platform.startswith('win') and os.path.exists(default_win_path):
                 tesseract_cmd = default_win_path

            if tesseract_cmd:
                 try:
                     pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
                     if not os.path.exists(tesseract_cmd):
                         st.warning(f"Tesseract path specified ({tesseract_cmd}) but not found. Trying system PATH.")
                         # Let it try the system path if specified path is invalid
                     else:
                          print(f"Using Tesseract at: {tesseract_cmd}") # Log path being used
                 except Exception as tess_path_e:
                      st.warning(f"Error setting Tesseract path '{tesseract_cmd}': {tess_path_e}. Trying system PATH.")
            elif sys.platform.startswith('win'):
                 st.warning(f"Tesseract executable not found at default {default_win_path} or via TESSERACT_PATH (secrets/env). Ensure Tesseract is installed and in PATH.")
            # For non-Windows, assume it's in PATH if no specific path is given
            return pytesseract # Return the library itself
        elif provider == "Groq": # For Groq Vision OCR
            if not GroqClient: return None # Use the client imported from utils
            api_key = st.secrets.get("GROQ_API_KEY")
            if not api_key:
                st.error("GROQ_API_KEY not found in st.secrets.")
                return None
            return GroqClient(api_key=api_key) # Instantiate the client
        # --- Add initializers for direct text extractors (even if not used by run_ocr) ---
        elif provider == "PyMuPDF":
            return fitz # Return the library module itself
        elif provider == "PyPDF2":
            return PyPDF2 # Return the library module itself
        # --- End direct extractor initializers ---
        return None
    except Exception as e:
        st.error(f"Error initializing client for {provider}: {str(e)}")
        return None

# --- Functions for specific OCR providers ---

def process_mistral(client, file_bytes, file_name, model="mistral-ocr-latest"): # Using dedicated OCR model
    """Process file with Mistral's dedicated OCR endpoint"""
    if not client: return None
    try:
        # Prepare file (may need adjustment based on prepare_file_for_mistral implementation)
        # Assuming prepare_file_for_mistral just returns bytes and name for now
        prepared_bytes, prepared_name = prepare_file_for_mistral(file_bytes, file_name)

        with st.spinner("Uploading file to Mistral..."):
            # Use client.files.upload (ensure method exists in your library version)
            # The exact structure {'file': (name, bytes)} might vary slightly
            uploaded_file = client.files.create(
                file=(prepared_name, prepared_bytes), # Pass as tuple for multipart/form-data
                purpose="ocr" # Specify purpose
            )
            st.info(f"File uploaded to Mistral (ID: {uploaded_file.id})")

        with st.spinner("Getting signed URL from Mistral..."):
            # Use client.files.get_signed_url (ensure method exists)
            signed_url_response = client.files.get_signed_url(file_id=uploaded_file.id)
            document_url = signed_url_response.url
            st.info("Signed URL obtained.")

        with st.spinner(f"Processing OCR with Mistral model '{model}'..."):
            # Use client.ocr.process (ensure method exists)
            ocr_response = client.ocr.process(
                model=model,
                document={
                    "type": "document_url",
                    "document_url": document_url,
                    # Include options as needed, matching ref file:
                    "include_image_base64": False, # Usually False unless needed downstream
                    "layout_info": True,
                    "tables": True
                }
            )
            st.info("Mistral OCR processing complete.")

        # Process the response object
        # Convert response to dict if it's a model object (like Pydantic)
        response_dict = ocr_response.model_dump() if hasattr(ocr_response, 'model_dump') else json.loads(str(ocr_response))

        # Use the helper function to extract text from the response structure
        return process_ocr_response(response_dict, os.path.splitext(file_name)[0])

    except Exception as e:
        st.error(f"Mistral processing error: {str(e)}")
        # st.exception(e) # Uncomment for full traceback during debugging
        return None

def process_google(client, file_bytes, file_name, model):
    if not client: return None
    processed_text = [] # Store results per page
    success_count = 0
    total_pages = 0

    try:
        # --- Updated Prompt ---
        prompt = """Extract all text from this document image/page. Preserve document structure (headings using #, ##, etc., paragraphs, lists using * or -) using standard Markdown. **Identify tables.** When you encounter a table, represent it as a valid JSON string (a list of objects, where each object is a row and keys are column headers). Embed this JSON string within the Markdown output using a JSON code block, like ```json [ { "Header1": "Value1", "Header2": "Value2" }, { ... } ] ```, exactly where the table originally appeared in the document flow. Extract all other text as standard Markdown. Preserve special characters. Output ONLY the resulting Markdown content, including the embedded JSON for tables. Do not add any introductory text, commentary, or explanations outside the Markdown itself."""
        # --- End Updated Prompt ---

        if file_name.lower().endswith('.pdf') and PILImage: # Also check if PILImage is available
            # Use the locally defined helper function
            images = render_pdf_pages(file_bytes) # Render all pages first
            total_pages_rendered = len(images)
            if total_pages_rendered == 0:
                 st.warning("No pages rendered from PDF.")
                 return None # Nothing to process

            # --- Limit processing to first 5 pages ---
            pages_to_process = min(total_pages_rendered, 100) # Limit pages if necessary
            st.write(f"Processing first {pages_to_process} of {total_pages_rendered} pages with Google Gemini...")
            total_pages_for_status = pages_to_process # Use a different variable for status message

            for i in range(pages_to_process): # Loop only up to pages_to_process
                image = images[i] # Get the specific page image
                page_num = i + 1
                try:
                    with st.spinner(f"Processing page {page_num}/{total_pages} with Google Gemini..."):
                        img_bytes_io = io.BytesIO()
                        image.save(img_bytes_io, format='PNG') # Ensure image is saved before getting value
                        img_bytes_val = img_bytes_io.getvalue()

                        # --- Call Google API ---
                        response = client.generate_content([
                            prompt, # Use updated prompt
                            {"mime_type": "image/png", "data": img_bytes_val}
                        ])

                        # Check if response has text (handle potential empty responses or errors)
                        page_text = None
                        if response and hasattr(response, 'text'):
                             page_text = response.text
                        elif response and hasattr(response, 'parts'): # Handle potential multi-part response
                             page_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))

                        if page_text and page_text.strip():
                             processed_text.append(page_text)
                             # --- Post-process to convert JSON tables ---
                             processed_text[-1] = replace_json_table_blocks(processed_text[-1])
                             # --- End Post-processing ---

                             success_count += 1
                        else:
                             # Handle cases where API returns success but no text or empty text
                             st.warning(f"Google processed page {page_num} but returned no significant text.")
                             processed_text.append(f"[Page {page_num}: No text returned by Google]")

                    # --- REMOVED time.sleep(1) ---
                    # Rely on API client or handle rate limits more dynamically if needed

                except Exception as page_e:
                    # --- Log specific error for the failing page ---
                    st.error(f"Error processing page {page_num} with Google: {page_e}")
                    print(f"Detailed Error on page {page_num}: {repr(page_e)}") # Print detailed error to console/log
                    processed_text.append(f"[Page {page_num}: Processing Error - {type(page_e).__name__}]")
                    # Continue to the next page

            # --- Add truncation note if applicable ---
            if total_pages_rendered > pages_to_process:
                processed_text.append(f"\n\n---\n[Note: Processing limited to the first {pages_to_process} pages.]\n---\n")

            # --- End of loop ---

            # Combine results
            final_text = "\n\n---\n[Page Break]\n---\n\n".join(processed_text)

            # Report overall success/failure based on pages processed (use total_pages_for_status)
            if success_count == total_pages_for_status:
                 st.success(f"Successfully processed {success_count}/{total_pages_for_status} pages with Google.")
            elif success_count > 0:
                 st.warning(f"Processed {success_count}/{total_pages} pages successfully with Google. Some pages failed.")
            else:
                 st.error(f"Failed to process any pages ({total_pages} attempted) with Google.")
                 return None # Return None if nothing succeeded

            return final_text

        else: # Single image processing
            try:
                response = client.generate_content([prompt, {"mime_type": "image/png", "data": file_bytes}]) # Use updated prompt
                # Check response structure as above
                img_text = None
                if response and hasattr(response, 'text'):
                    img_text = response.text
                elif response and hasattr(response, 'parts'):
                    img_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))

                if img_text and img_text.strip():
                    # --- Post-process to convert JSON tables ---
                    img_text = replace_json_table_blocks(img_text)
                    # --- End Post-processing ---
                    return img_text
                else:
                    st.warning("Google processed image but returned no significant text.")
                    return None
            except Exception as img_e:
                 st.error(f"Error processing single image with Google: {img_e}")
                 print(f"Detailed Error on image: {repr(img_e)}")
                 return None

    except Exception as e:
        # Catch errors outside the loop (e.g., during render_pdf_pages if not caught there)
        st.error(f"Google processing error (outside page loop): {str(e)}")
        print(f"Detailed Error (outside loop): {repr(e)}")
        return None

def process_tesseract(pytesseract_module, file_bytes, file_name):
    if not pytesseract_module or not PILImage: return None # Use PILImage from utils
    try:
        if file_name.lower().endswith('.pdf'):
            # Use the locally defined helper function
            images = render_pdf_pages(file_bytes)
            all_text = []
            total_pages = len(images)
            if total_pages == 0:
                 st.warning("No pages rendered from PDF for Tesseract.")
                 return None
            st.write(f"Processing {total_pages} pages with Tesseract...")
            for i, image in enumerate(images):
                 with st.spinner(f"Processing page {i+1}/{total_pages} with Tesseract..."):
                    try:
                        # Add config options for better results if needed, e.g., --psm 6
                        text = pytesseract_module.image_to_string(image, lang='eng') # Add more languages if needed: lang='eng+ind'
                        all_text.append(text if text else f"[Page {i+1}: No text found by Tesseract]")
                    except Exception as tess_page_e:
                         st.warning(f"Tesseract error on page {i+1}: {tess_page_e}")
                         all_text.append(f"[Page {i+1}: Tesseract processing error]")
            return "\n\n---\n[Page Break]\n---\n\n".join(all_text)
        else: # Single image
            image = PILImage.open(io.BytesIO(file_bytes)) # Use PILImage from utils
            return pytesseract_module.image_to_string(image, lang='eng') # Use the correct parameter name
    except Exception as e:
        st.error(f"Tesseract processing error: {str(e)}")
        return None

def process_groq(client, file_bytes, file_name, model="meta-llama/llama-4-maverick-17b-128e-instruct"): # Using vision model from reference
    """Process file with Groq Vision for OCR"""
    if not client: return None
    try:
        # --- Updated System Prompt ---
        system_prompt = """You are an expert OCR system. Extract text from the provided image.
        1. Preserve document structure (headings using #, ##, etc., paragraphs, lists using * or -) using standard Markdown.
        2. **Identify tables.** When you encounter a table, represent it as a valid JSON string (a list of objects, where each object is a row and keys are column headers). Embed this JSON string within the Markdown output using a JSON code block, like ```json [ { "Header1": "Value1", "Header2": "Value2" }, { ... } ] ```, exactly where the table originally appeared in the document flow.
        3. Extract all other text normally.
        4. Preserve special characters and symbols.
        5. Output ONLY the resulting Markdown content, including the embedded JSON for tables. Do not add any introductory text, commentary, or explanations outside the Markdown itself."""
        # --- End Updated System Prompt ---

        if file_name.lower().endswith('.pdf'):
            # Use the locally defined helper function, ensure PILImage is available
            images = render_pdf_pages(file_bytes)
            all_text = []
            total_pages = len(images)
            if total_pages == 0:
                 st.warning("No pages rendered from PDF for Groq.")
                 return None
            st.write(f"Processing {total_pages} pages with Groq Vision (Model: {model})...")
            for i, image in enumerate(images):
                page_num = i + 1
                with st.spinner(f"Processing page {page_num}/{total_pages} with Groq Vision..."):
                    img_bytes_io = io.BytesIO()
                    try:
                        image.save(img_bytes_io, format='PNG')
                        img_base64 = base64.b64encode(img_bytes_io.getvalue()).decode()
                    except Exception as img_err:
                        st.warning(f"Could not process page {page_num} for Groq: {img_err}")
                        all_text.append(f"[Page {page_num}: Error converting page to PNG]")
                        continue

                    try:
                        chat_completion = client.chat.completions.create(
                            model=model, # Use the model passed to the function
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": [
                                    {"type": "text", "text": "Extract text from this image:"},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                                ]}
                            ],
                            temperature=0.1, max_tokens=4096
                        )
                        page_text = chat_completion.choices[0].message.content
                        # --- Post-process to convert JSON tables ---
                        if page_text:
                            page_text = replace_json_table_blocks(page_text)
                        # --- End Post-processing ---
                        all_text.append(page_text if page_text else f"[Page {page_num}: No text returned by Groq]")
                    except Exception as groq_api_err:
                         st.warning(f"Groq API error on page {page_num}: {groq_api_err}")
                         all_text.append(f"[Page {page_num}: Groq API Error during processing]")
                    # Add delay if needed for rate limits
                    # time.sleep(0.5)
            return "\n\n---\n[Page Break]\n---\n\n".join(all_text)
        else:  # Single image
            img_bytes_io = io.BytesIO()
            try:
                # Ensure image is in a format Groq accepts (like PNG)
                img = PILImage.open(io.BytesIO(file_bytes)) # Use PILImage from utils
                img.save(img_bytes_io, format='PNG')
                img_base64 = base64.b64encode(img_bytes_io.getvalue()).decode()
            except Exception as img_err:
                 st.error(f"Could not process image for Groq: {img_err}")
                 return None
            try:
                chat_completion = client.chat.completions.create(
                    model=model, # Use the model passed to the function
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": "Extract text from this image:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                        ]}
                    ],
                    temperature=0.1, max_tokens=4096
                )
                img_text = chat_completion.choices[0].message.content
                # --- Post-process to convert JSON tables ---
                if img_text:
                    img_text = replace_json_table_blocks(img_text)
                # --- End Post-processing ---
                return img_text if img_text else None
            except Exception as groq_api_err:
                 st.error(f"Groq API error processing image: {groq_api_err}")
                 return None
    except Exception as e:
        st.error(f"Groq processing error: {str(e)}")
        return None


# --- Functions for Direct Text Extraction (Separate from OCR) ---
# These extract text embedded digitally in PDFs, not via image recognition.

def extract_direct_text_pymupdf(file_bytes, file_name) -> str | None:
    """Extracts embedded text directly from digital PDFs using PyMuPDF."""
    if not fitz or not markdownify: # fitz from utils
        st.error("PyMuPDF (fitz) not available for direct text extraction.")
        return None
    if not file_name.lower().endswith('.pdf'):
        st.warning("Direct text extraction only applicable to PDF files.")
        return None

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        all_text = []
        total_pages = len(doc)
        if total_pages == 0:
            st.warning("PDF has 0 pages.")
            doc.close()
            return None
        st.write(f"Directly extracting text from {total_pages} pages using PyMuPDF...")
        for i in range(total_pages):
            page_num = i + 1
            with st.spinner(f"Extracting text from page {page_num}/{total_pages}..."):
                try:
                    # --- Get HTML output ---
                    html_text = doc[i].get_text("html")
                    if html_text:
                        # --- Convert HTML to Markdown ---
                        md_text = markdownify.markdownify(html_text, heading_style="ATX")
                        all_text.append(md_text)
                    else:
                        all_text.append(f"[Page {page_num}: No embedded text found]")
                except Exception as page_extract_e:
                     st.warning(f"Error extracting text from page {page_num}: {page_extract_e}")
                     all_text.append(f"[Page {page_num}: Error during text extraction]")
        doc.close()
        result_text = "\n\n---\n[Page Break]\n---\n\n".join(all_text)
        # --- Update success message ---
        st.success(f"Successfully extracted embedded text and converted to Markdown ({len(result_text)} characters).")
        return result_text
    except Exception as e:
        st.error(f"PyMuPDF direct text extraction error: {str(e)}")
        return None

def extract_direct_text_pypdf2(file_bytes, file_name) -> str | None:
    """Extracts embedded text directly from digital PDFs using PyPDF2."""
    if not PyPDF2:
        st.error("PyPDF2 not available for direct text extraction.")
        return None
    if not file_name.lower().endswith('.pdf'):
        st.warning("Direct text extraction only applicable to PDF files.")
        return None

    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        all_text = []
        total_pages = len(pdf_reader.pages)
        if total_pages == 0:
            st.warning("PDF has 0 pages.")
            return None
        st.write(f"Directly extracting text from {total_pages} pages using PyPDF2...")
        for i in range(total_pages):
            page_num = i + 1
            with st.spinner(f"Extracting text from page {page_num}/{total_pages}..."):
                try:
                    text = pdf_reader.pages[i].extract_text()
                    all_text.append(text if text else f"[Page {page_num}: No embedded text found]")
                except Exception as page_extract_e:
                     st.warning(f"Error extracting text from page {page_num} with PyPDF2: {page_extract_e}")
                     all_text.append(f"[Page {page_num}: Error during text extraction]")
        result_text = "\n\n---\n[Page Break]\n---\n\n".join(all_text)
        st.success(f"Successfully extracted embedded text with PyPDF2 ({len(result_text)} characters).")
        return result_text
    except Exception as e:
        st.error(f"PyPDF2 direct text extraction error: {str(e)}")
        return None

# --- Main OCR Dispatch Function ---
def run_ocr(file_bytes, file_name, provider) -> str | None:
    """Runs the selected *OCR* provider (image-to-text) and returns the extracted text."""
    if not file_bytes:
        st.error("Empty file provided for OCR")
        return None

    client = get_ocr_client(provider)
    if not client and provider not in ["Tesseract", "PyMuPDF", "PyPDF2"]: # Libs returned directly
        # Error message handled in get_ocr_client or provider check below
        if provider != "Tesseract": # Avoid duplicate error if Tesseract failed path setup
             st.error(f"Failed to initialize client for {provider}.")
        return None
    elif provider == "Tesseract" and not pytesseract: # Explicit check for Tesseract library
         st.error("Tesseract library (pytesseract) not available.")
         return None
    # --- Add checks for direct extractor libraries ---
    elif provider == "PyMuPDF" and not fitz:
         st.error("PyMuPDF library (fitz) not available.")
         return None
    # --- End checks ---


    result = None
    spinner_msg = f"Performing OCR with {provider}..."
    if file_name.lower().endswith('.pdf'):
         spinner_msg = f"Performing OCR on PDF with {provider} (this may take time)..."

    with st.spinner(spinner_msg):
        try:
            if provider == "Mistral":
                # Use the dedicated OCR model name from the reference
                result = process_mistral(client, file_bytes, file_name, "mistral-ocr-latest")
            elif provider == "Google":
                result = process_google(client, file_bytes, file_name, 'gemini-2.5-flash-preview-04-17') # Correct model from ref
            elif provider == "Tesseract":
                # Pass the pytesseract module itself
                result = process_tesseract(pytesseract, file_bytes, file_name)
            elif provider == "Groq":
                # Use the specific Groq vision model from ref
                result = process_groq(client, file_bytes, file_name, "meta-llama/llama-4-maverick-17b-128e-instruct") # Match ref model
            # --- IMPORTANT: Do NOT call direct extractors here ---
            # PyMuPDF/PyPDF2 are handled by extract_direct_text_* functions separately
            else:
                st.error(f"Invalid *OCR* provider selected for run_ocr: {provider}. Use extract_direct_text functions for PyMuPDF/PyPDF2.")
                return None
            # --- End Provider Calls ---

            # --- Result Validation ---
            if result is not None and isinstance(result, str) and result.strip():
                st.success(f"Successfully extracted text using {provider} ({len(result)} characters).")
                return result
            elif result is None:
                 # Error likely already shown in the specific process_ function
                 st.error(f"OCR process with {provider} failed or returned no result.")
                 return None
            else: # Empty result or only whitespace
                 st.warning(f"OCR process with {provider} returned empty or whitespace-only text.")
                 # Return None for empty results to be consistent
                 return None

        except Exception as e:
            st.error(f"Error during OCR processing pipeline for {provider}: {str(e)}")
            # st.exception(e) # Uncomment for debugging traceback
            return None

# --- Quality Evaluation (Adjusted) ---
def calculate_and_store_quality(text_result, provider, processing_state_dict):
    """
    Calculates quality metrics and updates the provided processing state dictionary.

    Args:
        text_result (str | None): The extracted text.
        provider (str): The name of the OCR provider used.
        processing_state_dict (dict): The dictionary (e.g., st.session_state.admin_processing_state)
                                      to store the quality results in.
    """
    if not evaluate_ocr_quality: # Check if evaluation function is available
         st.warning("OCR metrics evaluation function not available. Skipping calculation.")
         processing_state_dict["metrics"] = None # Store metrics under 'metrics' key
         return

    if not text_result:
        processing_state_dict["metrics"] = None # Clear metrics if no text
        return

    try:
        metrics = evaluate_ocr_quality(text_result, provider) # Now returns only metrics dict

        # Ensure metrics are JSON serializable (convert numpy types if necessary)
        serializable_metrics = {}
        for k, v in metrics.items():
            if hasattr(v, 'item'): # Handle numpy types
                serializable_metrics[k] = v.item()
            elif isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                serializable_metrics[k] = v
            else:
                serializable_metrics[k] = str(v) # Convert unknown types to string

        metrics_data = {
            "metrics": serializable_metrics, # Store the serializable metrics
            "provider": provider # Store which provider this quality belongs to
        }
        # Update the dictionary passed as argument
        processing_state_dict["metrics"] = metrics_data # Store under 'metrics' key
        st.info(f"Basic text metrics calculated for {provider}.")
    except Exception as e:
        st.warning(f"Could not calculate text metrics for {provider}: {str(e)}")
        processing_state_dict["metrics"] = None # Clear metrics on error

# --- Keyword Extraction (Basic Example) ---
def extract_keywords(text: str, language: str = "en", max_ngram_size: int = 3, deduplication_threshold: float = 0.9, num_keywords: int = 10) -> list[str]:
    """
    Extracts keywords using the YAKE! algorithm.

    Args:
        text: The input text.
        language: Language code (e.g., "en", "id").
        max_ngram_size: Maximum size of keyword n-grams.
        deduplication_threshold: Threshold for deduplicating similar keywords (lower means more deduplication).
        num_keywords: The maximum number of keywords to return.

    Returns:
        A list of extracted keywords (strings).
    """
    if not text or not text.strip():
        return []
    if not yake: # Fallback if YAKE is not installed
        st.warning("YAKE not installed, using rudimentary keyword extraction.")
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        # Basic stop words - consider a more robust list if YAKE isn't used
        stop_words = {"the", "a", "an", "is", "are", "in", "on", "it", "and", "to", "of", "page", "break"}
        filtered_words = [word for word in words if word not in stop_words]
        from collections import Counter
        return [word for word, count in Counter(filtered_words).most_common(num_keywords)]

    try:
        kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=num_keywords, features=None)
        keywords_with_scores = kw_extractor.extract_keywords(text)
        # Return only the keyword strings
        return [kw for kw, score in keywords_with_scores]
    except Exception as e:
        st.warning(f"Keyword extraction failed: {e}")
        return []
