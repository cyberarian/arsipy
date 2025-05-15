# d:\pandasai\idp\idp_extraction.py
import streamlit as st
import os
import json
from io import BytesIO # To handle file bytes in memory

# --- Libraries for Text Extraction ---
try:
    import PyPDF2 # For PDF text extraction (basic)
except ImportError:
    st.warning("PyPDF2 library not found. PDF text extraction will be limited. Install with: pip install PyPDF2")
    PyPDF2 = None

try:
    from docx import Document # For DOCX text extraction
except ImportError:
    st.warning("python-docx library not found. DOCX text extraction will be limited. Install with: pip install python-docx")
    Document = None

try:
    from groq import Groq, APIError
except ImportError:
    st.error("Groq library not found. Please install it: pip install groq")
    Groq = None
    APIError = None

# --- Import Gemini client utilities for OCR ---
try:
    from ocr_providers import get_gemini_client, GEMINI_SAFETY_SETTINGS_NONE
    # GEMINI_GENERATION_CONFIG_JSON might not be needed for plain text OCR
except ImportError:
    st.warning("IDP Extraction: Could not import all Gemini utilities from ocr_providers.py for OCR. Ensure it's correctly set up.")
    def get_gemini_client(api_key_name="GEMINI_API_KEY"): return None
    GEMINI_SAFETY_SETTINGS_NONE = None


# --- NEW FUNCTION: Perform OCR on PDF using Gemini ---
def perform_gemini_ocr_on_pdf(file_bytes: bytes, filename: str, api_key_name: str = "MDR_GEMINI_API_KEY") -> str | None:
    """
    Performs OCR on a PDF file using a Gemini multimodal model.

    Args:
        file_bytes: The raw bytes of the PDF file.
        filename: The name of the file (for logging).
        api_key_name: The name of the API key to use from st.secrets.

    Returns:
        The extracted text as a string, or None if OCR fails.
    """
    st.info(f"Attempting Gemini OCR for PDF: {filename}")
    client = get_gemini_client(api_key_name=api_key_name)
    if not client:
        st.error(f"Failed to initialize Gemini client for OCR. Ensure API key '{api_key_name}' is set.")
        return None

    # Use a model known for multimodal capabilities, e.g., gemini-1.5-flash or gemini-1.5-pro
    # Adjust model_name if you have a specific one provisioned or preferred.
    # The user's metadata extraction model is "gemini-2.5-flash-preview-04-17".
    # For direct PDF ingestion for OCR, gemini-1.5-flash is a good choice.
    ocr_model_name = "models/gemini-2.5-flash-preview-04-17"

    try:
        model = client.GenerativeModel(ocr_model_name)
    except Exception as e:
        st.error(f"Failed to get Gemini OCR model '{ocr_model_name}'. Error: {e}")
        return None

    # Prepare the PDF part for the multimodal request
    pdf_part = {"mime_type": "application/pdf", "data": file_bytes}
    prompt = "Extract all text content from this PDF document."

    try:
        response = model.generate_content(
            [prompt, pdf_part], # Content can be a list of [text_prompt, image/pdf_part]
            safety_settings=GEMINI_SAFETY_SETTINGS_NONE # Apply safety settings if configured
            # No specific generation_config for plain text OCR usually
        )
        
        extracted_text = response.text
        if extracted_text and extracted_text.strip():
            st.success(f"Gemini OCR successful for {filename}.")
            return extracted_text
        else:
            st.warning(f"Gemini OCR for {filename} returned no text or only whitespace.")
            if response.prompt_feedback:
                st.warning(f"Gemini OCR Prompt Feedback: {response.prompt_feedback}")
            return None
    except Exception as e:
        st.error(f"Error during Gemini OCR API call for {filename}: {e}")
        if 'response' in locals() and hasattr(response, 'prompt_feedback') and response.prompt_feedback:
             st.error(f"Gemini OCR prompt feedback: {response.prompt_feedback}")
        return None

# --- NEW FUNCTION: General Text Extraction from Document ---
def get_text_from_document(file_bytes: bytes, filename: str, file_extension: str, **kwargs) -> str | None:
    """
    Extracts text content from a given document file.
    Handles various file types (TXT, PDF, DOCX, images via OCR).

    Args:
        file_bytes: The raw bytes of the file.
        filename: The original name of the file (for context or logging).
        file_extension: The lowercased file extension (e.g., "pdf", "docx", "txt", "png").
        **kwargs: Additional arguments, e.g., ocr_provider_name for image/scanned PDF OCR.

    Returns:
        The extracted text as a string, or None if extraction fails or is not supported.
    """
    st.info(f"Attempting text extraction for {filename} (type: {file_extension})")
    text_content = None
    try:
        if file_extension == "txt":
            text_content = file_bytes.decode('utf-8', errors='replace') # Use 'replace' for robustness
        
        elif file_extension == "pdf":
            if PyPDF2:
                try:
                    pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
                    extracted_pages = []
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        extracted_pages.append(page.extract_text())
                    text_content = "\n".join(filter(None, extracted_pages)) # Filter out None results from empty pages
                    
                    if not text_content or not text_content.strip():
                        st.warning(f"PyPDF2 extracted no text (or only whitespace) from {filename}. It might be an image-based PDF or have complex encoding.")
                        # --- Fallback to Gemini OCR ---
                        if get_gemini_client(): # Check if Gemini client can be initialized
                            st.info(f"Attempting OCR with Gemini for {filename} as PyPDF2 failed.")
                            # Use the existing perform_gemini_ocr_on_pdf function
                            text_content = perform_gemini_ocr_on_pdf(file_bytes, filename) 
                        else:
                            st.error("Gemini client not available for OCR fallback. Ensure ocr_providers.py is set up and API key is valid.")

                except Exception as e_pypdf:
                    st.error(f"PyPDF2 extraction failed for {filename}: {e_pypdf}. "
                             "This PDF might be scanned, encrypted, or have an unsupported format for PyPDF2. "
                             "Attempting Gemini OCR as fallback.")
                    text_content = perform_gemini_ocr_on_pdf(file_bytes, filename) if get_gemini_client() else None
            else:
                st.warning(f"PyPDF2 library not available. Cannot extract text from PDF: {filename}")
                text_content = f"Placeholder: PDF text extraction for '{filename}' requires PyPDF2 or similar."
            # Note: For scanned PDFs or PDFs where PyPDF2 fails, you'd integrate OCR here.
            # Example:
            # if not text_content and kwargs.get("ocr_on_pdf_failure", False):
            #     st.info(f"Attempting OCR for PDF {filename} as text extraction failed/was empty.")
            #     text_content = perform_ocr(file_bytes, filename, kwargs.get("ocr_provider_name"))

        elif file_extension == "docx":
            if Document:
                try:
                    doc = Document(BytesIO(file_bytes))
                    full_text = [para.text for para in doc.paragraphs]
                    text_content = "\n".join(full_text)
                except Exception as e_docx:
                    st.error(f"Error extracting text from DOCX {filename}: {e_docx}")
                    text_content = None
            else:
                st.warning(f"python-docx library not available. Cannot extract text from DOCX: {filename}")
                text_content = f"Placeholder: DOCX text extraction for '{filename}' requires python-docx."

        elif file_extension in ["png", "jpg", "jpeg", "tiff", "bmp"]:
            # This requires OCR. You should have an OCR mechanism.
            # ocr_provider = kwargs.get("ocr_provider_name")
            # if ocr_provider:
            #     st.info(f"Attempting OCR for image {filename} using {ocr_provider}...")
            #     # text_content = perform_ocr(file_bytes, filename, ocr_provider) # Your OCR function call
            # else:
            #     st.warning(f"OCR provider not specified for image {filename}. Cannot extract text.")
            #     text_content = None
            st.warning(f"Image OCR for {filename} needs full implementation in idp_extraction.py (e.g., using Tesseract, Google Vision). Using placeholder.")
            text_content = f"Placeholder: OCR text from image '{filename}' would be here. Implement with pytesseract or cloud OCR."
            
        else:
            st.warning(f"Unsupported file type for text extraction: {file_extension} for file {filename}")
            text_content = None

        if text_content and text_content.strip():
            st.success(f"Text successfully extracted from {filename}.")
        elif text_content is not None and not text_content.strip(): # Explicitly empty string after extraction
             st.info(f"Text extraction from {filename} resulted in empty content (possibly a blank or non-textual document).")
        # else text_content is None, error/warning already shown by specific handlers

        return text_content

    except Exception as e:
        st.error(f"General error during text extraction for {filename}: {e}")
        return None

# --- Example of how you might structure an OCR call (if using Google Vision, for instance) ---
# def perform_ocr(file_bytes, filename, provider_name="google_vision"):
#     if provider_name == "google_vision":
#         try:
#             from ocr_providers import get_google_vision_ocr_text # Assuming this exists
#             # This function in ocr_providers.py would handle the Google Vision API call
#             ocr_result = get_google_vision_ocr_text(file_bytes)
#             if ocr_result and "text" in ocr_result:
#                 return ocr_result["text"]
#             elif ocr_result and "error" in ocr_result:
#                 st.error(f"OCR for {filename} failed: {ocr_result['error']}")
#                 return None
#             return None
#         except ImportError:
#             st.error("Google Vision OCR utility not found in ocr_providers.py")
#             return None
#         except Exception as e:
#             st.error(f"Error during Google Vision OCR for {filename}: {e}")
#             return None
#     # Add other providers like Tesseract, Azure, etc.
#     else:
#         st.error(f"Unsupported OCR provider: {provider_name}")
#         return None

# --- Existing Function: Analyze Content for RM Suggestions ---
def analyze_content_for_rm_suggestions(text_content: str) -> dict | None:
    """
    Uses an LLM via Groq to analyze text content and suggest RM-related metadata.

    Args:
        text_content: The extracted text of the document.

    Returns:
        A dictionary containing suggestions (e.g., classification, keywords, sensitivity)
        or None if analysis fails. Expected format:
        {
            "suggested_classification": "...",
            "suggested_keywords": ["...", "..."],
            "sensitivity_flags": [
                {"type": "PII", "reason": "..."},
                {"type": "Confidential", "reason": "..."}
            ]
        }
    """
    if not Groq or not APIError:
        st.warning("Groq library not available for RM suggestion analysis.")
        return None
    if not text_content or not text_content.strip():
        st.warning("Cannot analyze empty text content for RM suggestions.")
        return None

    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found in st.secrets for RM analysis.")
        return None

    client = Groq(api_key=api_key)

    # Limit context size to avoid exceeding model limits (adjust as needed)
    max_context_length = 15000 # Example limit, depends on model
    truncated_content = text_content[:max_context_length]
    if len(text_content) > max_context_length:
        st.info(f"RM Analysis: Content truncated to {max_context_length} chars for LLM.")


    # --- Prompt for RM Suggestion Generation ---
    prompt = f"""
**Peran Anda:**
Anda adalah AI Analis Manajemen Rekod (Records Management Analyst). Tugas Anda adalah menganalisis teks dokumen yang diberikan dan memberikan saran terkait manajemen rekod HANYA berdasarkan konten teks tersebut.

**Instruksi Penting:**
1.  **Analisis Konten:** Baca dan pahami teks berikut dengan cermat.
2.  **Saran Klasifikasi:** Sarankan satu jenis klasifikasi rekod yang paling mungkin untuk dokumen ini (misalnya: "Kontrak", "Notulen Rapat", "Laporan Keuangan", "Korespondensi Internal", "Materi Pemasaran", "Dokumen Teknis", "Kebijakan/Prosedur", "Data Personalia"). Jika tidak yakin, gunakan "Umum/Tidak Spesifik".
3.  **Saran Kata Kunci:** Ekstrak 5-7 kata kunci atau frasa kunci yang paling relevan dan representatif dari isi dokumen.
4.  **Deteksi Sensitivitas:** Identifikasi potensi jenis informasi sensitif dalam teks. Jenis yang mungkin termasuk: "PII" (Informasi Identitas Pribadi - nama, NIK, alamat, kontak pribadi), "Financial" (Data keuangan rahasia, anggaran, gaji), "Confidential Project" (Nama proyek rahasia, detail strategis), "Legal" (Informasi hukum sensitif, litigasi). Untuk setiap jenis yang terdeteksi, berikan alasan singkat (1 kalimat) berdasarkan teks. Jika tidak ada yang terdeteksi, berikan array kosong.
5.  **Format Output:** Kembalikan hasil analisis Anda HANYA sebagai objek JSON yang valid dengan struktur berikut:
    ```json
    {{
      "suggested_classification": "...",
      "suggested_keywords": ["...", "..."],
      "sensitivity_flags": [
        {{"type": "...", "reason": "..."}},
        {{"type": "...", "reason": "..."}}
      ]
    }}
    ```
    Pastikan JSON valid dan hanya berisi struktur ini. Jangan tambahkan teks atau penjelasan lain di luar JSON.

**Teks Dokumen untuk Dianalisis:**
---
{truncated_content}
---

**Output JSON Anda:**
"""

    try:
        # Use a capable model, adjust if needed
        model_to_use = "llama3-70b-8192" # Llama3 70b is good for complex instructions & JSON
        # model_to_use = "mixtral-8x7b-32768" # Fallback option

        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, # Slightly higher temp for creative suggestion, but still factual
            max_tokens=512, # Adjust based on expected output size
            response_format={"type": "json_object"}, # Request JSON output if model supports it
        )
        raw_json_output = response.choices[0].message.content

        # Parse the JSON response
        try:
            suggestions = json.loads(raw_json_output)
            # Basic validation of structure (can be more thorough)
            if isinstance(suggestions, dict) and \
               "suggested_classification" in suggestions and \
               "suggested_keywords" in suggestions and \
               "sensitivity_flags" in suggestions:
                return suggestions
            else:
                st.warning(f"RM Analysis: LLM response was valid JSON but lacked expected structure. Response: {raw_json_output}")
                return None
        except json.JSONDecodeError as json_err:
            st.warning(f"RM Analysis: Failed to parse LLM response as JSON. Error: {json_err}. Response: {raw_json_output}")
            return None

    except APIError as e:
        st.error(f"Groq API Error during RM Analysis: {e}")
        print(f"Groq API Error during RM Analysis: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during RM Analysis: {e}")
        print(f"An unexpected error occurred during RM Analysis: {e}")
        return None


# --- Existing Function: answer_query_with_context (No changes needed here) ---
def answer_query_with_context(user_query: str, context_chunks: list[str]) -> dict | None:
    """
    Uses an LLM via Groq to answer a user's query based ONLY on provided context chunks.
    Retrieves API key from st.secrets.

    Args:
        user_query: The user's natural language question or information request.
        context_chunks: A list of text strings retrieved as relevant context.

    Returns:
        A dictionary containing the answer or an error message.
        {'answer': 'The generated answer.'} or {'error': 'Error message.'}
    """
    if not Groq or not APIError:
        return {"error": "Groq library not installed or import failed."}

    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return {"error": "GROQ_API_KEY not found in st.secrets."}

    if not user_query:
        return {"error": "No query provided."}
    if not context_chunks:
        # Return a specific answer indicating no context was found
        return {"answer": "Maaf, informasi yang Anda cari tidak ditemukan dalam konteks dokumen yang diberikan."} # Specific Bahasa response

    client = Groq(api_key=api_key)

    # --- Combine context chunks ---
    context_string = "\n\n---\n\n".join(context_chunks)

    # --- Enhanced System Prompt for Contextual Q&A (Bahasa) ---
    prompt = f"""
**Peran Anda:**
Anda adalah asisten AI yang sangat teliti dan membantu. Tugas Anda adalah menjawab pertanyaan pengguna secara akurat HANYA berdasarkan informasi yang terdapat dalam dokumen.

**Instruksi Penting:**
1.  **Context is Key:** Base your answer strictly and solely on the information contained within the provided documents.
2.  **Structured Data Priority:** When a query asks for specific data points, figures, dates, or comparisons (e.g., monthly values, sector shares, top products, growth rates), prioritize searching for and extracting this information from tables, lists, or clearly structured data blocks within the context.
3.  **Table Detail:** If extracting from tables, pay close attention to:
    *   Table headers (columns and rows) to understand the data's meaning.
    *   Units mentioned (e.g., USD Miliar, %, Juta USD, Rupiah).
    *   The specific time periods or categories requested in the query.
4.  **Synthesize When Necessary:** If the complete answer requires combining information from different parts of the provided context (e.g., multiple rows in a table, or text and a table), carefully synthesize these pieces of information accurately.
5.  **Handle Missing Information:**
    *   If the specific information needed to answer the query is *not found* within the provided context snippets, state clearly: "Maaf, informasi yang Anda cari tidak ditemukan dalam konteks dokumen yang diberikan."
    *   Do *not* guess or infer information that isn't explicitly present in the context.
6.  **Acknowledge Limitations:** If you encounter potential issues within the context that might affect the answer (e.g., noted parsing errors like '[Error parsing table JSON: ...]', incomplete tables, ambiguous text), briefly mention this limitation in your response if relevant to the query. For example: "Data untuk [bagian spesifik] mungkin tidak lengkap karena ada catatan error parsing dalam konteks dokumen."
7.  **Be Concise & Use Bahasa Indonesia:** Provide direct answers in clear and correct Bahasa Indonesia.
8.  **Formatting:**
    *   Present answers clearly and concisely. Use short paragraphs.
    *   If the query asks for data suitable for a table (e.g., comparisons, lists with attributes, structured numbers) AND the information exists in the context, **format the answer as a neat Markdown table.**
        Example Markdown Table:
        ```markdown
        | Header 1 | Header 2 | Angka |
        |----------|----------|-------|
        | Data A   | Info X   | 10    |
        | Data B   | Info Y   | 25    |
        ```
    *   Use Markdown emphasis (**bold**, *italic*) sparingly for key points.

**Konteks Dokumen yang Disediakan:**
---
{context_string}
---

**Pertanyaan Pengguna:**
{user_query}

**Jawaban Anda (Hanya berdasarkan konteks, dalam Bahasa Indonesia, format sesuai instruksi di atas):**
"""

    try:
        # Using a capable model like llama3 is recommended for better instruction following and formatting
        # model_to_use = "llama3-70b-8192" # Example: Consider using Llama 3 70b if available/suitable
        # model_to_use = "deepseek-r1-distill-llama-70b" # Or stick with Mixtral if Llama 3 isn't available/preferred
        # model_to_use = "meta-llama/llama-4-maverick-17b-128e-instruct" # Keep if this is the intended model
        model_to_use = "gemini-2.5-pro-preview-05-06" 
        st.info(f"Using model for Q&A: {model_to_use}") # Optional: Log which model is used

        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                # System message can sometimes help reinforce the role, but instructions are clear in user prompt
                # {"role": "system", "content": "Anda adalah asisten AI yang menjawab pertanyaan hanya berdasarkan konteks yang diberikan, dalam Bahasa Indonesia."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, # Keep low for factuality based on context
            max_tokens=1024, # Increase if complex answers or tables are expected
            # top_p=0.9, # Can adjust if needed, but low temp is usually sufficient
            # stop=None # No specific stop sequences needed usually
        )
        answer = response.choices[0].message.content

        # Basic cleanup - remove potential leading/trailing whitespace
        cleaned_answer = answer.strip()

        # Optional: Add post-processing if the model sometimes includes preamble before the actual answer
        # e.g., if it sometimes starts with "Berikut adalah jawabannya:" you might remove it.
        # However, the prompt tries to prevent this by asking for the answer directly.

        return {"answer": cleaned_answer}

    except APIError as e:
        st.error(f"Groq API Error during Q&A: {e}") # Show error in Streamlit UI
        print(f"Groq API Error during Q&A: {e}")
        return {"error": f"Groq API Error: {str(e)}"}
    except Exception as e:
        st.error(f"An unexpected error occurred during Q&A: {e}") # Show error in Streamlit UI
        print(f"An unexpected error occurred during Q&A: {e}")
        return {"error": f"An unexpected error occurred during Q&A: {str(e)}"}
