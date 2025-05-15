# d:\pandasai\idp\utils.py
import streamlit as st
import os
import json
import time
from datetime import datetime
import io
import pickle
import pandas as pd
from db_utils import load_records_to_dataframe # Assuming db_utils is accessible
from constants import PROCESSED_DIR # Assuming constants.py defines this
from constants import PROCESSED_DOCS_MANIFEST_PATH # Also import this from constants
# --- Optional imports with error handling (needed for metadata/loading) ---
try:
    import fitz # PyMuPDF
except ImportError:
    st.warning("PyMuPDF (fitz) not found. PDF metadata/direct extraction will be limited.")
    fitz = None

try:
    from PIL import Image
except ImportError:
    st.warning("Pillow not found. Image metadata extraction will be limited.")
    Image = None

try:
    import faiss
except ImportError:
    st.error("FAISS is required for vector indexing. Please install it: pip install faiss-cpu")
    faiss = None

try:
    import pdfplumber
except ImportError:
    st.warning("pdfplumber not found. PDF table extraction will be disabled.")
    pdfplumber = None

# --- Optional import for HyDE ---
try:
    from groq import Groq, APIError
except ImportError:
    st.warning("Groq library not found. HyDE feature will be disabled.")
    Groq = None # Placeholder if Groq is not available

# --- Helper Function for Metadata Extraction ---
def extract_metadata(file_bytes, file_name, file_type):
    """Extracts metadata from PDF or Image files."""
    metadata = {}
    metadata["Filename"] = file_name
    metadata["File Type"] = file_type
    # Add file size later if needed (requires uploaded_file object)

    if file_type == "application/pdf" and fitz:
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                meta = doc.metadata
                metadata["Page Count"] = doc.page_count
                metadata["Title"] = meta.get("title", "N/A")
                metadata["Author"] = meta.get("author", "N/A")
                metadata["Subject"] = meta.get("subject", "N/A")
                try:
                    if meta.get("creationDate"):
                         raw_date = meta["creationDate"][2:16] # Format D:YYYYMMDDHHMMSS
                         metadata["Creation Date"] = datetime.strptime(raw_date, '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
                    else: metadata["Creation Date"] = "N/A"
                except (ValueError, TypeError, IndexError): metadata["Creation Date"] = meta.get("creationDate", "N/A") # Keep raw if parsing fails
                try:
                     if meta.get("modDate"):
                         raw_date = meta["modDate"][2:16] # Format D:YYYYMMDDHHMMSS
                         metadata["Modification Date"] = datetime.strptime(raw_date, '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
                     else: metadata["Modification Date"] = "N/A"
                except (ValueError, TypeError, IndexError): metadata["Modification Date"] = meta.get("modDate", "N/A") # Keep raw if parsing fails
        except Exception as e:
            # Use print or logging here as st context might not be reliable
            print(f"Warning: Could not extract PDF metadata: {e}")
            metadata["Error"] = "Failed to read PDF metadata"
    elif file_type.startswith("image/") and Image:
        try:
            with Image.open(io.BytesIO(file_bytes)) as img:
                metadata["Dimensions (WxH)"] = f"{img.width} x {img.height}"
                metadata["Image Format"] = img.format
                # Extract EXIF data if available (can be extensive)
                # exif_data = img._getexif()
                # if exif_data:
                #     metadata["EXIF"] = {TAGS.get(tag, tag): value for tag, value in exif_data.items()} # Requires from PIL import ExifTags as TAGS
        except Exception as e:
            print(f"Warning: Could not extract image metadata: {e}")
            metadata["Error"] = "Failed to read image metadata"
    return metadata

# --- Placeholder for HyDE ---
def generate_hypothetical_answer(query: str) -> str | None:
    """
    Generates a hypothetical answer for the given query using an LLM via Groq.
    Used for the HyDE retrieval strategy.
    """
    if not Groq:
        st.warning("HyDE disabled: Groq library not available.")
        return None

    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.warning("HyDE disabled: GROQ_API_KEY not found in st.secrets.")
        return None

    client = Groq(api_key=api_key)

    prompt = f"""
    **Peran Anda:** Anda adalah AI yang bertugas membuat jawaban hipotetis singkat (1-2 kalimat) untuk sebuah pertanyaan, seolah-olah jawaban tersebut ada di dalam sebuah dokumen. Jawaban ini akan digunakan untuk mencari dokumen yang relevan.

    **Instruksi:**
    1. Baca pertanyaan pengguna.
    2. Buatlah jawaban yang paling mungkin dan relevan untuk pertanyaan tersebut, seakan-akan Anda mengutip dari sebuah dokumen.
    3. Jaga agar jawaban tetap singkat dan fokus pada inti pertanyaan.
    4. Jangan menambahkan basa-basi atau menyatakan bahwa ini adalah jawaban hipotetis. Langsung berikan jawabannya.

    **Pertanyaan Pengguna:**
    {query}

    **Jawaban Hipotetis Anda (singkat, relevan, seolah dari dokumen):**
    """

    try:
        # Use a fast model suitable for short generation
        model_to_use = "llama3-8b-8192" # Good balance of speed and quality
        # model_to_use = "mixtral-8x7b-32768" # Another good option

        response = client.chat.completions.create(
            model=model_to_use,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, # Slightly creative but still grounded
            max_tokens=100, # Limit output length
        )
        hypothetical_answer = response.choices[0].message.content.strip()
        st.info(f"HyDE: Generated hypothetical answer snippet.") # Keep user informed
        return hypothetical_answer
    except APIError as e:
        st.warning(f"HyDE Error: Groq API failed during hypothetical answer generation: {e}")
        return None
    except Exception as e:
        st.warning(f"HyDE Error: Unexpected error during hypothetical answer generation: {e}")
        return None

# --- Authentication Function ---
def check_login(username, password) -> str | None:
    """
    Checks credentials against Streamlit secrets.
    Returns role ("admin", "creator") or None if login fails.
    """
    try:
        # Ensure secrets are loaded correctly
        if "admin_credentials" not in st.secrets:
             st.error("Missing 'admin_credentials' section in Streamlit secrets.")
             return None
        if "creator_credentials" not in st.secrets: # Added for creator role
            st.warning("Missing 'creator_credentials' section in Streamlit secrets. Creator login disabled.")

        correct_username_admin = st.secrets["admin_credentials"]["username"]
        correct_password_admin = st.secrets["admin_credentials"]["password"]

        if username == correct_username_admin and password == correct_password_admin:
            return "admin"

        if "creator_credentials" in st.secrets and username == st.secrets["creator_credentials"]["username"] and password == st.secrets["creator_credentials"]["password"]:
            return "creator"

    except KeyError as e:
        st.error(f"Missing key in admin_credentials secrets: {e}")
    except Exception as e:
        st.error(f"Error reading secrets: {e}")
    return None

# --- Functions to Load/Manage the Manifest ---
def load_or_create_manifest():
    """Loads the processed documents manifest or creates an empty one."""
    if os.path.exists(PROCESSED_DOCS_MANIFEST_PATH):
        try:
            with open(PROCESSED_DOCS_MANIFEST_PATH, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
                if not isinstance(manifest, dict):
                    st.warning(f"Manifest file '{PROCESSED_DOCS_MANIFEST_PATH}' is not a valid JSON dictionary. Creating a new one.")
                    return {}
                return manifest
        except (json.JSONDecodeError, FileNotFoundError, TypeError) as e:
            st.warning(f"Could not load or parse manifest file '{PROCESSED_DOCS_MANIFEST_PATH}': {e}. Creating a new one.")
            return {}
    else:
        return {}

def save_manifest(manifest_data):
    """Saves the processed documents manifest."""
    try:
        # Ensure the directory exists before saving
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        with open(PROCESSED_DOCS_MANIFEST_PATH, 'w', encoding='utf-8') as f:
            # Save the entire manifest_data dictionary
            json.dump(manifest_data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        st.error(f"Failed to save manifest file '{PROCESSED_DOCS_MANIFEST_PATH}': {e}")

# --- Function to find available processed documents from Manifest ---
def find_available_documents_from_db() -> dict:
    """
    Queries the database for processed records and checks if corresponding
    index/chunk files exist. Returns a dictionary for the chatbot dropdown.

    Returns:
        dict: Dictionary where keys are user-facing filenames (from DB)
              and values are dicts containing 'base_filename', 'chunk_size',
              'chunk_overlap', 'provider'. Returns empty dict on error or if no valid docs.
    """
    available_docs = {}
    try:
        records_df = load_records_to_dataframe()
        if records_df.empty:
            return {}

        # Ensure required columns exist after loading
        required_cols = ['filename', 'base_filename', 'chunk_size', 'chunk_overlap', 'provider']
        if not all(col in records_df.columns for col in required_cols):
             print("[ERROR] Database is missing required columns for document discovery (filename, base_filename, chunk_size, chunk_overlap, provider).")
             # Maybe raise an error or return empty dict?
             return {} # Return empty for now

        for index, row in records_df.iterrows():
            display_name = row['filename']
            base_name = row['base_filename']
            chunk_size = row['chunk_size']
            chunk_overlap = row['chunk_overlap']
            provider = row['provider']
            processed_at_ts = row['processed_at'] # Get the timestamp from the DB row

            # Construct expected file paths
            index_path = os.path.join(PROCESSED_DIR, f"{base_name}.index")
            chunks_path = os.path.join(PROCESSED_DIR, f"{base_name}_chunks.pkl") # Original chunks
            tokenized_chunks_path = os.path.join(PROCESSED_DIR, f"{base_name}_tok_chunks.pkl") # Tokenized chunks

            # Check if ALL THREE files exist
            if os.path.exists(index_path) and os.path.exists(chunks_path) and os.path.exists(tokenized_chunks_path):
                available_docs[display_name] = {
                    "base_filename": base_name,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "provider": provider, # Store provider if needed later
                    "processed_at": processed_at_ts, # <-- Add the timestamp
                    "index_path": index_path, # <-- Add the full index path
                    "chunks_path": chunks_path, # <-- Add the full original chunks path
                    "tokenized_chunks_path": tokenized_chunks_path # <-- Add the full tokenized chunks path
                    # Add other relevant info from the DB row if needed by the chatbot tab
                }
            else:
                # Optional: Log a warning if DB record exists but files are missing
                print(f"[WARN] Files missing for DB record ID {row.get('id', 'N/A')} (base: {base_name}). Skipping.")

    except ImportError:
         print("[ERROR] Could not import db_utils in find_available_documents_from_db.")
         return {}
    except Exception as e:
        print(f"[ERROR] Failed to find available documents from DB: {e}")
        return {} # Return empty dict on any error

    # Sort alphabetically by display name (which should already be the case from DB query)
    return dict(sorted(available_docs.items()))

# --- Function to load selected document index and chunks ---
@st.cache_resource(show_spinner="Loading selected document...", max_entries=5)
def load_document_data(_index_path: str, _chunks_path: str, _tokenized_chunks_path: str):
    """Loads FAISS index, original chunks, and tokenized chunks from specified paths."""
    if not faiss:
        st.error("FAISS is not available. Cannot load index.")
        return None, None, None
    if not _index_path or not _chunks_path or not _tokenized_chunks_path:
        st.error("Index, chunks, or tokenized chunks path missing.")
        return None, None, None

    try:
        index = faiss.read_index(_index_path)
        with open(_chunks_path, 'rb') as f:
            original_chunks = pickle.load(f)
        # Load tokenized chunks
        with open(_tokenized_chunks_path, 'rb') as f:
            tokenized_chunks = pickle.load(f)
        return index, original_chunks, tokenized_chunks
    except FileNotFoundError:
         st.error(f"Error loading document data: File not found. "
                  f"Index: '{_index_path}', Chunks: '{_chunks_path}', "
                  f"Tokenized: '{_tokenized_chunks_path}'")
         # Clear the cache for this specific input if file not found
         # Note: This might require more specific cache invalidation depending on Streamlit version.
         # Return three Nones to match the expected unpacking.
         return None, None, None
    except Exception as e:
        st.error(f"Error loading document data (Index: {_index_path}, Chunks: {_chunks_path}): {e}")
        return None, None, None # Ensure three Nones are returned on general exception too
