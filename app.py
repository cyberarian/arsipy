__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import time
import fitz  # PyMuPDF
import pandas as pd
import logging
import traceback
# import gc # Removed gc import
import sys
import shutil
from stqdm import stqdm
# from contextlib import contextmanager # Removed contextmanager import
from typing import List, Any, Dict, Optional, Set, Tuple, Union
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain # LCEL
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import create_retrieval_chain # LCEL - Removed LLMChain, RetrievalQA, StuffDocumentsChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.globals import set_verbose
from dotenv import load_dotenv
from streamlit.runtime.caching import cache_data, cache_resource
from datetime import datetime
import toml
import chromadb
import sqlite3
# from image_analyzer import image_analyzer_main # Assuming not used based on context
from huggingface_hub import InferenceClient
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.language_models.llms import LLM
from langchain_core.retrievers import BaseRetriever
import re # Import re for text cleaning
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests # Ensure requests is imported for fetch_url_content
from bs4 import BeautifulSoup # Ensure bs4 is imported for fetch_url_content
from urllib.parse import urlparse, urljoin # Ensure urljoin is imported
from document_processor import UnifiedDocumentProcessor # Ensure this is imported

# Utility imports
from utils.cache_manager import CacheManager
from utils.security import SecurityManager
from utils.monitoring import SystemMonitor
from document_processor import UnifiedDocumentProcessor
from landing_page import show_landing_page
from utils.web_search import ArchivalWebSearch

# Agentic utilities (Using original names as per files)
from utils.agentic_ai import SearchAgent, ResponseAgent, KnowledgeAgent

# --- Constants ---
CHROMA_DB_DIR = "chroma_db"
DEFAULT_MODEL_ID = "compound-beta"
DEFAULT_ENRICHMENT_MODEL_ID = "llama3-70b-8192"
EMBEDDING_MODEL_ID = "models/embedding-001"
WEB_FILTER_SIMILARITY_THRESHOLD = 0.90

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Custom LLM Class (DeepSeek) ---
class DeepSeekLLM(LLM):
    """Custom LLM class for DeepSeek models from HuggingFace"""
    # (Keep the DeepSeekLLM class definition as provided in the original file)
    client: InferenceClient
    model: str
    temperature: float = 0.6
    max_tokens: int = 512

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.6,
        max_tokens: int = 512,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        super().__init__(callback_manager=callback_manager)
        self.client = InferenceClient(token=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        # Assuming the original implementation is correct
        response = self.client.text_generation(
            prompt,
            model=self.model,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            stop_sequences=stop or [],
            **kwargs
        )
        return response
    pass

# --- LLM Initialization ---
# @cache_resource # Caching LLM instances can be beneficial
def get_llm_model(model_name: str) -> Union[LLM, Any]:
    """Initialize and return the specified LLM model"""
    # (Keep the get_llm_model function definition as provided, including DeepSeek, Gemini, Groq)
    groq_api_key = os.getenv('GROQ_API_KEY')
    google_api_key = os.getenv("GOOGLE_API_KEY")
    huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')

    if model_name.startswith("llama3") or model_name == "compound-beta":
        if not groq_api_key:
            raise ValueError(f"GROQ_API_KEY not found in environment variables for model {model_name}.")
        return ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name
        )
    elif model_name == "DeepSeek-Prover-V2-671B":
        if not huggingface_api_key:
            raise ValueError(f"HUGGINGFACE_API_KEY not found for model {model_name}.")
        return DeepSeekLLM(
            model="deepseek-ai/DeepSeek-Prover-V2-671B",
            api_key=huggingface_api_key,
            temperature=0.5, max_tokens=512
        )
    elif model_name == "DeepSeek-Prover-V2-671B":
        if not huggingface_api_key:
            raise ValueError(f"HUGGINGFACE_API_KEY not found for model {model_name}.")
        return DeepSeekLLM(
            model="deepseek-ai/DeepSeek-Prover-V2-671B", # Confirm model if specific coder exists
            api_key=huggingface_api_key,
            temperature=0.5, max_tokens=512
        )
    elif model_name == "smallthinker":
        if not huggingface_api_key:
            raise ValueError(f"HUGGINGFACE_API_KEY not found for model {model_name}.")
        return DeepSeekLLM(
            model="PowerInfer/SmallThinker-3B-Preview",
            api_key=huggingface_api_key,
            temperature=0.5, max_tokens=512
        )
    elif model_name == "gemini-2.5-flash-preview-04-17":
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found for Gemini model.")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17",
            google_api_key=google_api_key,
            temperature=0.6
        )
    else:
        logger.warning(f"Model {model_name} not explicitly handled, falling back to {DEFAULT_MODEL_ID}")
        if not groq_api_key:
            raise ValueError(f"GROQ_API_KEY not found for fallback model {DEFAULT_MODEL_ID}.")
        return ChatGroq(
            groq_api_key=groq_api_key,
            model_name=DEFAULT_MODEL_ID
        )
    pass

# --- RAG Chain Setup (LCEL - Updated) ---
def get_rag_chain_lcel(llm: Union[LLM, Any], retriever: BaseRetriever) -> Any: # Return type is LCEL Runnable
    """Sets up the RAG chain using LangChain Expression Language (LCEL)."""

    # System prompt remains the same
    SYSTEM_PROMPT = """You are Arsipy, an expert archival documentation assistant.
    Analyze queries using this internal process (do not show in response):
    1. Topic identification
    2. Context evaluation
    3. Evidence gathering
    4. Response formulation

    Keep responses focused, clear, and professional."""

    # QA prompt updated for LCEL structure (uses 'input' and 'context')
    QA_CHAIN_PROMPT = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", """
        Context:
        {context}

        Question: {input}

        Instructions:
        1. INTERNAL REASONING (Do not include in response):
        {{"analysis": {{
            "topic": "Identify main topic",
            "requirements": "List key requirements",
            "evidence": "Locate supporting context",
            "reasoning": "Connect evidence to answer"
        }}}}

        2. RESPONSE FORMAT:
        - Clear direct answer
        - Supporting evidence (summarized from context)
        - Source citations (from context metadata if available)
        - Additional context (if needed)

        Response in id-ID:
        """)
    ])

    # Create the stuff documents chain
    question_answer_chain = create_stuff_documents_chain(llm, QA_CHAIN_PROMPT)

    # Create the retrieval chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    logger.info("RAG chain created using LCEL (create_retrieval_chain).")
    return rag_chain
    pass

# --- Vectorstore Helper ---
def _is_vectorstore_ready_and_populated(vectorstore: Optional[Chroma]) -> bool:
    """Checks if the vectorstore object exists and contains documents."""
    if not vectorstore:
        logger.debug("Vectorstore check: Object is None.")
        return False
    try:
        # Use a non-blocking, low-level check if possible
        count = vectorstore._collection.count()
        logger.debug(f"Vectorstore check: Collection count = {count}")
        return count > 0
    except Exception as vs_err:
        logger.error(f"Error checking vectorstore count: {vs_err}")
        # Avoid showing error directly in UI repeatedly, just log
        # st.warning("Tidak dapat memverifikasi status dokumen di database vektor.")
        return False
    pass

# --- Admin Sidebar and Controls ---
def setup_admin_sidebar() -> None:
    """Setup admin authentication and controls in sidebar"""
    # (Keep setup_admin_sidebar function definition as provided)
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False

    with st.sidebar:
        st.title("Admin Panel")

        if not st.session_state.admin_authenticated:
            input_password = st.text_input("Admin Password", type="password", key="admin_pw_input")
            if st.button("Login", key="admin_login_button"):
                admin_password = os.getenv('ADMIN_PASSWORD') # Fetch here
                if input_password and admin_password and input_password == admin_password:
                    st.session_state.admin_authenticated = True
                    st.success("Admin authenticated!")
                    st.rerun()
                elif not admin_password:
                     st.warning("Admin password not set in .env. Login disabled.")
                else:
                    st.error("Incorrect password")
        else:
            st.write("✅ Admin authenticated")
            if st.button("Logout", key="admin_logout_button"):
                st.session_state.admin_authenticated = False
                st.rerun()

            st.divider()
            show_admin_controls()
    pass

def show_admin_controls() -> None:
    """Display admin controls (Document Management) - ONLY shown when authenticated."""
    st.sidebar.header("Document Management")

    # --- Document Upload Section (Keep UI as is) ---
    uploaded_files = st.sidebar.file_uploader(
        "Upload/Replace Documents (Admin Only)", # Modified label slightly
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="admin_file_uploader"
    )

    if uploaded_files:
        st.sidebar.subheader("Document Metadata (for Title/Source)") # Clarified purpose
        st.sidebar.info("""
        Provide metadata for better organization and source identification.
        This will update `module_titles.json` automatically.
        Example: 'Modul 1, Manajemen Kearsipan di Indonesia, Drs. Syauki Hadiwardoyo'
        """)
        metadata_inputs = {}
        for file in uploaded_files:
            file_key_base = f"{file.name}_{file.size}" # Keep key unique per upload instance
            with st.sidebar.expander(f"Metadata for {file.name}"):
                metadata_inputs[file.name] = {
                    'judul': st.text_input("Judul Modul", key=f"title_{file_key_base}", placeholder="e.g., Manajemen Kearsipan"),
                    'pengajar': st.text_input("Nama Pengajar", key=f"author_{file_key_base}", placeholder="e.g., Drs. Syauki H."),
                    'deskripsi': st.text_area("Deskripsi (Optional)", key=f"desc_{file_key_base}", placeholder="Deskripsi singkat")
                }
        if st.sidebar.button("Process/Replace Documents", key="admin_process_docs_button"): # Modified label
            if 'doc_processor' in st.session_state and st.session_state.doc_processor:
                 # Call the modified processing function
                 process_uploaded_files(uploaded_files, metadata_inputs)
            else:
                 st.sidebar.error("Document processor not ready.")
    # --- End Document Upload Section ---

    st.sidebar.divider()

    # --- Document Deletion Section (Keep as is) ---
    # This still allows deleting files explicitly, independent of replacement
    st.sidebar.subheader("Delete Document Explicitly")
    processed_files_list = sorted(list(st.session_state.get('uploaded_file_names', set())))
    if not processed_files_list:
        st.sidebar.caption("No documents processed in this session to delete.")
    else:
        doc_to_delete = st.sidebar.selectbox(
            "Select Document to Delete",
            options=[""] + processed_files_list,
            index=0,
            key="admin_delete_doc_select",
            help="Select a document processed in this session to remove its content from the vector database."
        )
        if doc_to_delete:
            if st.sidebar.button(f"Delete '{doc_to_delete}'", key="admin_delete_doc_button"):
                if 'doc_processor' in st.session_state and st.session_state.doc_processor:
                    with st.spinner(f"Deleting {doc_to_delete}..."):
                        success = st.session_state.doc_processor.delete_document_by_filename(doc_to_delete)
                    if success:
                        st.sidebar.success(f"Successfully deleted document: {doc_to_delete}")
                        if doc_to_delete in st.session_state.uploaded_file_names:
                            st.session_state.uploaded_file_names.remove(doc_to_delete)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.sidebar.error(f"Failed to delete document: {doc_to_delete}")
                else:
                    st.sidebar.error("Document processor not ready. Cannot delete.")
    # --- End Document Deletion Section ---


# --- Document Processing ---
def extract_text_from_pdf(pdf_file: Any) -> str:
    """Extract text content from a PDF file"""
    # (Keep extract_text_from_pdf function definition as provided)
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text()
        if not text.strip():
            logger.warning(f"Extracted text from PDF '{getattr(pdf_file, 'name', 'unknown')}' is empty.")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF '{getattr(pdf_file, 'name', 'unknown')}': {str(e)}")
        raise
    finally:
        if 'pdf_document' in locals() and pdf_document:
            pdf_document.close()

def get_document_text(file: Any) -> str:
    """Get text content from a file based on its type"""
    # (Keep get_document_text function definition as provided)
    try:
        if file.type == "application/pdf":
            text = extract_text_from_pdf(file)
        elif file.type == "text/plain":
            try:
                text = file.getvalue().decode('utf-8')
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decoding failed for {file.name}, trying latin-1.")
                file.seek(0)
                text = file.getvalue().decode('latin-1')
        else:
            raise ValueError(f"Unsupported file type: {file.type}")

        if not text.strip():
            logger.warning(f"Extracted text from {file.name} is empty.")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {file.name}: {str(e)}")
        raise

# --- MODIFIED process_uploaded_files ---
def process_uploaded_files(uploaded_files: List[Any], metadata_inputs: Dict) -> None:
    """
    Process uploaded files: Delete existing version (if any) then add new version.
    Updates module_titles.json via doc_processor if metadata is provided.
    """
    try:
        if not uploaded_files:
            st.sidebar.warning("No files selected for processing")
            return

        if 'doc_processor' not in st.session_state or not st.session_state.doc_processor:
             st.sidebar.error("Document processor not initialized. Cannot process.")
             return

        processed_count = 0
        replaced_count = 0
        error_count = 0
        with st.spinner('Processing documents (Admin)...'):
            for file in stqdm(uploaded_files, desc="Processing Files", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
                try:
                    # --- BEGIN REPLACEMENT LOGIC ---
                    # Always attempt to delete the old version based on filename before adding the new one.
                    logger.info(f"Attempting pre-emptive deletion for potential replacement: {file.name}")
                    delete_success = st.session_state.doc_processor.delete_document_by_filename(file.name)
                    if delete_success:
                         logger.info(f"Pre-emptive deletion successful (or file didn't exist) for {file.name}.")
                         # We don't know for sure if it *was* replaced unless we query count before/after,
                         # but the action was performed. We can infer replacement if it was already in session state.
                         if file.name in st.session_state.get('uploaded_file_names', set()):
                             replaced_count += 1
                    else:
                        # Log warning, but proceed with adding the new version anyway
                        logger.warning(f"Pre-emptive deletion command failed for {file.name}. Proceeding with upload.")
                    # --- END REPLACEMENT LOGIC ---

                    # Now, process the newly uploaded file
                    metadata = metadata_inputs.get(file.name, {})
                    # Pass the original filename explicitly if needed, though doc_processor gets it from file.name
                    # metadata['original_filename'] = file.name

                    result = st.session_state.doc_processor.process_document(
                        file,
                        metadata=metadata # Pass the metadata dict
                    )

                    if result['success']:
                        # Use the title from the result metadata for the success message
                        display_name = result.get('metadata', {}).get('title', file.name)
                        st.sidebar.success(f"Processed/Replaced: {display_name}")
                        # Add/Update the filename in the session state set (used for the explicit delete dropdown)
                        st.session_state.uploaded_file_names.add(file.name)
                        processed_count += 1
                    else:
                        st.sidebar.error(f"Error processing {file.name}: {result['error']}")
                        error_count += 1

                except Exception as proc_err:
                     st.sidebar.error(f"Critical error processing {file.name}: {proc_err}")
                     logger.error(f"Critical error during file processing loop for {file.name}: {traceback.format_exc()}")
                     error_count += 1
                # REMOVED: The check to skip already processed files in this session
                # else:
                #    st.sidebar.info(f"Skipped (already processed this session): {file.name}")

        if processed_count > 0:
            st.sidebar.success(f"{processed_count} document(s) processed/replaced successfully!")
            if replaced_count > 0:
                 st.sidebar.info(f"({replaced_count} likely replaced existing versions)")
            # Persist changes after the batch
            if hasattr(st.session_state.vectorstore, 'persist'):
                try:
                    # It's often better to persist once after all additions/deletions if possible
                    st.session_state.vectorstore.persist()
                    logger.info("Vectorstore changes persisted after batch processing.")
                except Exception as persist_err:
                    st.sidebar.error(f"Error persisting vectorstore changes: {persist_err}")
                    logger.error(f"Failed to persist vectorstore: {traceback.format_exc()}")
        elif error_count > 0:
             st.sidebar.warning("Processing complete, but some files had errors.")
        else:
             st.sidebar.info("No documents were processed in this batch.") # Should not happen if files were uploaded

    except Exception as e:
        st.sidebar.error(f"An unexpected error occurred during file processing: {str(e)}")
        logger.error(f"Error in process_uploaded_files: {traceback.format_exc()}")


# --- Cache Clearing ---
def clear_cache() -> None:
    """Clear all cached data"""
    # (Keep clear_cache function definition as provided)
    cache_data.clear()
    cache_resource.clear()
    st.success("Application caches cleared.")
    pass
# ==============================================================================
# REFACTORED CORE CHAT RESPONSE GENERATION HELPERS
# ==============================================================================

def _perform_rag(query: str, llm: Union[LLM, Any], retriever: BaseRetriever) -> Optional[Dict]:
    """Performs RAG using the LCEL chain and handles errors."""
    try:
        logger.info(f"Attempting RAG for query: '{query}' using {getattr(llm, 'model_name', 'Unknown LLM')}")
        # Use the updated LCEL chain function
        rag_chain = get_rag_chain_lcel(llm, retriever)
        # LCEL chain expects 'input' key
        response = rag_chain.invoke({'input': query})
        logger.info(f"Local RAG response received: Keys={response.keys()}")
        # Extract relevant parts (answer and source documents)
        return {
            'result': response.get('answer', '').strip(),
            'source_documents': response.get('context', []) # 'context' holds retrieved docs
        }
    except Exception as rag_err:
        logger.error(f"Error during RAG chain execution: {rag_err}")
        logger.error(traceback.format_exc())
        return None
    pass
def _is_rag_response_adequate(rag_response: Optional[Dict]) -> bool:
    """Checks if the RAG response is sufficient based on content and sources."""
    if not rag_response:
        logger.info("RAG Adequacy Check: Failed (No response object).")
        return False

    result_text = rag_response.get('result', '').strip().lower()
    local_source_docs = rag_response.get('source_documents', [])

    # Check for empty or very short response
    if not result_text or len(result_text) < 30:
        logger.info(f"RAG Adequacy Check: Failed (Short or empty result: '{result_text[:50]}...')")
        return False

    # Check for explicit inadequacy phrases
    inadequate_phrases = ['tidak ditemukan', 'tidak tersedia', 'tidak dapat menemukan informasi', 'saya tidak tahu', 'i don\'t know', 'tidak relevan', 'kurang informasi']
    if any(phrase in result_text for phrase in inadequate_phrases):
        logger.info(f"RAG Adequacy Check: Failed (Inadequate phrase found in '{result_text[:100]}...')")
        return False

    # Check if source documents were found (essential for RAG)
    if not local_source_docs:
        logger.info("RAG Adequacy Check: Failed (No source documents found).")
        return False

    logger.info(f"RAG Adequacy Check: Passed. Result='{result_text[:100]}...', SourceDocsCount={len(local_source_docs)}")
    return True
    pass
def _perform_web_search(query: str, web_searcher: ArchivalWebSearch, max_results: int) -> List[Dict]:
    """Performs web search using the ArchivalWebSearch utility."""
    try:
        logger.info(f"Performing web search for query: '{query}' (max_results={max_results})")
        results = web_searcher.search(query, total_max_results=max_results)
        logger.info(f"Web search returned {len(results)} results.")
        return results
    except Exception as web_err:
        logger.error(f"Error during web search execution: {web_err}")
        logger.error(traceback.format_exc())
        return []
    pass
# --- Moved Web Result Filtering Function ---
def _filter_redundant_web_results(web_results: List[Dict], threshold: float = WEB_FILTER_SIMILARITY_THRESHOLD) -> List[Dict]:
    """
    Filters web results for redundancy using semantic similarity.
    NOTE: This adds latency and requires a Google API Key for embeddings.
    Consider simpler filtering (e.g., URL domain) if performance is critical.
    """
    if not web_results or len(web_results) <= 1:
        return web_results # No filtering needed

    snippets = [r.get('content', '') for r in web_results]
    valid_snippets_indices = [i for i, s in enumerate(snippets) if s and len(s) > 10] # Indices with actual content

    if len(valid_snippets_indices) <= 1:
        logger.info("Web result filtering skipped: Not enough valid snippets to compare.")
        return web_results

    valid_snippets = [snippets[i] for i in valid_snippets_indices]

    try:
        # Ensure GOOGLE_API_KEY is set in environment or this will fail
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logger.error("GOOGLE_API_KEY not found. Cannot perform embedding-based filtering.")
            return web_results # Return original list if key is missing

        logger.info(f"Filtering {len(web_results)} web results using embeddings (threshold={threshold})...")
        embeddings_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_ID, google_api_key=google_api_key)
        embeddings = embeddings_model.embed_documents(valid_snippets)
        similarity_matrix = cosine_similarity(embeddings)

        indices_to_keep = []
        kept_indices_set = set()

        for i in range(len(valid_snippets_indices)):
            original_index_i = valid_snippets_indices[i]
            is_redundant = False
            # Compare against already kept items' original indices
            for kept_original_index in indices_to_keep:
                try:
                    matrix_idx_i = valid_snippets_indices.index(original_index_i)
                    matrix_idx_j = valid_snippets_indices.index(kept_original_index)
                except ValueError:
                    logger.warning(f"Index mapping error during filtering. Skipping comparison for index {original_index_i}.")
                    continue

                if matrix_idx_i == matrix_idx_j: continue # Avoid self-comparison

                if matrix_idx_i < similarity_matrix.shape[0] and matrix_idx_j < similarity_matrix.shape[1]:
                    similarity = similarity_matrix[matrix_idx_i][matrix_idx_j]
                    if similarity > threshold:
                        is_redundant = True
                        logger.debug(f"Result {original_index_i} redundant with {kept_original_index}. Sim: {similarity:.2f}")
                        break
                else:
                     logger.warning(f"Index out of bounds: i={matrix_idx_i}, j={matrix_idx_j}, shape={similarity_matrix.shape}")

            if not is_redundant:
                indices_to_keep.append(original_index_i)
                kept_indices_set.add(original_index_i)

        # Include results that had no/short snippet content
        final_indices_to_keep = sorted(list(kept_indices_set) + [i for i, s in enumerate(snippets) if not s or len(s) <= 10])

        filtered_results = [web_results[i] for i in final_indices_to_keep]
        logger.info(f"Filtered web results from {len(web_results)} down to {len(filtered_results)}.")
        return filtered_results

    except Exception as filter_err:
        logger.error(f"Error during web result filtering: {filter_err}. Returning original results.")
        logger.error(traceback.format_exc())
        return web_results # Return original list on error
    pass
def _generate_fallback_response(query: str, web_results: List[Dict], llm: Union[LLM, Any]) -> str:
    """Generates a response using only web search results as context."""
    if not web_results:
        logger.warning("Fallback response generation skipped: No web results provided.")
        return "_Maaf, saya tidak dapat menemukan informasi yang relevan dari pencarian web._"

    web_context = "\n\n".join([
        f"Sumber: {r['source']} ({r.get('title', 'N/A')})\nURL: {r.get('url', 'N/A')}\nKonten: {r['content']}"
        for r in web_results
    ])

    # Using the same prompt structure as before
    web_prompt = f"""Berdasarkan informasi *hanya* dari sumber kearsipan terpercaya berikut:
--- Mulai Konteks Web ---
{web_context}
--- Akhir Konteks Web ---
Jawab pertanyaan berikut secara jelas dan ringkas dalam Bahasa Indonesia: "{query}"

Format Jawaban:
- Gunakan paragraf yang terstruktur dengan baik.
- Jika perlu membuat daftar, gunakan bullet points (•).
- Jika konteks tidak memuat jawaban, nyatakan dengan jelas bahwa informasi tidak ditemukan dalam konteks yang diberikan. Jangan gunakan pengetahuan sebelumnya.
- Sebutkan sumber (misal, Menurut ICA: ...) secara alami dalam teks jika memungkinkan untuk menambah kredibilitas.

Jawaban (Bahasa Indonesia):
"""
    try:
        logger.info(f"Invoking LLM ({getattr(llm, 'model_name', 'Unknown LLM')}) with web context for fallback.")
        response_obj = llm.invoke(web_prompt)
        response_text = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
        logger.info("Received web-based fallback response from LLM.")
        return response_text.strip()
    except Exception as llm_err:
        logger.error(f"Error invoking LLM for fallback response: {llm_err}")
        logger.error(traceback.format_exc())
        return "_Maaf, terjadi kesalahan saat menghasilkan jawaban dari sumber web._"
    pass
def _generate_enriched_response(query: str, rag_response: Dict, web_results: List[Dict], enrichment_llm: Union[LLM, Any]) -> str:
    """Generates an enriched response by synthesizing RAG results and web context."""
    if not web_results:
        logger.info("Enrichment skipped: No web results provided. Returning original RAG result.")
        return rag_response.get('result', "_Informasi dari dokumen lokal tidak tersedia._")

    web_context = "\n\n".join([
        f"Sumber: {r['source']} ({r.get('title', 'N/A')})\nURL: {r.get('url', 'N/A')}\nKonten: {r['content']}"
        for r in web_results
    ])
    local_context_summary = rag_response.get('result', '_Informasi dari dokumen lokal tidak tersedia._')

    # Using the same enrichment prompt structure as before
    enrichment_prompt = f"""Anda adalah asisten riset ahli kearsipan yang sangat terampil dalam mensintesis informasi dari berbagai sumber menjadi jawaban yang koheren dan alami.

Pertanyaan Pengguna: "{query}"

Informasi Utama dari Dokumen Internal (Ini adalah dasar jawaban Anda):
--- Mulai Konteks Dokumen ---
{local_context_summary}
--- Akhir Konteks Dokumen ---

Informasi Tambahan dari Sumber Web Terpercaya (Gunakan secara selektif untuk melengkapi, mengkonfirmasi, atau memberi update jika relevan dan menambah nilai):
--- Mulai Konteks Web ---
{web_context}
--- Akhir Konteks Web ---

Tugas Anda:
1.  Tulis jawaban komprehensif dalam Bahasa Indonesia untuk Pertanyaan Pengguna, prioritaskan Informasi Utama dari Dokumen Internal.
2.  Integrasikan Informasi Tambahan dari Sumber Web secara mulus **hanya jika** menambah nilai signifikan. **Jangan** memaksakan informasi web jika tidak relevan atau hanya mengulang.
3.  **Gaya Penulisan Sangat Penting:** Susun jawaban akhir agar terdengar alami, jelas, dan ditulis oleh seorang ahli manusia. **Hindari** secara eksplisit menyebut "Menurut dokumen..." atau "Dari web...". Gabungkan wawasan seolah-olah berasal dari satu basis pengetahuan yang koheren.
4.  **Struktur & Format:** Gunakan paragraf yang terstruktur dengan baik. Jika daftar diperlukan untuk kejelasan, gunakan bullet points (•) atau penomoran.
5.  **Kredibilitas:** Jika relevan, sebutkan nama sumber spesifik (misal, standar ISO 15489, ANRI, ICA) secara alami dalam teks.
6.  Fokus pada kualitas, akurasi, dan relevansi jawaban.

Jawaban Sintesis Alami (Bahasa Indonesia):
"""
    try:
        logger.info(f"Invoking enrichment LLM ({getattr(enrichment_llm, 'model_name', 'Unknown LLM')}) for synthesis.")
        enriched_response_obj = enrichment_llm.invoke(enrichment_prompt)
        enriched_response_text = enriched_response_obj.content if hasattr(enriched_response_obj, 'content') else str(enriched_response_obj)
        logger.info("Received enriched response from LLM.")
        return enriched_response_text.strip()
    except Exception as enrich_llm_err:
         logger.error(f"Error invoking enrichment LLM: {enrich_llm_err}. Falling back to original RAG response.")
         logger.error(traceback.format_exc())
         # Return the original RAG result as a fallback if enrichment fails
         return rag_response.get('result', "_Maaf, terjadi kesalahan saat memperkaya jawaban._")
    pass
# --- Orchestrator Function for Chat Response ---
def get_orchestrated_response(
    query: str,
    base_llm: Union[LLM, Any],
    enrichment_llm: Union[LLM, Any],
    vectorstore: Optional[Chroma],
    web_searcher: ArchivalWebSearch,
    attempt_rag: bool
    ) -> Dict:
    """
    Orchestrates the process: RAG -> Adequacy Check -> Web Search -> Fallback/Enrichment.
    """
    final_response = {
        'query': query,
        'result': "_Maaf, saya tidak dapat memproses permintaan Anda saat ini._",
        'source_documents': [],
        'web_source_documents': [],
        'from_web': False,
        'enriched': False
    }
    rag_response = None
    is_inadequate = True

    # 1. Attempt RAG if requested and vectorstore is available
    if attempt_rag and vectorstore:
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        rag_response = _perform_rag(query, base_llm, retriever) # Use base_llm for RAG
        is_inadequate = not _is_rag_response_adequate(rag_response)
    else:
        logger.info("Skipping RAG attempt (vectorstore unavailable or attempt_rag=False).")
        is_inadequate = True # Treat as inadequate if RAG wasn't attempted

    # 2. Perform Web Search (for fallback or enrichment)
    max_web_results = 3 if is_inadequate else 2 # Fetch more for fallback
    web_results_raw = _perform_web_search(query, web_searcher, max_results=max_web_results)

    # 3. Filter Web Results (if any found)
    web_results_filtered = _filter_redundant_web_results(web_results_raw) if web_results_raw else []
    final_response['web_source_documents'] = web_results_filtered # Store filtered results

    # 4. Branching Logic: Fallback or Enrichment
    if is_inadequate:
        # --- Fallback Path ---
        logger.info("RAG inadequate or skipped. Generating response from web results.")
        final_response['result'] = _generate_fallback_response(query, web_results_filtered, base_llm) # Use base_llm for fallback
        final_response['from_web'] = True
        final_response['source_documents'] = [] # Ensure local sources are empty
    else:
        # --- Enrichment Path ---
        logger.info("RAG adequate. Attempting enrichment with web results.")
        # Use the potentially more powerful enrichment_llm
        final_response['result'] = _generate_enriched_response(query, rag_response, web_results_filtered, enrichment_llm)
        final_response['source_documents'] = rag_response.get('source_documents', []) # Keep RAG sources
        final_response['enriched'] = bool(web_results_filtered) # Mark enriched only if web results were used

    # Final check for empty/error messages and potential fallback to original RAG
    if not final_response['result'] or "terjadi kesalahan" in final_response['result'].lower():
         if not final_response['from_web'] and rag_response and rag_response.get('result'):
             logger.warning("Fallback/Enrichment resulted in error/empty, using original RAG result.")
             final_response['result'] = rag_response['result']
             final_response['enriched'] = False # Not actually enriched if we fell back here
         elif not web_results_filtered and is_inadequate: # RAG failed AND web failed
             final_response['result'] = "_Maaf, saya tidak dapat menemukan informasi yang relevan baik dari dokumen lokal maupun pencarian web._"

    return final_response
    pass
# ==============================================================================
# CORE CHAT INTERFACE LOGIC (PUBLIC ACCESS) - Updated
# ==============================================================================
def show_chat_interface(
    default_llm: Union[LLM, Any], # Default base LLM passed in
    security_manager: SecurityManager,
    search_agent: SearchAgent,
    response_agent: ResponseAgent,
    knowledge_agent: KnowledgeAgent
    ) -> None:
    """Display the main chat interface (Public Access)"""

    # Add logo
    col1, col2, col3 = st.columns([1,100,1])
    with col2:
        logo_path = "assets/logo-transparent3.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=350)
        else:
            st.warning("Logo file not found at assets/logo-transparent3.png")
            st.title("Arsipy")

    # Create tabs
    tab1, tab5, tab3, tab2, tab4 = st.tabs([
        "💬 Chatbot", "🌐 Web Insights", "❓ Panduan", "ℹ️ Tentang", "📚 Resources"
    ])

    # Initialize web searcher (used in public tabs)
    web_searcher = ArchivalWebSearch()

    # --- Tab 1: Chatbot ---
    with tab1:
        # --- LLM Selection and Instantiation ---
        # Define model options available to the user
        model_options = {
            "Compound Beta (Groq - Default)": "compound-beta",
            "Llama-4-maverick (Groq - Powerful)": "meta-llama/llama-4-maverick-17b-128e-instruct",
            "Gemini 2.5 Flash (Google)": "gemini-2.5-flash-preview-04-17",
            "DeepSeek V3 (HuggingFace)": "deepseek-ai/DeepSeek-V3-0324",
        }
        default_model_display_key = "Compound Beta (Groq - Default)"
        if default_model_display_key not in model_options:
            default_model_display_key = list(model_options.keys())[0]

        selected_model_display = st.selectbox(
            "Pilih Model AI (Dasar)",
            options=list(model_options.keys()),
            index=list(model_options.keys()).index(default_model_display_key),
            key='model_selector',
            help=f"Model dasar untuk RAG/fallback. Sintesis diperkaya menggunakan {DEFAULT_ENRICHMENT_MODEL_ID}."
        )
        selected_model_id = model_options[selected_model_display]

        # Instantiate selected base LLM (or use default)
        try:
            # Use session state to cache the instantiated LLM based on selection
            if 'current_base_llm' not in st.session_state or st.session_state.get('selected_model_id') != selected_model_id:
                st.session_state.current_base_llm = get_llm_model(selected_model_id)
                st.session_state.selected_model_id = selected_model_id
                logger.info(f"Chatbot base LLM instantiated: {selected_model_id}")
            current_base_llm = st.session_state.current_base_llm
        except Exception as model_init_err:
            st.error(f"Gagal memuat model {selected_model_display}: {model_init_err}")
            current_base_llm = default_llm # Fallback to the initially passed default LLM
            logger.error(f"Base LLM init failed for {selected_model_id}. Falling back.")

        # Instantiate enrichment LLM (cached in session state)
        try:
            if 'enrichment_llm' not in st.session_state:
                st.session_state.enrichment_llm = get_llm_model(DEFAULT_ENRICHMENT_MODEL_ID)
                logger.info(f"Enrichment LLM instantiated: {DEFAULT_ENRICHMENT_MODEL_ID}")
            enrichment_llm = st.session_state.enrichment_llm
        except Exception as enrich_llm_err:
            st.error(f"Gagal memuat model enrichment ({DEFAULT_ENRICHMENT_MODEL_ID}): {enrich_llm_err}")
            enrichment_llm = current_base_llm # Fallback enrichment to current base LLM
            logger.error(f"Enrichment LLM init failed. Falling back to base LLM.")
        # --- End LLM Instantiation ---

        # Check vectorstore status using the helper function
        vectorstore = st.session_state.get('vectorstore')
        vectorstore_ready = _is_vectorstore_ready_and_populated(vectorstore)

        if not vectorstore_ready:
            st.info("👋 Selamat datang! Basis data dokumen lokal kosong/tidak dapat diakses. Pertanyaan akan dijawab menggunakan pencarian web. Admin dapat mengunggah dokumen.")
        else:
             st.info("👋 Selamat datang! Ajukan pertanyaan tentang dokumen yang telah diunggah atau topik kearsipan umum.")

        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history (Keep existing logic)
        st.markdown("---")
        st.subheader("Riwayat Percakapan")
        if not st.session_state.chat_history:
            st.caption("Belum ada percakapan.")
        else:
            for i, (q, response_data) in enumerate(st.session_state.chat_history):
                 with st.container(border=False):
                    st.markdown(f"👤 **Anda:** {q}") # Display original query
                    st.markdown(f"🤖 **Arsipy:**")
                    st.markdown(response_data.get('result', "_Error menampilkan jawaban._"), unsafe_allow_html=True)

                    # Display sources (simplified)
                    with st.expander("Lihat Sumber", expanded=False):
                        local_sources_hist = response_data.get('source_documents', [])
                        web_sources_hist = response_data.get('web_source_documents', [])

                        if local_sources_hist:
                            st.markdown("**Sumber Dokumen Lokal:**")
                            seen_local_sources = set()
                            for doc in local_sources_hist:
                                source_name = doc.metadata.get('source', 'Dokumen Lokal')
                                if source_name not in seen_local_sources:
                                     st.caption(f"- {source_name}")
                                     seen_local_sources.add(source_name)
                            if web_sources_hist: st.divider() # Add divider only if both exist

                        if web_sources_hist:
                            st.markdown("**Sumber Web:**")
                            for src in web_sources_hist:
                                title = src.get('title', 'Sumber Web')
                                url = src.get('url', '#')
                                source_site = src.get('source', 'N/A')
                                st.caption(f"- [{title}]({url}) ({source_site})")

                        if not local_sources_hist and not web_sources_hist:
                             st.caption("Tidak ada sumber spesifik yang dirujuk.")
                    st.divider()

        # Chat input form
        with st.form(key='chat_form'):
            prompt_raw = st.text_input("Masukkan pertanyaan Anda:", key='question_input', placeholder="Contoh: Apa saja komponen utama ISO 15489?")
            submit_button = st.form_submit_button("Kirim Pertanyaan")

        # =================== CORE PROCESSING LOGIC ===================
        if submit_button and prompt_raw:
            # --- Sanitize Input ---
            prompt_sanitized = security_manager.sanitize_input(prompt_raw)
            if prompt_sanitized != prompt_raw:
                logger.warning("User input sanitized.")
                # Optionally inform user? For now, just log.

            try:
                # --- Query Optimization & Clarification (Using Agents) ---
                enhanced_query = search_agent.optimize_query(prompt_sanitized)
                logger.info(f"Original query: '{prompt_sanitized}', Enhanced query: '{enhanced_query}'")

                if response_agent.should_ask_clarification(enhanced_query):
                    questions = response_agent.generate_clarifying_questions(enhanced_query)
                    if questions:
                        st.info("Untuk hasil yang lebih akurat, mohon perjelas:")
                        for q_clarify in questions:
                            st.write(f"• {q_clarify}")
                        st.stop() # Stop processing until clarification

                # --- Generate Response using Orchestrator ---
                spinner_messages = [
                    "Menganalisis pertanyaan Anda...",
                    "Mencari di dokumen lokal..." if vectorstore_ready else "Melewati pencarian dokumen lokal...",
                    "Memeriksa sumber web...",
                    "Mensintesis jawaban..."
                ]
                final_response = {}
                with st.spinner(spinner_messages[0]): # Initial message
                    start_time = time.time()

                    # Call the main orchestrator function with instantiated LLMs
                    final_response = get_orchestrated_response(
                        query=enhanced_query,
                        base_llm=current_base_llm, # Pass instantiated base LLM
                        enrichment_llm=enrichment_llm, # Pass instantiated enrichment LLM
                        vectorstore=vectorstore,
                        web_searcher=web_searcher,
                        attempt_rag=vectorstore_ready
                    )
                    elapsed_time = time.time() - start_time

                # --- Post-processing with Agents ---
                if 'result' in final_response and final_response['result']:
                    # Identify knowledge gaps
                    gaps = knowledge_agent.identify_knowledge_gaps(
                        enhanced_query,
                        final_response['result']
                    )
                    if gaps:
                        suggested_sources = knowledge_agent.suggest_sources(enhanced_query)
                        with st.expander("📚 Saran Pengembangan Pengetahuan"):
                            st.write("Area yang dapat ditambahkan/ditingkatkan:")
                            for gap in gaps: st.write(f"• {gap}")
                            if suggested_sources:
                                st.write("\nSumber yang disarankan:")
                                for source in suggested_sources:
                                    st.write(f"• {source['name']} (Reliabilitas: {source['reliability']})")

                    # Show related queries
                    related = search_agent.suggest_related_queries(enhanced_query)
                    if related:
                        with st.expander("🔍 Pertanyaan Terkait"):
                            for rel_query in related: st.write(f"• {rel_query}")

                # Store response (using original prompt for history key) and rerun
                st.session_state.chat_history.append((prompt_raw, final_response))
                st.success(f"Jawaban dihasilkan dalam {elapsed_time:.2f} detik.")
                time.sleep(0.5) # Brief pause before rerun
                st.rerun()

            except Exception as e:
                st.error(f"Terjadi kesalahan tak terduga: {str(e)}")
                logger.error(traceback.format_exc())
                # Store error message in history
                st.session_state.chat_history.append((prompt_raw, {'result': f"_Error: {str(e)}_", 'source_documents': [], 'web_source_documents': []}))
                st.rerun()

        # Clear chat history button
        if st.session_state.chat_history:
             if st.button("Hapus Riwayat Percakapan", key="clear_chat_hist_button"):
                st.session_state.chat_history = []
                st.success("Riwayat percakapan dihapus.")
                time.sleep(1)
                st.rerun()
    
    # --- Footer ---
    st.markdown("---")
    st.markdown("Dibangun oleh Adnuri Mohamidi dengan bantuan AI 🧡", help="cyberariani@gmail.com")

    with tab2: # Tentang
        st.write("""
        ### 🎯 Tentang Arsipy
        Arsipy adalah asisten AI yang dirancang untuk membantu para profesional kearsipan dan manajemen rekod dalam mengakses, memahami, dan menerapkan informasi dari manual, standar, dan panduan praktik terbaik. Aplikasi ini memanfaatkan kekuatan Retrieval-Augmented Generation (RAG) untuk memberikan jawaban yang relevan dari dokumen yang diunggah, dan secara cerdas **memperkaya** jawaban tersebut dengan konteks dari sumber web terpercaya (ICA, SAA, ANRI, ISO) bila diperlukan, atau **mencari jawaban** dari web jika informasi tidak tersedia secara lokal.

        ### 🔍 Fitur Utama
        - **Chatbot Berbasis RAG:** Menjawab pertanyaan berdasarkan dokumen lokal (PDF, TXT).
        - **Enrichment Cerdas:** Memperkaya jawaban lokal dengan konteks relevan dari web.
        - **Web Fallback:** Mencari jawaban dari sumber web terpercaya jika dokumen lokal tidak memadai.
        - **Web Insights:** Menganalisis topik umum, URL web, atau dokumen yang diunggah secara terpisah.
        - **Manajemen Dokumen:** Panel admin untuk mengunggah dan mengelola basis pengetahuan lokal.

        ### 💻 Teknologi
        - **Backend**: Python, ChromaDB, LangChain
        - **AI Models**: Groq (Llama 4 Maverick dan compound-beta), Google AI (Gemini, Embeddings), HuggingFace (DeepSeek)
        - **Frontend**: Streamlit
        - **Web Scraping**: Requests, BeautifulSoup
        """)
        st.subheader("⚠️ Penting")
        st.info("""
        * Aplikasi ini tidak menyimpan riwayat percakapan Anda secara permanen (hanya dalam sesi).
        * Jangan mengunggah dokumen yang berisi informasi yang sangat sensitif atau rahasia.
        * Jawaban dihasilkan oleh AI dan mungkin memerlukan verifikasi silang, terutama untuk keputusan kritis.
        * Berminat membangun solusi serupa untuk manajemen dokumen di organisasi Anda? Hubungi pengembang.
        """)

    with tab3: # Panduan
        st.markdown("""
        ### Chat dengan Arsipy (Tab 💬 Chatbot)
        1.  **Ajukan Pertanyaan:** Masukkan pertanyaan Anda tentang topik kearsipan atau konten dokumen yang telah diunggah di kotak input.
        2.  **Proses Jawaban:**
            *   Arsipy pertama-tama mencari jawaban di **dokumen lokal** yang diunggah Admin.
            *   Jika jawaban ditemukan dan dianggap memadai, Arsipy akan **memeriksa sumber web terpercaya** (ICA, SAA, ANRI, ISO) untuk **konteks tambahan** yang relevan. Jawaban akhir akan **mensintesis** informasi dari kedua sumber secara alami.
            *   Jika jawaban lokal **tidak ditemukan atau tidak memadai**, Arsipy akan **mencari jawaban langsung** dari sumber web terpercaya (fallback).
        3.  **Hasil:** Jawaban akan ditampilkan beserta sumber yang digunakan (dokumen lokal dan/atau web). Sumber dapat dilihat dengan mengklik "Lihat Sumber".
        4.  **Interaksi:** Gunakan riwayat percakapan untuk pertanyaan lanjutan atau klarifikasi.

        ### Web Insights (Tab 🌐 Web Insights)
        Gunakan tab ini untuk analisis mendalam tentang topik spesifik, konten URL, atau dokumen yang Anda unggah langsung di tab ini, terpisah dari chatbot utama.
        1.  Pilih jenis analisis (Pencarian Online, Konten Web, Konten Dokumen).
        2.  Masukkan input yang diperlukan (kata kunci, URL, atau unggah file).
        3.  Klik tombol "Buat Analisis...", "Analisis Konten Web", atau "Analisis Dokumen".

        ### Tips Hasil Optimal
        - Gunakan bahasa Indonesia yang jelas dan spesifik.
        - Fokus pada satu topik utama per pertanyaan di chatbot.
        - Periksa sumber yang diberikan untuk verifikasi.

        ### ❗ Troubleshooting
        - **Tidak Merespon:** Refresh halaman (F5 / Cmd+R). Periksa koneksi internet.
        - **Jawaban Kurang Tepat:** Coba formulasikan ulang pertanyaan Anda. Pastikan dokumen relevan telah diunggah oleh Admin.
        - **Error:** Mungkin ada masalah sementara dengan API AI atau web scraping. Coba lagi nanti atau laporkan ke Admin.
        """)

    with tab4: # Resources
        st.title("📚 Sumber Dokumen & Web")
        st.markdown("""
            Arsipy dirancang untuk memberikan informasi dari sumber-sumber berikut:

            ### 📄 Dokumen Lokal (Diunggah oleh Admin)
            Ini adalah basis pengetahuan utama Arsipy. Sistem menjawab pertanyaan berdasarkan konten dokumen PDF atau TXT yang diunggah melalui Panel Admin. Contoh sumber yang ideal:
            - Modul Ajar Kearsipan Universitas Terbuka (UT) [url](http://repository.ut.ac.id)
            - Regulasi Kearsipan Indonesia (UU, PP, Peraturan ANRI)
            - WBG Records Management Roadmap [url](https://www.worldbank.org/en/archive/aboutus/records-management-roadmap)
            - Archive Principles and Practice, Introduction to archives for non-archivists [url](https://cdn.nationalarchives.gov.uk/documents/archives/archive-principles-and-practice-an-introduction-to-archives-for-non-archivists.pdf)
            - Buku "Archive Fever: A Freudian Impression", karya Derrida dan "Manuscripts and Archives: Comparative views on Record-keeping" oleh Alessandro Bausi et al. (editor)

            *Catatan: Kualitas dan kelengkapan jawaban sangat bergantung pada relevansi dan kualitas dokumen yang diunggah.*

            ### 🌐 Sumber Web Terpercaya (Untuk Fallback & Enrichment)
            Jika jawaban tidak ditemukan di dokumen lokal, atau jika jawaban lokal dapat diperkaya dengan konteks tambahan, Arsipy akan secara otomatis mencari dari domain berikut:
            - **International Council on Archives (ICA):** `ica.org`
            - **Society of American Archivists (SAA):** `archivists.org`
            - **Arsip Nasional Republik Indonesia (ANRI):** `anri.go.id`
            - **International Organization for Standardization (ISO):** `iso.org` (Terutama untuk standar seperti ISO 15489, ISO 30300 series, dll.)

            Arsipy memprioritaskan informasi dari dokumen lokal tetapi menggunakan sumber web ini secara cerdas untuk memberikan jawaban yang lebih komprehensif dan terkini bila memungkinkan.
        """)

    # --- Tab 5 (Web Insights) ---
    with tab5:
        # Add REFINED CSS styling for Web Insights
        st.markdown("""
            <style>
            /* --- Web Insights Tab Styling --- */

            /* Overall container for each analysis section */
            .response-section {
                background-color: #262730; /* Dark background */
                border: 1px solid #383838; /* Slightly lighter border */
                border-radius: 8px;        /* Rounded corners */
                padding: 18px 22px;       /* More padding (top/bottom, left/right) */
                margin: 18px 0;           /* Vertical margin between sections */
                box-shadow: 0 1px 3px rgba(0,0,0,0.2); /* Subtle shadow for depth */
            }

            /* Section Title (e.g., "Ringkasan Eksekutif") */
            .section-title {
                color: #2A9DF4; /* Brighter, more modern blue */
                font-size: 1.1em; /* Slightly larger */
                font-weight: 600; /* Semi-bold weight */
                margin-top: 0;    /* Remove default top margin */
                margin-bottom: 2px; /* Space below title */
                padding-bottom: 6px; /* Space between text and border */
                border-bottom: 1px solid #444; /* Subtle bottom border */
            }

            /* Main content area within a section */
            .section-content {
                color: #E0E0E0;       /* Softer off-white text */
                font-size: 0.98em;    /* Slightly adjust base font size */
                line-height: 1.5;     /* Reduced line height for tighter lines */
                text-align: justify;  /* Justified text alignment */
                white-space: pre-wrap;/* Preserve line breaks from LLM output */
            }

            /* --- Specific Element Styling INSIDE .section-content --- */

            /* Paragraphs: Control spacing between paragraphs */
            .section-content p {
                margin-top: 0.5em;    /* Space above paragraph */
                margin-bottom: 0.3em; /* Reduced space below paragraph */
            }
            /* Remove margin for first/last paragraphs for cleaner section spacing */
            .section-content p:first-child {
                margin-top: 0;
            }
             .section-content p:last-child {
                margin-bottom: 0;
            }

            /* Lists (Unordered and Ordered): Tighter spacing */
            .section-content ul,
            .section-content ol {
                margin-top: 0.6em;    /* Space above list */
                margin-bottom: 0.3em; /* Space below list */
                padding-left: 22px;   /* Indentation for list items */
            }

            /* List Items: *** FURTHER REDUCED space between items *** */
            .section-content li {
                margin-top: 0;        /* Ensure no extra top margin */
                margin-bottom: 0.3em; /* <<< SIGNIFICANTLY REDUCED bottom margin */
                padding-left: 5px;    /* Small space between bullet/number and text */
            }
            /* Optional: If list items contain paragraphs, control their spacing too */
            .section-content li p {
                 margin-top: 0.1em;
                 margin-bottom: 0.32em; /* Keep paragraphs within list items tight */
            }


            /* Optional: Style bold text for emphasis */
            .section-content strong,
            .section-content b {
                color: #F0F0F0; /* Slightly brighter bold text */
                font-weight: 600;
            }

            /* Optional: Style code blocks if they appear */
            .section-content pre,
            .section-content code {
                background-color: #1E1E1E;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 8px 12px;
                font-size: 0.9em;
                white-space: pre-wrap; /* Ensure wrapping */
                word-wrap: break-word; /* Break long words */
                display: block; /* Make pre take block layout */
                margin: 0.8em 0;
            }
            .section-content code:not(pre code) { /* Inline code */
                 padding: 0.2em 0.4em;
                 margin: 0;
                 font-size: 85%;
                 background-color: rgba(110,118,129,0.4);
                 border-radius: 3px;
                 display: inline; /* Make inline code display inline */
            }

            </style>
        """, unsafe_allow_html=True)

        # Instantiate LLM for Web Insights (cached)
        try:
            if 'web_insights_llm' not in st.session_state:
                # Use the default enrichment model for these potentially complex tasks
                st.session_state.web_insights_llm = get_llm_model(DEFAULT_ENRICHMENT_MODEL_ID)
                logger.info(f"Web Insights LLM instantiated: {DEFAULT_ENRICHMENT_MODEL_ID}")
            web_insights_llm = st.session_state.web_insights_llm
        except Exception as wi_llm_err:
            st.error(f"Gagal memuat model AI untuk Web Insights: {wi_llm_err}")
            web_insights_llm = None # Disable Web Insights if LLM fails

        analysis_type = st.radio(
            "Pilih Sumber Analisis",
            ["Pencarian Online", "Konten Web (URL)", "Konten Dokumen (Unggah)"],
            key="web_insights_analysis_type",
            horizontal=True,
            disabled=(web_insights_llm is None) # Disable radio if LLM failed
        )

        # --- Web Insights: Online Search ---
        if analysis_type == "Pencarian Online":
            search_query_raw = st.text_input("Masukkan topik atau kata kunci pencarian:", key="online_search_query", placeholder="Contoh: ISO 15489, preservasi digital", disabled=(web_insights_llm is None))
            citation_style = st.selectbox(
                "Gaya Sitasi untuk Referensi",
                ["APA", "Chicago", "Harvard", "MLA"],
                key="online_search_citation",
                disabled=(web_insights_llm is None)
            )

            if search_query_raw and st.button("Buat Analisis dari Web", key="online_search_button", disabled=(web_insights_llm is None)):
                # --- Sanitize Input ---
                search_query = security_manager.sanitize_input(search_query_raw)
                if search_query != search_query_raw: logger.warning("Web Insights search query sanitized.")

                try:
                    with st.spinner("🔍 Menganalisis topik dari web..."):
                        # Use the pre-instantiated LLM
                        model = web_insights_llm
                        # Keep the detailed prompt as provided
                        prompt = f"""Anda adalah asisten riset ahli di bidang kearsipan dan manajemen rekod. Tugas Anda adalah melakukan riset mendalam (berdasarkan pengetahuan internal Anda atau simulasi pencarian web) mengenai topik berikut: "{search_query}"

                        Sajikan hasil riset Anda dalam format terstruktur berikut, menggunakan penanda bagian yang **tepat** seperti di bawah ini. Pastikan setiap bagian diisi dengan informasi yang relevan dan akurat.

                        === DEFINISI ===
                        Berikan definisi yang jelas dan ringkas tentang topik "{search_query}". Jelaskan konsep utamanya.

                        === PEMBAHASAN ===
                        Jelaskan topik "{search_query}" secara mendalam. Mencakup aspek-aspek penting, relevansi dalam konteks kearsipan/manajemen rekod, prinsip dasar, tantangan umum, dan perkembangan terkini jika ada. Berikan analisis yang komprehensif.

                        === STANDAR & PRAKTIK TERBAIK ===
                        Sebutkan standar internasional atau nasional yang relevan (misal, ISO 15489, ISO 30300 series, SNI terkait), serta praktik terbaik (best practices) yang umum diadopsi berkaitan dengan topik "{search_query}". Jelaskan secara singkat relevansi masing-masing standar/praktik.

                        === REFERENSI ===
                        Sediakan daftar sumber (minimal 3-5 jika memungkinkan) yang kredibel (buku, artikel jurnal, standar resmi, situs web organisasi profesional seperti ICA/SAA/ANRI/ISO) yang mendukung analisis Anda di bagian sebelumnya. Format daftar referensi ini secara akurat menggunakan gaya sitasi **{citation_style}**.

                        Instruksi Tambahan:
                        - Gunakan Bahasa Indonesia yang formal, jelas, dan profesional.
                        - Pastikan informasi yang disajikan akurat dan relevan dengan bidang kearsipan/manajemen rekod.
                        - Fokus pada penyajian informasi yang berguna bagi praktisi atau akademisi di bidang ini.
                        - Jangan sertakan header atau footer tambahan di luar format yang diminta.

                        Hasil Analisis (Bahasa Indonesia):
                        """

                        logger.info(f"Invoking LLM ({getattr(model, 'model_name', 'Unknown')}) for Online Search: {search_query}")
                        response = model.invoke(prompt)
                        logger.info(f"LLM response received (type: {type(response)})")

                        # Keep the robust response handling and parsing logic as provided
                        if response and hasattr(response, 'content'):
                            content = response.content.strip()
                            # ... (rest of the parsing and display logic) ...
                            sections = {}
                            current_section = None
                            current_content = []
                            section_pattern = re.compile(r"^===\s*([\w\s&]+?)\s*===$")
                            processed_lines = content.replace('\r\n', '\n').split('\n')

                            for line in processed_lines:
                                match = section_pattern.match(line.strip())
                                if match:
                                    if current_section is not None: sections[current_section] = '\n'.join(current_content).strip()
                                    current_section = match.group(1).strip().upper().replace(' ', '_')
                                    current_content = []
                                else:
                                    if current_section is not None: current_content.append(line)
                            if current_section is not None: sections[current_section] = '\n'.join(current_content).strip()

                            expected_keys = ['DEFINISI', 'PEMBAHASAN', 'STANDAR_&_PRAKTIK_TERBAIK', 'REFERENSI']
                            displayed_something = False
                            for key in expected_keys:
                                section_title = key.replace('_', ' ').title()
                                section_content_parsed = sections.get(key, "").strip()
                                if section_content_parsed:
                                    st.markdown(f"<div class='response-section'><h3 class='section-title'>{section_title}</h3><div class='section-content'>{section_content_parsed}</div></div>", unsafe_allow_html=True)
                                    displayed_something = True
                            if not displayed_something:
                                 st.warning("Format respons AI tidak sesuai. Menampilkan mentah:")
                                 st.markdown(f"<div class='response-section'><h3 class='section-title'>Respons Mentah</h3><div class='section-content' style='white-space: pre-wrap;'>{content}</div></div>", unsafe_allow_html=True)

                            references_content = sections.get('REFERENSI', '').strip()
                            if references_content:
                                # ... (download button logic) ...
                                col_ref1, col_ref2 = st.columns([3,1])
                                with col_ref1: st.caption("Verifikasi referensi disarankan.")
                                with col_ref2:
                                    try:
                                        st.download_button(label="📥 Unduh Referensi", data=references_content, file_name=f"ref_{search_query[:20].replace(' ','_')}.txt", mime="text/plain")
                                    except Exception as dl_error: st.error("Gagal buat tombol unduh.")

                        elif response: st.error("Respons AI tidak dapat diproses.")
                        else: st.error("Tidak ada respons dari AI.")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat analisis online: {str(e)}")
                    logger.error(f"Online Search error: {traceback.format_exc()}")

        # --- Web Insights: URL Content ---
        elif analysis_type == "Konten Web (URL)":
            url_raw = st.text_input("Masukkan URL halaman web:", key="web_content_url", placeholder="https://www.example.com/article", disabled=(web_insights_llm is None))
            if url_raw and st.button("Analisis Konten Web", key="web_content_button", disabled=(web_insights_llm is None)):
                 # --- Sanitize Input ---
                url = security_manager.sanitize_input(url_raw) # Basic sanitization
                if url != url_raw: logger.warning("Web Insights URL sanitized.")

                try:
                    with st.spinner(f"Mengambil konten dari {url}..."):
                        content = fetch_url_content(url) # Use the existing utility
                        if not content or len(content) < 50:
                            st.error("Tidak dapat mengekstrak konten dari URL.")
                            st.stop()

                    with st.spinner("🔍 Menganalisis konten web..."):
                        model = web_insights_llm # Use pre-instantiated LLM
                        prompt = f"""Anda adalah analis konten web ahli, khususnya dalam mengevaluasi relevansi dan isi halaman web untuk bidang kearsipan dan manajemen rekod.

                        Tugas Anda adalah menganalisis teks berikut yang diekstrak dari URL: {url}

                        Konten Web (Maks 8000 karakter):
                        ---
                        {content[:8000]}
                        ---

                        Sajikan hasil analisis Anda dalam format terstruktur berikut, menggunakan penanda bagian yang **tepat** seperti di bawah ini. Pastikan setiap bagian diisi dengan informasi yang relevan berdasarkan teks yang diberikan.

                        === RINGKASAN UTAMA ===
                        Berikan ringkasan singkat (2-4 kalimat) mengenai isi utama dari konten web tersebut.

                        === TOPIK/TEMA KUNCI ===
                        Identifikasi dan sebutkan topik atau tema utama yang dibahas dalam teks. Gunakan bullet points jika lebih dari satu.

                        === RELEVANSI DENGAN KEARSIPAN/MANAJEMEN REKOD ===
                        Evaluasi dan jelaskan sejauh mana konten ini relevan dengan bidang kearsipan, manajemen rekod, preservasi digital, standar terkait, atau topik sejenis. Jika tidak relevan, nyatakan demikian.

                        === POIN PENTING/KESIMPULAN ===
                        Sebutkan poin-poin penting, argumen utama, atau kesimpulan yang disampaikan dalam teks. Gunakan bullet points jika perlu.

                        Instruksi Tambahan:
                        - Gunakan Bahasa Indonesia yang formal dan jelas.
                        - Fokus analisis **hanya** pada teks yang diberikan di atas. Jangan gunakan pengetahuan eksternal.
                        - Jika suatu bagian tidak dapat diisi karena informasi tidak ada dalam teks, tulis "Informasi tidak ditemukan dalam teks yang diberikan."
                        - Jangan sertakan header atau footer tambahan di luar format yang diminta.

                        Hasil Analisis (Bahasa Indonesia):
                        """
                        response = model.invoke(prompt)

                        if response and hasattr(response, 'content'):
                            analysis_content = response.content.strip()
                            st.success("Analisis selesai!")
                            st.markdown(f"<div class='response-section'><h3 class='section-title'>Analisis Konten Web</h3><div class='section-content'>{analysis_content}</div></div>", unsafe_allow_html=True)
                        else: st.error("Tidak ada analisis yang dihasilkan.")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat menganalisis URL: {str(e)}")
                    logger.error(f"Web Content Analysis error: {traceback.format_exc()}")

        # --- Web Insights: Document Content ---
        elif analysis_type == "Konten Dokumen (Unggah)":
            uploaded_file = st.file_uploader(
                "Unggah dokumen (PDF/TXT):",
                type=["pdf", "txt"],
                key="doc_analysis_uploader",
                disabled=(web_insights_llm is None)
            )
            if uploaded_file and st.button("Analisis Dokumen", key="doc_analysis_button", disabled=(web_insights_llm is None)):
                # NOTE: No sanitization applied to uploaded file content itself. Assumed trusted source in this context.
                try:
                    with st.spinner(f"📄 Memproses dokumen {uploaded_file.name}..."):
                        text = get_document_text(uploaded_file) # Use existing utility

                    with st.spinner("🔍 Menganalisis dokumen..."):
                        model = web_insights_llm # Use pre-instantiated LLM
                        prompt = f"""Anda adalah analis dokumen ahli, dengan spesialisasi dalam mengevaluasi dokumen (seperti artikel, laporan, modul, standar) dalam konteks kearsipan dan manajemen rekod.

                        Tugas Anda adalah menganalisis teks berikut yang diekstrak dari dokumen yang diunggah: {uploaded_file.name}

                        Konten Dokumen (Maks 8000 karakter):
                        ---
                        {text[:8000]}
                        ---

                        Sajikan hasil analisis Anda dalam format terstruktur berikut, menggunakan penanda bagian yang **tepat** seperti di bawah ini. Pastikan setiap bagian diisi dengan informasi yang relevan berdasarkan teks dokumen yang diberikan.

                        === RINGKASAN EKSEKUTIF ===
                        Berikan ringkasan eksekutif (3-5 kalimat) yang mencakup tujuan, cakupan, dan temuan/kesimpulan utama dari dokumen tersebut.

                        === TOPIK UTAMA & SUB-TOPIK ===
                        Identifikasi topik utama yang dibahas. Jika memungkinkan, sebutkan juga sub-topik penting di bawah topik utama. Gunakan struktur hierarki atau bullet points.

                        === ARGUMEN/TEMUAN KUNCI ===
                        Sebutkan argumen utama, temuan penelitian, rekomendasi, atau poin-poin kunci yang paling signifikan dari dokumen ini. Gunakan bullet points.

                        === POTENSI APLIKASI/IMPLIKASI DALAM KEARSIPAN ===
                        Jelaskan bagaimana isi dokumen ini dapat diterapkan atau apa implikasinya bagi praktik, teori, atau standar dalam bidang kearsipan dan manajemen rekod.

                        Instruksi Tambahan:
                        - Gunakan Bahasa Indonesia yang formal, akademis, dan jelas.
                        - Fokus analisis **hanya** pada teks dokumen yang diberikan. Jangan gunakan pengetahuan eksternal.
                        - Jika suatu bagian tidak dapat diisi karena informasi tidak ada dalam teks, tulis "Informasi tidak ditemukan dalam teks yang diberikan."
                        - Jangan sertakan header atau footer tambahan di luar format yang diminta.

                        Hasil Analisis (Bahasa Indonesia):
                        """
                        response = model.invoke(prompt)

                        if response and hasattr(response, 'content'):
                            analysis_content = response.content.strip()
                            st.success("Analisis selesai!")
                            st.markdown(f"<div class='response-section'><h3 class='section-title'>Analisis Dokumen: {uploaded_file.name}</h3><div class='section-content'>{analysis_content}</div></div>", unsafe_allow_html=True)
                        else: st.error("Tidak ada analisis yang dihasilkan.")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat menganalisis dokumen: {str(e)}")
                    logger.error(f"Document Analysis error: {traceback.format_exc()}")
    pass
# --- Utility Instantiation ---
cache_manager = CacheManager()
security_manager = SecurityManager()
system_monitor = SystemMonitor()

# Removed unused process_chat_query function

# --- Vectorstore Initialization ---
# @cache_resource # Caching vectorstore can be complex due to connection state
def initialize_or_load_vectorstore() -> Optional[Chroma]:
    """Initialize or load the vector store. Returns None on failure."""
    try:
        # Ensure GOOGLE_API_KEY is available for embeddings
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            st.error("❌ Kunci API Google (untuk embeddings) tidak ditemukan!")
            logger.error("GOOGLE_API_KEY not found, cannot initialize embeddings.")
            return None

        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_ID, google_api_key=google_api_key)
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings
        )
        # Perform a quick check
        try:
            vectorstore._collection.peek(limit=1)
            logger.info(f"Vectorstore loaded/initialized successfully from {CHROMA_DB_DIR}")
            return vectorstore
        except Exception as vs_conn_err:
             logger.error(f"Vectorstore loaded but connection check failed: {vs_conn_err}")
             st.error(f"Gagal terhubung ke database vektor di {CHROMA_DB_DIR}.")
             return None

    except ImportError as imp_err:
         logger.error(f"Import error during vectorstore init (check chromadb, google-generativeai): {imp_err}")
         st.error(f"Kesalahan library saat memuat database: {imp_err}")
         return None
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Gagal memuat basis data vektor: {e}")
        return None

# --- URL Fetching Utility ---
def fetch_url_content(url: str) -> str:
    """Fetch and extract meaningful content from URL, handling HTML and PDF."""
    # (Keep fetch_url_content function definition as provided)
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.8,*/*;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9,id;q=0.8',
            'Connection': 'keep-alive', 'DNT': '1', 'Upgrade-Insecure-Requests': '1'
        }
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        response = requests.get(url, headers=headers, timeout=20, verify=True, allow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        logger.info(f"Fetching URL: {url}, Content-Type: {content_type}")

        if 'application/pdf' in content_type:
            logger.info("Detected PDF content.")
            import io
            try:
                pdf_file = io.BytesIO(response.content)
                doc = fitz.open(stream=pdf_file, filetype="pdf")
                text = "".join(page.get_text("text", flags=fitz.TEXT_INHIBIT_SPACES) + "\n" for page in doc)
                doc.close()
                logger.info(f"Extracted {len(text)} characters from PDF.")
                return text.strip()
            except Exception as pdf_err:
                 logger.error(f"Error processing PDF from URL {url}: {pdf_err}")
                 raise Exception(f"Gagal memproses PDF dari URL: {pdf_err}")

        elif 'text/html' in content_type:
            logger.info("Detected HTML content.")
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'button', 'iframe', 'img', 'svg', 'link', 'meta']):
                tag.decompose()
            main_content = soup.find('main') or soup.find('article') or \
                           soup.find('div', id='content') or \
                           soup.find('div', class_=lambda c: c and any(k in c.lower() for k in ['content', 'main', 'article', 'body', 'post']))
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
            else:
                text = soup.body.get_text(separator='\n', strip=True) if soup.body else ""
            text = re.sub(r'\n{3,}', '\n\n', text)
            logger.info(f"Extracted {len(text)} characters from HTML.")
            return text.strip()
        else:
            logger.warning(f"Unsupported content type '{content_type}' for URL {url}. Attempting text decode.")
            try: return response.text
            except: return ""

    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request error fetching URL {url}: {req_err}")
        raise Exception(f"Gagal mengambil konten dari URL: {req_err}")
    except Exception as e:
        logger.error(f"Unexpected error fetching or parsing URL {url}: {e}")
        logger.error(traceback.format_exc())
        raise Exception(f"Terjadi kesalahan tidak terduga saat memproses URL.")

# --- Main Application Entry Point ---
# @system_monitor.monitor_performance # Apply decorator here if desired
def main() -> None:
    st.set_page_config(
        page_title="Arsipy",
        page_icon="assets/arsipy-favicon.png", # Optional: Add a favicon
        layout="wide"  # <--- THIS IS THE IMPORTANT CHANGE
    )
    #"""Main application entry point"""
    os.environ['ANONYMIZED_TELEMETRY'] = 'False'
    set_verbose(False) # Langchain verbosity
    load_dotenv()

    # --- Landing Page Logic ---
    if 'show_app' not in st.session_state:
         st.session_state['show_app'] = False
    query_params = st.query_params
    if "page" in query_params and query_params.get("page") == "app":
        st.session_state['show_app'] = True
    elif not st.session_state.get('show_app', False):
        show_landing_page()
        return # Stop execution if showing landing page

    logger.info("Loading main application interface.")

    # --- API Key Validation ---
    groq_api_key = os.getenv('GROQ_API_KEY')
    google_api_key = os.getenv("GOOGLE_API_KEY")
    admin_password_env = os.getenv('ADMIN_PASSWORD')

    keys_ok = True
    if not groq_api_key:
        st.error("❌ Kunci API Groq tidak ditemukan! Periksa file .env.")
        keys_ok = False
    if not google_api_key:
        st.error("❌ Kunci API Google tidak ditemukan! Periksa file .env.")
        keys_ok = False
    # Add check for HuggingFace key if DeepSeek models are intended to be default/selectable
    # if not os.getenv('HUGGINGFACE_API_KEY'):
    #     st.warning("⚠️ Kunci API HuggingFace tidak ditemukan. Model DeepSeek tidak akan berfungsi.")
        # keys_ok = False # Decide if this is critical

    if not keys_ok:
        st.warning("Beberapa fitur AI mungkin tidak berfungsi.")

    if not admin_password_env:
         st.sidebar.warning("⚠️ Password Admin tidak diatur di .env. Panel Admin tidak aman.")

    # Set Google API Key for LangChain components that need it directly
    os.environ["GOOGLE_API_KEY"] = google_api_key if google_api_key else ""

    # --- ChromaDB Directory Setup ---
    if not os.path.exists(CHROMA_DB_DIR):
        try:
            os.makedirs(CHROMA_DB_DIR)
            logger.info(f"Created ChromaDB directory: {CHROMA_DB_DIR}")
        except OSError as e:
             st.error(f"Gagal membuat direktori database: {e}. Fitur dokumen lokal tidak akan berfungsi.")

    # --- Session State Initialization ---
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = initialize_or_load_vectorstore()

    if 'doc_processor' not in st.session_state:
        if st.session_state.get('vectorstore'):
            st.session_state.doc_processor = UnifiedDocumentProcessor(st.session_state.vectorstore)
            logger.info("UnifiedDocumentProcessor initialized.")
        else:
            logger.warning("Vectorstore not available. Document processor not initialized.")
            st.session_state.doc_processor = None

    if 'uploaded_file_names' not in st.session_state:
        st.session_state.uploaded_file_names = set()

    # Initialize Agentic Utilities (Using original names)
    if 'search_agent' not in st.session_state:
        st.session_state.search_agent = SearchAgent()
    if 'response_agent' not in st.session_state:
        st.session_state.response_agent = ResponseAgent()
    if 'knowledge_agent' not in st.session_state:
        st.session_state.knowledge_agent = KnowledgeAgent()

    # --- Initialize Default LLM ---
    try:
        # Use session state to cache the default LLM instance
        if 'default_llm' not in st.session_state:
            st.session_state.default_llm = get_llm_model(DEFAULT_MODEL_ID)
            logger.info(f"Default LLM ({DEFAULT_MODEL_ID}) initialized and cached.")
        default_llm = st.session_state.default_llm
    except Exception as e:
        st.error(f"❌ Gagal menginisialisasi model AI default ({DEFAULT_MODEL_ID}): {str(e)}")
        logger.error(traceback.format_exc())
        st.warning("Fungsionalitas AI mungkin terbatas.")
        default_llm = None

    # --- Load Custom CSS ---
    try:
        # Assuming config.toml is in .streamlit/ relative to app.py
        config_path = os.path.join(os.path.dirname(__file__), ".streamlit", "config.toml")
        if os.path.exists(config_path):
            config_css = toml.load(config_path)
            st.markdown(f"<style>{config_css.get('custom_css', {}).get('css', '')}</style>", unsafe_allow_html=True)
        else:
             logger.warning(f"Custom CSS file not found at {config_path}")
    except Exception as e:
        logger.error(f"Error loading config.toml CSS: {e}")

    # --- Setup Sidebar (Handles Admin Login/Controls) ---
    setup_admin_sidebar()

    # --- Show Main Interface ---
    if default_llm:
        show_chat_interface(
            default_llm=default_llm,
            security_manager=security_manager, # Pass instantiated utilities
            search_agent=st.session_state.search_agent,
            response_agent=st.session_state.response_agent,
            knowledge_agent=st.session_state.knowledge_agent
        )
    else:
        st.error("Model AI utama tidak dapat dimuat. Fungsi chatbot tidak tersedia.")
        # Optionally show static tabs even if LLM fails
        # show_static_tabs()

if __name__ == "__main__":
    main()
