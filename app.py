#__import__('pysqlite3')
#import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import time
import fitz  # PyMuPDF
import pandas as pd
import logging
import traceback
import gc
import sys
import shutil
from stqdm import stqdm
from contextlib import contextmanager
from typing import List, Any, Dict, Optional, Set, Tuple, Union
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import create_retrieval_chain, LLMChain, RetrievalQA, StuffDocumentsChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.globals import set_verbose
from dotenv import load_dotenv
from streamlit.runtime.caching import cache_data, cache_resource
from datetime import datetime
import toml
import chromadb
#import sqlite3
from image_analyzer import image_analyzer_main
from huggingface_hub import InferenceClient
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.language_models.llms import LLM
from langchain_core.retrievers import BaseRetriever
import re # Import re for text cleaning
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add these imports at the top
from utils.cache_manager import CacheManager
from utils.security import SecurityManager
from utils.monitoring import SystemMonitor
from document_processor import UnifiedDocumentProcessor

# Change the import at the top
from landing_page import show_landing_page

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin # Ensure urljoin is imported

# New imports for agents
from utils.agentic_ai import SearchAgent, ResponseAgent, KnowledgeAgent

# New import for web search
from utils.web_search import ArchivalWebSearch

class DeepSeekLLM(LLM):
    """Custom LLM class for DeepSeek models from HuggingFace"""

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
        response = self.client.text_generation(
            prompt,
            model=self.model,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            stop_sequences=stop or [],
            **kwargs
        )
        return response

# Modify the get_llm_model function
def get_llm_model(model_name: str) -> Union[LLM, Any]:
    """
    Initialize and return the specified LLM model
    """
    # Simplified for brevity - assume original implementation is correct
    if model_name.startswith("llama3") or model_name == "compound-beta":
        return ChatGroq(
            groq_api_key=os.getenv('GROQ_API_KEY'),
            model_name=model_name
        )
    elif model_name == "deepseek-coder":
         return DeepSeekLLM(
            model="deepseek-ai/DeepSeek-V3-0324",
            api_key=os.getenv('HUGGINGFACE_API_KEY'),
            temperature=0.5,
            max_tokens=512
        )
    elif model_name == "smallthinker":
        return DeepSeekLLM(
            model="PowerInfer/SmallThinker-3B-Preview",
            api_key=os.getenv('HUGGINGFACE_API_KEY'),
            temperature=0.5,
            max_tokens=512
        )
    else:
        # Fallback to a default if needed, or raise error
        logger.warning(f"Model {model_name} not explicitly handled, falling back to compound-beta")
        return ChatGroq(
            groq_api_key=os.getenv('GROQ_API_KEY'),
            model_name="compound-beta"
        )

def get_rag_chain(llm: Union[LLM, Any], retriever: BaseRetriever) -> RetrievalQA:
    """Enhanced RAG chain setup (without fallback/enrichment logic here)"""

    # Define system prompt with hidden reasoning
    SYSTEM_PROMPT = """You are Arsipy, an expert archival documentation assistant.
    Analyze queries using this internal process (do not show in response):
    1. Topic identification
    2. Context evaluation
    3. Evidence gathering
    4. Response formulation

    Keep responses focused, clear, and professional."""

    # Define the QA prompt template with internal reasoning
    QA_CHAIN_PROMPT = ChatPromptTemplate.from_template("""
    System: {system_prompt}

    Context: {context}

    Question: {question}

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
    - Supporting evidence
    - Source citations
    - Additional context (if needed)

    Response in id-ID:
    """)

    # Set up the chain
    llm_chain = LLMChain(
        llm=llm,
        prompt=QA_CHAIN_PROMPT.partial(system_prompt=SYSTEM_PROMPT),
        output_key="answer" # Note: RetrievalQA might expect 'result' by default, check compatibility
    )

    document_prompt = PromptTemplate(
        template="Content: {page_content}\nSumber: {source}",
        input_variables=["page_content", "source"]
    )

    combine_docs_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=document_prompt,
            document_variable_name="context",
            document_separator="\n\n"
        )

    # Using the deprecated RetrievalQA as per the original code structure.
    # Ideally, this should be refactored to use create_retrieval_chain.
    qa_chain = RetrievalQA(
        combine_documents_chain=combine_docs_chain, # Use the created StuffDocumentsChain
        retriever=retriever,
        return_source_documents=True,
    )

    return qa_chain


st.set_page_config(layout="wide")
try:
    config = toml.load(".streamlit/config.toml")
    st.markdown(f"<style>{config.get('custom_css', {}).get('css', '')}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    # logger.warning(".streamlit/config.toml not found. Skipping custom CSS.") # Logger not defined yet here
    config = {}
except Exception as e:
    # logger.error(f"Error loading config.toml: {e}") # Logger not defined yet here
    config = {}

admin_password = os.getenv('ADMIN_PASSWORD')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@contextmanager
def memory_track():
    try:
        gc.collect()
        yield
    finally:
        gc.collect()

# --- Admin Sidebar and Controls ---
def setup_admin_sidebar() -> None:
    """Setup admin authentication and controls in sidebar"""
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False

    with st.sidebar:
        st.title("Admin Panel")

        # Admin authentication
        if not st.session_state.admin_authenticated:
            input_password = st.text_input("Admin Password", type="password", key="admin_pw_input")
            if st.button("Login", key="admin_login_button"):
                # Use the admin password from the .env file
                if input_password and input_password == admin_password:
                    st.session_state.admin_authenticated = True
                    st.success("Admin authenticated!")
                    st.rerun() # Rerun to update sidebar display
                elif not admin_password:
                     st.warning("Admin password not set in .env. Login disabled.")
                else:
                    st.error("Incorrect password")
        else:
            st.write("✅ Admin authenticated")
            if st.button("Logout", key="admin_logout_button"):
                st.session_state.admin_authenticated = False
                # Clear potentially sensitive admin-related session state if needed
                # e.g., del st.session_state['some_admin_data']
                st.rerun() # Rerun to update sidebar display

            # Show admin controls only when authenticated
            st.divider()
            show_admin_controls() # This function contains the admin-only features

def show_admin_controls() -> None:
    """Display admin controls (Document Management) - ONLY shown when authenticated."""
    # This function is only called from setup_admin_sidebar when authenticated
    st.sidebar.header("Document Management")

    # File uploader section
    uploaded_files = st.sidebar.file_uploader(
        "Upload Documents (Admin Only)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="admin_file_uploader" # Unique key
    )

    if uploaded_files:
        st.sidebar.subheader("Document Metadata")
        st.sidebar.info("""
        Please provide metadata for better document organization.
        Example: 'Modul 1, Manajemen Kearsipan di Indonesia, Drs. Syauki Hadiwardoyo'
        """)

        metadata_inputs = {}
        for file in uploaded_files:
            # Use file.id or file.file_id if available and stable across reruns, otherwise name is fallback
            file_key_base = f"{file.name}_{file.size}" # Make key more unique
            with st.sidebar.expander(f"Metadata for {file.name}"):
                metadata_inputs[file.name] = { # Still use name as dict key for lookup
                    'judul': st.text_input(
                        "Judul Modul",
                        key=f"title_{file_key_base}",
                        placeholder="e.g., Manajemen Kearsipan di Indonesia"
                    ),
                    'pengajar': st.text_input(
                        "Nama Pengajar",
                        key=f"author_{file_key_base}",
                        placeholder="e.g., Drs. Syauki Hadiwardoyo"
                    ),
                    'deskripsi': st.text_area(
                        "Deskripsi (Optional)",
                        key=f"desc_{file_key_base}",
                        placeholder="Deskripsi singkat tentang modul ini"
                    )
                }

        # Process documents button
        if st.sidebar.button("Process Documents", key="admin_process_docs_button"):
            # Ensure processor is ready before attempting to process
            if 'doc_processor' in st.session_state and st.session_state.doc_processor:
                 process_uploaded_files(uploaded_files, metadata_inputs)
            else:
                 st.sidebar.error("Document processor not ready. Cannot process files.")

def extract_text_from_pdf(pdf_file: Any) -> str:
    """Extract text content from a PDF file"""
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text()
        if not text.strip():
            # Don't raise error, just return empty string or log warning
            logger.warning(f"Extracted text from PDF '{getattr(pdf_file, 'name', 'unknown')}' is empty.")
            # raise ValueError("Extracted text from PDF is empty")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF '{getattr(pdf_file, 'name', 'unknown')}': {str(e)}")
        raise # Re-raise to be caught by caller
    finally:
        if 'pdf_document' in locals() and pdf_document:
            pdf_document.close()

def get_document_text(file: Any) -> str:
    """Get text content from a file based on its type"""
    try:
        if file.type == "application/pdf":
            text = extract_text_from_pdf(file)
        elif file.type == "text/plain":
            # Ensure correct decoding, handle potential errors
            try:
                text = file.getvalue().decode('utf-8')
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decoding failed for {file.name}, trying latin-1.")
                file.seek(0) # Reset pointer after failed read
                text = file.getvalue().decode('latin-1')
        else:
            raise ValueError(f"Unsupported file type: {file.type}")

        if not text.strip():
            logger.warning(f"Extracted text from {file.name} is empty.")
            # raise ValueError("Extracted text is empty")

        return text
    except Exception as e:
        logger.error(f"Error extracting text from {file.name}: {str(e)}")
        raise # Re-raise

def process_uploaded_files(uploaded_files: List[Any], metadata_inputs: Dict) -> None:
    """Process uploaded files with enhanced metadata handling (Admin action)"""
    # This function is now only triggered by the admin button
    try:
        if not uploaded_files:
            st.sidebar.warning("No files selected for processing")
            return

        if 'doc_processor' not in st.session_state or not st.session_state.doc_processor:
             st.sidebar.error("Document processor not initialized. Cannot process.")
             return

        processed_count = 0
        error_count = 0
        with st.spinner('Processing documents (Admin)...'):
            # Use stqdm for progress bar within the sidebar spinner context
            for file in stqdm(uploaded_files, desc="Processing Files", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
                # Check if file already processed (using a simple name check for this session)
                # A more robust check might involve content hashing if needed
                if file.name not in st.session_state.get('uploaded_file_names', set()):
                    try:
                        metadata = metadata_inputs.get(file.name, {})
                        metadata['source'] = file.name # Ensure source filename is in metadata

                        # Process document using the unified processor
                        result = st.session_state.doc_processor.process_document(
                            file,
                            metadata=metadata
                        )

                        if result['success']:
                            st.sidebar.success(f"Processed: {metadata.get('judul', file.name)}")
                            st.session_state.uploaded_file_names.add(file.name)
                            processed_count += 1
                        else:
                            st.sidebar.error(f"Error processing {file.name}: {result['error']}")
                            error_count += 1
                    except Exception as proc_err:
                         st.sidebar.error(f"Critical error processing {file.name}: {proc_err}")
                         logger.error(f"Critical error during file processing loop for {file.name}: {traceback.format_exc()}")
                         error_count += 1
                         # Optionally continue to next file or stop? Continue for now.
                else:
                    st.sidebar.info(f"Skipped (already processed this session): {file.name}")


        if processed_count > 0:
            st.sidebar.success(f"{processed_count} document(s) processed successfully!")
            # Optionally trigger vectorstore persistence if needed by Chroma setup
            if hasattr(st.session_state.vectorstore, 'persist'):
                try:
                    st.session_state.vectorstore.persist()
                    logger.info("Vectorstore changes persisted.")
                except Exception as persist_err:
                    st.sidebar.error(f"Error persisting vectorstore changes: {persist_err}")
                    logger.error(f"Failed to persist vectorstore: {traceback.format_exc()}")
        elif error_count > 0:
             st.sidebar.warning("Processing complete, but some files had errors.")
        else:
             st.sidebar.info("No new documents were processed.")


    except Exception as e:
        st.sidebar.error(f"An unexpected error occurred during file processing: {str(e)}")
        logger.error(f"Error in process_uploaded_files: {traceback.format_exc()}")

def clear_cache() -> None:
    """Clear all cached data"""
    # This might be an admin function too - consider adding a button in show_admin_controls
    cache_data.clear()
    cache_resource.clear()
    st.success("Application caches cleared.")

# ==============================================================================
# CORE CHAT INTERFACE LOGIC (PUBLIC ACCESS)
# ==============================================================================
def show_chat_interface(llm: Union[LLM, Any]) -> None:
    """Display the main chat interface (Public Access)"""
    # Initialize agents if not present
    if 'search_agent' not in st.session_state:
        st.session_state.search_agent = SearchAgent()
    if 'response_agent' not in st.session_state:
        st.session_state.response_agent = ResponseAgent()
    if 'knowledge_agent' not in st.session_state:
        st.session_state.knowledge_agent = KnowledgeAgent()

    # Add logo
    col1, col2, col3 = st.columns([1,100,1])
    with col2:
        # Check if logo exists
        logo_path = "assets/logo-transparent3.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=350)
        else:
            st.warning("Logo file not found at assets/logo-transparent3.png")
            st.title("Arsipy") # Fallback title

    # Create tabs for the main interface
    tab1, tab5, tab3, tab2, tab4 = st.tabs([
        "💬 Chatbot",
        "🌐 Web Insights",
        "❓ Panduan",
        "ℹ️ Tentang",
        "📚 Resources"
    ])

    # Initialize web searcher (used in public tabs)
    web_searcher = ArchivalWebSearch()

    # --- Tab 1: Chatbot ---
    with tab1:
        # Model selection
        model_options = {
            "compound-beta (Groq - Default)": "compound-beta",
            "Llama3 70B (Groq - Powerful)": "llama3-70b-8192",
            # Add other models as needed
        }
        default_model_key = "compound-beta (Groq - Default)"
        if default_model_key not in model_options:
            default_model_key = list(model_options.keys())[0]

        selected_model_display = st.selectbox(
            "Pilih Model AI (Dasar)",
            options=list(model_options.keys()),
            index=list(model_options.keys()).index(default_model_key),
            key='model_selector',
            help="Model dasar untuk RAG/fallback. Sintesis diperkaya menggunakan Llama3 70B."
        )
        selected_model_id = model_options[selected_model_display]

        # Attempt to re-initialize LLM based on selection
        try:
            current_llm = get_llm_model(selected_model_id)
            logger.info(f"Chatbot LLM set to: {selected_model_id}")
        except Exception as model_init_err:
            st.error(f"Gagal memuat model {selected_model_display}: {model_init_err}")
            current_llm = llm # Fallback to the initially passed default LLM
            logger.error(f"LLM initialization failed for {selected_model_id}. Falling back to default.")

        # Check vectorstore status for greeting message
        vectorstore_ready = 'vectorstore' in st.session_state and st.session_state.vectorstore is not None
        vectorstore_has_docs_init = False
        if vectorstore_ready:
             try:
                 # Check if the collection actually has items
                 vectorstore_has_docs_init = st.session_state.vectorstore._collection.count() > 0
             except Exception as vs_check_err:
                 logger.error(f"Error checking vectorstore count: {vs_check_err}")
                 st.warning("Tidak dapat memverifikasi status dokumen di database vektor.")


        if not vectorstore_has_docs_init:
            st.info("👋 Selamat datang di Arsipy! Basis data dokumen lokal kosong atau tidak dapat diakses. Pertanyaan akan dijawab menggunakan pencarian web. Admin dapat mengunggah dokumen melalui Panel Admin di sidebar.")
        else:
             st.info("👋 Selamat datang di Arsipy! Ajukan pertanyaan tentang dokumen yang telah diunggah atau topik kearsipan umum.")

        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        st.markdown("---")
        st.subheader("Riwayat Percakapan")
        if not st.session_state.chat_history:
            st.caption("Belum ada percakapan.")
        else:
            # Display history (Keep existing logic)
            for i, (q, response_data) in enumerate(st.session_state.chat_history):
                 with st.container(border=False):
                    st.markdown(f"👤 **Anda:** {q}")
                    st.markdown(f"🤖 **Arsipy:**")
                    st.markdown(response_data.get('result', "_Maaf, terjadi kesalahan dalam menampilkan jawaban ini._"), unsafe_allow_html=True)

                    local_sources_hist = response_data.get('source_documents', [])
                    web_sources_hist = response_data.get('web_source_documents', [])

                    if local_sources_hist or web_sources_hist:
                        with st.expander(f"Lihat Sumber Jawaban #{i+1}", expanded=False):
                            if local_sources_hist:
                                st.markdown("**Sumber Dokumen Lokal:**")
                                valid_local_sources_hist = [doc for doc in local_sources_hist if isinstance(doc, Document) and hasattr(doc, 'metadata')]
                                if not valid_local_sources_hist:
                                     st.caption("_Tidak ada sumber dokumen lokal yang valid._")
                                else:
                                    for doc in valid_local_sources_hist:
                                        source_meta = doc.metadata.get('source', 'Sumber Tidak Diketahui')
                                        title_meta = doc.metadata.get('judul', source_meta)
                                        page_meta = doc.metadata.get('page')
                                        page_info = f", Hal: {page_meta}" if page_meta is not None else ""
                                        st.markdown(f"📄 **{title_meta}**{page_info}")
                                        st.markdown("---")

                            if web_sources_hist:
                                st.markdown("**Sumber Web (Konteks Tambahan/Fallback):**")
                                if not web_sources_hist:
                                     st.caption("_Tidak ada sumber web yang digunakan._")
                                else:
                                    for src in web_sources_hist:
                                        title = src.get('title', 'Judul Tidak Tersedia')
                                        url = src.get('url')
                                        display_url = f" ([Link]({url}))" if url and isinstance(url, str) and url.startswith('http') else ""
                                        source_name = src.get('source', 'Web')
                                        st.markdown(f"🌐 **{source_name}**: *{title}*{display_url}")
                                        snippet = src.get('content', 'N/A')
                                        st.caption(f"    {snippet}")
                                        st.markdown("---")
                    st.divider()

        # Chat input form
        with st.form(key='chat_form'):
            prompt1 = st.text_input("Masukkan pertanyaan Anda:", key='question_input', placeholder="Contoh: Apa saja komponen utama ISO 15489?")
            submit_button = st.form_submit_button("Kirim Pertanyaan")

        # =================== CORE PROCESSING LOGIC ===================
        if submit_button and prompt1:
            try:
                # --- (Keep the existing query optimization and clarification logic) ---
                enhanced_query = st.session_state.search_agent.optimize_query(prompt1)
                logger.info(f"Original query: '{prompt1}', Enhanced query: '{enhanced_query}'")

                if st.session_state.response_agent.should_ask_clarification(enhanced_query):
                    questions = st.session_state.response_agent.generate_clarifying_questions(enhanced_query)
                    if questions:
                        st.info("Untuk hasil yang lebih akurat, mohon perjelas:")
                        for q_clarify in questions:
                            st.write(f"• {q_clarify}")
                        st.stop()

                with memory_track():
                    # --- (Keep the vectorstore check logic) ---
                    vectorstore = st.session_state.get('vectorstore') # Use .get for safety
                    vectorstore_has_docs = False
                    if vectorstore:
                        try:
                            vectorstore_has_docs = vectorstore._collection.count() > 0
                        except Exception as vs_err:
                            logger.error(f"Error checking vectorstore count during query: {vs_err}")
                            st.warning("Tidak dapat memverifikasi status dokumen di database vektor.")
                    logger.info(f"Vectorstore status for query: Ready={vectorstore is not None}, HasDocs={vectorstore_has_docs}")

                    # --- (Keep the get_response_with_enrichment function definition) ---
                    # This function correctly handles the logic based on vectorstore_has_docs
                    def get_response_with_enrichment(query: str, attempt_rag: bool) -> Dict:
                        local_response = {}
                        local_source_docs = []
                        result_text = ""
                        is_inadequate = True # Assume inadequate initially

                        # 1. Attempt RAG if vectorstore has docs and flag is true
                        if attempt_rag and vectorstore: # Added check for vectorstore existence
                            logger.info(f"Attempting RAG for query: {query} using {current_llm.model_name}")
                            try:
                                retriever = vectorstore.as_retriever(
                                    search_type="similarity",
                                    search_kwargs={"k": 4}
                                )
                                qa_chain = get_rag_chain(current_llm, retriever)
                                local_response = qa_chain.invoke({'query': query})
                                logger.info(f"Local RAG response received: Keys={local_response.keys()}")

                                result_text = local_response.get('result', '').strip().lower()
                                local_source_docs = [
                                    doc for doc in local_response.get('source_documents', [])
                                    if isinstance(doc, Document) and hasattr(doc, 'metadata')
                                ]

                                # 2. Check if RAG response is adequate
                                inadequate_phrases = ['tidak ditemukan', 'tidak tersedia', 'tidak dapat menemukan informasi', 'saya tidak tahu', 'i don\'t know', 'tidak relevan', 'kurang informasi']
                                is_inadequate = not result_text or (
                                    any(phrase in result_text for phrase in inadequate_phrases) or
                                    len(result_text) < 30
                                ) or not local_source_docs
                                logger.info(f"RAG Adequacy Check: Inadequate={is_inadequate}, Result='{result_text[:100]}...', SourceDocsCount={len(local_source_docs)}")

                            except Exception as rag_err:
                                logger.error(f"Error during RAG chain execution: {rag_err}")
                                logger.error(traceback.format_exc())
                                is_inadequate = True
                                local_response = {'result': "_Terjadi kesalahan saat mencari di dokumen lokal._", 'source_documents': []}
                            pass # Placeholder for brevity, keep existing RAG code
                        else:
                             logger.info("Skipping RAG attempt as vectorstore is empty or unavailable.")
                             is_inadequate = True

                        # --- Function to filter web results ---
                        def filter_redundant_web_results(web_results: List[Dict], threshold: float = 0.90) -> List[Dict]:
                            if not web_results or len(web_results) <= 1:
                                return web_results # No filtering needed

                            snippets = [r.get('content', '') for r in web_results]
                            valid_snippets_indices = [i for i, s in enumerate(snippets) if s] # Indices with actual content

                            if len(valid_snippets_indices) <= 1:
                                return web_results # Not enough content to compare

                            valid_snippets = [snippets[i] for i in valid_snippets_indices]

                            try:
                                # Initialize embeddings (ensure GOOGLE_API_KEY is available)
                                embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                                embeddings = embeddings_model.embed_documents(valid_snippets)

                                # Calculate cosine similarity
                                similarity_matrix = cosine_similarity(embeddings)

                                # Filter based on similarity
                                indices_to_keep = []
                                kept_indices_set = set() # Track original indices we decide to keep

                                for i in range(len(valid_snippets_indices)):
                                    original_index_i = valid_snippets_indices[i]
                                    is_redundant = False
                                    for j_idx in range(len(indices_to_keep)): # Compare against already kept items
                                        kept_original_index = indices_to_keep[j_idx]
                                        # Find the corresponding index in the similarity matrix
                                        matrix_idx_i = valid_snippets_indices.index(original_index_i)
                                        matrix_idx_j = valid_snippets_indices.index(kept_original_index)

                                        if i != matrix_idx_j and similarity_matrix[matrix_idx_i][matrix_idx_j] > threshold:
                                            is_redundant = True
                                            logger.debug(f"Result {original_index_i} ('{web_results[original_index_i]['title'][:30]}...') is redundant with kept result {kept_original_index} ('{web_results[kept_original_index]['title'][:30]}...'). Similarity: {similarity_matrix[matrix_idx_i][matrix_idx_j]:.2f}")
                                            break

                                    if not is_redundant:
                                        indices_to_keep.append(original_index_i)
                                        kept_indices_set.add(original_index_i)

                                # Include results that had no snippet content (can't compare them)
                                final_indices_to_keep = sorted(list(kept_indices_set) + [i for i, s in enumerate(snippets) if not s])

                                filtered_results = [web_results[i] for i in final_indices_to_keep]
                                logger.info(f"Filtered web results from {len(web_results)} down to {len(filtered_results)} (Similarity threshold: {threshold})")
                                return filtered_results

                            except Exception as filter_err:
                                logger.error(f"Error during web result filtering: {filter_err}. Returning original results.")
                                logger.error(traceback.format_exc())
                                return web_results # Return original list on error
                            
                            
                        # --- Branching Logic: Fallback vs. Enrichment ---
                        if is_inadequate:
                            # --- Fallback Path ---
                            logger.info(f"Response inadequate or RAG skipped/failed. Falling back to web search for: {query}")
                            web_results_raw = web_searcher.search(query, total_max_results=3) # Fetch results

                            # Filter the results
                            web_results = filter_redundant_web_results(web_results_raw)

                            if web_results: # Use filtered results
                                logger.info(f"Found {len(web_results)} non-redundant web results for fallback.")
                                web_context = "\n\n".join([
                                    f"Sumber: {r['source']} ({r.get('title', 'N/A')})\nURL: {r.get('url', 'N/A')}\nKonten: {r['content']}"
                                    for r in web_results # Use filtered results
                                ])
                                # --- (Keep the rest of the fallback LLM call logic as is) ---
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
                                logger.info(f"Invoking LLM ({current_llm.model_name}) with web context for fallback.")
                                web_response_obj = current_llm.invoke(web_prompt)
                                web_response_text = web_response_obj.content if hasattr(web_response_obj, 'content') else str(web_response_obj)
                                logger.info("Received web-based fallback response from LLM.")
                                return {
                                    'query': query,
                                    'result': web_response_text.strip(),
                                    'source_documents': [],
                                    'web_source_documents': web_results, # Store filtered results
                                    'from_web': True,
                                    'enriched': False
                                }
                            else:
                                 logger.info("Web search returned no results for fallback.")
                                 final_result = local_response.get('result', "_Maaf, saya tidak dapat menemukan informasi yang relevan baik di dokumen lokal maupun dari pencarian web._")
                                 if "Terjadi kesalahan" in final_result or not final_result.strip() or len(final_result.strip()) < 20:
                                     final_result = "_Maaf, saya tidak dapat menemukan informasi yang relevan saat ini._"

                                 return {
                                     'query': query,
                                     'result': final_result,
                                     'source_documents': [],
                                     'web_source_documents': [],
                                     'from_web': False,
                                     'enriched': False
                                 }
                        else:
                            # --- Enrichment Path ---
                            logger.info("Local response deemed adequate. Attempting enrichment via web search.")
                            web_results_raw = web_searcher.search(query, total_max_results=2) # Fetch results

                            # Filter the results
                            web_results = filter_redundant_web_results(web_results_raw)

                            if web_results: # Use filtered results
                                logger.info(f"Found {len(web_results)} non-redundant web results for enrichment.")
                                web_context = "\n\n".join([
                                    f"Sumber: {r['source']} ({r.get('title', 'N/A')})\nURL: {r.get('url', 'N/A')}\nKonten: {r['content']}"
                                    for r in web_results # Use filtered results
                                ])
                                local_context_summary = local_response.get('result', '_Informasi dari dokumen lokal tidak tersedia._')

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
                                    enrichment_llm = ChatGroq(
                                        groq_api_key=os.getenv('GROQ_API_KEY'),
                                        model_name="llama3-70b-8192",
                                        temperature=0.5
                                    )
                                    logger.info("Using llama3-70b-8192 for enrichment synthesis.")
                                    enriched_response_obj = enrichment_llm.invoke(enrichment_prompt)
                                    enriched_response_text = enriched_response_obj.content if hasattr(enriched_response_obj, 'content') else str(enriched_response_obj)
                                    logger.info("Received enriched response from LLM.")

                                    return {
                                        'query': query,
                                        'result': enriched_response_text.strip(),
                                        'source_documents': local_source_docs,
                                        'web_source_documents': web_results,
                                        'from_web': False,
                                        'enriched': True
                                    }
                                except Exception as enrich_llm_err:
                                     logger.error(f"Error invoking enrichment LLM: {enrich_llm_err}. Falling back to local response.")
                                     return {
                                        **local_response,
                                        'web_source_documents': [],
                                        'from_web': False,
                                        'enriched': False
                                    }
                            else:
                                logger.info("No relevant web results found for enrichment. Returning original local response.")
                                return {
                                    **local_response,
                                    'web_source_documents': [],
                                    'from_web': False,
                                    'enriched': False
                                }
                    # --- END get_response_with_enrichment FUNCTION ---

                    # --- (Keep spinner and response processing logic) ---
                    spinner_messages = [
                        "Menganalisis pertanyaan Anda...",
                        "Mencari di dokumen lokal..." if vectorstore_has_docs else "Melewati pencarian dokumen lokal (kosong)...",
                        "Memeriksa sumber web untuk konteks tambahan/fallback...",
                        "Mensintesis jawaban..."
                    ]
                    final_response = {}
                    with st.spinner(spinner_messages[0]):
                        start_time = time.time()
                        # Simulate work for spinner updates
                        for i, message in enumerate(spinner_messages[1:], 1):
                            time.sleep(0.2) # Short delay for visual effect
                            # Update spinner text dynamically if possible with st.spinner context manager
                            # st.spinner(message) # This might replace the spinner, not update it.
                            # The default spinner behavior might just show the last message.

                        # Call the main response function
                        final_response = get_response_with_enrichment(enhanced_query, attempt_rag=vectorstore_has_docs)
                        elapsed_time = time.time() - start_time

                    # --- (Keep Post-processing with Agents logic) ---
                    if 'result' in final_response and final_response['result']:
                        # Identify knowledge gaps
                        gaps = st.session_state.knowledge_agent.identify_knowledge_gaps(
                            enhanced_query,
                            final_response['result']
                        )
                        if gaps:
                            suggested_sources = st.session_state.knowledge_agent.suggest_sources(enhanced_query)
                            with st.expander("📚 Saran Pengembangan Pengetahuan"):
                                st.write("Area yang dapat ditambahkan/ditingkatkan:")
                                for gap in gaps:
                                    st.write(f"• {gap}")
                                if suggested_sources:
                                    st.write("\nSumber yang disarankan untuk eksplorasi lebih lanjut:")
                                    for source in suggested_sources:
                                        st.write(f"• {source['name']} (Reliabilitas: {source['reliability']})")

                        # Show related queries
                        related = st.session_state.search_agent.suggest_related_queries(enhanced_query)
                        if related:
                            with st.expander("🔍 Pertanyaan Terkait"):
                                for rel_query in related:
                                    st.write(f"• {rel_query}")

                    # Store response and rerun
                    st.session_state.chat_history.append((prompt1, final_response))
                    st.success(f"Jawaban dihasilkan dalam {elapsed_time:.2f} detik.")
                    time.sleep(0.5)
                    st.rerun()

            except Exception as e:
                st.error(f"Terjadi kesalahan tak terduga saat memproses pertanyaan: {str(e)}")
                logger.error(traceback.format_exc())
                st.session_state.chat_history.append((prompt1, {'result': f"_Error: Terjadi kesalahan - {str(e)}_", 'source_documents': [], 'web_source_documents': []}))
                st.rerun()
                pass # Placeholder
            
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


    # --- Tabs 2, 3, 4 (Tentang, Panduan, Resources) ---
    # Update descriptions slightly to reflect enrichment
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
        - **AI Models**: Groq (Llama3), Google AI Embeddings
        - **Frontend**: Streamlit
        - **Web Scraping**: Requests, BeautifulSoup
        """)
        st.subheader("⚠️ Penting")
        st.info("""
        * Aplikasi ini tidak menyimpan riwayat percakapan Anda secara permanen (hanya dalam sesi).
        * Jangan mengunggah dokumen yang berisi informasi pribadi yang sangat sensitif atau rahasia.
        * Jawaban dihasilkan oleh AI dan mungkin memerlukan verifikasi silang, terutama untuk keputusan kritis.
        * Berminat membangun solusi serupa untuk organisasi Anda? Hubungi pengembang.
        """)

    with tab3: # Panduan
        st.markdown("""
        ### Chat dengan Arsipy (Tab 💬 Chatbot)
        1.  **Ajukan Pertanyaan:** Masukkan pertanyaan Anda tentang topik kearsipan atau konten dokumen yang telah diunggah di kotak input.
        2.  **Proses Jawaban:**
            *   Arsipy pertama-tama mencari jawaban di **dokumen lokal** yang diunggah Admin.
            *   Jika jawaban ditemukan dan dianggap memadai, Arsipy akan **memeriksa sumber web terpercaya** (ICA, SAA, ANRI, ISO) untuk **konteks tambahan** yang relevan. Jawaban akhir akan **mensintesis** informasi dari kedua sumber secara alami.
            *   Jika jawaban lokal **tidak ditemukan atau tidak memadai**, Arsipy akan **mencari jawaban langsung** dari sumber web terpercaya (fallback).
        3.  **Hasil:** Jawaban akan ditampilkan beserta sumber yang digunakan (dokumen lokal dan/atau web).
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
            - Modul Ajar Kearsipan Universitas Terbuka (UT)
            - Regulasi Kearsipan Indonesia (UU, PP, Peraturan ANRI)
            - Standar Operasional Prosedur (SOP) internal
            - Panduan Praktik Terbaik internal
            - Materi pelatihan kearsipan

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
    # Ensure formatting and LLM calls are robust
    with tab5:
        # Add the CSS styling here for consistency if not globally applied
        st.markdown("""
            <style>
            .response-section { background-color: #262730; padding: 15px; border-radius: 8px; margin: 15px 0; border: 1px solid #333; }
            .section-title { color: #1F77B4; border-bottom: 1px solid #1F77B4; padding-bottom: 8px; margin-bottom: 15px; font-size: 1.1em; font-weight: bold; }
            .section-content { color: #FAFAFA; font-size: 1em; line-height: 1.7; text-align: justify; white-space: pre-wrap; }
            .section-content ul, .section-content ol { padding-left: 25px; margin-top: 10px; }
            .section-content li { margin-bottom: 8px; }
            </style>
        """, unsafe_allow_html=True)

        analysis_type = st.radio(
            "Pilih Sumber Analisis",
            ["Pencarian Online", "Konten Web (URL)", "Konten Dokumen (Unggah)"],
            key="web_insights_analysis_type",
            horizontal=True
        )

        if analysis_type == "Pencarian Online":
            search_query = st.text_input("Masukkan topik atau kata kunci pencarian:", key="online_search_query", placeholder="Contoh: ISO 15489, preservasi digital, metadata arsip")
            citation_style = st.selectbox(
                "Gaya Sitasi untuk Referensi",
                ["APA", "Chicago", "Harvard", "MLA"],
                key="online_search_citation"
            )

            if search_query and st.button("Buat Analisis dari Web", key="online_search_button"):
                try:
                    with st.spinner("🔍 Mencari informasi dari sumber terpercaya dan menganalisis... Ini mungkin memerlukan waktu beberapa saat."):
                        # Use powerful model for this complex task
                        model = ChatGroq(
                            groq_api_key=os.getenv('GROQ_API_KEY'),
                            model_name="llama3-70b-8192",
                            temperature=0.6,
                            max_tokens=4096
                        )

                        # Refined Prompt for Online Search
                        prompt = f"""Anda adalah asisten riset ahli di bidang kearsipan. Buat analisis komprehensif dalam Bahasa Indonesia tentang topik berikut: "{search_query}"

Gunakan format **WAJIB** berikut dengan pemisah yang jelas (===NAMA_BAGIAN===). Jangan tambahkan teks atau salam pembuka/penutup di luar format ini. Pastikan setiap bagian memiliki konten yang relevan dan berkualitas.

===DEFINISI===
Berikan definisi formal yang jelas dan ringkas dalam Bahasa Indonesia.
Sertakan istilah relevan dalam Bahasa Inggris (jika ada) dalam kurung.
Jelaskan konsep-konsep kunci terkait definisi menggunakan bullet points (•) untuk kejelasan.

===PEMBAHASAN===
Berikan analisis mendalam dalam minimal 3 paragraf yang terstruktur dan informatif:
1.  **Latar Belakang & Konteks:** Jelaskan asal-usul, perkembangan historis, atau signifikansi topik dalam dunia kearsipan.
2.  **Analisis Inti:** Bahas aspek-aspek fundamental, komponen utama, metode, teori, atau prinsip kerja terkait topik secara mendalam dan jelas.
3.  **Implikasi & Tantangan:** Uraikan penerapan praktisnya, manfaat yang dihasilkan, tantangan yang dihadapi dalam implementasi, atau isu-isu kontemporer terkait topik ini.

===STANDAR & PRAKTIK TERBAIK===
Identifikasi dan sebutkan standar internasional atau nasional yang paling relevan (misal: seri ISO 15489, ISO 30300, SNI terkait, panduan ANRI).
Jika tidak ada standar formal, sebutkan prinsip atau praktik terbaik (best practices) yang diakui secara luas di komunitas kearsipan.
Jelaskan secara singkat poin-poin utama dari standar/praktik tersebut menggunakan bullet points (•).
Jika benar-benar tidak ada standar atau praktik yang relevan secara langsung, nyatakan: "Tidak ada standar atau praktik terbaik spesifik yang secara langsung mengatur topik ini, namun prinsip-prinsip umum [sebutkan prinsip umum jika ada] dapat diterapkan."

===REFERENSI===
Sajikan minimal 3 (maksimal 5) referensi kredibel yang mendukung analisis (utamakan dari domain .org, .gov, .edu, publikasi ilmiah, atau standar resmi). Gunakan format sitasi **{citation_style}** secara konsisten.
Contoh Format {citation_style}: [Berikan contoh singkat format {citation_style} yang benar di sini, misal: Author, A. A. (Year). *Title of work*. Publisher.]

"""

                        logger.info(f"Invoking LLM (llama3-70b) for Online Search with query: {search_query}")
                        response = model.invoke(prompt)
                        logger.info(f"LLM response received (type: {type(response)})")

                        # Robust Response Handling and Parsing
                        if response and hasattr(response, 'content'):
                            content = response.content.strip()
                            logger.debug(f"Raw LLM content received:\n{content[:500]}...") # Log beginning of content

                            sections = {}
                            current_section = None
                            current_content = []
                            # Use regex to find section headers more reliably, allowing spaces
                            section_pattern = re.compile(r"^===\s*([\w\s&]+?)\s*===$")

                            processed_lines = content.replace('\r\n', '\n').split('\n')

                            for line in processed_lines:
                                match = section_pattern.match(line.strip())
                                if match:
                                    if current_section is not None:
                                        sections[current_section] = '\n'.join(current_content).strip()
                                        logger.debug(f"Saved section '{current_section}'")
                                    # Normalize section key (uppercase, replace space with underscore)
                                    current_section = match.group(1).strip().upper().replace(' ', '_')
                                    current_content = []
                                    logger.info(f"Found section marker, normalized key: {current_section}")
                                else:
                                    if current_section is not None:
                                        current_content.append(line) # Keep original line for formatting

                            if current_section is not None:
                                sections[current_section] = '\n'.join(current_content).strip()
                                logger.debug(f"Saved final section '{current_section}'")

                            logger.info(f"Parsed section keys: {list(sections.keys())}")

                            # Displaying Sections with checks
                            expected_keys = ['DEFINISI', 'PEMBAHASAN', 'STANDAR_&_PRAKTIK_TERBAIK', 'REFERENSI']
                            displayed_something = False
                            for key in expected_keys:
                                section_title = key.replace('_', ' ').title()
                                section_content_parsed = sections.get(key, "").strip()
                                if section_content_parsed:
                                    st.markdown(f"""
                                        <div class="response-section">
                                            <h3 class="section-title">{section_title}</h3>
                                            <div class="section-content">{section_content_parsed}</div>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    displayed_something = True
                                else:
                                    # Optionally show missing sections
                                    # st.markdown(f"<div class='response-section'><h3 class='section-title'>{section_title}</h3><div class='section-content'><i>Bagian ini tidak dihasilkan oleh AI.</i></div></div>", unsafe_allow_html=True)
                                    logger.info(f"Expected section '{key}' was empty or missing in parsed response.")

                            # Fallback if parsing failed or no expected sections found
                            if not displayed_something:
                                 st.warning("Format respons dari AI tidak sepenuhnya sesuai harapan. Menampilkan konten mentah:")
                                 logger.warning(f"Could not display expected sections. Found keys: {list(sections.keys())}. Displaying raw content.")
                                 st.markdown(f"""
                                     <div class="response-section">
                                         <h3 class="section-title">Respons Mentah dari AI</h3>
                                         <div class="section-content" style="white-space: pre-wrap;">{content}</div>
                                     </div>
                                 """, unsafe_allow_html=True)


                            # Handle references download button (using normalized key)
                            references_content = sections.get('REFERENSI', '').strip()
                            if references_content:
                                col_ref1, col_ref2 = st.columns([3,1])
                                with col_ref1:
                                    st.caption("Verifikasi sumber referensi yang disajikan sangat disarankan.")
                                with col_ref2:
                                    try:
                                        st.download_button(
                                            label="📥 Unduh Referensi",
                                            data=references_content,
                                            file_name=f"referensi_{search_query.replace(' ','_')}_{citation_style.lower()}.txt",
                                            mime="text/plain",
                                            key="download_references_button"
                                        )
                                    except Exception as dl_error:
                                        st.error("Gagal membuat tombol unduh.")
                                        logger.error(f"Error creating download button: {dl_error}")
                        elif response:
                             st.error("Respons AI diterima, tetapi formatnya tidak dapat diproses (bukan objek konten yang diharapkan).")
                             logger.error(f"Unexpected LLM response format: Type={type(response)}, Value={response}")
                        else:
                            st.error("Tidak ada respons yang diterima dari AI. Periksa kunci API dan status layanan Groq.")
                            logger.error("LLM invocation returned None or empty.")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat melakukan analisis online: {str(e)}")
                    logger.error(f"Online Search error: {traceback.format_exc()}")

        elif analysis_type == "Konten Web (URL)":
            # ... (Keep logic as is, ensure model is powerful e.g., llama3-70b) ...
            url = st.text_input("Masukkan URL halaman web untuk dianalisis:", key="web_content_url", placeholder="https://www.example.com/article")
            if url and st.button("Analisis Konten Web", key="web_content_button"):
                try:
                    with st.spinner(f"Mengambil konten dari {url}..."):
                        content = fetch_url_content(url) # Ensure fetch_url_content is defined and robust
                        if not content or len(content) < 50:
                            st.error("Tidak dapat mengekstrak konten yang cukup dari URL tersebut. Periksa URL atau coba yang lain.")
                            st.stop()

                    with st.spinner("🔍 Menganalisis konten web..."):
                        model = ChatGroq(
                            groq_api_key=os.getenv('GROQ_API_KEY'),
                            model_name="llama3-70b-8192", # Use powerful model
                            temperature=0.5
                        )
                        prompt = f"""Anda adalah analis konten web yang ahli. Analisis konten berikut dari URL {url} dalam Bahasa Indonesia. Fokus pada poin-poin utama, argumen kunci, dan kesimpulan jika ada. Abaikan elemen navigasi, iklan, atau komentar. Sajikan analisis dalam paragraf yang jelas dan terstruktur.

                        Konten Web (Maks 8000 karakter):
                        ---
                        {content[:8000]}
                        ---

                        Hasil Analisis (Bahasa Indonesia):
                        """
                        response = model.invoke(prompt)

                        if response and hasattr(response, 'content'):
                            analysis_content = response.content.strip()
                            st.success("Analisis selesai!")
                            st.markdown(f"""
                                <div class="response-section">
                                    <h3 class="section-title">Analisis Konten Web</h3>
                                    <div class="section-content">{analysis_content}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error("Tidak ada analisis yang dihasilkan. Coba lagi.")
                            logger.error(f"Web content analysis failed. Response: {response}")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat menganalisis konten web: {str(e)}")
                    logger.error(f"Web Content Analysis error: {traceback.format_exc()}")


        elif analysis_type == "Konten Dokumen (Unggah)":
            # ... (Keep logic as is, ensure model is powerful e.g., llama3-70b) ...
            uploaded_file = st.file_uploader(
                "Unggah dokumen (PDF/TXT) untuk dianalisis:",
                type=["pdf", "txt"],
                key="doc_analysis_uploader"
            )
            if uploaded_file and st.button("Analisis Dokumen", key="doc_analysis_button"):
                try:
                    with st.spinner(f"📄 Memproses dokumen {uploaded_file.name}..."):
                        text = get_document_text(uploaded_file) # Ensure this handles errors

                    with st.spinner("🔍 Menghasilkan analisis dokumen..."):
                        model = ChatGroq(
                            groq_api_key=os.getenv('GROQ_API_KEY'),
                            model_name="llama3-70b-8192", # Use powerful model
                            temperature=0.5
                        )
                        prompt = f"""Anda adalah analis dokumen yang ahli. Analisis konten dokumen berikut ({uploaded_file.name}) dalam Bahasa Indonesia. Identifikasi topik utama, poin-poin kunci, argumen utama, dan berikan ringkasan singkat. Sajikan analisis dalam format yang terstruktur dan mudah dibaca.

                        Konten Dokumen (Maks 8000 karakter):
                        ---
                        {text[:8000]}
                        ---

                        Hasil Analisis (Bahasa Indonesia):
                        """
                        response = model.invoke(prompt)

                        if response and hasattr(response, 'content'):
                            analysis_content = response.content.strip()
                            st.success("Analisis selesai!")
                            st.markdown(f"""
                                <div class="response-section">
                                    <h3 class="section-title">Analisis Dokumen: {uploaded_file.name}</h3>
                                    <div class="section-content">{analysis_content}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error("Tidak ada analisis yang dihasilkan. Coba lagi.")
                            logger.error(f"Document content analysis failed. Response: {response}")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat menganalisis dokumen: {str(e)}")
                    logger.error(f"Document Analysis error: {traceback.format_exc()}")

# --- (Keep initializers: cache_manager, security_manager, system_monitor) ---
cache_manager = CacheManager()
security_manager = SecurityManager()
system_monitor = SystemMonitor()

@system_monitor.monitor_performance
@cache_manager.cache_query(ttl=3600)
def process_chat_query(prompt: str, vectorstore: Any, llm: Any) -> dict:
    # This wrapper is still not directly integrated into the main flow above
    # If intended for use, the call inside show_chat_interface needs refactoring
    prompt = security_manager.sanitize_input(prompt)
    logger.warning("process_chat_query wrapper called, but core logic is currently in show_chat_interface.")
    raise NotImplementedError("process_chat_query wrapper needs integration")

def initialize_or_load_vectorstore() -> Optional[Chroma]: # Return Optional
    """Initialize or load the vector store. Returns None on failure."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        global CHROMA_DB_DIR # Ensure global variable is accessible
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings
        )
        # Perform a quick check to see if connection is valid
        try:
            vectorstore._collection.peek(limit=1) # Try a low-level operation
            logger.info(f"Vectorstore loaded/initialized successfully from {CHROMA_DB_DIR}")
            return vectorstore
        except Exception as vs_conn_err:
             logger.error(f"Vectorstore loaded but connection check failed: {vs_conn_err}")
             st.error(f"Gagal terhubung ke database vektor yang ada di {CHROMA_DB_DIR}.")
             return None

    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Gagal memuat basis data vektor: {e}")
        return None # Return None on failure

def fetch_url_content(url: str) -> str:
    """Fetch and extract meaningful content from URL, handling HTML and PDF."""
    # --- (Keep existing robust fetch_url_content logic) ---
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
                text = "".join(page.get_text("text", flags=fitz.TEXT_INHIBIT_SPACES) + "\n" for page in doc) # Slightly cleaner text extraction
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
def main() -> None:
    """Main application entry point"""
    os.environ['ANONYMIZED_TELEMETRY'] = 'False'
    set_verbose(False)
    load_dotenv()

    # Landing page logic (remains the same)
    if 'show_app' not in st.session_state:
         st.session_state['show_app'] = False # Use a more descriptive name
    query_params = st.query_params
    if "page" in query_params and query_params.get("page") == "app":
        st.session_state['show_app'] = True
    elif not st.session_state.get('show_app', False):
        show_landing_page()
        return # Stop execution if showing landing page

    logger.info("Loading main application interface.")

    # API Key Validation (remains the same)
    groq_api_key = os.getenv('GROQ_API_KEY')
    google_api_key = os.getenv("GOOGLE_API_KEY")
    admin_password_env = os.getenv('ADMIN_PASSWORD')

    keys_ok = True
    if not groq_api_key:
        st.error("❌ Kunci API Groq tidak ditemukan! Periksa file .env Anda.")
        keys_ok = False
    if not google_api_key:
        st.error("❌ Kunci API Google tidak ditemukan! Periksa file .env Anda.")
        keys_ok = False
    if not keys_ok:
        st.warning("Beberapa fitur AI mungkin tidak berfungsi.")
        # Decide whether to stop or continue with limited functionality
        # st.stop() # Uncomment to halt execution if keys are critical

    if not admin_password_env:
         st.sidebar.warning("⚠️ Password Admin tidak diatur di .env. Panel Admin tidak akan aman/fungsional.")

    os.environ["GOOGLE_API_KEY"] = google_api_key if google_api_key else ""

    # ChromaDB Directory Setup (remains the same)
    global CHROMA_DB_DIR
    CHROMA_DB_DIR = "chroma_db"
    if not os.path.exists(CHROMA_DB_DIR):
        try:
            os.makedirs(CHROMA_DB_DIR)
            logger.info(f"Created ChromaDB directory: {CHROMA_DB_DIR}")
        except OSError as e:
             st.error(f"Gagal membuat direktori database: {e}. Fitur dokumen lokal tidak akan berfungsi.")
             # Don't stop, allow web features to work

    # Session State Initialization (Vectorstore, Doc Processor, uploaded files)
    if 'vectorstore' not in st.session_state:
        # Attempt initialization, will be None if fails
        st.session_state.vectorstore = initialize_or_load_vectorstore()

    if 'doc_processor' not in st.session_state:
        if st.session_state.get('vectorstore'): # Check if vectorstore loaded successfully
            st.session_state.doc_processor = UnifiedDocumentProcessor(st.session_state.vectorstore)
            logger.info("UnifiedDocumentProcessor initialized.")
        else:
            # Allow app to run but warn that local doc processing won't work
            logger.warning("Vectorstore not available. Document processor not initialized.")
            st.session_state.doc_processor = None # Indicate processor is not ready

    if 'uploaded_file_names' not in st.session_state:
        st.session_state.uploaded_file_names = set()

    # Initialize Default LLM (remains the same)
    try:
        default_llm = get_llm_model("compound-beta") # Use the function
        logger.info("Default LLM (compound-beta) initialized.")
    except Exception as e:
        st.error(f"❌ Gagal menginisialisasi model AI default: {str(e)}")
        logger.error(traceback.format_exc())
        st.warning("Fungsionalitas AI mungkin terbatas.")
        default_llm = None # Set to None if fails

    # --- CORE CHANGE HERE ---
    # Always setup the sidebar (which handles admin login/controls internally)
    setup_admin_sidebar()

    # Always show the main chat interface, regardless of admin authentication
    # Ensure default_llm is available before calling
    if default_llm:
        show_chat_interface(default_llm)
    else:
        # Handle case where even the default LLM failed
        st.error("Model AI utama tidak dapat dimuat. Fungsi chatbot tidak tersedia.")
        # Optionally show other tabs like Panduan/Tentang here if desired
        # show_static_tabs() # Example function call

    # The previous check for admin authentication before showing the chat interface is removed.
    # The admin-only features (upload, process) are now correctly gated within show_admin_controls,
    # which is only called by setup_admin_sidebar when authenticated.

if __name__ == "__main__":
    main()