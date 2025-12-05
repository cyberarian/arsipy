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
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.globals import set_verbose
from dotenv import load_dotenv
from streamlit.runtime.caching import cache_data, cache_resource
from datetime import datetime
import toml
import chromadb
from langchain_chroma import Chroma
import markdown
import sqlite3
from image_analyzer import image_analyzer_main
from huggingface_hub import InferenceClient
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.language_models.llms import LLM
from langchain_core.retrievers import BaseRetriever

# Add these imports at the top
from utils.cache_manager import CacheManager
from utils.security import SecurityManager
from utils.monitoring import SystemMonitor
from document_processor import UnifiedDocumentProcessor

# Change the import at the top
from landing_page import show_landing_page

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

class NvidiaEmbeddings(OpenAIEmbeddings):
    """
    Custom LangChain embedding class for NVIDIA's embedding models.
    
    This class inherits from OpenAIEmbeddings and is configured to work with
    NVIDIA's API endpoint by setting a custom base_url. It also handles the
    `input_type` parameter required by NVIDIA, differentiating between
    'query' and 'passage' embeddings.
    """
    def __init__(self, model: str, nvidia_api_key: str, input_type: str = "passage", **kwargs):
        # Pass remaining kwargs to the parent OpenAIEmbeddings constructor FIRST
        super().__init__(
            model=model,
            api_key=nvidia_api_key,
            base_url="https://integrate.api.nvidia.com/v1",
            **kwargs
        )
        # Store NVIDIA-specific parameters AFTER parent init
        object.__setattr__(self, '_nvidia_api_key', nvidia_api_key)
        object.__setattr__(self, '_nvidia_input_type', input_type)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed search docs using NVIDIA's embedding API with input_type='passage'.
        """
        from openai import OpenAI
        
        # Retrieve stored values using object.__getattribute__
        nvidia_api_key = object.__getattribute__(self, '_nvidia_api_key')
        nvidia_input_type = object.__getattribute__(self, '_nvidia_input_type')
        
        client = OpenAI(api_key=nvidia_api_key, base_url="https://integrate.api.nvidia.com/v1")
        
        embeddings = []
        for text in texts:
            response = client.embeddings.create(
                input=[text],
                model=self.model,
                encoding_format="float",
                extra_body={"input_type": nvidia_input_type, "truncate": "NONE"}
            )
            embeddings.append(response.data[0].embedding)
        
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed query using NVIDIA's embedding API with input_type='query'.
        """
        from openai import OpenAI
        
        # Retrieve stored values using object.__getattribute__
        nvidia_api_key = object.__getattribute__(self, '_nvidia_api_key')
        
        client = OpenAI(api_key=nvidia_api_key, base_url="https://integrate.api.nvidia.com/v1")
        
        response = client.embeddings.create(
            input=[text],
            model=self.model,
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "NONE"}
        )
        
        return response.data[0].embedding

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
    models = {
        "openai/gpt-oss-120b": lambda: ChatGroq(
            groq_api_key=os.getenv('GROQ_API_KEY'),
            model_name="openai/gpt-oss-120b"
        ),
        "compound-beta": lambda: ChatGroq(
            groq_api_key=os.getenv('GROQ_API_KEY'),
            model_name="compound-beta"
        ),
        "deepseek-coder": lambda: DeepSeekLLM(
            model="deepseek-ai/DeepSeek-V3.2",
            api_key=os.getenv('HUGGINGFACE_API_KEY'),
            temperature=0.5,
            max_tokens=512
        ),
        "Kimi-K2-Thinking": lambda: DeepSeekLLM(
            model="moonshotai/Kimi-K2-Thinking",
            api_key=os.getenv('HUGGINGFACE_API_KEY'),
            temperature=0.5,
            max_tokens=512
        ),       
    }
    
    if model_name not in models:
        raise ValueError(f"Unsupported model: {model_name}")
        
    return models[model_name]()

def get_rag_chain(llm: Union[LLM, Any], retriever: BaseRetriever) -> RetrievalQA:
    """
    Create an enhanced RAG pipeline with internal reasoning
    """
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
        output_key="answer"
    )
    
    document_prompt = PromptTemplate(
        template="Content: {page_content}\nSumber: {source}",
        input_variables=["page_content", "source"]
    )
    
    qa_chain = RetrievalQA(
        combine_documents_chain=StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=document_prompt,
            document_variable_name="context",
            document_separator="\n\n"
        ),
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain
# Set the page layout to wide
st.set_page_config(layout="wide")

# Load the config.toml file
config = toml.load(".streamlit/config.toml")

# Apply the custom CSS
st.markdown(f"<style>{config['custom_css']['css']}</style>", unsafe_allow_html=True)

# Load the admin password from the .env file
admin_password = os.getenv('ADMIN_PASSWORD')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Memory management context
@contextmanager
def memory_track():
    try:
        gc.collect()
        yield
    finally:
        gc.collect()

def setup_admin_sidebar() -> None:
    """Setup admin authentication and controls in sidebar"""
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False

    with st.sidebar:
        st.title("Admin Panel")

        # Admin authentication
        if not st.session_state.admin_authenticated:
            input_password = st.text_input("Admin Password", type="password")
            if st.button("Login"):
                # Use the admin password from the .env file
                if input_password == admin_password:
                    st.session_state.admin_authenticated = True
                    st.success("Admin authenticated!")
                    st.rerun()
                else:
                    st.error("Incorrect password")
        else:
            st.write("✅ Admin authenticated")
            if st.button("Logout"):
                st.session_state.admin_authenticated = False
                st.rerun()

            # Show admin controls only when authenticated
            st.divider()
            show_admin_controls()

def show_admin_controls() -> None:
    """Display admin controls with enhanced metadata input"""
    st.sidebar.header("Document Management")
    
    # File uploader section
    uploaded_files = st.sidebar.file_uploader(
        "Upload Documents", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.sidebar.subheader("Document Metadata")
        st.sidebar.info("""
        Please provide metadata for better document organization.
        Example: 'Modul 1, Manajemen Kearsipan di Indonesia, Drs. Syauki Hadiwardoyo'
        """)
        
        metadata_inputs = {}
        for file in uploaded_files:
            with st.sidebar.expander(f"Metadata for {file.name}"):
                metadata_inputs[file.name] = {
                    'judul': st.text_input(
                        "Judul Modul",
                        key=f"title_{file.name}",
                        placeholder="e.g., Manajemen Kearsipan di Indonesia"
                    ),
                    'pengajar': st.text_input(
                        "Nama Pengajar",
                        key=f"author_{file.name}",
                        placeholder="e.g., Drs. Syauki Hadiwardoyo"
                    ),
                    'deskripsi': st.text_area(
                        "Deskripsi (Optional)",
                        key=f"desc_{file.name}",
                        placeholder="Deskripsi singkat tentang modul ini"
                    )
                }
        
        # Process documents button
        if st.sidebar.button("Process Documents", key="process_docs_button"):
            process_uploaded_files(uploaded_files, metadata_inputs)

def process_uploaded_files(uploaded_files: List[Any], metadata_inputs: Dict) -> None:
    """Process uploaded files with enhanced metadata handling"""
    if not uploaded_files:
        st.sidebar.warning("No files selected for processing")
        return

    success_count = 0
    error_count = 0

    with st.spinner('Processing documents...'):
        for file in stqdm(uploaded_files):
            if file.name not in st.session_state.uploaded_file_names:
                metadata = metadata_inputs.get(file.name, {})
                result = st.session_state.doc_processor.process_document(file, metadata=metadata)
                if result['success']:
                    success_count += 1
                    st.sidebar.success(f"Processed: {result['metadata']['title']}")
                    st.session_state.uploaded_file_names.add(file.name)
                else:
                    error_count += 1
                    st.sidebar.error(f"Error processing {file.name}: {result['error']}")
    
    if error_count == 0 and success_count > 0:
        st.sidebar.success("All documents processed successfully!")
    elif success_count > 0 and error_count > 0:
        st.sidebar.warning(f"Processing complete. {success_count} succeeded, {error_count} failed.")


def extract_text_from_pdf(pdf_file: Any) -> str:
    """
    Extract text content from a PDF file
    
    Args:
        pdf_file: File-like object containing PDF data
        
    Returns:
        str: Extracted text from PDF
        
    Raises:
        ValueError: If extracted text is empty
        Exception: For PDF processing errors
    """
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text()
        if not text.strip():
            raise ValueError("Extracted text from PDF is empty")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise
    finally:
        if 'pdf_document' in locals():
            pdf_document.close()

def get_document_text(file: Any) -> str:
    """
    Get text content from a file based on its type
    
    Args:
        file: File-like object to extract text from
        
    Returns:
        str: Extracted text content
        
    Raises:
        ValueError: For unsupported file types or empty content
        Exception: For text extraction errors
    """
    try:
        if file.type == "application/pdf":
            text = extract_text_from_pdf(file)
        elif file.type == "text/plain":
            text = file.getvalue().decode('utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file.type}")
            
        if not text.strip():
            raise ValueError("Extracted text is empty")
            
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {file.name}: {str(e)}")
        raise

def clear_cache() -> None:
    """Clear all cached data"""
    cache_data.clear()
    cache_resource.clear()
    
def show_chat_interface(llm: Union[LLM, Any]) -> None:
    """Display the main chat interface"""
    # Centralized error handling for vectorstore initialization
    if 'vectorstore_error' in st.session_state and st.session_state.vectorstore_error:
        st.error(f"Failed to initialize the document database: {st.session_state.vectorstore_error}")
        st.info("This might be due to an API key issue or rate limiting. Please check your Google AI Platform billing and quotas.")
        st.stop()

    # Add logo
    col1, col2, col3 = st.columns([1,100,1])
    with col2:
        st.image("assets/logo-transparent3.png", width=350)
    
    # Create tabs for the main interface
    tab1, tab5, tab3, tab2, tab4 = st.tabs([
        "💬 Chatbot", 
        "🌐 Web Insights", 
        "❓ Panduan", 
        "ℹ️ Tentang",
        "📚 Resources"
    ])    
    
    with tab1:
        # Model selection inside chatbot tab
        model_options = {
            "gpt-oss-120b (Groq)": "openai/gpt-oss-120b",
            "DeepSeek-V3.2 (HuggingFace)": "deepseek-ai/DeepSeek-V3.2",
            "Kimi-K2-Thinking (HuggingFace)": "moonshotai/Kimi-K2-Thinking",
        }
        
        selected_model = st.selectbox(
            "Select AI Model",
            options=list(model_options.keys()),
            key='model_selector'
        )
        
        # Get the actual model identifier
        model_id = model_options[selected_model]
        
        # Add a greeting message
        if not st.session_state.uploaded_file_names:
            st.info("👋 Welcome to Arsipy, your AI-powered guide to archive manuals and analysis of trusted web insights.")
        
        # Initialize chat history in session state if it doesn't exist
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
        # Create a form for the chat input
        with st.form(key='chat_form'):
            prompt1 = st.text_input("Enter your question about the documents", key='question_input')
            submit_button = st.form_submit_button("Submit Question")
            
        # Updated chat history display
        for q, a in st.session_state.chat_history:
            st.write("Question:", q)
            
            if isinstance(a, dict) and 'result' in a and 'source_documents' in a:
                st.write("Answer:", a['result'])
                

            else:
                st.write("Answer:", a)
            
            st.divider()
        
        if submit_button and prompt1:
            try:
                with memory_track():
                    vectorstore = st.session_state.get('vectorstore')
                    if len(vectorstore.get()['ids']) > 0:
                        retriever = vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 4}
                        )
                        
                        qa_chain = get_rag_chain(llm, retriever)
                        
                        # Enhanced spinner messages
                        spinner_messages = [
                            "Analyzing your question...",
                            "Searching through documents...",
                            "Processing relevant information...",
                            "Formulating response..."
                        ]
                        
                        with st.spinner(spinner_messages[0]):
                            # Start timer
                            start_time = time.time()  # Changed from time.process_time()
                            
                            # Simulate analysis steps with multiple spinners
                            for i, message in enumerate(spinner_messages[1:], 1):
                                time.sleep(0.5)  # Brief pause for UX
                                st.spinner(message)
                            
                            # Get actual response
                            response = qa_chain.invoke({'query': prompt1})
                            elapsed_time = time.time() - start_time  # Calculate elapsed time here
                            
                            # Ensure we have valid Document objects
                            valid_sources = []
                            if 'source_documents' in response:
                                for doc in response['source_documents']:
                                    if isinstance(doc, Document) and hasattr(doc, 'metadata'):
                                        valid_sources.append(doc)
                            
                            # Store only valid documents in chat history
                            chat_response = {
                                'result': response['result'],
                                'source_documents': valid_sources
                            }
                            
                            st.session_state.chat_history.append((prompt1, chat_response))
                            
                            # Display current response
                            st.write("Response:", chat_response['result'])
                            
                            # Display sources if available
                            if valid_sources:
                                with st.expander("Sumber Referensi"):
                                    for doc in valid_sources:
                                        source = doc.metadata.get('source', 'Unknown Source')
                                        st.markdown(f"**Sumber:** {source}")
                                        st.markdown("---")
                            
                            st.write(f"Response time: {elapsed_time:.2f} seconds")
                            st.rerun()
                            
                    else:
                        st.warning("No documents found in the database. Please ask an admin to upload some documents.")
                        
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str and "quota" in error_str:
                    st.error("🚫 API Quota Exceeded")
                    st.warning("The free allowance for the embedding service has been used up. To continue, you must enable billing on your Google Cloud project.")
                    st.info("This is a billing issue, not an invalid API key. Please visit your Google AI Platform dashboard to check your plan and billing details.")
                else:
                    st.error(f"An unexpected error occurred: {e}")
                logger.error(f"Error processing question: {traceback.format_exc()}")

    # Add a clear chat history button
    if st.session_state.chat_history and st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
        
    # Footer
    st.markdown("---")
    st.markdown("Developed by Adnuri Mohamidi with help from AI :orange_heart:", help="cyberariani@gmail.com")
    with tab2:
        st.write("""
        ### 🎯 Tentang Arsipy
        Unit Kearsipan dan Manajemen Rekod sering kali menghadapi manual, prosedur, dan panduan yang kompleks serta krusial untuk menjaga integritas dan aksesibilitas rekod. Namun, banyaknya informasi yang harus diingat seringkali menyulitkan dalam mengingat langkah-langkah atau detail tertentu, terutama ketika menangani tugas-tugas yang jarang atau tidak sering dilakukan. Hal ini dapat mengakibatkan ketidakefisienan atau kesalahan dalam proses kearsipan dan pengelolaan rekod.

        Sebagai contoh, bayangkan seorang staf yang sedang mempersiapkan dokumen langka untuk digitalisasi tetapi lupa dengan prosedur penanganan yang spesifik. Alih-alih harus membuka-buka halaman manual, mereka cukup menanyakan pada chatbot, "Apa proses untuk mendigitalisasi dokumen yang rapuh?" Aplikasi ini akan langsung mengambil langkah-langkah yang relevan—seperti menggunakan sarung tangan, menyiapkan pemindai, dan mengatur pencahayaan—dan menyajikannya dalam format yang jelas dan mudah diikuti.

        Arsipy adalah asisten AI yang dibangun untuk itu, membantu Anda mengakses dan memahami manual arsip dengan lebih efektif. 

        ### 🔍 Fitur Utama
        **RAG-based Chatbot**
        - Menjawab pertanyaan tentang kearsipan
        - Referensi dari sumber terpercaya
        - Konteks yang akurat dan relevan

        ### 💻 Teknologi
        - **Backend**: Python, ChromaDB, LangChain
        - **AI Models**: gpt-oss-120b, DeepSeek-V3.2
        - **OCR**: pytesseract, Tesseract OCR
        - **Frontend**: Streamlit
        - **Database**: Vector Store dengan Google AI Embeddings

        """)
        
        st.subheader("⚠️ Penting")
        st.info("""
        * Aplikasi ini tidak merekam percakapan
        * Chatbot hanya menjawab pertanyaan seputar isi dari dokumen manual arsip
        * Untuk informasi lebih lanjut, silakan hubungi developer
        """)
    
    with tab3:
        st.markdown("""
        ### Chat dengan Manual Arsip
        1. Buka tab **💬 Chat**
        2. Ketik pertanyaan tentang manual arsip
        3. Tunggu jawaban dari chatbot
        4. Lihat referensi sumber yang disertakan

        ### Web Insights
        1. Buka tab **🖼️ Web Insights**
        2. Pilih opsi
        3. Analyze
       
        ### Untuk Hasil Chat Optimal
        - Gunakan bahasa yang jelas
        - Berikan konteks spesifik
        - Tanyakan satu topik per chat
        - Manfaatkan history chat untuk mempertajam pertanyaan

        ### ❗ Troubleshooting bila Chat tidak merespon
        - Refresh halaman
        - Periksa koneksi internet
        - Batas token harian telah terlampaui     

            """)
            
    with tab4:
        st.title("📚 Sumber Dokumen")
        
        st.markdown("""
        Sistem ini menggunakan sumber-sumber terpercaya untuk memastikan keakuratan informasi dalam pengelolaan arsip:
        
        ### 🏛️ Sumber Utama
        
        #### Repositori Universitas Terbuka (UT)
        - Modul ajar kearsipan dan dokumen akademik
        - Panduan pengelolaan arsip institusional
        - Standar prosedur kearsipan
        - [Akses Repositori UT](http://repository.ut.ac.id)
        
        #### WBG Records Management Roadmap
        - Peta Jalan Manajemen Rekod
        - Standar internasional pengelolaan arsip
        - Praktik terbaik preservasi dokumen
        - Best practices global
        - [Akses WBG Roadmap](https://www.worldbank.org/en/archive/aboutus/records-management-roadmap)
        
        #### Archive Principles and Practice: an introduction to archives for non-archivists
        - [url](https://cdn.nationalarchives.gov.uk/documents/archives/archive-principles-and-practice-an-introduction-to-archives-for-non-archivists.pdf)
        
        #### Guide to Archiving Electronic Records: Edition 2 (Health Sciences Records and Archives Association, UK)
        - [url](https://the-hsraa.org/wp-content/uploads/2023/10/A-Guide-to-the-Archiving-of-Electronic-Records-A4-Version-for-Publication.pdf)
        
        #### Principles of Access to Archives (International Council on Archives, 2012)
        - [url](https://ica.org/app/uploads/2023/12/ICA_Access-principles_EN.pdf)
        
        #### Regulasi Indonesia
        - UU No. 43 Tahun 2009 tentang Kearsipan
        - Peraturan Pemerintah No. 28/2012 tentang Pelaksanaan Undang-Undang Nomor 43 Tahun 2009 tentang Kearsipan;
        - Peraturan Arsip Nasional Republik Indonesia No. 3/2024 tentang Pedoman Penyelenggaraan Pelatihan Kearsipan
        - Pedoman teknis kearsipan nasional     

        """)
               
    with tab5:        
        analysis_type = st.radio(
            "Select Analysis Source",
            ["Web Content", "Document Content", "Online Search"]
        )
        
        if analysis_type == "Online Search":
            st.markdown("""
                <style>
                .search-result {
                    background-color: #000000;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 20px 0;
                }
                </style>
            """, unsafe_allow_html=True)
            
            search_query = st.text_input("Masukkan kata kunci pencarian:")
            citation_style = st.selectbox(
                "Gaya Sitasi",
                ["APA", "Chicago", "Harvard"]
            )
            
            if search_query and st.button("Cari"):
                try:
                    with st.spinner("🔍 Mencari..."):
                        # Create model
                        compound_model = ChatGroq(
                            groq_api_key=os.getenv('GROQ_API_KEY'),
                            model_name="compound-beta"
                        )
                        
                        # Updated prompt
                        prompt = f"""Berikan analisis dalam Bahasa Indonesia tentang: {search_query}

Format jawaban HARUS menggunakan pemisah seperti ini:

===DEFINISI===
Definisi dan penjelasan istilah dalam Bahasa Indonesia

===PEMBAHASAN===
Penjelasan detail dalam paragraf yang utuh

===STANDAR===
Standar dan praktik terbaik yang berlaku

===REFERENSI===
Referensi dalam format {citation_style}"""

                        # Get and clean response
                        response = compound_model.invoke(prompt)
                        
                        if response:
                            # Get the content directly from the AIMessage object
                            content = response.content.replace('\\n', '\n')
                            
                            # Parse sections
                            sections: Dict[str, List[str]] = {}
                            current_section = None
                            
                            for line in content.split('\n'):
                                line = line.strip()
                                if line.startswith('===') and line.endswith('==='):
                                    current_section = line.strip('=')
                                    sections[current_section] = []
                                elif current_section and line:
                                    sections[current_section].append(line)
                            
                            # Display sections
                            for section, lines in sections.items():
                                if lines:
                                    # Join lines, preserving paragraph breaks
                                    markdown_content = "\n".join(lines).strip()
                                    section_content = markdown.markdown(markdown_content)
                                    
                                    # Use default alignment for references, justify for others
                                    text_align_style = "left" if section == "REFERENSI" else "justify"

                                    st.markdown(f"""
                                        <div style='background-color: #000000; 
                                            padding: 20px; 
                                            border-radius: 10px; 
                                            margin: 20px 0;'>
                                            <h3 style='color: #1F77B4; 
                                                margin-bottom: 15px;
                                                border-bottom: 2px solid #1F77B4;
                                                padding-bottom: 10px;'>
                                                {section}
                                            </h3>
                                            <div style='font-size: 1rem; 
                                                line-height: 1.8; 
                                                text-align: justify;'>
                                            <div style='font-size: 1rem; line-height: 1.8; text-align: {text_align_style};'>
                                                {section_content}
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)
                            
                            # Handle citations
                            if "REFERENSI" in sections and sections["REFERENSI"]:
                                citations = ' '.join(sections["REFERENSI"])
                            # The references are already displayed above. This is just for the download button.
                            if "REFERENSI" in sections:
                                # Join with newlines to preserve list format in the downloaded file
                                citations = "\n".join(sections["REFERENSI"])
                                if st.download_button(
                                    "📥 Unduh Referensi",
                                    citations,
                                    file_name=f"referensi_{citation_style.lower()}.txt",
                                    mime="text/plain"
                                ):
                                    st.success("Referensi berhasil diunduh!")
                                    
                        else:
                            st.error("Tidak ada hasil yang ditemukan.")
                            
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Search error: {traceback.format_exc()}")

        elif analysis_type == "Web Content":
            url = st.text_input("Enter URL to analyze:")
            if url and st.button("Analyze Web Content"):
                try:
                    # Fetch content
                    with st.spinner("📥 Fetching content..."):
                        parsed_url = urlparse(url)
                        if not all([parsed_url.scheme, parsed_url.netloc]):
                            st.error("Invalid URL format")
                            return
                            
                        content = fetch_url_content(url)
                        if not content:
                            st.error("Could not extract content from URL")
                            return
                            
                        st.success("Content fetched successfully!")

                    # Initialize model and analyze
                    with st.spinner("🔍 Analyzing content..."):
                        compound_model = ChatGroq(
                            groq_api_key=os.getenv('GROQ_API_KEY'),
                            model_name="compound-beta"
                        )

                        prompt = f"""Analyze this web content and provide a comprehensive analysis in Indonesian:

{content[:2000]}

Format your response as follows:

1. A clear summary in 2-3 well-structured paragraphs

2. A list of key points marked with asterisks (*)

Keep the format clean and consistent. Avoid any special characters except asterisks for bullet points."""

                        # Get and display analysis
                        response = compound_model.invoke(prompt)
                        
                        if response:
                            st.success("Analysis completed!")
                            
                            # Get content and replace escaped newlines
                            clean_response = response.content.replace('\\n', '\n')
                            
                            # Split content into sections
                            parts = clean_response.split("\n*")
                            
                            # Display main paragraphs
                            summary_html = markdown.markdown(parts[0].strip())

                            st.markdown("""
                                <div style='background-color: #000000; 
                                    padding: 20px; 
                                    border-radius: 10px; 
                                    margin: 20px 0;
                                    text-align: justify;
                                    line-height: 1.6;'>
                                    {}
                                </div><br>
                            """.format(summary_html), unsafe_allow_html=True)
                            
                            # Display key points
                            if len(parts) > 1:
                                st.markdown("""
                                    <div style='background-color: #000000; 
                                        padding: 20px; 
                                        border-radius: 10px; 
                                        margin: 20px 0;'>
                                        <h4 style='color: #1F77B4; margin-bottom: 15px;'>
                                            Poin-poin kunci:
                                        </h4>
                                """, unsafe_allow_html=True)
                                
                                for point in parts[1:]:
                                    if point.strip():
                                        st.markdown(f"&bull; {point.strip()}")
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.error("No analysis generated. Please try again.")
                            
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Analysis error: {traceback.format_exc()}")

        elif analysis_type == "Document Content":
            uploaded_file = st.file_uploader(
                "Upload document for analysis", 
                type=["pdf", "txt", "docx"]
            )
            
            if uploaded_file and st.button("Analyze Document"):
                try:
                    with st.spinner("📄 Processing document..."):
                        text = get_document_text(uploaded_file)
                        
                        # Initialize model
                        compound_model = ChatGroq(
                            groq_api_key=os.getenv('GROQ_API_KEY'),
                            model_name="compound-beta"
                        )
                        
                        prompt = f"""Analyze this document content in Indonesian language:

Text: {text[:3000]}... [truncated]

Provide a clear analysis with these exact sections:

Ringkasan Konten:
[Write a clear 3-4 paragraph summary]

[Write key points]
"""
                        
                        with st.spinner("🔍 Generating analysis..."):
                            response = compound_model.invoke(prompt)
                            
                            if response:
                                st.success("Analysis completed!")
                                
                                # Get content and replace escaped newlines
                                clean_response = response.content.replace('\\n', '\n')
                                clean_response = clean_response.replace("\n\n", "\n").strip()
                                
                                # Dynamically parse sections based on titles ending with a colon
                                lines = clean_response.split('\n')
                                sections = {}
                                current_section_title = None
                                current_section_content = []
                                
                                for line in lines:
                                    if line.strip().endswith(':') and len(line.strip().split()) < 5: # Likely a title
                                        if current_section_title:
                                            sections[current_section_title] = '\n'.join(current_section_content)
                                        current_section_title = line.strip()
                                        current_section_content = []
                                    elif current_section_title:
                                        current_section_content.append(line)
                                if current_section_title: # Add the last section
                                    sections[current_section_title] = '\n'.join(current_section_content)

                                # Display the parsed sections
                                for title, content in sections.items():
                                    # Convert markdown content to HTML
                                    content_html = markdown.markdown(content.strip())
                                    st.markdown(f"""
                                        <div style='background-color: #000000; padding: 20px; border-radius: 10px; margin: 20px 0; border: 1px solid #dee2e6;'>
                                            <h3 style='color: #1F77B4; margin-bottom: 15px; border-bottom: 2px solid #1F77B4; padding-bottom: 10px;'>
                                                {title.replace(":", "")}
                                            </h3>
                                            <div style='font-size: 1rem; line-height: 1.8; text-align: justify;'>{content_html}</div>
                                        </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.error("No analysis generated. Please try again.")
                                
                except Exception as e:
                    st.error(f"Error analyzing document: {str(e)}")
                    logger.error(f"Document analysis error: {traceback.format_exc()}")

# Initialize the new components
cache_manager = CacheManager()
security_manager = SecurityManager()
system_monitor = SystemMonitor()

# Then modify the main chat interface function to use these components:
@system_monitor.monitor_performance
@cache_manager.cache_query(ttl=3600)
def process_chat_query(prompt: str, vectorstore: Any, llm: Any) -> dict:
    # Sanitize input
    prompt = security_manager.sanitize_input(prompt)
    
    # Check rate limiting
    if not security_manager.rate_limiter(st.session_state.get('client_ip', '0.0.0.0')):
        raise Exception("Rate limit exceeded. Please wait before sending more requests.")
    
    # Rest of the existing chat processing code
    # ...existing code...

def initialize_or_load_vectorstore() -> Chroma:
    """
    Initialize or load the vector store for document embeddings
    
    Returns:
        Chroma: Vector store instance
        
    Raises:
        Exception: For initialization errors
    """
    try:
        # Initialize NVIDIA embeddings using the custom class.
        # This requires an NVIDIA_API_KEY in your .env file.
        nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        if not nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY not found in environment variables.")
            
        embeddings = NvidiaEmbeddings(
            model="nvidia/nv-embed-v1",
            nvidia_api_key=nvidia_api_key,
            input_type="passage"
        )

        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings
        )
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
def fetch_url_content(url: str) -> str:
    """Fetch and extract content from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        response = requests.get(url, headers=headers, timeout=15, verify=True)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        
        if 'application/pdf' in content_type:
            # Handle PDF content
            import io
            pdf_file = io.BytesIO(response.content)
            doc = fitz.open(stream=pdf_file, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
            
        # Handle HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'iframe']):
            tag.decompose()
            
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=['content', 'main', 'article'])
        if main_content:
            return main_content.get_text(separator=' ', strip=True)
        
        # Fallback to all paragraphs
        return ' '.join([p.get_text(strip=True) for p in soup.find_all(['p', 'h1', 'h2', 'h3'])])
        
    except Exception as e:
        raise Exception(f"Error fetching URL: {str(e)}")

def main() -> None:
    """Main application entry point"""
    # Disable ChromaDB telemetry
    os.environ['ANONYMIZED_TELEMETRY'] = 'False'
    
    set_verbose(True)
    load_dotenv()
    
    # Initialize session state for showing admin panel
    if 'show_admin' not in st.session_state:
        st.session_state['show_admin'] = False

    # Show landing page if not accessing admin panel
    if not st.session_state['show_admin']:
        show_landing_page()
        return
    
    # Load and validate API keys
    groq_api_key = os.getenv('GROQ_API_KEY')
    nvidia_api_key = os.getenv('NVIDIA_API_KEY')
    # google_api_key is still used by image_analyzer, but not for embeddings.
    if not groq_api_key or not nvidia_api_key:
        st.error("Missing GROQ_API_KEY or NVIDIA_API_KEY. Please check your .env file.")
        st.stop()
    google_api_key = os.getenv("GOOGLE_API_KEY")

    # Google API key is now optional for the core RAG functionality
    if not google_api_key:
        logger.warning("GOOGLE_API_KEY not found. Image analysis features may be disabled.")
        st.stop()

    os.environ["GOOGLE_API_KEY"] = google_api_key
    
    # Create ChromaDB directory
    global CHROMA_DB_DIR
    CHROMA_DB_DIR = "chroma_db"
    if not os.path.exists(CHROMA_DB_DIR):
        os.makedirs(CHROMA_DB_DIR)

    # Initialize session state
    if 'uploaded_file_names' not in st.session_state:
        st.session_state.uploaded_file_names = set()
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = initialize_or_load_vectorstore()
    
    # Initialize doc_processor in session state
    if 'doc_processor' not in st.session_state:
        # Only initialize if vectorstore was created successfully
        if st.session_state.get('vectorstore'):
            st.session_state.doc_processor = UnifiedDocumentProcessor(st.session_state.vectorstore)
        else:
            # Handle case where vectorstore failed to initialize
            st.error("Document processor could not be initialized.")
            st.stop()

    # Initialize LLM and prompt template
    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="openai/gpt-oss-120b"
        )
        
        
            
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        st.stop()

    # Setup sidebar with admin controls
    setup_admin_sidebar()
    
    # Show main chat interface
    show_chat_interface(llm)
    
if __name__ == "__main__":
    main()
