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
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.globals import set_verbose
from dotenv import load_dotenv
from streamlit.runtime.caching import cache_data, cache_resource
from datetime import datetime
import toml
import chromadb
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
        "llama-3.3-70b-versatile": lambda: ChatGroq(
            groq_api_key=os.getenv('GROQ_API_KEY'),
            model_name="llama-3.3-70b-versatile"
        ),
        "deepseek-coder": lambda: DeepSeekLLM(
            model="deepseek-ai/deepseek-r1",
            api_key=os.getenv('HUGGINGFACE_API_KEY'),
            temperature=0.5,
            max_tokens=512
        ),
        "smallthinker": lambda: DeepSeekLLM(
            model="PowerInfer/SmallThinker-3B-Preview",
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
        template="Context:\ncontent:{page_content}\nsource:{source}",
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

def show_landing_page() -> None:
    """Display the landing page with all content inside a centered box container."""
    # Set the background image URL
    background_image_url = "assets/logo-transparent3.png"  # Update this path to your image

    # Open the wrapper for centering
    st.markdown('<div class="landing-container-wrapper">', unsafe_allow_html=True)

    # Open the box container
    st.markdown('<div class="landing-container">', unsafe_allow_html=True)

    # Logo (centered)
    st.image("assets/logo-transparent3.png", width=350)  # Adjust width as needed

    # Title and Introduction
    st.title("Selamat Datang di Arsipy")
    st.write("""
    ### Chatbot Pintar untuk Referensi Manual Arsip dan Analisis Tulisan Tangan
    **Arsipy** adalah aplikasi berbasis AI yang memudahkan Anda untuk:
    - Mempelajari dan merujuk manual arsip melalui chatbot interaktif.
    - Menganalisis dokumen tulisan tangan untuk ekstraksi teks dan referensi.

    Dengan teknologi RAG (Retrieval-Augmented Generation) dan analisis gambar, Arsipy memastikan Anda mendapatkan informasi yang akurat dan terstruktur, baik dari teks digital maupun tulisan tangan.
    """)

    # Key Features
    st.write("""
    ### Kenapa Memilih Arsipy?
    - **Chatbot Interaktif**: Ajukan pertanyaan tentang manual arsip dan dapatkan jawaban instan.
    - **Analisis Tulisan Tangan**: Unggah gambar dokumen tulisan tangan, dan Arsipy akan mengekstrak teks untuk referensi.
    - **Referensi Terpercaya**: Jawaban diambil dari koleksi manual arsip yang terstruktur dan terverifikasi.
    - **Mudah Digunakan**: Cukup ketik pertanyaan atau unggah gambar, dan Arsipy akan membantu.
    - **Pembelajaran Cepat**: Pelajari prosedur dan tahapan pengarsipan dengan cepat.
    - **Akses 24/7**: Tersedia kapan saja, di mana saja.
    """)

    # Call-to-Action
    st.write("""
    ### Mulai Jelajahi Arsipy!
    Manfaatkan chatbot pintar dan analisis tulisan tangan untuk mempelajari manual arsip dengan cara yang lebih mudah dan interaktif. Klik tombol di bawah untuk memulai.
    """)

    # Centered "Access Admin Panel" button
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button("Klik disini lebih lanjut", key="access_admin_button"):
        st.session_state['show_admin'] = True
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Close the box container
    st.markdown('</div>', unsafe_allow_html=True)

    # Close the wrapper
    st.markdown('</div>', unsafe_allow_html=True)

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
    """Display admin controls when authenticated"""
    st.sidebar.header("Document Management")
    
    # File uploader
    uploaded_files = st.sidebar.file_uploader(
        "Upload Documents", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    # Process documents button
    if uploaded_files:
        if st.sidebar.button("Process Documents", key="process_docs_button"):
            process_uploaded_files(uploaded_files)
    
    # Show currently processed files
    if st.session_state.uploaded_file_names:
        st.sidebar.write("Processed Documents:")
        for filename in st.session_state.uploaded_file_names:
            st.sidebar.write(f"- {filename}")
    
    # Reset system
    st.sidebar.divider()
    st.sidebar.header("System Reset")
    if st.sidebar.button("Reset Everything", key="reset_everything_button"):
        if st.sidebar.checkbox("Are you sure? This will delete all processed documents."):
            try:
                # Clear cache first
                clear_cache()
                
                # Clear vector store
                if os.path.exists(CHROMA_DB_DIR):
                    shutil.rmtree(CHROMA_DB_DIR)
                    os.makedirs(CHROMA_DB_DIR)
                    st.session_state.uploaded_file_names.clear()
                    st.session_state.vectorstore = None
                
                st.sidebar.success("Complete reset successful!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error during reset: {str(e)}")
                logger.error(traceback.format_exc())

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

def process_uploaded_files(uploaded_files: List[Any]) -> None:
    """
    Process uploaded files and add them to the vector store
    
    Args:
        uploaded_files: List of uploaded file objects
    """
    try:
        # Validate input files
        if not uploaded_files:
            st.sidebar.warning("No files selected for processing")
            return
            
        # Initialize components
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=embeddings
            )
        
        vectorstore = st.session_state.vectorstore
        
        with st.spinner('Processing documents...'):
            for file in stqdm(uploaded_files):
                if file.name not in st.session_state.uploaded_file_names:
                    # Extract and validate text
                    try:
                        text = get_document_text(file)
                    except ValueError as e:
                        st.sidebar.warning(f"Skipping {file.name}: {str(e)}")
                        continue
                    
                    # Create and validate chunks
                    chunks = text_splitter.create_documents([text])
                    if not chunks:
                        st.sidebar.warning(f"No valid chunks created from {file.name}")
                        continue
                        
                    # Add metadata
                    for chunk in chunks:
                        chunk.metadata = {
                            "source": file.name,
                            "chunk_size": len(chunk.page_content)
                        }
                    
                    # Add to vectorstore
                    try:
                        vectorstore.add_documents(chunks)
                        st.session_state.uploaded_file_names.add(file.name)
                    except ValueError as e:
                        st.sidebar.error(f"Error processing {file.name}: {str(e)}")
                        continue
                        
        st.sidebar.success(f"Successfully processed {len(uploaded_files)} documents!")
        
    except Exception as e:
        st.sidebar.error(f"Error processing files: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def clear_cache() -> None:
    """Clear all cached data"""
    cache_data.clear()
    cache_resource.clear()
    
def show_chat_interface(llm: Union[LLM, Any]) -> None:
    """Display the main chat interface"""
    # Add logo
    col1, col2, col3 = st.columns([1,100,1])
    with col2:
        st.image("assets/logo-transparent3.png", width=350)
    
    # Create tabs for the main interface
    tab1, tab5, tab3, tab2, tab4 = st.tabs([
        "💬 Chatbot", 
        "🖼️ Analisis tulisan tangan", 
        "❓ Panduan", 
        "ℹ️ Tentang",
        "📚 Resources"
    ])    
    
    with tab1:
        # Model selection inside chatbot tab
        model_options = {
            "Llama-3.3-70b-versatile (Groq)": "llama-3.3-70b-versatile",
            "DeepSeek-R1 (HuggingFace)": "deepseek-coder",
            "SmallThinker-3B (HuggingFace)": "smallthinker",
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
            st.info("👋 Welcome to Arsipy, your AI-powered guide to archive manuals and handwriting analysis")
        
        # Initialize chat history in session state if it doesn't exist
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
        # Create a form for the chat input
        with st.form(key='chat_form'):
            prompt1 = st.text_input("Enter your question about the documents", key='question_input')
            submit_button = st.form_submit_button("Submit Question")
            
        # Display chat history
        for q, a in st.session_state.chat_history:
            st.write("Question:", q)
            st.write("Answer:", a)
            st.divider()
        
        if submit_button and prompt1:
            try:
                with memory_track():
                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = initialize_or_load_vectorstore()
                    
                    vectorstore = st.session_state.vectorstore
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
                            start = time.process_time()
                            
                            # Simulate analysis steps with multiple spinners
                            for i, message in enumerate(spinner_messages[1:], 1):
                                time.sleep(0.5)  # Brief pause for UX
                                st.spinner(message)
                            
                            # Get actual response
                            response = qa_chain.invoke({'query': prompt1})
                            elapsed_time = time.process_time() - start
                            
                            # Update chat history and display
                            st.session_state.chat_history.append((prompt1, response['result']))
                            st.write("Response:")
                            st.write(response['result'])
                            st.write(f"Response time: {elapsed_time:.2f} seconds")
                            
                            st.rerun()
                            
                    else:
                        st.warning("No documents found in the database. Please ask an admin to upload some documents.")
                        
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                logger.error(traceback.format_exc())

    # Add a clear chat history button
    if st.session_state.chat_history and st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
        
    # Footer
    st.markdown("---")
    st.markdown("Built by Adnuri Mohamidi with help from AI :orange_heart:", help="cyberariani@gmail.com")
    with tab2:
        st.write("""
        ### 🎯 Tentang Arsipy
        Unit Kearsipan dan Manajemen Rekod sering kali menghadapi manual, prosedur, dan panduan yang kompleks serta krusial untuk menjaga integritas dan aksesibilitas rekod. Namun, banyaknya informasi yang harus diingat seringkali menyulitkan dalam mengingat langkah-langkah atau detail tertentu, terutama ketika menangani tugas-tugas yang jarang atau tidak sering dilakukan. Hal ini dapat mengakibatkan ketidakefisienan atau kesalahan dalam proses kearsipan dan pengelolaan rekod.

        Sebagai contoh, bayangkan seorang staf yang sedang mempersiapkan dokumen langka untuk digitalisasi tetapi lupa dengan prosedur penanganan yang spesifik. Alih-alih harus membuka-buka halaman manual, mereka cukup menanyakan pada chatbot, "Apa proses untuk mendigitalisasi dokumen yang rapuh?" Aplikasi ini akan langsung mengambil langkah-langkah yang relevan—seperti menggunakan sarung tangan, menyiapkan pemindai, dan mengatur pencahayaan—dan menyajikannya dalam format yang jelas dan mudah diikuti.

        Arsipy adalah asisten AI yang dibangun untuk itu, membantu Anda mengakses dan memahami manual arsip dengan lebih efektif. Aplikasi ini menggunakan teknologi RAG (Retrieval-Augmented Generation) plus analisis gambar untuk menguraikan dan mengekstraksi teks dari dokumen tulisan tangan.
        
        Dengan Arsipy, Anda dapat:

        - Mempelajari dan merujuk manual arsip melalui chatbot interaktif yang mudah digunakan.
        - Menganalisis dokumen tulisan tangan untuk mengekstraksi teks dan referensi yang relevan.
        
        Arsipy memastikan Anda mendapatkan informasi yang akurat dan terstruktur, baik dari teks digital maupun tulisan tangan. Dengan demikian, Anda dapat lebih mudah mengakses dan memahami isi manual arsip, serta meningkatkan efisiensi dalam mencari informasi.

        ### 🔍 Fitur Utama
        **RAG-based Chatbot**
        - Menjawab pertanyaan tentang manual arsip
        - Referensi dari sumber terpercaya
        - Konteks yang akurat dan relevan

        **Analisis Tulisan Tangan**
        - Ekstraksi teks dari gambar
        - Pemrosesan dokumen tulisan tangan
        - Format hasil yang terstruktur

        **Manajemen Arsip**
        - Penyimpanan dokumen digital
        - Pencarian cepat
        - Pengorganisasian otomatis

        ### 💻 Teknologi
        - **Backend**: Python, ChromaDB, LangChain
        - **AI Models**: llama-3.3-70b-versatile (Groq's API), Google Gemini 2.0 Flash
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
        st.subheader("📝 Panduan Dasar")
        st.markdown("""
        #### 1️⃣ Chat dengan Manual Arsip
        1. Buka tab **💬 Chat**
        2. Ketik pertanyaan tentang manual arsip
        3. Tunggu jawaban dari chatbot
        4. Lihat referensi sumber yang disertakan

        #### 2️⃣ Analisis Tulisan Tangan
        1. Buka tab **🖼️ Analisis Gambar**
        2. Upload gambar tulisan tangan (.jpg, .png, .jpeg)
        3. Periksa teks yang diekstrak

        ### 📋 Format File yang Didukung

        #### Gambar Tulisan Tangan
        - Format: JPG, PNG, JPEG
        - Ukuran max: 5MB
        - Resolusi yang dianjurkan: 300dpi
        - Background: Putih/terang

        #### Manual Arsip
        - Format: PDF, TXT
        - Ukuran max: 10MB
        - Bahasa: Indonesia dan asing

        ### 💡 Tips Penggunaan

        #### Untuk Hasil Chat Optimal
        - Gunakan bahasa yang jelas
        - Berikan konteks spesifik
        - Tanyakan satu topik per chat
        - Manfaatkan history chat untuk mempertajam pertanyaan

        #### Untuk Analisis Gambar Optimal
        - Pastikan pencahayaan baik
        - Hindari bayangan
        - Foto tegak lurus
        - Fokuskan pada teks

        ### ❗ Troubleshooting

        #### Masalah Umum
        1. **Chat tidak merespon**
        - Refresh halaman
        - Periksa koneksi internet
        - Batas token model telah terlampaui. Mohon bersabar beberapa saat lagi
        
        2. **Analisis gambar gagal**
        - Coba format file berbeda
        - Kurangi ukuran file
        - Perbaiki kualitas gambar
        - Coba analisis ulang
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
       
        ### 🔒 Jaminan Kualitas
        - Validasi sumber
        - Pembaruan berkala
        - Verifikasi konten
        - Audit kepatuhan
        """)      
        
    with tab5:
        st.subheader("⚠️ Penting")
        st.info("""
        Fitur ini rencananya akan dikembangkan menjadi aplikasi mandiri yang dirancang khusus untuk arsiparis, sejarawan, dan peneliti guna memenuhi kebutuhan unik mereka. Mengingat mereka sering bekerja dengan gabungan materi cetak dan tulisan tangan, aplikasi ini akan menyediakan alat dan fitur khusus untuk mempermudah alur kerja, mendukung penelitian, dan meningkatkan temu balik informasi.
        
        """)
        image_analyzer_main()  # Call the imported image analyzer function        

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
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Initialize or load the existing Chroma database
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings
        )
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
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
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not groq_api_key or not google_api_key:
        st.error("Missing API keys. Please check your .env file.")
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
        st.session_state.vectorstore = None

    # Initialize LLM and prompt template
    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile"
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
