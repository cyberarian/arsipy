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
from typing import List
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
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

def show_landing_page():
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

    # How It Works
    st.write("""
    ### Cara Menggunakan Arsipy:
    1. **Ajukan Pertanyaan**: Ketik pertanyaan Anda tentang manual arsip di kolom chat.
    2. **Unggah Gambar**: Unggah gambar dokumen tulisan tangan untuk analisis dan ekstraksi teks.
    3. **Dapatkan Jawaban**: Arsipy akan memberikan jawaban yang relevan dari manual arsip atau hasil analisis tulisan tangan.
    4. **Pelajari Lebih Lanjut**: Jelajahi referensi yang diberikan untuk pemahaman mendalam.
    5. **Simpan Referensi**: Simpan jawaban penting untuk referensi di masa mendatang.
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

def setup_admin_sidebar():
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

def show_admin_controls():
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

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text content from a PDF file"""
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

def get_document_text(file) -> str:
    """Get text content from a file based on its type"""
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

def process_uploaded_files(uploaded_files: List):
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

def clear_cache():
    """Clear all cached data"""
    cache_data.clear()
    cache_resource.clear()
    
def show_chat_interface(llm, prompt):
    """Display the main chat interface"""
    # Add logo
    col1, col2, col3 = st.columns([1,100,1])
    with col2:
        st.image("assets/logo-transparent3.png", width=350)
    
    # Create tabs for the main interface
    tab1, tab5, tab3, tab2, tab4 = st.tabs([
        "💬 Chat", 
        "🖼️ Analisis Gambar", 
        "❓ How-to", 
        "ℹ️ About",
        "📚 Resources"
    ])    
    
    with tab1:
        # Add a greeting message
        if not st.session_state.uploaded_file_names:
            st.info("👋 Welcome to Digital Archive Manual System - Your Document Management Solution")

        
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
        
        if submit_button and prompt1:  # Only process if there's a question and the button is clicked
            try:
                with memory_track():
                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = initialize_or_load_vectorstore()
                    
                    vectorstore = st.session_state.vectorstore
                    if len(vectorstore.get()['ids']) > 0:
                        document_chain = create_stuff_documents_chain(llm, prompt)
                        retriever = vectorstore.as_retriever()
                        retrieval_chain = create_retrieval_chain(retriever, document_chain)
                        
                        with st.spinner('Searching through documents...'):
                            start = time.process_time()
                            response = retrieval_chain.invoke({'input': prompt1})
                            elapsed_time = time.process_time() - start
                            
                            # Add the new Q&A to the chat history
                            st.session_state.chat_history.append((prompt1, response['answer']))
                            
                            # Display the latest response
                            st.write("Latest Response:")
                            st.write(response['answer'])
                            st.write(f"Response time: {elapsed_time:.2f} seconds")
                            
                            # Clear the input box by rerunning the app
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
        Arsipy adalah asisten AI yang dikembangkan khusus untuk referensi manual arsip dan analisis tulisan tangan.
        
        Aplikasi ini menggabungkan teknologi RAG (Retrieval Augmented Generation) dengan kemampuan analisis 
        tulisan tangan untuk memberikan pengalaman pencarian dan referensi yang lebih efisien.
        
        - Mempelajari dan merujuk manual arsip melalui chatbot interaktif.
        - Menganalisis dokumen tulisan tangan untuk ekstraksi teks dan referensi.

        Dengan teknologi RAG (Retrieval-Augmented Generation) dan analisis gambar, Arsipy memastikan Anda mendapatkan informasi yang akurat dan terstruktur, baik dari teks digital maupun tulisan tangan.

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
        - **AI Models**: llama-3.3-70b-versatile, Google Gemini, Tesseract OCR
        - **Frontend**: Streamlit
        - **Database**: Vector Store dengan Google AI Embeddings

        """)
        # Footer
        st.markdown("---")
        st.markdown("Built by Adnuri Mohamidi with help from AI :orange_heart:", help="cyberariani@gmail.com")
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
        3. Pilih metode analisis:
        - Tesseract OCR: Ekstraksi teks dasar
        - Google Gemini: Analisis AI lanjutan
        - Hybrid: Kombinasi OCR dan AI
        4. Tunggu hasil analisis
        5. Periksa teks yang diekstrak

        ### 📋 Format File yang Didukung

        #### Gambar Tulisan Tangan
        - Format: JPG, PNG, JPEG
        - Ukuran max: 5MB
        - Resolusi minimal: 300dpi
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
        - Manfaatkan history chat

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
        
        2. **Analisis gambar gagal**
        - Coba format file berbeda
        - Kurangi ukuran file
        - Perbaiki kualitas gambar

        3. **Hasil tidak akurat**
        - Gunakan mode Hybrid
        - Tingkatkan kualitas input
        - Coba analisis ulang
            """)
        
        # Footer
        st.markdown("---")
        st.markdown("Built by Adnuri Mohamidi with help from AI :orange_heart:", help="cyberariani@gmail.com")
    
    with tab4:
        st.title("📚 Sumber Dokumen")
        
        st.markdown("""
        Sistem ini menggunakan sumber-sumber terpercaya untuk memastikan keakuratan informasi dalam pengelolaan arsip:
        
        ### 🏛️ Sumber Utama
        
        #### Repositori Universitas Terbuka (UT)
        - Manual arsip dan dokumen akademik
        - Panduan pengelolaan arsip institusional
        - Standar prosedur kearsipan
        - [Akses Repositori UT](http://repository.ut.ac.id)
        
        #### Kerangka Kerja Bank Dunia
        - Peta Jalan Manajemen Rekaman
        - Standar internasional pengelolaan arsip
        - Praktik terbaik preservasi dokumen
        - Best practices global
        
        #### Regulasi Indonesia
        - **UU No. 43 Tahun 2009** tentang Kearsipan
        - Peraturan pelaksanaan terkait
        - Standar Nasional Indonesia (SNI) Kearsipan
        - Pedoman teknis kearsipan nasional
        
        ### 📋 Standar yang Diterapkan
        - Autentikasi dokumen
        - Integritas data
        - Preservasi digital
        - Pengelolaan metadata
        - Sistem klasifikasi arsip
        
        ### 🔒 Jaminan Kualitas
        - Validasi sumber
        - Pembaruan berkala
        - Verifikasi konten
        - Audit kepatuhan
        """)
        
        # Footer
        st.markdown("---")
        st.markdown("Built by Adnuri Mohamidi with help from AI :orange_heart:", help="cyberariani@gmail.com")    
        
    with tab5:
        
        image_analyzer_main()  # Call the imported image analyzer function
        
        # Footer
        st.markdown("---")
        st.markdown("Built by Adnuri Mohamidi with help from AI :orange_heart:", help="cyberariani@gmail.com")
def initialize_or_load_vectorstore():
    """Initialize or load the vector store for document embeddings"""
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
    
def main():
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
        
        prompt = ChatPromptTemplate.from_template("""
           Your role: Your name is Arsipy, Chatbot Pintar untuk Referensi Manual Arsip dan Analisis Tulisan Tangan
            Language: Dynamically adapt your responses to match the user's language with formal tone
            Function: Assist the user in finding relevant information within provided documents, including names, titles, locations, history, tables, images, and other relevant texts. Keep responses brief and accurate; provide detailed explanations only when specifically requested.
            Greetings: respond to the greetings accordingly.
            
            When a user inquires about the source of the document, provide the exact title as written in the top or beginning of the document, including any subtitles or headings. Ensure that the title is:

            Accurately extracted: Retrieve the title from the document with high precision, without introducing any errors or modifications.
            Exactly as written: Preserve the original wording, punctuation, and formatting of the title, including any special characters or symbols.
            Complete and comprehensive: Include any subtitles, headings, or other relevant information that appears at the beginning of the document, to provide context and clarity.
            Example Response:

            If the document begins with the title "2022 Annual Report: Financial Highlights and Strategic Outlook", the system should respond with the exact same title, without any modifications or abbreviations.

            Input:

            User query: "What is the source of this document?"
            Document text: "2022 Annual Report: Financial Highlights and Strategic Outlook"
            Output:

            System response: "The source of this document is: 2022 Annual Report: Financial Highlights and Strategic Outlook"

            Guidelines:
            1. Format responses as follows:
            - Use bullet points for steps, procedures, or stages
            - Present sequential information in numbered lists
            - Break down complex answers into clear bullet points
            
            2. Response Structure:
            - Start with direct answer
            - List steps/procedures with bullets
            - End with relevant context if needed
                                
            3. Content Rules:
            - Base your responses strictly on the document's content and context
            - Provide answers that directly address the user's question only
            - Do not respond user's questions with irrelevant, misleading, or incomplete information
            - Present table data in a clear and logical format for easy understanding
            - Strive for accuracy and relevance in all responses
            
            lang:id-ID
            
            Context:
            {context}
            
            Question: {input}
            """)
            
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        st.stop()

    # Setup sidebar with admin controls
    setup_admin_sidebar()
    
    # Show main chat interface
    show_chat_interface(llm, prompt)
    
if __name__ == "__main__":
    main()