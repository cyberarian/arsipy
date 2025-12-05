import streamlit as st

def show_about_content() -> None:
    """Display about content"""
    st.write("""
    ### 🎯 Tentang Arsipy
    Unit Kearsipan dan Manajemen Rekod sering kali menghadapi manual, prosedur, dan panduan yang kompleks serta krusial untuk menjaga integritas dan aksesibilitas rekod. Namun, banyaknya informasi yang harus diingat seringkali menyulitkan dalam mengingat langkah-langkah atau detail tertentu, terutama ketika menangani tugas-tugas yang jarang atau tidak sering dilakukan.

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

# Footer
    st.markdown("---")
    st.markdown("Developed by Adnuri Mohamidi with help from AI :orange_heart:", help="cyberariani@gmail.com")