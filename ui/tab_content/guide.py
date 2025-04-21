import streamlit as st

def show_guide_content() -> None:
    """Display guide content"""
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
    """)

# Footer
    st.markdown("---")
    st.markdown("Built by Adnuri Mohamidi with help from AI :orange_heart:", help="cyberariani@gmail.com")