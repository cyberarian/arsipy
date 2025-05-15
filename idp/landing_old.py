# d:\pandasai\idp\landing_page.py
import streamlit as st

def show_landing_page():
    """Displays the landing page content for the gekDocs IDP/Q&A application."""

    # Updated title reflecting the core Q&A functionality
    st.title("🚀 IDocspy: Tanya Jawab & Pahami Dokumen Anda dengan AI")

    st.markdown("""
    ## Dapatkan Jawaban Langsung dari Dokumen Anda, Tanpa Membaca Semuanya!

    Punya laporan panjang, notulen rapat berlembar-lembar, atau dokumen teknis yang rumit? Lelah mencari informasi spesifik secara manual?

    **IDocspy hadir sebagai solusi cerdas untuk:**

    1.  **Membaca & Memproses:** Mengubah dokumen (PDF, gambar) menjadi teks digital menggunakan teknologi OCR canggih atau ekstraksi langsung. Hasil proses **disimpan** untuk penggunaan berikutnya!
    2.  **Membuat Indeks Pemahaman:** Membangun 'peta' makna (indeks semantik) dari isi dokumen Anda menggunakan AI.
    3.  **Menjawab Pertanyaan Anda:** Memungkinkan Anda bertanya dalam bahasa natural dan mendapatkan jawaban akurat yang **dihasilkan AI berdasarkan konteks dokumen**, bukan dari pengetahuan umum internet.

    Lupakan pencarian manual yang melelahkan. Cukup unggah, proses sekali, lalu tanyakan apa saja tentang isi dokumen Anda!
    """)

    st.markdown("---")

    st.markdown("""
    ### ✨ Fitur Utama IDocspy:

    *   📄 **Unggah Dokumen Fleksibel:** Mendukung PDF, PNG, JPG, dan format lainnya.
    *   ℹ️ **Tampilkan Info Dokumen:** Lihat metadata penting seperti judul, penulis, jumlah halaman (PDF), atau dimensi (gambar).
    *   🧠 **Ekstraksi Teks Cerdas:**
        *   Pilih metode: OCR (Groq Vision, Google Gemini, Mistral, Tesseract) atau Ekstraksi Langsung (untuk PDF digital).
        *   **Hemat Waktu!** Hasil teks disimpan, tidak perlu proses ulang saat dokumen yang sama diunggah lagi (kecuali Anda memilih 'Force Reprocessing').
    *   🗺️ **Indeks Semantik Otomatis:** Teks dipecah dan diindeks untuk pemahaman makna yang mendalam oleh AI (menggunakan Sentence Transformers & FAISS).
    *   ❓ **Tanya Jawab Kontekstual (RAG):**
        *   Ajukan pertanyaan dalam bahasa biasa tentang isi dokumen.
        *   AI (Groq Llama 3) akan mencari bagian relevan dan merumuskan jawaban **hanya berdasarkan informasi dalam dokumen Anda**.
    *   📊 **Analisis Kualitas Teks:** Dapatkan skor dan metrik untuk menilai kualitas hasil ekstraksi teks.
    """)

    st.markdown("---")

    st.markdown("""
    ### ✅ Manfaat Utama Menggunakan IDocspy:

    *   **Temukan Informasi Super Cepat:** Dapatkan jawaban spesifik dari dokumen puluhan atau ratusan halaman dalam hitungan detik.
    *   **Pahami Dokumen Kompleks:** Ekstrak poin-poin penting, ringkasan, atau jawaban atas pertanyaan Anda tanpa perlu membaca keseluruhan teks.
    *   **Jawaban Terpercaya & Terkontekstual:** AI menjawab berdasarkan isi dokumen Anda, mengurangi risiko informasi yang tidak relevan atau salah.
    *   **Hemat Waktu & Tenaga:** Otomatiskan proses pencarian informasi dan analisis dokumen. Fokus pada tugas yang lebih strategis.
    *   **Proses Sekali, Tanya Berkali-kali:** Manfaatkan teks yang sudah diproses dan disimpan untuk pertanyaan-pertanyaan berikutnya.
    """)

    st.markdown("---")

    st.markdown("""
    ### 💡 Contoh Kasus Penggunaan:

    *   **Analisis Laporan:** "Apa saja risiko utama yang disebutkan di laporan kuartal 3?"
    *   **Review Notulen Rapat:** "Sebutkan keputusan penting terkait Proyek Alpha."
    *   **Riset Cepat:** "Jelaskan metodologi yang digunakan dalam studi [Nama Studi] berdasarkan jurnal ini."
    *   **Pengecekan Kepatuhan:** "Apakah kontrak ini menyebutkan klausul tentang kerahasiaan data?"
    *   **Dukungan Teknis:** "Bagaimana cara reset perangkat X menurut manual ini?"
    """)

    st.markdown("---")

    st.markdown("""
    ### 🛠️ Teknologi yang Digunakan:

    *   **OCR:** Groq Vision, Google Gemini, Mistral AI, Tesseract
    *   **Ekstraksi Langsung:** PyMuPDF (fitz)
    *   **Embeddings:** Sentence Transformers
    *   **Vector Search:** FAISS (CPU)
    *   **Language Model (Q&A):** Groq Llama 3 (70B)
    *   **Framework:** Streamlit
    """)

    st.markdown("---")

    st.markdown("""
    ### ⚠️ Disclaimer

    Aplikasi ini dikembangkan untuk tujuan edukasi dan demonstrasi kapabilitas IDP & RAG.
    Pengembang tidak bertanggung jawab atas penyalahgunaan atau dampak dari penggunaan aplikasi ini. Jangan mengunggah dokumen sensitif atau rahasia.
    """)

    # Footer
    st.markdown("---")
    st.markdown(
        "Built by Adnuri Mohamidi with help from AI :orange_heart: #OpenToWork #HireMe",
        help="cyberariani@gmail.com"
    )

    # Button to proceed to the dashboard
    st.markdown("---") # Separator before button
    if st.button("🚀 Mulai Proses & Tanya Dokumen!", key="start_processing_button"): # Updated button text slightly
         st.session_state.show_dashboard = True
         st.rerun() # Force rerun to update the view immediately
