# d:\pandasai\idp\landing_page.py
import streamlit as st
import os
import base64

# Define assets directory (needed by get_base64_image)
ASSETS_DIR = "assets"
# Note: main.py ensures this directory exists.

# --- Helper Function for Base64 Image Encoding ---
def get_base64_image(img_filename="gekdocs1.png"):
    """Loads an image from the assets folder and returns its base64 representation."""
    img_path = os.path.join(ASSETS_DIR, img_filename)
    try:
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        print(f"Warning: Landing page background image not found at: {img_path}")
        # Consider st.warning if context is guaranteed
        return None
    except Exception as e:
        print(f"Warning: Error loading background image '{img_filename}': {e}")
        return None

# --- Landing Page Function (Integrated and Refined) ---
def show_landing_page():
    """Displays the styled landing page content."""

    img_base64 = get_base64_image()
    bg_image_style = "none"
    if img_base64:
        bg_image_style = f"url('data:image/jpg;base64,{img_base64}')"

    # Inject custom CSS
    st.markdown(f"""
        <style>
        /* Hide default Streamlit elements */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}} /* Hide header bar on landing page */
        div[data-testid='stSidebar'] {{display: none;}} /* Hide sidebar specifically */

        /* Apply background */
        .stApp {{
            background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)),
                        {bg_image_style} !important;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
            background-attachment: fixed !important;
            background-color: #1E1E1E !important; /* Fallback color */
        }}

        /* Style the main content box */
        .landing-content {{
            animation: fadeIn 1s ease-in;
            background: rgba(255, 255, 255, 0.1); /* Slightly transparent white */
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem 3rem; /* Adjusted padding */
            max-width: 850px; /* Slightly wider */
            margin: 3rem auto; /* Adjusted margin */
            color: white;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            border: 1px solid rgba(255, 255, 255, 0.18);
            text-align: center; /* Center align text */
        }}

        .landing-content h1 {{
            font-size: 2.8rem; /* Larger title */
            margin-bottom: 0.5rem;
            color: #FF4B4B; /* Highlight color for title */
        }}

        .landing-content h3 {{
            font-size: 1.5rem;
            margin-bottom: 1.5rem;
            font-weight: 500;
        }}

        .landing-content p {{
            font-size: 1.1rem;
            line-height: 1.6;
            margin-bottom: 1rem;
            text-align: justify; /* Justify paragraphs */
        }}

        /* Style the button */
        .stButton > button {{
            background-color: #FF4B4B !important;
            color: white !important;
            padding: 12px 30px !important;
            border-radius: 10px !important;
            border: none !important;
            font-size: 1.2rem !important; /* Slightly larger font */
            font-weight: 600 !important; /* Bolder */
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            margin-top: 2rem !important;
            display: inline-block !important; /* Allows centering via parent */
        }}

        .stButton > button:hover {{
            background-color: #E03030 !important; /* Darker red on hover */
            transform: translateY(-2px) !important;
            box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4) !important; /* Enhanced shadow */
        }}

        /* Animation */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* Responsive adjustments */
        @media (max-width: 768px) {{
            .landing-content {{
                margin: 2rem 1rem;
                padding: 1.5rem 2rem;
            }}
            .landing-content h1 {{ font-size: 2.2rem; }}
            .landing-content h3 {{ font-size: 1.3rem; }}
            .landing-content p {{ font-size: 1rem; }}
            .stButton > button {{ font-size: 1.1rem; padding: 10px 25px !important; }}
        }}

        /* Container to center the button */
        .button-container {{
            text-align: center;
            padding-top: 1rem;
        }}
        </style>
    """, unsafe_allow_html=True)

    # --- Refined Landing Page Content ---
    st.markdown('<div class="landing-content">', unsafe_allow_html=True)

    st.markdown("# IDocspy: Tanya Jawab Dokumen Anda")
    st.markdown("### Pahami Lebih Cepat. Dapatkan Jawaban Akurat. Didukung AI.")

    st.markdown("""
        Punya tumpukan laporan, notulen rapat, atau dokumen teknis? Lelah mencari informasi spesifik secara manual? **IDocspy** mengubah cara Anda berinteraksi dengan dokumen.
    """)

    st.markdown("""
        **Proses Sekali, Tanya Berkali-kali:**
        Unggah dokumen Anda (PDF, gambar), dan biarkan IDocspy mengekstrak teks menggunakan OCR canggih atau ekstraksi langsung. Hasil teks dan indeks pemahaman AI **disimpan secara permanen**. Anda dapat memilih dokumen mana saja yang sudah diproses untuk diajak bicara kapan pun, tanpa perlu mengunggah ulang!
    """)

    st.markdown("""
        **Tanya Jawab Cerdas & Kontekstual:**
        Ajukan pertanyaan dalam bahasa natural. AI akan mencari bagian paling relevan dalam dokumen yang Anda pilih dan merumuskan jawaban akurat **hanya berdasarkan konteks dokumen tersebut**. Atur parameter pencarian (jumlah hasil, skor minimum, re-ranking) melalui panel Admin untuk hasil yang lebih presisi.
    """)

    st.markdown("""
        **Manfaat Utama:**
        *   **Hemat Waktu:** Temukan informasi spesifik dalam hitungan detik.
        *   **Pahaman Mendalam:** Ekstrak poin penting tanpa membaca keseluruhan teks.
        *   **Jawaban Terpercaya:** Dapatkan jawaban yang relevan dan bersumber langsung dari dokumen.
        *   **Fleksibel:** Pilih dari berbagai dokumen yang telah Anda proses.
    """)

    # --- Button Container ---
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button("Masuk ke Sistem", key="enter_system_button"):
        st.session_state.show_dashboard = True
        st.rerun() # Rerun to switch view
    st.markdown('</div>', unsafe_allow_html=True) # Close button container

    st.markdown('</div>', unsafe_allow_html=True) # Close landing-content div
