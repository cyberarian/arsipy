import streamlit as st
import base64
import os

def get_base64_image():
    img_path = os.path.join(os.path.dirname(__file__), "assets", "enhanced-logo-landing.png")
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def show_landing_page():
    # Initialize session state
    if 'show_admin' not in st.session_state:
        st.session_state['show_admin'] = False

    try:
        img_base64 = get_base64_image()
        bg_image = f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"Error loading background image: {e}")
        bg_image = "none"

    # Inject custom HTML/CSS with proper background handling
    st.markdown(f"""
        <style>
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        
        .stApp {{
            background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), 
                        url("{bg_image}") !important;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
            background-attachment: fixed !important;
            background-color: #1E1E1E !important;
        }}
        
        .landing-content {{
            animation: fadeIn 1s ease-in;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 3rem;
            max-width: 1200px;
            margin: 4rem auto;
            color: white;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }}
        
        .stButton > button {{
            background-color: #FF4B4B !important;
            color: white !important;
            padding: 12px 30px !important;
            border-radius: 10px !important;
            border: none !important;
            font-size: 1.1rem !important;
            font-weight: 500 !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            margin-top: 2rem !important;
            display: block !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }}
        
        .stButton > button:hover {{
            background-color: #FF3333 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 5px 15px rgba(255, 75, 75, 0.3) !important;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        @media (max-width: 768px) {{
            .landing-content {{
                margin: 2rem 1rem;
                padding: 2rem;
            }}
        }}

        .button-container {{
            text-align: center;
            padding-top: 1rem;
        }}
        </style>

        <div class="landing-content">
            <h1>Arsipy</h1>
            <h3>Selamat Datang di Arsipy, AI-Powered Archive Assistant</h3>
            <p>
                Mengelola arsip dengan baik bukan hanya soal keteraturan, tetapi juga tentang efisiensi, akurasi, dan akuntabilitas. Di sinilah Arsipy hadir, sebagai asisten cerdas berbasis teknologi AI dan Retrieval-Augmented Generation (RAG), untuk membantu Anda mengakses dan memahami manual arsip secara lebih cepat dan terpercaya.
            </p>
            <p>
                Melalui chatbot interaktif, Anda bisa mengajukan pertanyaan langsung dan mendapatkan jawaban instan yang bersumber dari koleksi manual arsip yang terstruktur dan terverifikasi.  Fitur web-search membantu Anda menggali wawasan tambahan dari berbagai sumber daring terpercaya, melengkapi informasi yang tersedia di dalam manual arsip.
            </p>
            <p>Dengan antarmuka yang sederhana dan akses 24/7, Arsipy dirancang untuk siapa saja yang ingin belajar pengarsipan dengan cara yang lebih efektif.  Mari jelajahi Arsipy dan temukan pengalaman baru dalam memahami dunia kearsipan, langsung dari ujung jari Anda.
            </p>
        <div class="button-container">
    """, unsafe_allow_html=True)

    # Add Streamlit button
    # Use columns to center the button, which is more reliable than CSS hacks.
    _, col2, _ = st.columns([1, 1, 1])
    with col2:
        if st.button("Masuk ke Sistem", key="enter_system", use_container_width=True):
            st.session_state['show_admin'] = True
            st.rerun()

    # Close the containers
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    show_landing_page()
