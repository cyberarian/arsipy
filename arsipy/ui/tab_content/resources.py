import streamlit as st

def show_resources_content() -> None:
    """Display resources content"""
    st.title("📚 Sumber Dokumen")
    st.markdown("""
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
    - [Akses WBG Roadmap](https://www.worldbank.org/en/archive/aboutus/records-management-roadmap)
    
    #### Archive Principles and Practice
    - [Introduction to archives for non-archivists](https://cdn.nationalarchives.gov.uk/documents/archives/archive-principles-and-practice-an-introduction-to-archives-for-non-archivists.pdf)
    
    #### Regulasi Indonesia
    - UU No. 43 Tahun 2009 tentang Kearsipan
    - PP No. 28/2012 tentang Pelaksanaan UU Kearsipan
    - Peraturan ANRI No. 3/2024
    """)

# Footer
    st.markdown("---")
    st.markdown("Developed by Adnuri Mohamidi with help from AI :orange_heart:", help="cyberariani@gmail.com")