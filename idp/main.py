import streamlit as st
import pandas as pd
import time
import os # Make sure os is imported if using file_name parts

# --- Set page config FIRST ---
st.set_page_config(page_title="IDocspy - IDP & Q&A", page_icon="🚀", layout="wide")


# --- Import Local Modules ---
try:
    from landing_page import show_landing_page
    from utils import (
        load_or_create_manifest, save_manifest, # Removed find_available_documents
        check_login, faiss, find_available_documents_from_db # Import the new DB function
        # Removed PROCESSED_DIR import from utils
    )
    # Import tab display functions
    from tabs.chatbot_tab import display_chatbot_tab
    from tabs.admin_tab import display_admin_tab
    from tabs.database_tab import display_database_tab # Import the new tab function
    from tabs.mdr_tab import display_mdr_tab # Import the new MDR tab function
    from tabs.about_tab import display_about_tab
    # Import other core logic modules (ensure they exist)
    import ocr_providers
    import idp_extraction
    import rag_processor
    # --- Import constants including PROCESSED_DIR ---
    from constants import PROCESSED_DIR, ASSETS_DIR # Import core dirs from constants
    # Create directories if they don't exist (idempotent) - Moved here after import
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)
    # --- Import DB utils ---
    from db_utils import init_db, load_records_to_dataframe, delete_record, insert_record
except ImportError as e:
    st.error(f"Fatal Error: Failed to import core modules: {e}. "
             f"Please ensure `utils.py`, `landing_page.py`, `ocr_providers.py`, "
             f"`idp_extraction.py`, `rag_processor.py`, `constants.py`, `db_utils.py`, and the `tabs/` directory "
             f"with `chatbot_tab.py`, `admin_tab.py`, `database_tab.py`, `mdr_tab.py`, `about_tab.py` exist and are correct.")
    st.stop() # Stop execution if core modules are missing
except NameError as e:
     # Catch if faiss wasn't successfully imported in utils
    if 'faiss' in str(e):
        st.error("FAISS library failed to load or import correctly in utils.py. The application cannot run.")
        st.stop()
    else:
        st.error(f"Fatal Error: A required name is not defined during import: {e}")
        st.stop()


# --- Main Application Logic ---
def main():
    # --- Initialize Database ---
    init_db() # Ensure DB and table exist

    # Initialize session state keys if they don't exist
    # Consolidate all session state keys used across the app here
    defaults = {
        # App Flow / UI State
        "show_dashboard": False,
        "admin_logged_in": False,
        "current_user_role": None,      # To store "admin" or "creator"

        # Chatbot State
        "selected_doc_display_name": None, # User's selection in the selectbox
        "current_doc_index": None,      # Loaded FAISS index for chat
        "current_doc_chunks": None,     # Loaded chunks for chat
        "current_doc_tokenized_chunks": None, # Loaded tokenized chunks for BM25
        "current_doc_display_name": None,# Display name of the *actually* loaded doc
        "current_doc_base_filename": None,# Base filename of the loaded doc
        "current_doc_chunk_size": None, # Chunk size of loaded doc
        "current_doc_chunk_overlap": None,# Chunk overlap of loaded doc
        "last_query_result": None,      # Stores result of the last chat query
        # "current_doc_rm_suggestions": None, # RM suggestions - loaded on demand or stored elsewhere? Let's remove from default state for now.
        "current_doc_original_metadata": None, # Original metadata for the loaded doc

        # Admin Tab State
        "admin_uploaded_metadata": None, # Metadata of the file in the uploader
        "admin_processing_state": {},   # Stores details of the last processing run (includes text, quality, rm_suggestions, original_metadata)
        "admin_current_file_id": None,  # ID of the file currently in the uploader
        "admin_file_bytes": None,       # Bytes of the file in the uploader
        "admin_next_chunk_size": 500,   # Chunk size setting for next processing
        "admin_next_chunk_overlap": 50, # Chunk overlap setting for next processing

        # Global Settings (set by Admin, used by Chatbot)
        "global_retrieval_k": 7, # Increased default k
        "global_retrieval_threshold": 0.3,
        "global_retrieval_reranking": False,
        "global_retrieval_hyde": False,

        # Data / Manifest State
        "manifest_data": None,          # Loaded manifest content
        "available_documents": {},      # Dictionary of {display_name: info} for valid docs (now includes RM info)

        # Processed Records DataFrame (loaded from DB)
        "records_df": None, # Will be loaded from DB
        "original_records_df": None, # To store original for comparison in DB tab

        # MDR Tab State
        "mdr_extracted_data": pd.DataFrame(columns=["Document No.", "Revision No.", "Document Title (English)", "Document Type", "Status", "Required Handover Status", "Phase", "System", "Area/Unit/Location", "Discipline", "Document Type (Detail)", "Sub-Type", "Contractor Return Code", "SDR Code", "Originating Company", "Review Category", "Revision Date (DD-MMM-YY)", "Planned Submission Date (DD-MMM-YY)", "Planned Completion Date (DD-MMM-YY)", "Forecast Submission Date (DD-MMM-YY)", "Forecast Completion Date (DD-MMM-YY)", "Actual Completion Date (DD-MMM-YY)", "Tag Number(s)", "PO Number", "Supplier Name", "Alternative Doc No.", "Alternative Rev No.", "Document Has Holds (Y/N)", "Start-Up Critical (Y/N)", "Export Control (Y/N)", "Remarks/Notes"]), # Initialize as empty DataFrame
        "mdr_document_text": "",    # Text extracted from document for MDR
        "mdr_last_uploaded_filename": None, # To track if file changed
        "mdr_last_prompt": "",      # Last prompt sent to AI for MDR
        "mdr_last_ai_response": ""  # Last raw response from AI for MDR
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # --- Show Landing Page OR Dashboard ---
    if not st.session_state.get("show_dashboard", False):
        show_landing_page() # Function from landing_page.py
        # CSS to hide sidebar is now inside show_landing_page()
        st.stop() # Stop execution here, only show landing page

    # --- If show_dashboard is True, proceed with the main app ---

    # Ensure FAISS is available before showing the dashboard (checked during import now)
    if faiss is None:
         # This check might be redundant if import fails, but safe to keep
        st.error("FAISS library failed to load. The main application cannot run.")
        st.stop()

    # --- Load Manifest and Find Available Documents ONCE per session start (for dashboard) ---
    # Or if manifest_data is somehow reset
    if st.session_state.manifest_data is None:
        st.session_state.manifest_data = load_or_create_manifest() # Keep loading manifest for other potential uses (metadata, recovery)
        # --- Use the NEW function to populate available documents from the DATABASE ---
        st.session_state.available_documents = find_available_documents_from_db() # From utils.py
        # Set initial document selection if none is selected or selection is invalid
        if st.session_state.selected_doc_display_name not in st.session_state.available_documents:
            if st.session_state.available_documents:
                st.session_state.selected_doc_display_name = list(st.session_state.available_documents.keys())[0]
            else:
                st.session_state.selected_doc_display_name = None # No docs available

    # --- Load DB Records into Session State DataFrame ---
    # Load once or if it's None (e.g., after deletion/insertion)
    if st.session_state.records_df is None:
        st.session_state.records_df = load_records_to_dataframe() # Load from db_utils

    # --- Sidebar (Login/Logout - appears only on dashboard) ---
    with st.sidebar:
        # Optional Sidebar Logo
        try:
            # Try a specific small logo first
            logo_path_sidebar = os.path.join(ASSETS_DIR, "idocspy1.png")
            if os.path.exists(logo_path_sidebar):
                 st.image(logo_path_sidebar, width=250)
            else:
                 # Fallback to the main logo if small one not found
                 main_logo_path = os.path.join(ASSETS_DIR, "HKI-Sketch2.jpg")
                 if os.path.exists(main_logo_path):
                    st.image(main_logo_path, width=180) # Adjust width as needed
        except Exception as e:
            st.warning(f"Could not load sidebar logo: {e}")

        # Admin Login Section
        st.subheader("Admin Access")
        if not st.session_state.admin_logged_in:
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login", key="login_button"):
                role = check_login(username, password) # From utils.py, now returns role
                if role:
                    st.session_state.admin_logged_in = True
                    st.session_state.current_user_role = role
                    st.success("Login Successful!")
                    st.rerun() # Rerun to update UI reflecting login
                else:
                    st.error("Invalid username or password.")
        else:
            role_display = st.session_state.get("current_user_role", "Unknown").capitalize()
            st.success(f"✅ Logged in as {role_display}")
            if st.button("Logout", key="logout_button"):
                # Clear admin-specific session state on logout
                st.session_state.admin_logged_in = False
                st.session_state.admin_processing_state = {}
                st.session_state.admin_current_file_id = None
                st.session_state.admin_file_bytes = None
                st.session_state.admin_uploaded_metadata = None
                st.session_state.current_user_role = None
                # Keep chat state and global settings
                st.rerun() # Rerun to update UI reflecting logout

        # Back to Landing Page Button
        st.markdown("---")
        if st.button("⬅️ Back to Landing Page", key="back_to_landing"):
             st.session_state.show_dashboard = False
             # Clear potentially sensitive/large state when going back
             # st.session_state.admin_file_bytes = None # Optional: clear large data
             # st.session_state.current_doc_index = None
             # st.session_state.current_doc_chunks = None
             # st.session_state.current_doc_tokenized_chunks = None
             st.rerun()


    # --- Main Content Area with Tabs (Dashboard) ---
    st.title("IDocspy - Intelligent Document Processing") # Title for the dashboard part

    tab_titles = ["💬 Chatbot", "Document Control (MDR)", "🛠️ Admin Management", "🗄️ Database", "ℹ️ About"]
    tab_chatbot, tab_doc_control_mdr, tab_admin, tab_database, tab_about = st.tabs(tab_titles)

    # --- Render Tabs by Calling Imported Functions ---
    with tab_chatbot:
        display_chatbot_tab() # Function from tabs/chatbot_tab.py

    with tab_admin:
        display_admin_tab() # Function from tabs/admin_tab.py

    with tab_doc_control_mdr: # Updated variable name for clarity, though not strictly necessary if order is maintained
        # This tab's content is conditional on admin login, handled within the function
        display_mdr_tab() # Function from tabs/mdr_tab.py

    with tab_database:
        display_database_tab() # Function from tabs/database_tab.py

    with tab_about:
        display_about_tab() # Function from tabs/about_tab.py


# --- Entry Point ---
if __name__ == "__main__":
    main()
