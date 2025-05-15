# d:\pandasai\idp\tabs\admin_tab.py
import streamlit as st
import os
import pandas as pd
import pickle
import time # Import the time module
from datetime import datetime
import json # Import json for displaying suggestions

# --- Import necessary functions and constants ---
try:
    # Import from parent directory modules
    from utils import (
        extract_metadata, load_or_create_manifest, save_manifest,
        load_document_data, find_available_documents_from_db # Use the DB version
        , PROCESSED_DOCS_MANIFEST_PATH # Import manifest path for deletion consistency
    )
    # --- Updated Import: Replace extract_direct_text with specific function ---
    from ocr_providers import (
        run_ocr, extract_direct_text_pymupdf,
        calculate_and_store_quality,
        extract_tables_from_pdf_bytes # Import new table extraction function
    )
    # If you plan to add PyPDF2 direct extraction later, add: , extract_direct_text_pypdf2

    from rag_processor import create_index_from_text
    # --- Import LLM functions ---
    from idp_extraction import analyze_content_for_rm_suggestions # Corrected import if needed
    # --- Import constants including PROCESSED_DIR ---
    from constants import OCR_METRICS, PROCESSED_DIR # <-- Make sure PROCESSED_DIR is imported here
    import faiss # FAISS is needed directly here for saving index
    # --- Import DB functions ---
    from db_utils import insert_record, load_records_to_dataframe, delete_record
except ImportError as e:
    st.error(f"Admin Tab Error: Failed to import required functions/modules: {e}")
    st.stop()
except NameError as e:
    # Catch if faiss wasn't successfully imported in utils
    if 'faiss' in str(e):
        st.error("FAISS library failed to load. Admin tab requires FAISS.")
        st.stop()
    else:
        st.error(f"Admin Tab Error: A required name is not defined: {e}")
        st.stop()


def display_admin_tab():
    """Renders the Admin Management tab content."""

    if not st.session_state.get("admin_logged_in", False):
        st.warning("🔒 Please log in via the sidebar to access Admin Management.")
        return # Stop rendering if not logged in

    # Determine if the current user is a 'creator'
    is_creator_role = st.session_state.get("current_user_role") == "creator"

    st.header("🛠️ Admin Management")
    st.markdown("---")

    # --- Section 1: Upload Document ---
    st.subheader("1. Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file to process",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp", "md", "txt"], # Added md and txt
        key="admin_file_uploader",
        # Reset processing state when a new file is uploaded
        on_change=lambda: st.session_state.update(
            admin_processing_state={}, admin_current_file_id=None,
            admin_file_bytes=None, admin_uploaded_metadata=None
        )
    )

    # Process uploaded file info only if a file is present
    if uploaded_file:
        # Check if it's a new file or the same file hasn't been processed yet
        if st.session_state.get("admin_file_bytes") is None or st.session_state.get("admin_current_file_id") != uploaded_file.file_id:
            try:
                file_bytes = uploaded_file.getvalue()
                st.session_state.admin_file_bytes = file_bytes
                st.session_state.admin_original_filename = uploaded_file.name # Store original filename
                st.session_state.admin_current_file_id = uploaded_file.file_id

                # --- Handle Metadata Extraction based on file type ---
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                if file_extension in [".md", ".txt"]:
                    # Basic metadata for text files
                    metadata = {
                        "Filename": uploaded_file.name,
                        "File Type": uploaded_file.type,
                        "File Size (KB)": f"{uploaded_file.size / 1024:.2f}",
                        "Page Count": "N/A", # Not applicable
                    }
                    st.session_state.admin_uploaded_metadata = metadata
                else:
                    # Use existing function for PDF/Images
                    metadata = extract_metadata(file_bytes, uploaded_file.name, uploaded_file.type)
                    if metadata is not None: metadata["File Size (KB)"] = f"{uploaded_file.size / 1024:.2f}"
                    st.session_state.admin_uploaded_metadata = metadata

            except Exception as e:
                st.error(f"Error reading file or extracting metadata: {e}")
                # Reset state if reading fails
                st.session_state.update(admin_file_bytes=None, admin_current_file_id=None, admin_uploaded_metadata=None)

        # Display metadata if available
        st.markdown("**Document Information (from Upload)**")
        doc_meta = st.session_state.get("admin_uploaded_metadata")
        if doc_meta:
            # Define preferred display order
            display_order = ["Filename", "File Type", "File Size (KB)", "Page Count", "Title", "Author", "Subject", "Dimensions (WxH)", "Image Format", "Creation Date", "Modification Date"]
            meta_col1, meta_col2 = st.columns(2)
            with meta_col1:
                for i, key in enumerate(display_order):
                    if i % 2 == 0 and key in doc_meta and doc_meta[key] != "N/A" and doc_meta[key] is not None:
                        st.markdown(f"**{key}:** {doc_meta[key]}")
            with meta_col2:
                 for i, key in enumerate(display_order):
                    if i % 2 != 0 and key in doc_meta and doc_meta[key] != "N/A" and doc_meta[key] is not None:
                        st.markdown(f"**{key}:** {doc_meta[key]}")

            # Display any other metadata found below columns
            other_keys = [k for k in doc_meta if k not in display_order and k != "Error"]
            if other_keys:
                 st.markdown("---")
                 st.markdown("**Other Metadata:**")
                 for key in other_keys:
                     st.markdown(f"**{key}:** {doc_meta[key]}")
            # Show error if metadata extraction failed partially
            if "Error" in doc_meta:
                st.warning(f"Metadata Extraction Note: {doc_meta['Error']}")
        elif st.session_state.get("admin_file_bytes"):
            st.info("Metadata could not be extracted for this file type or an error occurred.")
        else:
            st.info("File could not be read.") # Should not happen if uploaded_file is True, but good fallback
        st.markdown("---")

    # --- Section 2: Text Extraction & Indexing Settings ---
    st.subheader("2. Text Extraction & Indexing")

    # Disable options if no file is uploaded
    file_uploaded = uploaded_file is not None and st.session_state.get("admin_file_bytes") is not None

    # --- Settings Section - Conditionally Disable for Creator ---
    if is_creator_role:
        st.info("Processing settings are managed by the Administrator.")
        # For creators, these settings are not changeable but we still need their values for processing.
        # We rely on the session state defaults or admin-set values.
        # We can display them as disabled or just omit them. For simplicity, we'll omit direct interaction.
        # Ensure force_reprocess is available for creators if desired, or default it.
        force_reprocess = st.checkbox(
            "Force Reprocessing",
            key="admin_force_reprocess_cb_creator", # Use a different key if needed or manage state carefully
            help="If checked, ignores any saved text/index and re-runs OCR/extraction, RM analysis, and indexing.",
            disabled=(not file_uploaded)
        )
        # Creators will use the globally set OCR provider and chunking settings.
        # No need for them to see/change these options.
        # We'll fetch these from session_state when processing.
        use_direct_extraction = st.session_state.get("admin_direct_extract_cb_global", False) # Assume a global state if needed
        ocr_provider = st.session_state.get("admin_ocr_provider_select_global", "Groq") # Default to a provider
    else: # Full Admin
        force_reprocess = st.checkbox(
            "Force Reprocessing",
            key="admin_force_reprocess_cb",
            help="If checked, ignores any saved text/index and re-runs OCR/extraction, RM analysis, and indexing.",
            disabled=(not file_uploaded)
        )

        file_extension = os.path.splitext(uploaded_file.name)[1].lower() if file_uploaded else None
        is_pdf = file_extension == '.pdf'
        is_text_based = file_extension in ['.md', '.txt']
        use_direct_extraction = st.checkbox(
            "Attempt Direct Text Extraction (Digital PDFs Only)",
            key="admin_direct_extract_cb_global", # Key for global setting
            value=st.session_state.get("admin_direct_extract_cb_global", False),
            help="If checked and PDF is digital, skips OCR. Overrides OCR provider selection.",
            disabled=(not is_pdf or force_reprocess or is_text_based)
        )

        ocr_provider_options = ["Groq", "Google", "Mistral", "Tesseract"]
        ocr_provider = st.selectbox(
            "Select OCR Provider (if not Direct Extraction)",
            options=ocr_provider_options,
            index=ocr_provider_options.index(st.session_state.get("admin_ocr_provider_select_global", "Groq")),
            key="admin_ocr_provider_select_global", # Key for global setting
            disabled=(use_direct_extraction or not file_uploaded)
        )

        st.markdown("**Chunking Settings (for NEXT processing run):**")
        admin_next_chunk_size_val = st.slider(
            "Chunk Size (chars)", min_value=100, max_value=2000,
            value=st.session_state.get("admin_next_chunk_size", 500),
            step=50, key="admin_chunk_size_slider",
            help="Size of text chunks for indexing."
        )
        admin_next_chunk_overlap_val = st.slider(
            "Chunk Overlap (chars)", min_value=0, max_value=500,
            value=st.session_state.get("admin_next_chunk_overlap", 50),
            step=10, key="admin_chunk_overlap_slider",
            help="Overlap between consecutive chunks."
        )

    # --- Section 3: Process & Index Document ---
    st.subheader("3. Process & Index Document")
    st.warning("Privacy Notice: Cloud services (Groq, Google, Mistral) may be used for OCR/LLMs if selected. Ensure compliance with your data policies.")
    privacy_consent = st.checkbox("I understand and accept potential cloud usage.", key="admin_privacy_consent_cb")

    # Disable button if no file, no consent, or FAISS is missing
    process_button_disabled = (
        not file_uploaded or
        not privacy_consent or
        faiss is None # Check if FAISS loaded successfully
    )
    process_index_button = st.button(
        "⚙️ Process, Analyze & Index Document",
        key="admin_process_index_button",
        disabled=process_button_disabled
    )

    if process_index_button:
        # Re-check conditions just before processing
        if not file_uploaded or faiss is None:
             st.error("Cannot process: Missing uploaded file data or FAISS library.")
        else:
            # Get necessary info from session state and widgets
            file_bytes = st.session_state.admin_file_bytes
            file_id = st.session_state.admin_current_file_id # Unique ID for the upload instance
            original_filename = st.session_state.admin_original_filename # Use stored filename
            current_chunk_size = st.session_state.admin_next_chunk_size
            current_chunk_overlap = st.session_state.admin_next_chunk_overlap
            # For creator role, ensure these settings are pulled from session state (set by admin)
            # The `ocr_provider` and `use_direct_extraction` variables are already set based on role earlier.
            # If they were not set for creator, we'd fetch from session state here:
            # actual_ocr_provider = st.session_state.get("admin_ocr_provider_select_global", "Groq")
            # actual_use_direct_extraction = st.session_state.get("admin_direct_extract_cb_global", False)

            original_metadata_from_upload = st.session_state.get("admin_uploaded_metadata", {}) # Get metadata captured at upload

            # Construct a base filename incorporating settings for uniqueness
            base_filename = f"{file_id}_cs{current_chunk_size}_co{current_chunk_overlap}"
            text_save_path = os.path.join(PROCESSED_DIR, f"{base_filename}.md")
            index_save_path = os.path.join(PROCESSED_DIR, f"{base_filename}.index") # Path for FAISS index
            tokenized_chunks_save_path = os.path.join(PROCESSED_DIR, f"{base_filename}_tok_chunks.pkl") # Path for tokenized chunks
            chunks_save_path = os.path.join(PROCESSED_DIR, f"{base_filename}_chunks.pkl")

            # Reset processing state for this run
            st.session_state.admin_processing_state = {
                "base_filename": base_filename,
                "original_filename": original_filename,
                "chunk_size": current_chunk_size,
                "chunk_overlap": current_chunk_overlap,
                "result_text": None,
                "processing_provider": None,
                "quality": None,
                "rm_suggestions": None, # Initialize RM suggestions
                "original_metadata": original_metadata_from_upload # Store metadata from upload
                # No need to store index/chunks here, they are loaded/created below
            }

            text_result = None
            provider_used = None
            processing_successful = False
            index = None # Initialize index/chunks to None
            chunks = None
            tokenized_chunks = None # Initialize tokenized chunks
            rm_suggestions = None # Initialize RM suggestions for this run
            db_record_to_insert = None # Initialize DB record dict
            manifest = st.session_state.manifest_data # Get current manifest
            extracted_table_files = [] # To store paths of saved JSON table files

            # --- Check if already processed and exists (unless forcing reprocess) ---
            load_existing = False
            if base_filename in manifest and not force_reprocess:
                 existing_data = manifest[base_filename]
                 # Construct full paths from manifest relative paths
                 existing_index_path = os.path.join(PROCESSED_DIR, os.path.basename(existing_data.get("index_path","")))
                 existing_chunks_path = os.path.join(PROCESSED_DIR, os.path.basename(existing_data.get("chunks_path","")))
                 existing_tokenized_path = os.path.join(PROCESSED_DIR, os.path.basename(existing_data.get("tokenized_chunks_path",""))) # Path for tokenized chunks
                 # Check if the files actually exist
                 if os.path.exists(existing_index_path) and os.path.exists(existing_chunks_path) and os.path.exists(existing_tokenized_path):
                     load_existing = True
                 else:
                     st.warning(f"Manifest entry for '{base_filename}' exists, but files not found. Reprocessing.")
                     # Clean up potentially stale manifest entry
                     if base_filename in manifest: del manifest[base_filename] # Remove directly
                     save_manifest(manifest)
                     st.session_state.manifest_data = manifest # Update session state manifest

            # --- Attempt to Load Existing Data ---
            if load_existing:
                 st.info(f"Found existing processed data for '{original_filename}' with these settings. Loading...")
                 try:
                     # Use the loading function (which is cached)
                     index, chunks, tokenized_chunks = load_document_data(existing_index_path, existing_chunks_path, existing_tokenized_path) # Load all three files
                     if index is not None and chunks is not None and tokenized_chunks is not None:
                         # Try to load the corresponding text file as well
                         existing_text_path = os.path.join(PROCESSED_DIR, f"{base_filename}.md")
                         if os.path.exists(existing_text_path):
                             try:
                                 with open(existing_text_path, "r", encoding="utf-8") as f:
                                     text_result = f.read()
                                 provider_used = "Loaded from Saved Files"
                             except Exception as e:
                                 text_result = f"Index/Chunks loaded, but text file ({existing_text_path}) could not be read: {e}"
                                 provider_used = "Loaded Index/Chunks (Text Read Error)"
                         else:
                             text_result = f"Index/Chunks loaded, but original text file ({existing_text_path}) not found."
                             provider_used = "Loaded Index/Chunks Only"

                         # Load RM suggestions and original metadata from manifest
                         rm_suggestions = existing_data.get("rm_suggestions")
                         original_metadata_from_manifest = existing_data.get("original_metadata")
                         extracted_table_files = existing_data.get("table_files", []) # Load table file paths

                         # Update processing state with loaded info
                         st.session_state.admin_processing_state["result_text"] = text_result
                         st.session_state.admin_processing_state["processing_provider"] = provider_used
                         st.session_state.admin_processing_state["rm_suggestions"] = rm_suggestions
                         st.session_state.admin_processing_state["original_metadata"] = original_metadata_from_manifest # Use manifest version


                         # Calculate quality if text was loaded successfully
                         if provider_used == "Loaded from Saved Files":
                             calculate_and_store_quality(text_result, provider_used, st.session_state.admin_processing_state)

                         processing_successful = True
                         st.success(f"✅ Loaded existing processed data for {original_filename} (CS:{current_chunk_size}, CO:{current_chunk_overlap}) successfully!")
                     else:
                         st.error("Failed to load existing index/chunks despite manifest entry and file presence. Reprocessing...")
                         # Clean up stale manifest entry
                         if base_filename in manifest: del manifest[base_filename]
                         save_manifest(manifest)
                         st.session_state.manifest_data = manifest
                 except Exception as e:
                     st.error(f"Error loading saved index/chunks: {e}. Reprocessing...")
                     # Clean up stale manifest entry
                     if base_filename in manifest: del manifest[base_filename]
                     save_manifest(manifest)
                     st.session_state.manifest_data = manifest

            # --- Perform Full Processing if Not Loaded ---
            if not processing_successful:
                with st.spinner(f"Processing '{original_filename}'..."):
                    # 1. Get Text (Check saved text first unless forcing)
                    if os.path.exists(text_save_path) and not force_reprocess:
                        try:
                            st.info(f"Loading previously extracted text from {text_save_path}...")
                            with open(text_save_path, "r", encoding="utf-8") as f:
                                text_result = f.read()
                            st.success(f"Loaded saved text ({len(text_result)} chars).")
                            provider_used = "Loaded from Saved Text File"
                        except Exception as e:
                            st.error(f"Error loading saved text file {text_save_path}: {e}. Re-extracting text...")
                            text_result = None # Force re-extraction

                    # Extract text if not loaded from file
                    if text_result is None:
                        current_file_extension = os.path.splitext(original_filename)[1].lower()
                        # Use the globally set OCR provider and direct extraction setting
                        # These are either set by admin or default values from session_state
                        actual_ocr_provider = st.session_state.get("admin_ocr_provider_select_global", "Groq")
                        actual_use_direct_extraction = st.session_state.get("admin_direct_extract_cb_global", False)
                        if is_creator_role: # Ensure creator uses global settings
                            use_direct_extraction_for_processing = actual_use_direct_extraction
                            ocr_provider_for_processing = actual_ocr_provider
                        else: # Admin uses their current selection from UI
                            use_direct_extraction_for_processing = use_direct_extraction
                            ocr_provider_for_processing = ocr_provider

                        # --- Determine Extraction Method ---
                        if current_file_extension in [".md", ".txt"]:
                            provider_used = f"Direct Read ({current_file_extension})"
                            st.info(f"Reading text directly from {original_filename}...")
                            try:
                                text_result = file_bytes.decode("utf-8")
                            except UnicodeDecodeError:
                                st.error(f"Failed to decode {original_filename} as UTF-8. Trying latin-1.")
                                try:
                                    text_result = file_bytes.decode("latin-1")
                                except Exception as decode_err:
                                    st.error(f"Failed to decode file: {decode_err}")
                                    text_result = None # Ensure it's None on failure
                        elif use_direct_extraction_for_processing and current_file_extension == '.pdf':
                            provider_used = "Direct Extraction (PyMuPDF)"
                            st.info(f"Attempting {provider_used}...")
                            text_result = extract_direct_text_pymupdf(file_bytes, original_filename)
                        else: # Fallback to OCR for PDFs (if not direct) and Images
                            provider_used = f"OCR ({ocr_provider_for_processing})"
                            st.info(f"Running {provider_used}...")
                            text_result = run_ocr(file_bytes, original_filename, ocr_provider_for_processing)
                        # Save extracted text if successful
                        if text_result:
                            try:
                                os.makedirs(PROCESSED_DIR, exist_ok=True) # Ensure dir exists
                                with open(text_save_path, "w", encoding="utf-8") as f:
                                    f.write(text_result)
                                st.success(f"Processed text saved to {text_save_path}")
                            except Exception as e:
                                st.warning(f"Could not save processed text: {e}")
                        else:
                            st.error(f"Failed to get text using {provider_used}.")

                    # Update processing state with extraction results
                    st.session_state.admin_processing_state["result_text"] = text_result
                    st.session_state.admin_processing_state["processing_provider"] = provider_used

                    # --- NEW: Extract and Save Tables for PDFs ---
                    if text_result and current_file_extension == '.pdf': # Only for PDFs and if text extraction was successful
                        with st.spinner("Extracting tables from PDF..."):
                            raw_tables = extract_tables_from_pdf_bytes(file_bytes, original_filename)
                            if raw_tables:
                                st.info(f"Extracted {len(raw_tables)} tables. Saving and creating summaries...")
                                table_summaries_for_indexing = []
                                for i, table_data in enumerate(raw_tables):
                                    if not table_data: continue # Skip empty tables

                                    table_json_filename = f"{base_filename}_table_{i}.json"
                                    table_json_path = os.path.join(PROCESSED_DIR, table_json_filename)
                                    try:
                                        with open(table_json_path, 'w', encoding='utf-8') as tf:
                                            json.dump(table_data, tf, indent=2, ensure_ascii=False)
                                        extracted_table_files.append(os.path.basename(table_json_path)) # Store relative path

                                        # Create a textual summary for indexing
                                        # Marker: [IDOCSPY_TABLE_JSON_REF:filename.json]
                                        table_summary_text = f"\n\n--- TABLE START: {i} ---\n"
                                        table_summary_text += f"Content reference: [IDOCSPY_TABLE_JSON_REF:{os.path.basename(table_json_path)}]\n"
                                        if table_data:
                                            headers = table_data[0] if table_data else []
                                            table_summary_text += f"Headers: {str(headers)}\n"
                                            # Include first 1-2 data rows in summary for better embedding
                                            for row_idx, row_content in enumerate(table_data[1:3], 1):
                                                table_summary_text += f"Data Row {row_idx}: {str(row_content)}\n"
                                        table_summary_text += f"--- TABLE END: {i} ---\n\n"
                                        table_summaries_for_indexing.append(table_summary_text)
                                    except Exception as ts_e:
                                        st.warning(f"Could not save or summarize table {i}: {ts_e}")
                                text_result += "".join(table_summaries_for_indexing) # Append all table summaries to main text

                    # 2. Calculate Quality, Analyze for RM, Create Index (if text exists)
                    if text_result and text_result.strip():
                        st.info("Calculating text quality...")
                        calculate_and_store_quality(
                            text_result,
                            provider_used,
                            st.session_state.admin_processing_state # Pass the dict here
                        )

                        # --- NEW: Analyze for RM Suggestions ---
                        with st.spinner("Analyzing content for RM suggestions..."):
                            rm_suggestions = analyze_content_for_rm_suggestions(text_result)
                            st.session_state.admin_processing_state["rm_suggestions"] = rm_suggestions
                            if rm_suggestions:
                                st.success("RM suggestion analysis complete.")
                            else:
                                st.warning("RM suggestion analysis failed or returned no results.")

                        # --- Create Index ---
                        with st.spinner("Creating document index..."):
                            # Now returns FAISS index, original chunks, and tokenized chunks
                            index, chunks, tokenized_chunks = create_index_from_text( # Expecting three return values now
                                text_result,
                                chunk_size=current_chunk_size,
                                chunk_overlap=current_chunk_overlap
                            )

                        if index is not None and chunks is not None and tokenized_chunks is not None:
                            try:
                                # Save the index and chunks
                                os.makedirs(PROCESSED_DIR, exist_ok=True) # Ensure dir exists
                                faiss.write_index(index, index_save_path)
                                with open(chunks_save_path, 'wb') as f: # Save original chunks
                                    pickle.dump(chunks, f)
                                with open(tokenized_chunks_save_path, 'wb') as f: # Save tokenized chunks
                                    pickle.dump(tokenized_chunks, f)

                                processing_successful = True
                                st.success("✅ Document processed, analyzed, indexed, and saved successfully!")

                                # Update and save the manifest
                                manifest = st.session_state.manifest_data # Reload potentially modified manifest
                                manifest[base_filename] = {
                                    "original_filename": original_filename,
                                    "index_path": os.path.basename(index_save_path), # Store relative path
                                    "chunks_path": os.path.basename(chunks_save_path),# Store relative path for original chunks
                                    "tokenized_chunks_path": os.path.basename(tokenized_chunks_save_path), # Store relative path for tokenized
                                    "chunk_size": current_chunk_size,
                                    "chunk_overlap": current_chunk_overlap,
                                    "processed_timestamp": datetime.now().isoformat(), # Add timestamp
                                    "original_metadata": original_metadata_from_upload, # Save original metadata
                                    "rm_suggestions": rm_suggestions, # Save RM suggestions
                                    "table_files": extracted_table_files # Save list of table JSON filenames
                                }
                                save_manifest(manifest)
                                st.session_state.manifest_data = manifest # Update session state manifest
                                st.info("Updated processed documents manifest.")

                                # --- Prepare and Insert DB Record ---
                                db_record_to_insert = {
                                    "filename": original_filename,
                                    "processed_at": time.time(), # Use current Unix timestamp
                                    "base_filename": base_filename, # <-- Add base_filename
                                    "provider": provider_used,
                                    "chunk_size": current_chunk_size, # <-- Add chunk_size
                                    "chunk_overlap": current_chunk_overlap, # <-- Add chunk_overlap
                                    "keywords": rm_suggestions.get("suggested_keywords", []) if rm_suggestions else []
                                }
                                insert_record(db_record_to_insert)
                                # Invalidate the cached DataFrame in session state so it reloads
                                st.session_state.records_df = None

                            except Exception as e:
                                st.error(f"Error saving index or chunks: {e}")
                                # Clean up potentially corrupted files if saving failed
                                if os.path.exists(index_save_path): os.remove(index_save_path)
                                if os.path.exists(tokenized_chunks_save_path): os.remove(tokenized_chunks_save_path)
                                if os.path.exists(chunks_save_path): os.remove(chunks_save_path)
                                if os.path.exists(chunks_save_path): os.remove(chunks_save_path)
                        else:
                            st.error("Index creation failed. Check logs or text quality.")
                    elif text_result is None:
                        st.error("Text extraction failed. Cannot analyze or index.")
                    else:
                        st.warning("Extracted text is empty or whitespace only. Cannot analyze or index.")

            # --- Post-processing Actions (if successful or loaded) ---
            if processing_successful:
                # Refresh the list of available documents
                st.session_state.available_documents = find_available_documents_from_db() # Use the DB version

                # Find the display name of the newly processed/loaded document
                new_display_name = None
                for dn, info in st.session_state.available_documents.items():
                    if info["base_filename"] == base_filename:
                        new_display_name = dn
                        break

                # Automatically select the new document in the chatbot tab
                if new_display_name:
                    st.session_state.selected_doc_display_name = new_display_name
                    # Reset chatbot's loaded state to force reload with new data
                    st.session_state.current_doc_index = None
                    st.session_state.current_doc_chunks = None
                    st.session_state.current_doc_tokenized_chunks = None # Reset tokenized chunks too
                    st.session_state.current_doc_display_name = None
                    st.session_state.current_doc_base_filename = None
                    st.info(f"'{new_display_name}' will be selected in the Chatbot tab.")

                # Rerun to update the UI (especially the available docs list and chatbot selection)
                st.rerun()

    # --- Section 4: Global Retrieval Settings ---
    st.markdown("---")
    st.subheader("4. Retrieval Settings (Global for Chatbot)")
    st.caption("These settings control how the chatbot retrieves context for *all* documents.")
    if is_creator_role:
        st.info("Global retrieval settings are managed by the Administrator.")
        # Optionally display current settings as read-only for creator
        st.markdown(f"- **Number of Chunks (k):** `{st.session_state.get('global_retrieval_k', 7)}`")
        st.markdown(f"- **Min Similarity Score:** `{st.session_state.get('global_retrieval_threshold', 0.3)}`")
        st.markdown(f"- **Re-ranking Enabled:** `{'Yes' if st.session_state.get('global_retrieval_reranking', False) else 'No'}`")
        st.markdown(f"- **HyDE Search Enabled:** `{'Yes' if st.session_state.get('global_retrieval_hyde', False) else 'No'}`")
    else: # Full Admin
        admin_top_k = st.slider(
            "Number of Chunks (k)", min_value=1, max_value=15,
            value=st.session_state.get("global_retrieval_k", 7),
            key="admin_top_k_setter",
            help="How many text chunks to retrieve based on similarity."
        )
        admin_threshold = st.slider(
            "Min Similarity Score", min_value=0.0, max_value=1.0,
            value=st.session_state.get("global_retrieval_threshold", 0.3),
            step=0.05, key="admin_sim_thresh_setter",
            help="Minimum similarity score for a chunk to be considered relevant (0 = no threshold)."
        )
        admin_reranking = st.checkbox(
            "Enable Re-ranking",
            value=st.session_state.get("global_retrieval_reranking", False),
            key="admin_rerank_setter",
            help="Use a cross-encoder to re-rank initial chunks (slower, potentially more accurate, requires model)."
        )
        admin_hyde = st.checkbox(
            "Enable HyDE Search",
            value=st.session_state.get("global_retrieval_hyde", False),
            key="admin_hyde_setter",
            help="Generate hypothetical answer for search query embedding (requires LLM call)."
        )

    # --- Section 5: Last Processed Document Details ---
    st.markdown("---")
    st.subheader("5. Last Processed Document Details")
    proc_state = st.session_state.get("admin_processing_state", {})

    if proc_state.get("processing_provider"): # Check if any processing has happened
        provider_used = proc_state.get("processing_provider")
        st.markdown(f"**Document:** `{proc_state.get('original_filename', 'N/A')}`")
        st.markdown(f"**Action:** `{provider_used}`")
        st.markdown(f"**Settings Used:** Chunk Size=`{proc_state.get('chunk_size', 'N/A')}`, Overlap=`{proc_state.get('chunk_overlap', 'N/A')}`")

        # Use tabs for Raw Text, Quality, and RM Info
        detail_tab_titles = ["📄 Raw Text", "📊 Processing Quality", "🗂️ RM Suggestions & Metadata"]
        detail_tabs = st.tabs(detail_tab_titles)

        with detail_tabs[0]: # Raw Text Tab
            result_text = proc_state.get("result_text")
            if result_text:
                st.text_area("Extracted/Loaded Text", value=result_text, height=300, key="admin_raw_text_display", disabled=True)
            else:
                st.info(f"No text result available from the last action ({provider_used}).")

        with detail_tabs[1]: # Processing Quality Tab
            quality = proc_state.get("quality")
            if quality and isinstance(quality, dict):
                st.write(f"### Text Quality Assessment (`{quality.get('provider', provider_used)}`)")
                quality_score = quality.get("score", 0) # Overall score

                # Define thresholds (could be moved to constants.py)
                good_threshold = OCR_METRICS.get("text_quality", {}).get("good", 0.85)
                medium_threshold = OCR_METRICS.get("text_quality", {}).get("medium", 0.60)

                # Determine color based on score
                if quality_score >= good_threshold: quality_color = "🟢"
                elif quality_score >= medium_threshold: quality_color = "🟡"
                else: quality_color = "🔴"

                st.metric("Overall Quality Score", f"{quality_score:.2%}", delta=quality_color, delta_color="off") # Use delta for icon

                # Display detailed metrics if available
                if "metrics" in quality and quality["metrics"]:
                    metrics = quality["metrics"]
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.metric("Word Count", f"{metrics.get('word_count', 0):,}")
                        st.metric("Line Count", f"{metrics.get('line_count', 0):,}")
                        st.metric("Char Count", f"{metrics.get('char_count', 0):,}")
                    with m_col2:
                        # Display metrics conditionally if they exist
                        if 'confidence_score' in metrics: st.metric("Confidence", f"{metrics.get('confidence_score', 0):.1%}")
                        if 'structure_score' in metrics: st.metric("Structure", f"{metrics.get('structure_score', 0):.1%}")
                        if 'format_retention' in metrics: st.metric("Format", f"{metrics.get('format_retention', 0):.1%}")
                else:
                    st.info("Detailed metrics not calculated.")
            else:
                st.info("Quality metrics not calculated or available for the last action.")

        with detail_tabs[2]: # RM Suggestions & Metadata Tab
            st.write("### Records Management Analysis")
            rm_suggestions = proc_state.get("rm_suggestions")
            original_metadata = proc_state.get("original_metadata")

            st.markdown("**AI Suggestions (Based on Content):**")
            if rm_suggestions and isinstance(rm_suggestions, dict):
                st.markdown(f"**Suggested Classification:** `{rm_suggestions.get('suggested_classification', 'N/A')}`")

                keywords = rm_suggestions.get('suggested_keywords', [])
                if keywords:
                    st.markdown(f"**Suggested Keywords:**")
                    st.markdown(f"`{', '.join(keywords)}`")
                else:
                    st.markdown("**Suggested Keywords:** `None suggested`")

                sensitivity = rm_suggestions.get('sensitivity_flags', [])
                if sensitivity:
                    st.markdown("**Potential Sensitivity Flags:**")
                    for flag in sensitivity:
                        st.markdown(f"- **Type:** `{flag.get('type', 'Unknown')}` - **Reason:** *{flag.get('reason', 'N/A')}*")
                else:
                    st.markdown("**Potential Sensitivity Flags:** `None detected`")
            elif provider_used and "Loaded" not in provider_used: # Only show if analysis was expected
                 st.warning("RM suggestion analysis was not performed or failed.")
            else:
                 st.info("RM suggestion analysis not available for this action (e.g., loaded from existing).")

            st.markdown("---")
            st.markdown("**Original Metadata (from Upload/File):**")
            if original_metadata and isinstance(original_metadata, dict):
                 # Reuse display logic, maybe simplified
                 display_order = ["Filename", "File Type", "File Size (KB)", "Page Count", "Title", "Author", "Subject", "Dimensions (WxH)", "Image Format", "Creation Date", "Modification Date"]
                 displayed_any = False
                 for key in display_order:
                     if key in original_metadata and original_metadata[key] != "N/A" and original_metadata[key] is not None:
                         st.markdown(f"- **{key}:** {original_metadata[key]}")
                         displayed_any = True
                 if not displayed_any:
                     st.caption("No significant original metadata found or extracted.")
                 if "Error" in original_metadata:
                     st.caption(f"(Metadata Extraction Note: {original_metadata['Error']})")
            else:
                 st.caption("Original metadata not available.")

    elif not file_uploaded:
         st.info("Upload a document to see processing options and details.")
    else: # File uploaded but not processed yet
         st.info("Process the uploaded document using the button above to see details.")
    st.markdown("---")
    st.caption("IDocspy, Built by Adnuri Mohamidi with help from AI :orange_heart:", help="cyberariani@gmail.com")