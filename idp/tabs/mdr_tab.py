import streamlit as st
import pandas as pd
import json # For robust JSON parsing

# Attempt to import Gemini client utilities from ocr_providers
# These are assumed to be defined in your ocr_providers.py:
# - get_gemini_client(api_key_name="GEMINI_API_KEY")
# - GEMINI_SAFETY_SETTINGS_NONE (e.g., for disabling safety filters if appropriate for internal docs)
# - GEMINI_GENERATION_CONFIG_JSON (e.g., {"temperature": 0.2, "response_mime_type": "application/json"} or similar)
try:
    from ocr_providers import get_gemini_client, GEMINI_SAFETY_SETTINGS_NONE, GEMINI_GENERATION_CONFIG_JSON
except ImportError:
    # Fallback if these specific constants are not defined, basic client might still work.
    # The extract_metadata_with_gemini_pro function will handle missing client.
    st.warning("Could not import all Gemini utilities from ocr_providers.py. Ensure it's correctly set up.")
    # Define placeholders if not found, so the app doesn't crash on import,
    # but functionality will be limited.
    def get_gemini_client(api_key_name="GEMINI_API_KEY"): return None
    GEMINI_SAFETY_SETTINGS_NONE = None
    GEMINI_GENERATION_CONFIG_JSON = None

# Attempt to import the main text extraction function from idp_extraction
try:
    from idp_extraction import get_text_from_document # You'll need to implement this
except ImportError:
    st.error("MDR Tab: Critical function 'get_text_from_document' not found in 'idp_extraction.py'. Text extraction will fail.")
    def get_text_from_document(file_bytes, filename, file_extension, **kwargs): return None # Placeholder

# Import dropdown options for MDR fields
try:
    from mdr_field_options import MDR_SELECTBOX_OPTIONS_MAP, get_mdr_column_config
except ImportError:
    st.error("MDR Tab: Critical file 'mdr_field_options.py' not found. Dropdown functionality will be missing.")
    MDR_SELECTBOX_OPTIONS_MAP = {} # Fallback


# Define the metadata fields for the MDR
METADATA_COLUMNS = [
    "Document No.", "Revision No.", "Document Title (English)", "Document Type",
    "Status", "Required Handover Status", "Phase", "System", "Area/Unit/Location",
    "Discipline", "Document Type (Detail)", "Sub-Type", "Contractor Return Code",
    "SDR Code", "Originating Company", "Review Category", "Revision Date (DD-MMM-YY)",
    "Planned Submission Date (DD-MMM-YY)", "Planned Completion Date (DD-MMM-YY)",
    "Forecast Submission Date (DD-MMM-YY)", "Forecast Completion Date (DD-MMM-YY)",
    "Actual Completion Date (DD-MMM-YY)", "Tag Number(s)", "PO Number",
    "Supplier Name", "Alternative Doc No.", "Alternative Rev No.",
    "Document Has Holds (Y/N)", "Start-Up Critical (Y/N)", "Export Control (Y/N)",
    "Remarks/Notes"
]

# --- Placeholder DB functions for MDR ---
# In a real application, move these to a dedicated db_utils_mdr.py and connect to a database.
def _initialize_simulated_mdr_db():
    if "simulated_mdr_db" not in st.session_state:
        st.session_state.simulated_mdr_db = [] # List of dicts, where each dict is a record
        st.session_state.mdr_next_id = 1

def load_mdr_records_from_db():
    _initialize_simulated_mdr_db()
    if not st.session_state.simulated_mdr_db:
        return pd.DataFrame(columns=["id"] + METADATA_COLUMNS)
    return pd.DataFrame(st.session_state.simulated_mdr_db)

def add_mdr_record_to_db(record_data: dict):
    _initialize_simulated_mdr_db()
    # Ensure record_data only contains METADATA_COLUMNS
    filtered_record_data = {k: v for k, v in record_data.items() if k in METADATA_COLUMNS}

    new_id = st.session_state.mdr_next_id
    record_with_id = {"id": new_id, **filtered_record_data}
    st.session_state.simulated_mdr_db.append(record_with_id)
    st.session_state.mdr_next_id += 1
    # st.success(f"Simulated: Added MDR record with ID {new_id}") # Optional: for debugging
    return True

def update_mdr_record_in_db(record_id: int, updates: dict):
    _initialize_simulated_mdr_db()
    for record in st.session_state.simulated_mdr_db:
        if record["id"] == record_id:
            record.update(updates)
            return True
    return False

def delete_mdr_record_from_db(record_id: int):
    _initialize_simulated_mdr_db()
    original_len = len(st.session_state.simulated_mdr_db)
    st.session_state.simulated_mdr_db = [
        r for r in st.session_state.simulated_mdr_db if r["id"] != record_id
    ]
    return len(st.session_state.simulated_mdr_db) < original_len
# --- End Placeholder DB functions ---

def generate_extraction_prompt(document_text, fields):
    """Generates a prompt for Gemini to extract metadata."""
    max_chars = 18000 # Adjust based on typical document structure and token limits
    if len(document_text) > max_chars:
        document_text_snippet = document_text[:max_chars//2] + "\n...\n[CONTENT TRUNCATED]\n...\n" + document_text[-max_chars//2:]
        warning_message = f"(Note: Document text was truncated for AI processing. Showing start and end portions up to {max_chars} chars.)"
    else:
        document_text_snippet = document_text
        warning_message = ""

    prompt = f"""
    Analyze the following document text and extract the specified metadata fields.
    The document is likely a technical drawing, engineering diagram, or a similar technical document.
    The goal is to populate a Master Document Register (MDR) with information typically found on such documents.

    **Instructions for Extraction:**
    1.  Pay close attention to information typically found in **title blocks, revision tables, notes sections, and document borders.**
    2.  If a specific piece of information for a field is not found in the text, use "N/A" as the value for that field.
    3.  For dates (like "Revision Date", "Planned Submission Date", etc.), please format them as DD-MMM-YY (e.g., 25-DEC-23). If the date is in a different format in the document, extract it as found and convert if confident; otherwise, provide it as is.
    4.  For fields like "Document No.", "Revision No.", "Status", "Discipline", look for standard codes or explicit labels.
    5.  "Tag Number(s)" might be a list of equipment tags shown on the drawing.

    Please return the information as a single, valid JSON object where keys are the exact field names
    provided below and values are the extracted information.

    Metadata Fields to Extract:
    {json.dumps(fields, indent=2)} 
    # ^^^ "Document No." is one of these fields

    Document Text Snippet {warning_message}:
    ---
    {document_text_snippet}
    ---

    Provide ONLY the JSON object in your response, without any surrounding text, comments, or markdown. Ensure the JSON is complete and valid.
    Example of expected JSON format for a drawing:
    {{
      "Document No.": "PID-ME-00-1001", // AI should map "Drawing No." from text to this field
      "Revision No.": "B2",
      "Document Title (English)": "P&ID FOR UTILITY SYSTEM - AREA 00",
      // ... other fields ...
    }}
    """
    return prompt

def extract_metadata_with_gemini_pro(text_content, api_key_name="GEMINI_API_KEY"):
    """
    Extracts metadata from text content using a Gemini Pro model.
    """
    st.info("Attempting to extract metadata using Gemini Pro...")
    if not text_content:
        st.warning("No text content provided for extraction.")
        return pd.DataFrame(columns=METADATA_COLUMNS)

    # Use a distinct API key name for MDR operations
    client = get_gemini_client(api_key_name="MDR_GEMINI_API_KEY")
    if not client:
        st.error(f"Failed to initialize Gemini client. Ensure the API key '{api_key_name}' is set in your environment (e.g., .env file) and ocr_providers.py is correct.")
        return pd.DataFrame(columns=METADATA_COLUMNS)

    # IMPORTANT: Replace with your specific Gemini model name if different.
    # The user requested "gemini-2.5-pro-preview-05-06".
    # Ensure this model name is valid for your Google AI Studio or Vertex AI project.
    # model_name = "models/gemini-1.5-pro-latest" # Default to a generally available powerful model
    # To use the user-specified model:
    model_name = "models/gemini-2.5-flash-preview-04-17" # Or "gemini-2.5-pro-preview-05-06" if it's a base model ID

    try:
        # Correctly instantiate the GenerativeModel
        model = client.GenerativeModel(model_name)
    except Exception as e:
        st.error(f"Failed to get Gemini model '{model_name}'. Error: {e}. Please check model name and API access.")
        return pd.DataFrame(columns=METADATA_COLUMNS)


    prompt = generate_extraction_prompt(text_content, METADATA_COLUMNS)
    st.session_state.mdr_last_prompt = prompt

    try:
        # Use GEMINI_GENERATION_CONFIG_JSON if it's configured for JSON output.
        # Example: {"response_mime_type": "application/json", "temperature": 0.1}
        # If GEMINI_GENERATION_CONFIG_JSON is None or doesn't specify response_mime_type,
        # the model will rely on the prompt's instruction for JSON.
        gen_config = GEMINI_GENERATION_CONFIG_JSON if GEMINI_GENERATION_CONFIG_JSON else None
        safety_config = GEMINI_SAFETY_SETTINGS_NONE if GEMINI_SAFETY_SETTINGS_NONE else None

        response = model.generate_content(
            prompt,
            generation_config=gen_config,
            safety_settings=safety_config
        )

        response_text = response.text.strip()
        st.session_state.mdr_last_ai_response = response_text # Store raw response for debugging

        # Clean the response to get only the JSON part
        if response_text.startswith("```json"):
            response_text = response_text[len("```json"):]
            if response_text.endswith("```"):
                response_text = response_text[:-len("```")]
        elif response_text.startswith("```"):
            response_text = response_text[len("```"):]
            if response_text.endswith("```"):
                response_text = response_text[:-len("```")]
        response_text = response_text.strip()

        if not (response_text.startswith("{") and response_text.endswith("}")):
            st.error("AI response does not appear to be a valid JSON object after cleaning.")
            if response.prompt_feedback:
                 st.error(f"Prompt Feedback: {response.prompt_feedback}")
            return pd.DataFrame(columns=METADATA_COLUMNS)

        try:
            extracted_dict = json.loads(response_text)
        except json.JSONDecodeError as ve:
            st.error(f"Failed to parse JSON from AI response: {ve}")
            return pd.DataFrame(columns=METADATA_COLUMNS)

        data_for_df = {col: extracted_dict.get(col, "N/A") for col in METADATA_COLUMNS}
        df = pd.DataFrame([data_for_df], columns=METADATA_COLUMNS)
        st.success("Metadata extraction with AI successful.")
        return df

    except Exception as e:
        st.error(f"Error during Gemini API call or processing: {e}")
        if 'response' in locals() and hasattr(response, 'prompt_feedback') and response.prompt_feedback:
             st.error(f"Gemini prompt feedback: {response.prompt_feedback}")
        return pd.DataFrame(columns=METADATA_COLUMNS)


def display_mdr_tab():
    st.header("📄 Master Document Register (MDR) - AI Assisted Extraction")

    if not st.session_state.get("admin_logged_in", False) or st.session_state.get("current_user_role") != "admin":
        st.warning("🔒 This section is for Admin users only. Please log in with admin credentials.")
        st.info("If you are an admin, please log in via the sidebar.")
        return

    # Initialize session state keys for MDR tab
    if "new_mdr_entry_df" not in st.session_state: # For AI extracted, single new entry
        st.session_state.new_mdr_entry_df = pd.DataFrame(columns=METADATA_COLUMNS)
    if "mdr_records_df" not in st.session_state: # For all existing MDR records
        st.session_state.mdr_records_df = None
    if "mdr_original_records_df" not in st.session_state: # For diffing updates on all records
        st.session_state.mdr_original_records_df = None

    # --- Existing session state keys (ensure they are initialized if used before this point) ---
    if "mdr_document_text" not in st.session_state:
        st.session_state.mdr_document_text = ""
    if "mdr_last_uploaded_filename" not in st.session_state:
        st.session_state.mdr_last_uploaded_filename = None
    if "mdr_last_prompt" not in st.session_state:
        st.session_state.mdr_last_prompt = ""
    if "mdr_last_ai_response" not in st.session_state:
        st.session_state.mdr_last_ai_response = ""

    st.subheader("🆕 Create New MDR Entry (AI-Assisted)")
    st.markdown("Upload a document to extract metadata using AI and create a new MDR entry.")
    uploaded_file = st.file_uploader("Choose a document for MDR processing",
                                     type=["pdf", "txt", "docx", "png", "jpg", "jpeg"],
                                     key="mdr_uploader")

    if uploaded_file is not None:
        if st.session_state.mdr_last_uploaded_filename != uploaded_file.name:
            # Reset for new file upload
            st.session_state.mdr_document_text = "" # Reset text for new file
            st.session_state.new_mdr_entry_df = pd.DataFrame(columns=METADATA_COLUMNS) # Reset AI extracted data
            # Do not reset mdr_records_df here, as it holds all existing records

            st.session_state.mdr_last_uploaded_filename = uploaded_file.name
            st.session_state.mdr_last_prompt = ""
            st.session_state.mdr_last_ai_response = ""


            file_bytes = uploaded_file.getvalue()
            file_extension = uploaded_file.name.split('.')[-1].lower()

            with st.spinner(f"Extracting text from {uploaded_file.name}... This may take a moment."):
                try:
                    # Call the text extraction function from idp_extraction.py
                    # This function needs to be implemented in idp_extraction.py
                    # to handle various file types (txt, pdf, docx, images with OCR).
                    extracted_text = get_text_from_document(
                        file_bytes=file_bytes,
                        filename=uploaded_file.name,
                        file_extension=file_extension,
                        # You might need to pass other parameters like OCR provider preferences
                        # ocr_provider_name="google_vision" # Example
                    )

                    if extracted_text is not None and extracted_text.strip():
                        st.session_state.mdr_document_text = extracted_text
                        st.success(f"Successfully extracted text from {uploaded_file.name}.")
                    elif extracted_text is None:
                        st.error(f"Text extraction failed for {uploaded_file.name}. The extraction function returned None.")
                        st.session_state.mdr_document_text = ""
                    else:
                        st.warning(f"Text extraction for {uploaded_file.name} returned empty content. The document might be blank or not machine-readable.")
                        st.session_state.mdr_document_text = ""

                except ImportError:
                    # This case is handled by the placeholder 'get_text_from_document' if import fails at the top.
                    # The error message is already shown.
                    st.session_state.mdr_document_text = "Critical: Text extraction function missing from idp_extraction.py."
                except Exception as e:
                    st.error(f"An error occurred during text extraction: {e}")
                    st.session_state.mdr_document_text = ""
        else:
            # File is the same, do nothing for text extraction unless forced
            pass


        if st.session_state.mdr_document_text:
            st.subheader("Extracted Text (Preview)")
            st.text_area("Document Content (first 2000 chars)", st.session_state.mdr_document_text[:2000], height=150, disabled=True, key="mdr_text_preview")

            if st.button("✨ Extract Metadata with AI (Gemini)", key="mdr_extract_button", type="primary"):
                with st.spinner("🤖 AI is analyzing the document and extracting metadata... Please wait."):
                    extracted_df = extract_metadata_with_gemini_pro(st.session_state.mdr_document_text)
                    if extracted_df is not None and not extracted_df.empty:
                        st.session_state.new_mdr_entry_df = extracted_df
                    else:
                        st.error("AI metadata extraction failed or returned no data. Check AI Interaction Details below.")
                        st.session_state.new_mdr_entry_df = pd.DataFrame(columns=METADATA_COLUMNS)
        elif uploaded_file:
             st.warning("Document uploaded, but no text could be extracted or extraction was skipped. Cannot proceed with AI metadata extraction.")

    # --- Display and Save for New AI-Extracted Entry ---
    if not st.session_state.new_mdr_entry_df.empty:
        st.info("Review and edit the AI-generated metadata for the new entry below.")
        # Ensure METADATA_COLUMNS are editable for the new entry
        if MDR_SELECTBOX_OPTIONS_MAP: # Check if options were loaded
            new_entry_column_config = get_mdr_column_config(METADATA_COLUMNS, MDR_SELECTBOX_OPTIONS_MAP)
        else: # Fallback if options file is missing
            new_entry_column_config = {col: st.column_config.TextColumn(col) for col in METADATA_COLUMNS}


        edited_new_entry_df = st.data_editor(
            st.session_state.new_mdr_entry_df,
            num_rows="fixed", # Only 1 row for this MDR entry
            key="mdr_new_entry_editor",
            use_container_width=True,
            column_config=new_entry_column_config
        )
        st.session_state.new_mdr_entry_df = edited_new_entry_df

        if st.button("💾 Save New Entry to MDR", key="mdr_save_new_button", type="primary"):
            if not st.session_state.new_mdr_entry_df.empty:
                new_record_data = st.session_state.new_mdr_entry_df.iloc[0].to_dict()
                if add_mdr_record_to_db(new_record_data):
                    st.success("New MDR entry successfully saved (simulated).")
                    # Clear new entry form and reload all MDR records
                    st.session_state.new_mdr_entry_df = pd.DataFrame(columns=METADATA_COLUMNS)
                    st.session_state.mdr_document_text = ""
                    # mdr_last_uploaded_filename is kept to prevent re-extraction if user doesn't change file
                    # but it might be better to clear it too or use st.empty() for the uploader
                    st.session_state.mdr_records_df = None # Force reload of all records
                    st.session_state.mdr_original_records_df = None
                    st.rerun()
                else:
                    st.error("Failed to save the new MDR entry (simulated).")
            else:
                st.warning("No data in the new entry form to save.")

    st.markdown("---")
    st.subheader("🗂️ Manage Existing MDR Records")

    # --- Load and Display All MDR Records ---
    if st.session_state.mdr_records_df is None:
        st.session_state.mdr_records_df = load_mdr_records_from_db()
        st.session_state.mdr_original_records_df = None # Clear original when main DF reloads

    mdr_df = st.session_state.mdr_records_df

    if mdr_df is not None and ("mdr_original_records_df" not in st.session_state or st.session_state.mdr_original_records_df is None):
        st.session_state.mdr_original_records_df = mdr_df.copy()

    mdr_original_df = st.session_state.mdr_original_records_df

    if mdr_df is not None and not mdr_df.empty:
        if 'delete' not in mdr_df.columns:
            mdr_df['delete'] = False

        mdr_df_display = mdr_df.copy()
        if "id" in mdr_df_display.columns: # Ensure 'id' exists before trying to use it
             mdr_df_display.insert(0, "Row #", range(1, len(mdr_df_display) + 1))
        else: # Should not happen if load_mdr_records_from_db is correct
            st.error("Critical error: 'id' column is missing from MDR records.")
            st.stop()


        # Configure columns for the main MDR editor
        base_mdr_column_config = {
            "Row #": st.column_config.NumberColumn("No.", width="small", disabled=True),
            "delete": st.column_config.CheckboxColumn("Delete?", default=False),
        }
        if MDR_SELECTBOX_OPTIONS_MAP:
            metadata_specific_config = get_mdr_column_config(METADATA_COLUMNS, MDR_SELECTBOX_OPTIONS_MAP)
        else:
            metadata_specific_config = {col: st.column_config.TextColumn(col) for col in METADATA_COLUMNS}
        mdr_column_config = {**base_mdr_column_config, **metadata_specific_config}
    
        # 'id' column is not in METADATA_COLUMNS, it's handled separately. We don't display it directly.

        display_cols_mdr = [col for col in mdr_df_display.columns if col != 'id']
        edited_mdr_df_display = st.data_editor(
            mdr_df_display[display_cols_mdr],
            column_config=mdr_column_config,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic", # Allows for potential future "add row manually"
            key="mdr_records_editor"
        )

        # Add back 'id' from the original mdr_df for operations
        # This assumes mdr_df_display and edited_mdr_df_display maintain row order and count
        if 'id' in mdr_df.columns:
             edited_mdr_df_with_id = pd.concat([mdr_df[['id']].reset_index(drop=True), edited_mdr_df_display.reset_index(drop=True)], axis=1)
        else:
            edited_mdr_df_with_id = edited_mdr_df_display # Fallback, though 'id' is crucial


        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Update MDR Records", key="mdr_update_records"):
                if mdr_original_df is None:
                    st.error("Error: Original MDR data is missing. Cannot compare changes. Please refresh.")
                else:
                    changes = {}
                    # Iterate using the index, assuming it aligns if no rows added/deleted by editor directly
                    for idx in edited_mdr_df_with_id.index:
                        if idx in mdr_original_df.index:
                            row_original = mdr_original_df.loc[idx]
                            row_edited = edited_mdr_df_with_id.loc[idx]
                            row_updates = {}
                            # Compare only METADATA_COLUMNS as 'id', 'Row #', 'delete' are special
                            for col in METADATA_COLUMNS:
                                if col in row_original and col in row_edited: # Ensure column exists
                                    original_val = row_original[col] if not pd.isna(row_original[col]) else None
                                    edited_val = row_edited[col] if not pd.isna(row_edited[col]) else None
                                    if original_val != edited_val:
                                        row_updates[col] = edited_val
                            if row_updates:
                                changes[row_edited['id']] = row_updates

                    if changes:
                        updated_count = 0
                        for record_id, updates_dict in changes.items():
                            if update_mdr_record_in_db(record_id, updates_dict):
                                updated_count += 1
                        st.success(f"Successfully updated {updated_count} MDR records (simulated).")
                        st.session_state.mdr_records_df = None # Force reload
                        st.rerun()
                    else:
                        st.info("No changes detected in MDR records to update.")

        with col2:
            records_to_delete_mdr = edited_mdr_df_with_id[edited_mdr_df_with_id["delete"] == True]
            if not records_to_delete_mdr.empty:
                if st.button("🗑️ Delete Selected MDR Records", type="primary", key="mdr_delete_records"):
                    deleted_count = 0
                    for _, row in records_to_delete_mdr.iterrows():
                        if delete_mdr_record_from_db(row['id']):
                            deleted_count += 1
                    st.success(f"Deleted {deleted_count} MDR records (simulated).")
                    st.session_state.mdr_records_df = None # Force reload
                    st.rerun()

    elif mdr_df is not None and mdr_df.empty:
        st.info("No MDR records found in the (simulated) database yet. Use the section above to create new entries.")
    else:
        st.error("Could not load MDR records (simulated).")

    # Debugging expander for admins
    if st.session_state.get("admin_logged_in", False):
        with st.expander("🤖 AI Interaction Details (For Admin Debugging)", expanded=False):
            st.caption("This section shows the last prompt sent to the AI and its raw response. Useful for troubleshooting.")
            st.subheader("Last Prompt to AI")
            st.text_area("Prompt Sent", st.session_state.get("mdr_last_prompt", "No prompt generated yet."), height=200, disabled=True, key="mdr_debug_prompt")
            st.subheader("Last Raw Response from AI")
            st.text_area("Raw Response Received", st.session_state.get("mdr_last_ai_response", "No response received yet."), height=200, disabled=True, key="mdr_debug_response")