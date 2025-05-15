# d:\pandasai\idp\tabs\database_tab.py
import streamlit as st
import pandas as pd
import time

# --- Import DB functions ---
try:
    from db_utils import load_records_to_dataframe, delete_record, update_record
    from utils import find_available_documents_from_db, save_manifest # Use DB version for potential cleanup later
except ImportError as e:
    st.error(f"Database Tab Error: Failed to import required functions/modules: {e}")
    st.stop()

def display_database_tab():
    """Renders the Database Management tab content."""

    st.header("🗄️ Database Records Management")
    st.caption("View, update, and delete processed document records stored in the application database.")
    st.markdown("---")

    # --- Check User Role ---
    is_logged_in = st.session_state.get("admin_logged_in", False)
    is_full_admin_role = st.session_state.get("current_user_role") == "admin"

    # Load data if not in session state or if forced reload
    if "records_df" not in st.session_state or st.session_state.records_df is None:
        st.session_state.records_df = load_records_to_dataframe()
        # Clear original_records_df whenever records_df is reloaded to ensure it's refreshed below
        st.session_state.original_records_df = None

    # Use the DataFrame from session state
    records_df = st.session_state.records_df

    # Ensure original_records_df is populated if records_df exists and original is None or missing
    # This handles the case after a rerun where records_df might exist but original_df was cleared
    if records_df is not None and ("original_records_df" not in st.session_state or st.session_state.original_records_df is None):
         st.session_state.original_records_df = records_df.copy()

    # Retrieve original_records_df *after* potentially populating it
    original_records_df = st.session_state.original_records_df

    if records_df is not None and not records_df.empty:
        # --- Ensure the 'delete' column exists for the UI ---
        if 'delete' not in records_df.columns:
            records_df['delete'] = False # Initialize with False

        # --- Create a copy for display and add the sequential row number ---
        records_df_display = records_df.copy()
        # Insert Row # column at the beginning (position 0)
        records_df_display.insert(0, "Row #", range(1, len(records_df_display) + 1))

        # --- Configure columns for display and editing (conditionally disabled) ---
        column_config = {
            "Row #": st.column_config.NumberColumn("No. #", width="small", help="Visual row number (not the database ID)", disabled=True),
            # Remove the "id" config since we won't display it
            "filename": st.column_config.TextColumn("Display Filename", width="medium", help="Editable name shown in Chatbot", disabled=not is_full_admin_role),
            "base_filename": st.column_config.TextColumn("Internal ID", width="medium", help="Internal identifier linking to processed files (read-only)", disabled=True), # Display read-only
            "processed_at": st.column_config.DatetimeColumn("Processed At", format="YYYY-MM-DD HH:mm:ss", disabled=True),
            "provider": st.column_config.TextColumn("Method Used", disabled=not is_full_admin_role),
            "notes": st.column_config.TextColumn("Notes", help="Add custom notes about this record", disabled=not is_full_admin_role),
            "chunk_size": st.column_config.NumberColumn("Chunk Size", help="Chars per chunk (read-only)", disabled=True), # Display read-only
            "chunk_overlap": st.column_config.NumberColumn("Overlap", help="Chars overlap (read-only)", disabled=True), # Display read-only
            # ListColumn doesn't support 'disabled', editing is controlled by the Update button's state
            "keywords": st.column_config.ListColumn("Keywords (AI)", help="Keywords suggested by AI (editable by admin via Update button)"),
            "delete": st.column_config.CheckboxColumn("Delete?", default=False, disabled=not is_full_admin_role)
        }

        if not is_logged_in:
            st.warning("🔒 Please log in via the sidebar to manage database records.")
            # Optionally hide the editor or disable all columns if not logged in at all
            # For now, the column_config handles disabling based on full_admin_role
        elif is_full_admin_role:
            st.info("As an admin, you can edit the 'Display Filename', 'Method Used', 'Notes', and 'Keywords' columns. Use 'Update Records' to save. Use checkbox and 'Delete Selected' to remove.")
        elif st.session_state.get("current_user_role") == "creator":
            st.info("As a Records Creator, you can view records. Editing and deletion are restricted to Administrators.")
        else: # Should not happen if roles are 'admin' or 'creator' when logged in
            st.info("Viewing database records. Log in as admin to enable editing and deletion.")

        # --- Display the Data Editor ---
        # Pass the modified DataFrame with the "Row #" column, but exclude 'id' from display
        display_columns = [col for col in records_df_display.columns if col != 'id']
        edited_df = st.data_editor(
            records_df_display[display_columns], # Show all columns except 'id'
            column_config=column_config,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic", # Keep dynamic for potential future add row feature, but disable editing non-configured cols
            key="db_records_editor_main",
            # disabled parameter here disables editing for the whole df,
            # we rely on column_config's disabled=True/False instead.
        )

        # Add back the 'id' column from the original DataFrame for operations
        edited_df = pd.concat([records_df_display[['id']], edited_df], axis=1)

        # --- Update Logic ---
        if st.button("💾 Update Records", key="update_db_records", disabled=not is_full_admin_role):
            # --- Add check for original_records_df before proceeding ---
            if original_records_df is None:
                st.error("Error: Cannot compare changes because original data is missing. Please refresh the page.")
                st.stop() # Stop execution of the update logic

            changes = {}
            # Compare edited_df with the original DataFrame stored in session state
            # Iterate using the index of the edited_df (which matches original_records_df's index if no rows added/deleted)
            for idx in edited_df.index:
                # Check if this index exists in the original dataframe to prevent errors if rows were somehow added/deleted outside standard flow
                if idx in original_records_df.index:
                    # Get rows based on index. Note: original_records_df does NOT have "Row #"
                    row_original = original_records_df.loc[idx]
                    # edited_df HAS "Row #" but we ignore it in comparison logic below
                    row_edited = edited_df.loc[idx]
                    # Check only editable columns for changes
                    editable_cols = ["filename", "provider", "notes", "keywords"] # base_filename, chunk settings are NOT editable
                    row_updates = {}
                    for col in editable_cols:
                        # --- Special handling for list column 'keywords' ---
                        if col == 'keywords':
                            # Direct list comparison (order matters)
                            if row_original[col] != row_edited[col]:
                                row_updates[col] = row_edited[col]
                        # --- Handling for scalar columns (filename, provider, notes) ---
                        else:
                            # Use pd.isna for scalar comparison, converting potential NaNs for comparison
                            original_val = row_original[col] if not pd.isna(row_original[col]) else None
                            edited_val = row_edited[col] if not pd.isna(row_edited[col]) else None
                            if original_val != edited_val:
                             row_updates[col] = row_edited[col]

                    if row_updates:
                        # Use the correct database 'id' from the edited row
                        changes[row_edited['id']] = row_updates

            if changes:
                updated_count = 0
                failed_count = 0
                with st.spinner("Applying updates to the database..."):
                    for record_id, updates_dict in changes.items():
                        if update_record(record_id, updates_dict):
                            updated_count += 1
                        else:
                            failed_count += 1
                st.success(f"Successfully updated {updated_count} records.")
                if failed_count > 0:
                    st.warning(f"Failed to update {failed_count} records. Check logs.")
                # Invalidate cache and rerun to show updated data
                st.session_state.records_df = None
                # Also clear original_records_df to force reload on next run
                st.session_state.original_records_df = None
                st.rerun()
            else:
                st.info("No changes detected to update.")

        # --- Delete Logic ---
        # Use the edited_df (which includes the 'delete' column)
        records_to_delete = edited_df[edited_df["delete"]]
        if not records_to_delete.empty:
            if st.button("🗑️ Delete Selected Records", type="primary", key="delete_db_records", disabled=not is_full_admin_role):
                deleted_count = 0
                failed_count = 0
                with st.spinner("Deleting selected records..."):
                    for index, row in records_to_delete.iterrows():
                        # Use the correct database 'id' from the row to delete
                        if delete_record(row['id']):
                            deleted_count += 1
                        else:
                            failed_count += 1
                st.success(f"Deleted {deleted_count} records from the database.")
                if failed_count > 0:
                    st.warning(f"Failed to delete {failed_count} records from the database.")
                # Invalidate cache and rerun to refresh the table
                st.session_state.records_df = None
                # Also clear original_records_df after delete
                st.session_state.original_records_df = None
                st.rerun()

    elif records_df is not None and records_df.empty:
        st.info("No processed records found in the database yet.")
    else: # records_df is None (error during load)
        st.error("Could not load records from the database. Check connection and logs.")
    
    st.markdown("---")    
    st.caption("IDocspy, Built by Adnuri Mohamidi with help from AI :orange_heart:", help="cyberariani@gmail.com")
        