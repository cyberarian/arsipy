# d:\pandasai\idp\tabs\chatbot_tab.py
import streamlit as st
import os # Import the os module
import re # Import regex for table reference matching
import json # Import json for loading table data
import pandas as pd # Import pandas

# --- Import necessary functions from other modules ---
# Assume utils.py, idp_extraction.py, rag_processor.py are in the parent directory (d:\pandasai\idp\)
try:
    from utils import load_document_data, generate_hypothetical_answer
    from idp_extraction import answer_query_with_context
    from rag_processor import retrieve_relevant_chunks, rerank_chunks
except ImportError as e:
    st.error(f"Chatbot Tab Error: Failed to import required functions: {e}")
    st.stop()

# --- Define Constants (Adjust path as needed) ---
# Assuming 'processed_documents' is in the same parent directory as 'tabs'
PROCESSED_DIR_CHATBOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed_docs") # Use a distinct name if PROCESSED_DIR is also in constants

# --- Callback function for selectbox ---
def update_selected_doc():
    # The value from the selectbox is automatically passed via session state key
    st.session_state.selected_doc_display_name = st.session_state.doc_selector_chatbot

def _reset_chatbot_state():
    """Resets session state variables related to the currently loaded document."""
    st.session_state.current_doc_index = None
    st.session_state.current_doc_chunks = None
    st.session_state.current_doc_display_name = None
    st.session_state.current_doc_base_filename = None
    st.session_state.current_doc_chunk_size = None
    st.session_state.current_doc_chunk_overlap = None
    st.session_state.current_doc_rm_suggestions = None
    st.session_state.current_doc_tokenized_chunks = None # Add reset for tokenized chunks
    st.session_state.current_doc_original_metadata = None
    st.session_state.last_query_result = None

def display_chatbot_tab():
    """Renders the Chatbot tab content."""
    st.header("💬 Chat with Document")

    # Access available documents from session state (populated in main.py)
    available_docs_dict = st.session_state.get("available_documents", {})

    if not available_docs_dict:
        st.info("ℹ️ No processed documents found.")
        if st.session_state.get("admin_logged_in", False):
            st.warning("➡️ Go to the 'Admin Management' tab to process a document.")
        _reset_chatbot_state() # Reset state if no docs available
        return # Stop rendering the rest of the tab

    # --- Document Selection ---
    doc_options = list(available_docs_dict.keys())
    current_selection = st.session_state.get("selected_doc_display_name")

    # Determine the index for the selectbox
    try:
        # Use the session state value if it's valid, otherwise default to 0
        select_index = doc_options.index(current_selection) if current_selection in doc_options else 0
    except ValueError:
        select_index = 0 # Default to the first option if current_selection is somehow invalid

    selected_display_name_from_user = st.selectbox(
        "Select Document to Chat With:",
        options=doc_options,
        index=select_index,
        key="doc_selector_chatbot", # Unique key for this widget
        help="Choose from the list of processed documents.",
        on_change=update_selected_doc # Call the function when selection changes
    )

    # --- Display Details Below Dropdown ---
    # Use the value from session state, which is updated by the callback
    selected_display_name = st.session_state.get("selected_doc_display_name")
    if selected_display_name and selected_display_name in available_docs_dict:
        selected_doc_info = available_docs_dict[selected_display_name]
        processed_time = selected_doc_info.get('processed_at')
        provider = selected_doc_info.get('provider', 'N/A')
        base_filename = selected_doc_info.get('base_filename', 'N/A')

        # Format timestamp nicely if available
        processed_time_str = "N/A"
        if pd.notna(processed_time): # Check if it's a valid timestamp/datetime
            try:
                # Assuming it's already a datetime object from db_utils post-processing
                processed_time_str = processed_time.strftime('%Y-%m-%d %H:%M:%S')
            except AttributeError: # Handle if it's somehow not a datetime object
                processed_time_str = str(processed_time) # Fallback to string representation

        st.caption(f"**Selected:** `{selected_display_name}` | **Processed:** `{processed_time_str}` | **Method:** `{provider}` | **Internal ID:** `{base_filename}`")

    # --- Load Document Data if Selection Changes or Not Loaded ---
    # Check if the user's selection differs from the currently loaded doc OR if nothing is loaded
    needs_loading = (
        selected_display_name != st.session_state.get("current_doc_display_name") or
        st.session_state.get("current_doc_index") is None
    )

    if needs_loading and selected_display_name:
        # The selected_doc_display_name is already updated by the callback
        selected_doc_info = available_docs_dict[selected_display_name]
        selected_base_filename = selected_doc_info["base_filename"]

        # Check if the underlying base file has changed or if index is missing
        if selected_base_filename != st.session_state.get("current_doc_base_filename") or st.session_state.get("current_doc_index") is None:
            index_path = selected_doc_info["index_path"]
            chunks_path = selected_doc_info["chunks_path"]
            # Get the full path for tokenized chunks
            tokenized_chunks_path = selected_doc_info.get("tokenized_chunks_path")

            # Call the cached loading function from utils (now returns 3 items)
            index, chunks, tokenized_chunks = load_document_data(index_path, chunks_path, tokenized_chunks_path) # Load all three

            # Update session state with loaded data or handle loading failure
            if index is not None and chunks is not None and tokenized_chunks is not None: # Check all three
                st.session_state.current_doc_index = index
                st.session_state.current_doc_chunks = chunks
                st.session_state.current_doc_display_name = selected_display_name
                st.session_state.current_doc_base_filename = selected_base_filename
                st.session_state.current_doc_chunk_size = selected_doc_info.get("chunk_size", "N/A")
                st.session_state.current_doc_chunk_overlap = selected_doc_info.get("chunk_overlap", "N/A")
                # --- Load RM info for the selected doc ---
                # st.session_state.current_doc_rm_suggestions = selected_doc_info.get("rm_suggestions", {}) # This seems to be loaded from manifest in admin tab, maybe load from DB? For now, assume it's in available_docs_dict if needed.
                st.session_state.current_doc_tokenized_chunks = tokenized_chunks # Store tokenized chunks (needed for BM25)
                st.session_state.current_doc_original_metadata = selected_doc_info.get("original_metadata", {})
                st.session_state.last_query_result = None # Reset last query on doc change
                st.success(f"✅ Loaded: **{st.session_state.current_doc_display_name}**")
                # No rerun needed here, loading happens within the flow
            else:
                st.error(f"Failed to load data for {selected_display_name}. Check logs or manifest.")
                _reset_chatbot_state() # Reset state if loading failed

    # --- Q&A Section (only if a document is successfully loaded) ---
    if st.session_state.get("current_doc_index") is not None and st.session_state.get("current_doc_chunks") is not None and st.session_state.get("current_doc_tokenized_chunks") is not None:
        st.markdown("---")
        st.markdown(f"#### Ask a question about **{st.session_state.current_doc_display_name}**")
        st.caption(f"(Indexed with Chunk Size: {st.session_state.get('current_doc_chunk_size', 'N/A')}, Overlap: {st.session_state.get('current_doc_chunk_overlap', 'N/A')})")

        # --- Display RM Info in Expander ---
        with st.expander("Show Document Info & AI Suggestions"):
            rm_suggestions = st.session_state.get("current_doc_rm_suggestions", {})
            original_metadata = st.session_state.get("current_doc_original_metadata", {})

            st.markdown("**Original Metadata:**")
            orig_fn = original_metadata.get("Filename", st.session_state.current_doc_display_name) # Fallback
            st.markdown(f"- **Filename:** {orig_fn}")
            if original_metadata.get("Title") and original_metadata.get("Title") != "N/A":
                 st.markdown(f"- **Title:** {original_metadata.get('Title')}")
            if original_metadata.get("Author") and original_metadata.get("Author") != "N/A":
                 st.markdown(f"- **Author:** {original_metadata.get('Author')}")
            if original_metadata.get("Creation Date") and original_metadata.get("Creation Date") != "N/A":
                 st.markdown(f"- **Created:** {original_metadata.get('Creation Date')}")

            st.markdown("**AI Content Analysis:**")
            if rm_suggestions:
                st.markdown(f"- **Suggested Classification:** `{rm_suggestions.get('suggested_classification', 'N/A')}`")
                keywords = rm_suggestions.get('suggested_keywords', [])
                st.markdown(f"- **Suggested Keywords:** `{', '.join(keywords) if keywords else 'None'}`")
                sensitivity = rm_suggestions.get('sensitivity_flags', [])
                if sensitivity:
                    flags = [f"`{flag.get('type', '?')}` ({flag.get('reason', 'N/A')})" for flag in sensitivity]
                    st.markdown(f"- **Sensitivity Flags:** {'; '.join(flags)}")
                else:
                    st.markdown("- **Sensitivity Flags:** `None detected`")
            else:
                st.caption("AI suggestions not available for this document.")

        # --- User Query Input ---
        user_query = st.text_area("Your question:", key="user_query_input_main", height=100)
        ask_button = st.button("❓ Ask Question", key="ask_button_main", disabled=(not user_query.strip()))

        if ask_button:
            index = st.session_state.current_doc_index
            chunks = st.session_state.current_doc_chunks
            tokenized_chunks = st.session_state.current_doc_tokenized_chunks # Get tokenized chunks from state
            query = user_query.strip()

            if index and chunks and tokenized_chunks and query: # Check tokenized_chunks too
                # Retrieve global settings from session state
                k = st.session_state.get("global_retrieval_k", 5)
                threshold = st.session_state.get("global_retrieval_threshold", 0.3)
                use_reranking = st.session_state.get("global_retrieval_reranking", False)
                use_hyde = st.session_state.get("global_retrieval_hyde", False)

                search_term = query
                if use_hyde:
                    try:
                        hypo_answer = generate_hypothetical_answer(query) # From utils
                        if hypo_answer:
                            search_term = hypo_answer
                        else:
                            st.warning("HyDE generation failed, using original query.")
                    except Exception as hyde_e:
                         st.warning(f"Error during HyDE generation: {hyde_e}. Using original query.")

                # Initial retrieval size is handled inside retrieve_relevant_chunks now

                # 1. Retrieve relevant chunks
                # This function now performs hybrid search (FAISS + BM25) and RRF internally
                relevant_chunks_with_scores = retrieve_relevant_chunks(
                    search_term, index, chunks, tokenized_chunks, # Pass tokenized chunks
                    top_k=k, score_threshold=threshold) # Use 'k' directly here

                # 2. Optional Re-ranking
                if use_reranking and relevant_chunks_with_scores:
                    st.info("Re-ranking retrieved chunks...")
                    try:
                        # Pass the original query for re-ranking context
                        final_chunks_with_scores = rerank_chunks(query, relevant_chunks_with_scores, top_k=k) # Pass results from hybrid search
                    except Exception as rerank_e:
                        st.error(f"Error during re-ranking: {rerank_e}. Using initial retrieval order.")
                        # Fallback: sort by original score and take top k
                        relevant_chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
                        final_chunks_with_scores = relevant_chunks_with_scores[:k]
                else:
                    # If not re-ranking, just take the top k from initial retrieval
                    final_chunks_with_scores = relevant_chunks_with_scores # Already sliced to top_k in retrieve_relevant_chunks

                # --- Augment context with full table data if references are found ---
                augmented_context_parts = []
                processed_table_refs = set() # To avoid duplicating table data if summary is split

                for chunk_text, score in final_chunks_with_scores:
                    # Regex to find our table reference marker
                    # Marker: [IDOCSPY_TABLE_JSON_REF:filename.json]
                    match = re.search(r"\[IDOCSPY_TABLE_JSON_REF:([^\]]+)\]", chunk_text)
                    if match:
                        table_json_filename = match.group(1)
                        if table_json_filename not in processed_table_refs:
                            table_json_full_path = os.path.join(PROCESSED_DIR_CHATBOT, table_json_filename)
                            if os.path.exists(table_json_full_path):
                                try:
                                    with open(table_json_full_path, 'r', encoding='utf-8') as tjf:
                                        table_data_json = json.load(tjf) # This is list of lists
                                    # Convert list of lists to Markdown table string
                                    markdown_table = f"\n\n**Extracted Table ({table_json_filename}):**\n"
                                    if table_data_json:
                                        headers = table_data_json[0]
                                        markdown_table += "| " + " | ".join(map(str, headers)) + " |\n"
                                        markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                                        for row_data in table_data_json[1:]:
                                            markdown_table += "| " + " | ".join(map(str, row_data)) + " |\n"
                                    augmented_context_parts.append(markdown_table)
                                    processed_table_refs.add(table_json_filename)
                                except Exception as load_table_e:
                                    st.warning(f"Could not load or format table {table_json_filename}: {load_table_e}")
                                    augmented_context_parts.append(chunk_text) # Fallback to summary chunk
                            else: # Fallback if JSON file is missing
                                augmented_context_parts.append(chunk_text)
                        # else, table already added, original chunk_text (summary) might be redundant or add it if desired
                    else:
                        augmented_context_parts.append(chunk_text)

                final_context_for_llm = "\n\n---\n\n".join(augmented_context_parts)

                # 3. Generate Answer using the final context
                with st.spinner("Generating answer..."):
                    try:
                        # Call the function from idp_extraction
                        answer_result = answer_query_with_context(query, [final_context_for_llm]) # Pass as a list with one combined string
                        # Store the complete result for display
                        st.session_state.last_query_result = {
                            "query": query,
                            "answer_data": answer_result, # This might be a dict with 'answer' or 'error'
                            "retrieved_context": final_chunks_with_scores # Store chunks with scores
                        }
                    except NotImplementedError:
                         st.error("Answer generation function (answer_query_with_context) is not implemented.")
                         st.session_state.last_query_result = None
                    except Exception as answer_e:
                         st.error(f"Error generating answer: {answer_e}")
                         st.session_state.last_query_result = None

                # --- TEMPORARY DEBUG: Show final context sent to LLM ---
                # st.markdown("---")
                # st.subheader("DEBUG: Final Context for LLM")
                # st.text_area("Context String:", value=final_context_for_llm, height=300, key="debug_final_context")
                # --- END TEMPORARY DEBUG ---
            elif not query:
                st.warning("Please enter a question.")

        # --- Display Last Query Result ---
        st.markdown("---")
        if st.session_state.get("last_query_result"):
            query_result = st.session_state.last_query_result
            st.markdown("#### Last Query Result")
            st.markdown(f"**Your Query:** {query_result['query']}")

            # Display Answer
            answer_data = query_result.get("answer_data")
            if answer_data and isinstance(answer_data, dict):
                if "error" in answer_data:
                    st.error(f"Error generating answer: {answer_data['error']}")
                elif "answer" in answer_data:
                    st.success("**Generated Answer:**")
                    st.markdown(answer_data["answer"]) # Display the actual answer
                else:
                     # Handle cases where the function returns a dict without 'answer' or 'error'
                     st.info("Answer generation result received, but no specific 'answer' field found.")
                     st.json(answer_data) # Show the raw result dict
            elif ask_button: # Only show this warning if the button was just pressed
                st.warning("Could not generate an answer based on the retrieved context.") # Keep this warning if needed

            st.markdown("---")

    elif selected_display_name:
         # This case handles when a document was selected but failed to load (index, chunks, or tokenized_chunks)
         st.warning(f"Document '{selected_display_name}' could not be loaded. Please check logs or try reprocessing in the Admin tab.")

# Versi dan detail teknis
    st.caption("IDocspy, Built by Adnuri Mohamidi with help from AI :orange_heart:", help="cyberariani@gmail.com")