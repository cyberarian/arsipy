# d:\pandasai\idp\rag_processor.py
import streamlit as st
import re # Import regex for tokenizer
import numpy as np

# --- Import HyDE function ---
from utils import generate_hypothetical_answer

import time

# --- Embedding Model ---
# --- LangChain Text Splitter ---
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    st.error("LangChain library not found. Please install it: pip install langchain")
_embedding_model = None
_cross_encoder = None # Added for re-ranking
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    # Load a relatively small but effective model for demonstration
    _model_name = 'all-MiniLM-L6-v2'
    _cross_encoder_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2' # Common cross-encoder

    # Use caching for models to avoid reloading on every script run
    @st.cache_resource(show_spinner=False) # Hide spinner for background loading
    def load_embedding_model(model_name):
        try:
            # print(f"Attempting to load embedding model: {model_name}") # DEBUG ONLY
            model = SentenceTransformer(model_name)
            # print(f"Embedding model ({model_name}) loaded successfully.") # DEBUG ONLY
            return model
        except Exception as e:
            st.error(f"Failed to load embedding model '{model_name}': {e}")
            return None

    @st.cache_resource(show_spinner=False) # Hide spinner for background loading
    def load_cross_encoder_model(model_name):
        try:
            # print(f"Attempting to load cross-encoder model: {model_name}") # DEBUG ONLY
            model = CrossEncoder(model_name)
            # print(f"Cross-encoder model ({model_name}) loaded successfully.") # DEBUG ONLY
            return model
        except Exception as e:
            st.error(f"Failed to load cross-encoder model '{model_name}': {e}")
            return None

    # Load models during script initialization
    _embedding_model = load_embedding_model(_model_name)
    _cross_encoder = load_cross_encoder_model(_cross_encoder_name) # Load cross-encoder

    # --- REMOVED SUCCESS MESSAGES ---
    # The st.success messages previously here have been removed as requested.
    # Model loading now happens silently in the background thanks to st.cache_resource.
    # Errors during loading will still be displayed via st.error within the functions.

except ImportError:
    st.error("SentenceTransformers library not found. Please install it: pip install sentence-transformers")
except Exception as e:
    st.error(f"An error occurred during model loading: {e}")


# --- Vector Store (FAISS) ---
_faiss_index = None
try:
    import faiss
    _faiss_index = faiss # Assign the library itself
except ImportError:
    st.error("FAISS library not found. Please install it: pip install faiss-cpu")

# --- BM25 Library ---
try:
    from rank_bm25 import BM25Okapi
    # Simple tokenizer for BM25: lowercase and split by non-alphanumeric
    def bm25_tokenizer(text):
        return re.split(r'\W+', text.lower()) if text else []
except ImportError:
    st.error("rank_bm25 library not found. Please install it: pip install rank_bm25")


# --- Chunking Function ---
def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Splits text into overlapping chunks."""
    if not text:
        return []
    if chunk_overlap >= chunk_size:
        # Use st.warning for user feedback within Streamlit context
        st.warning(f"Chunk overlap ({chunk_overlap}) is greater than or equal to chunk size ({chunk_size}). Setting overlap to {chunk_size // 3}.")
        chunk_overlap = chunk_size // 3 # Prevent excessive overlap or errors

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # Ensure next_start calculation is safe
        next_start = start + chunk_size - chunk_overlap
        # Prevent infinite loop if overlap is too large or step is zero/negative
        if next_start <= start:
            if start + 1 < len(text): # Move forward by at least one char if possible
                 next_start = start + 1
            else: # Reached the end
                 break
        start = next_start

    # Filter out potentially empty chunks if overlap logic creates them
    return [chunk for chunk in chunks if chunk and chunk.strip()]

# --- REMOVED OLD chunk_text FUNCTION ---

# --- Indexing Function ---
def create_index_from_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """Chunks text using Markdown-aware splitter, creates embeddings, tokenizes for BM25, and builds a FAISS index."""
    if not _embedding_model:
        st.error("Cannot create index: Embedding model not loaded.")
        return None, None
    if not _faiss_index:
        st.error("Cannot create index: FAISS library not available.")
        return None, None
    if not text or not text.strip():
        st.warning("Cannot create index: Input text is empty.")
        return None, None

    start_time = time.time()
    # Use st.spinner for user feedback during potentially long operations
    with st.spinner(f"Chunking document text (size={chunk_size}, overlap={chunk_overlap})..."):
        # --- USE RecursiveCharacterTextSplitter ---
        markdown_splitter = RecursiveCharacterTextSplitter(
            # Define Markdown separators (from headers down to paragraphs/lines)
            separators=[
                "\n# ", "\n## ", "\n### ", "\n#### ", # Markdown headers
                "\n\n", "\n", " ", ""              # Paragraphs, lines, words
            ],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False, # Separators are plain strings
        )
        chunks = markdown_splitter.split_text(text)
        if not chunks: # Check if chunks list is empty
            st.warning("No text chunks generated after chunking.")
        st.info(f"Document split into {len(chunks)} chunks.")

    with st.spinner(f"Generating embeddings for {len(chunks)} chunks... (This may take time)"):
        try:
            # Use the loaded embedding model
            embeddings = _embedding_model.encode(chunks, show_progress_bar=False) # Disable progress bar for cleaner UI
            # Ensure embeddings are numpy array and float32 for FAISS
            if isinstance(embeddings, list): embeddings = np.array(embeddings)
            embeddings = embeddings.astype('float32')
        except Exception as e:
            st.error(f"Failed to generate embeddings: {e}")
            return None, None
    
    # --- Build FAISS Index ---
    bm25_index_data = None # Initialize BM25 data
    if embeddings.shape[0] > 0:
        dimension = embeddings.shape[1]
        # Using IndexFlatL2 - suitable for many sentence transformers including MiniLM
        index = _faiss_index.IndexFlatL2(dimension)
        try:
            # --- Tokenize for BM25 ---
            with st.spinner("Tokenizing chunks for BM25 index..."):
                tokenized_chunks = [bm25_tokenizer(chunk) for chunk in chunks]
                if not any(tokenized_chunks): # Check if all tokenized chunks are empty
                    st.warning("Tokenization resulted in empty lists for all chunks. BM25 index cannot be built.")
                    # Proceed without BM25
                else:
                    # Create BM25 index object (we'll save tokenized chunks, not the object directly for simplicity)
                    # bm25_index = BM25Okapi(tokenized_chunks) # Object created on the fly during retrieval
                    st.info("Tokenized chunks for BM25.")
                    bm25_index_data = tokenized_chunks # Save the tokenized data needed to build it

            # --- Add to FAISS ---
            index.add(embeddings)
            end_time = time.time()
            # Use st.success to indicate successful completion
            st.success(f"Document indexed successfully in {end_time - start_time:.2f} seconds.")
            # Return FAISS index, original chunks, and tokenized chunks for BM25
            return index, chunks, bm25_index_data
        except Exception as e:
            st.error(f"Failed to add embeddings to FAISS index: {e}")
            return None, None
    else:
        st.warning("No embeddings were generated to build the index.")
        return None, None, None # Return three Nones


# --- Constants for RRF ---
RRF_K = 60 # Parameter for Reciprocal Rank Fusion

# --- Retrieval Function (Hybrid Search + Threshold + RRF) ---
# --- Updated Retrieval Function ---
def retrieve_relevant_chunks(
    query: str,
    index,
    chunks: list[str],
    tokenized_chunks: list[list[str]] | None, # Add tokenized chunks for BM25
    top_k: int = 5, # Default k for initial retrieval
    score_threshold: float = 0.0 # Default threshold (0.0 means no filtering by score initially)
) -> list[tuple[str, float]]: # Return chunks and their RRF scores

    """
    Performs hybrid search (FAISS + BM25), combines results using RRF,
    filters by semantic score threshold, and returns top_k chunks with RRF scores.
    """
    # --- Initial Checks ---
    if not _embedding_model:
        st.error("Cannot retrieve chunks: Embedding model not loaded.")
        return []
    if not index:
        st.error("Cannot retrieve chunks: FAISS index not available.")
        return []
    # BM25 index is optional for now, but warn if missing when expected
    if not chunks or not tokenized_chunks: # Combine the checks
        st.warning("Cannot retrieve chunks: No document chunks provided.")
        return []
    if not query or not query.strip():
        st.warning("Cannot retrieve chunks: Query is empty.")
        return []

    try:
        # --- Determine number of candidates to fetch from each retriever ---
        # Fetch more candidates initially for better RRF results and potential re-ranking later
        k_to_retrieve = max(top_k * 5, 20) # Retrieve significantly more candidates
        # --- HyDE Logic ---
        text_to_embed = query # Default to original query
        use_hyde = st.session_state.get("global_retrieval_hyde", False)

        if use_hyde:
            with st.spinner("HyDE Enabled: Generating hypothetical answer..."):
                hypothetical_answer = generate_hypothetical_answer(query)
            if hypothetical_answer:
                text_to_embed = hypothetical_answer # Use hypothetical answer for embedding
                st.sidebar.caption(f"HyDE Query: _{hypothetical_answer[:100]}..._") # Show snippet in sidebar
            else:
                st.warning("HyDE enabled, but failed to generate hypothetical answer. Falling back to original query.")

        # --- Semantic Search (FAISS) ---
        semantic_results = {} # Store as {index: score}
        spinner_text = f"Finding relevant context using {'HyDE query' if use_hyde and hypothetical_answer else 'original query'}..."
        with st.spinner(spinner_text):
            query_embedding = _embedding_model.encode([text_to_embed]) # Embed either original query or hypothetical answer
            if isinstance(query_embedding, list): query_embedding = np.array(query_embedding)
            query_embedding = query_embedding.astype('float32')

            distances, indices = index.search(query_embedding, k_to_retrieve)

            if indices.size > 0 and distances.size > 0:
                if indices.ndim == 2 and distances.ndim == 2:
                    valid_indices = indices[0]
                    valid_distances = distances[0]
                    for i, dist in zip(valid_indices, valid_distances):
                        if i < 0 or i >= len(chunks): continue # Skip invalid indices
                        similarity_score = 1.0 / (1.0 + float(dist))
                        # Apply threshold *here* for semantic results
                        if similarity_score >= score_threshold:
                            semantic_results[int(i)] = similarity_score # Store index and score
                else:
                    st.warning(f"Unexpected shape from FAISS search: indices {indices.shape}, distances {distances.shape}")

        # --- Keyword Search (BM25) ---
        bm25_results = {} # Store as {index: score}
        if tokenized_chunks: # Only run if tokenized data exists
            with st.spinner("Performing keyword search (BM25)..."):
                try:
                    bm25 = BM25Okapi(tokenized_chunks)
                    tokenized_query = bm25_tokenizer(query) # Use the same tokenizer
                    bm25_scores = bm25.get_scores(tokenized_query)

                    # Get top k_to_retrieve indices based on BM25 scores
                    top_n_indices_bm25 = np.argsort(bm25_scores)[::-1][:k_to_retrieve]

                    for i in top_n_indices_bm25:
                        score = bm25_scores[i]
                        # BM25 scores can be negative, we only care about relative ranking for RRF
                        # No threshold applied here, RRF handles relevance fusion
                        if score > 0: # Often good to filter out non-matching docs
                            bm25_results[int(i)] = score
                except Exception as bm25_e:
                    st.warning(f"BM25 search failed: {bm25_e}")
        else:
            st.info("BM25 tokenized data not available. Skipping keyword search.")

        # --- Reciprocal Rank Fusion (RRF) ---
        rrf_scores = {}
        all_doc_indices = set(semantic_results.keys()) | set(bm25_results.keys())

        # Create rank lists (index: rank), lower rank is better (rank 1 is best)
        semantic_rank_list = {doc_id: rank + 1 for rank, doc_id in enumerate(sorted(semantic_results, key=semantic_results.get, reverse=True))}
        bm25_rank_list = {doc_id: rank + 1 for rank, doc_id in enumerate(sorted(bm25_results, key=bm25_results.get, reverse=True))}

        for doc_id in all_doc_indices:
            score = 0.0
            if doc_id in semantic_rank_list:
                score += 1.0 / (RRF_K + semantic_rank_list[doc_id])
            if doc_id in bm25_rank_list:
                score += 1.0 / (RRF_K + bm25_rank_list[doc_id])
            rrf_scores[doc_id] = score

        # Sort documents based on RRF score
        sorted_indices_rrf = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

        # --- Prepare Final Results ---
        final_results = []
        for i in sorted_indices_rrf[:top_k]: # Take top_k based on RRF score
            # Return chunk text and its RRF score
            final_results.append((chunks[i], rrf_scores[i]))

        return final_results

    except Exception as e:
        st.error(f"Error during context retrieval: {e}")
        # Optionally log the full traceback here for debugging
        # import traceback
        # traceback.print_exc()
        return []

# --- Re-ranking Function ---
def rerank_chunks(query: str, candidate_chunks_with_rrf_scores: list[tuple[str, float]], top_k: int) -> list[tuple[str, float]]:
    """Re-ranks candidate chunks using a cross-encoder model."""
    if not _cross_encoder:
        # Don't show error if re-ranking wasn't requested, just return original
        # st.warning("Re-ranking skipped: Cross-encoder model not loaded.")
        return candidate_chunks_with_scores[:top_k] # Apply top_k even if not re-ranking

    if not candidate_chunks_with_scores:
        return []

    candidate_texts = [chunk for chunk, score in candidate_chunks_with_scores]
    if not candidate_texts: # Check if list is empty after extraction
        return []

    try:
        # Use spinner for user feedback
        with st.spinner(f"Re-ranking {len(candidate_texts)} candidates..."):
            # Create pairs of [query, chunk_text] for the cross-encoder
            pairs = [[query, text] for text in candidate_texts]
            # Predict scores (these are typically relevance scores, higher is better)
            cross_scores = _cross_encoder.predict(pairs, show_progress_bar=False)

            # Combine original chunks with new cross-encoder scores
            reranked_results = []
            for i in range(len(candidate_texts)):
                 # Keep original chunk text, use new cross-score for sorting
                 reranked_results.append((candidate_texts[i], float(cross_scores[i])))

            # Sort by the new cross-encoder score, descending
            reranked_results.sort(key=lambda x: x[1], reverse=True)

            # Return the top_k re-ranked chunks (still as tuple with score)
            return reranked_results[:top_k]
    except Exception as e:
        st.error(f"Error during re-ranking: {e}")
        # Fallback to original candidates (sorted by initial score) on error
        candidate_chunks_with_scores.sort(key=lambda x: x[1], reverse=True) # Sort by RRF score as fallback
        return candidate_chunks_with_scores[:top_k]
