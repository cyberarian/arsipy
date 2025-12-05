import streamlit as st
from typing import Dict, List, Any, Union, Callable
from langchain_core.language_models.llms import LLM
from langchain.schema import Document

def show_chatbot_content(
    llm: Union[LLM, Any],
    model_options: Dict[str, str],
    chat_history: List,
    process_chat_func: Callable
) -> None:
    """Display chatbot interface content"""
    # Model selection
    selected_model = st.selectbox(
        "Select AI Model",
        options=list(model_options.keys()),
        key='model_selector'
    )
    model_id = model_options[selected_model]

    # Chat form
    with st.form(key='chat_form'):
        prompt = st.text_input("Enter your question about the documents", key='question_input')
        submit = st.form_submit_button("Submit Question")

        if submit and prompt:
            try:
                with st.spinner("Processing your question..."):
                    # Get vectorstore from session state
                    vectorstore = st.session_state.vectorstore
                    if vectorstore is None:
                        st.error("No documents loaded. Please add some documents first.")
                        return

                    # Process the query
                    response = process_chat_func(prompt, vectorstore, llm)
                    
                    if response and isinstance(response, dict):
                        st.session_state.chat_history.append((prompt, response))
                        st.rerun()

            except Exception as e:
                st.error(f"Error processing question: {str(e)}")

    # Display chat history
    display_chat_history(chat_history)

def display_chat_history(chat_history: List) -> None:
    """Display chat history with unique sources"""
    for q, a in chat_history:
        st.write("Question:", q)
        
        if isinstance(a, dict) and 'result' in a:
            st.write("Answer:", a['result'])
            
            if a.get('source_documents'):
                with st.expander("Sumber Referensi"):
                    shown_sources = set()
                    for doc in a['source_documents']:
                        source = doc.metadata.get('source', '')
                        if source and source not in shown_sources:
                            shown_sources.add(source)
                            st.markdown(f"**Sumber:** {source}")
                            st.markdown("---")
                            
        st.divider()

# Footer
    st.markdown("---")
    st.markdown("Developed by Adnuri Mohamidi with help from AI :orange_heart:", help="cyberariani@gmail.com")