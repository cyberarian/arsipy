import streamlit as st
from typing import Dict, List, Any, Union, Callable
from langchain_core.language_models.llms import LLM
from .tab_content.chatbot import show_chatbot_content
from .tab_content.guide import show_guide_content
from .tab_content.about import show_about_content
from .tab_content.resources import show_resources_content
from image_analyzer import image_analyzer_main

def show_all_tabs(
    llm: Union[LLM, Any],
    model_options: Dict[str, str],
    chat_history: List,
    process_chat_func: Callable
) -> None:
    """Display all tabs in the interface"""
    tab1, tab5, tab3, tab2, tab4 = st.tabs([
        "💬 Chatbot",
        "🖼️ Analisis tulisan tangan",
        "❓ Panduan",
        "ℹ️ Tentang",
        "📚 Resources"
    ])

    with tab1:
        show_chatbot_content(llm, model_options, chat_history, process_chat_func)
    with tab5:
        image_analyzer_main()
    with tab3:
        show_guide_content()
    with tab2:
        show_about_content()
    with tab4:
        show_resources_content()
