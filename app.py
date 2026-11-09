"""Streamlit interface for the local document assistant."""

from pathlib import Path
import logging

import streamlit as st

from src.retriever.query_engine import OllamaUnavailableError, RAGQueryEngine

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
VECTOR_STORE_PATH = PROJECT_ROOT / "data" / "processed" / "chroma_db"

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="centered",
)


@st.cache_resource
def load_rag_engine() -> RAGQueryEngine:
    """Create and cache the RAG engine for the current Streamlit process."""
    LOGGER.info("Initializing the RAG engine for the Streamlit application.")
    return RAGQueryEngine(persist_directory=VECTOR_STORE_PATH)


st.title("📄 AI Document Assistant")
st.caption(
    "A private, local document assistant powered by LangChain, ChromaDB, "
    "Hugging Face embeddings, and Ollama (Llama 3)."
)

try:
    rag_engine = load_rag_engine()
except Exception as error:
    LOGGER.error("Unable to initialize the RAG engine: %s", error)
    st.error(
        "The document assistant could not be initialized. Confirm that the "
        "vector database is available and the required local dependencies "
        "are installed."
    )
else:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about your documents...")
    if prompt:
        st.session_state["messages"].append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(
                "Searching the vector database and generating a local response..."
            ):
                try:
                    response_data = rag_engine.query(prompt)
                except OllamaUnavailableError:
                    st.warning(
                        "The local Ollama service could not be reached. Start "
                        "Ollama and confirm that the configured model is available."
                    )
                except Exception as error:
                    LOGGER.error("Unable to process the user query: %s", error)
                    st.error(
                        "The request could not be completed. Please try again "
                        "or review the application logs."
                    )
                else:
                    answer = response_data["result"]
                    st.markdown(answer)

                    with st.expander("📚 View source excerpts"):
                        for index, document in enumerate(
                            response_data["source_documents"], start=1
                        ):
                            page_number = document.metadata.get("page", "Unknown")
                            st.markdown(
                                f"**Source {index} — Page {page_number}**"
                            )
                            st.text(document.page_content)
                            st.divider()

                    st.session_state["messages"].append(
                        {"role": "assistant", "content": answer}
                    )
