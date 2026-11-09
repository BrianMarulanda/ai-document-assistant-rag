"""Local retrieval-augmented generation query engine."""

from pathlib import Path
from typing import Dict, List, Union

import httpx
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama service cannot be reached."""


class RAGQueryEngine:
    """Retrieve relevant document chunks and generate answers with Ollama."""

    def __init__(
        self,
        persist_directory: Union[str, Path] = "data/processed/chroma_db",
        model_name: str = "llama3",
    ) -> None:
        """Initialize the local embeddings, Chroma store, and Ollama chain."""
        project_root = Path(__file__).resolve().parents[2]
        configured_path = Path(persist_directory).expanduser()
        if not configured_path.is_absolute():
            configured_path = project_root / configured_path
        self.persist_directory = configured_path.resolve()
        if not self.persist_directory.is_dir():
            raise FileNotFoundError(
                f"Vector database not found at {self.persist_directory}. "
                "Run run_ingest.py before starting the assistant."
            )

        print("Loading the local embedding model for document retrieval...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        print(f"Connecting to the vector database at {self.persist_directory}...")
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )

        print(f"Configuring the local Ollama model ({model_name})...")
        self.llm = ChatOllama(model=model_name, temperature=0.0)
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )

        system_prompt = (
            "You are a professional corporate document assistant. Answer the "
            "question using only the following context from the indexed documents. "
            "If the context does not contain the answer, say so clearly.\n\n"
            "{context}"
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        def format_docs(documents: List[Document]) -> str:
            """Combine the retrieved document text into a single context."""
            return "\n\n".join(document.page_content for document in documents)

        self.chain = (
            {
                "context": self.retriever | format_docs,
                "input": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str) -> Dict[str, Union[str, List[Document]]]:
        """Answer a question and return the retrieved source documents."""
        print(f'Querying the indexed documents: "{question}"...')

        try:
            response_text = self.chain.invoke(question)
        except (httpx.ConnectError, httpx.ConnectTimeout) as error:
            raise OllamaUnavailableError(
                "The local Ollama service could not be reached."
            ) from error

        source_documents = self.retriever.invoke(question)
        return {
            "result": response_text,
            "source_documents": source_documents,
        }
