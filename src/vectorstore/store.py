"""Local Chroma vector-store management."""

from pathlib import Path
from typing import List, Union

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStoreManager:
    """Create and load a persistent Chroma vector store."""

    def __init__(
        self, persist_directory: Union[str, Path] = "data/processed/chroma_db"
    ) -> None:
        """Initialize the local Hugging Face embedding model."""
        project_root = Path(__file__).resolve().parents[2]
        configured_path = Path(persist_directory).expanduser()
        if not configured_path.is_absolute():
            configured_path = project_root / configured_path
        self.persist_directory = configured_path.resolve()

        print("Loading the local all-MiniLM-L6-v2 embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def create_vector_store(self, chunks: List[Document]) -> Chroma:
        """Create and persist a Chroma vector store from document chunks."""
        if not chunks:
            raise ValueError("At least one document chunk is required for indexing.")

        self.persist_directory.mkdir(parents=True, exist_ok=True)
        print(
            "Generating local embeddings and saving the Chroma database to "
            f"{self.persist_directory}..."
        )
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        print("The local Chroma vector database was created successfully.")
        return vector_store

    def load_vector_store(self) -> Chroma:
        """Load the persistent Chroma vector store from disk."""
        if not self.persist_directory.is_dir():
            raise FileNotFoundError(
                f"Vector database not found at {self.persist_directory}. "
                "Run the ingestion script first."
            )

        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )
