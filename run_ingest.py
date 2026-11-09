"""Ingest the sample PDF into the local Chroma vector database."""

from pathlib import Path

from src.ingester.loader import DocumentIngester
from src.vectorstore.store import VectorStoreManager

PROJECT_ROOT = Path(__file__).resolve().parent
PDF_PATH = PROJECT_ROOT / "data" / "raw" / "sample.pdf"
VECTOR_STORE_PATH = PROJECT_ROOT / "data" / "processed" / "chroma_db"


def main() -> int:
    """Load, split, and index the sample PDF."""
    print("Starting document ingestion and vectorization...")
    if not PDF_PATH.is_file():
        print(
            f"Error: The PDF was not found at {PDF_PATH}. Place the source "
            "document at data/raw/sample.pdf."
        )
        return 1

    ingester = DocumentIngester(chunk_size=500, chunk_overlap=50)
    chunks = ingester.load_and_split_pdf(PDF_PATH)
    print(f"Document chunks ready for indexing: {len(chunks)}")
    print(f"Metadata for the first chunk: {chunks[0].metadata}")

    vector_manager = VectorStoreManager(persist_directory=VECTOR_STORE_PATH)
    vector_manager.create_vector_store(chunks)
    print("Document ingestion completed. The local vector database is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
