"""Run a sample document query from the command line."""

from pathlib import Path

from src.retriever.query_engine import OllamaUnavailableError, RAGQueryEngine

PROJECT_ROOT = Path(__file__).resolve().parent
VECTOR_STORE_PATH = PROJECT_ROOT / "data" / "processed" / "chroma_db"


def main() -> int:
    """Run a sample query and print its answer and source excerpts."""
    print("Initializing the local document assistant...")
    engine = RAGQueryEngine(persist_directory=VECTOR_STORE_PATH)
    question = (
        "What was the total amount of Apple's common stock repurchases "
        "in fiscal year 2026?"
    )

    print(f'\nSearching the indexed documents for: "{question}"\n')
    try:
        result = engine.query(question)
    except OllamaUnavailableError:
        print(
            "Error: The local Ollama service could not be reached. Start Ollama "
            "and confirm that the configured model is available."
        )
        return 1

    print("=" * 50)
    print(f"QUESTION: {question}")
    print("=" * 50)
    print(f"ANSWER:\n{result['result']}")
    print("=" * 50)
    print("SOURCES:")

    for index, document in enumerate(result["source_documents"], start=1):
        page_number = document.metadata.get("page", "Unknown")
        excerpt = document.page_content[:300]
        print(f"\n[Source {index} — Page {page_number}]:")
        print(f"{excerpt}...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
