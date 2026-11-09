# Local RAG Document Assistant

A fully local retrieval-augmented generation (RAG) application for ingesting,
indexing, and querying corporate documents. The application uses local models
and a persistent vector database to keep document processing on the user's
machine.

## Architecture

- **Document ingestion:** PyPDFLoader and a recursive character text splitter
  load and segment PDF documents.
- **Vector storage:** ChromaDB persists document vectors locally.
- **Embeddings:** Hugging Face `all-MiniLM-L6-v2` generates embeddings.
- **Language model:** Ollama runs the `llama3` model locally.
- **User interface:** Streamlit provides a chat interface with source excerpts.

## Project structure

```text
.
├── app.py
├── run_ingest.py
├── run_query.py
├── data/
│   ├── raw/                 # Source PDF documents
│   └── processed/chroma_db/ # Persistent Chroma database
└── src/
    ├── ingester/
    ├── retriever/
    └── vectorstore/
```

## Setup

1. Create and activate a virtual environment.
2. Install the project dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Install [Ollama](https://ollama.com/) and download the configured model:

   ```powershell
   ollama pull llama3
   ```

4. Place a PDF named `sample.pdf` in `data/raw/`.

No external API key is required. The Hugging Face embedding model is downloaded
on first use; subsequent application interactions reuse the cached RAG engine.

## Usage

From the project root, build or refresh the local vector database:

```powershell
python run_ingest.py
```

Run a sample query from the terminal:

```powershell
python run_query.py
```

Launch the Streamlit application:

```powershell
streamlit run app.py
```

The application is available at `http://localhost:8501`. If Ollama is not
running, the interface displays a clear warning with startup guidance.
