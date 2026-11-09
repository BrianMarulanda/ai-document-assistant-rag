"""PDF loading and text chunking utilities."""

from pathlib import Path
from typing import List, Union

import pypdf
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentIngester:
    """Load PDF documents and split their pages into overlapping text chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        """Configure the text splitter."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def load_and_split_pdf(self, pdf_path: Union[str, Path]) -> List[Document]:
        """Load a PDF, split its contents, and return the resulting chunks."""
        pdf_path = Path(pdf_path).expanduser().resolve()
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        print(f"Loading document: {pdf_path}")
        try:
            pages = PyPDFLoader(pdf_path).load()
        except Exception as error:
            print(
                f"Standard PDF loading failed ({error}); retrying with the "
                "direct PDF reader."
            )
            reader = pypdf.PdfReader(pdf_path)
            if reader.is_encrypted and not reader.decrypt(""):
                raise ValueError(
                    f"The PDF is encrypted and could not be opened: {pdf_path}"
                ) from error

            pages = [
                Document(
                    page_content=page.extract_text() or "",
                    metadata={"source": str(pdf_path), "page": index},
                )
                for index, page in enumerate(reader.pages)
            ]

        print(
            f"Splitting document (chunk size: {self.chunk_size}, "
            f"overlap: {self.chunk_overlap})..."
        )
        chunks = self.text_splitter.split_documents(pages)
        if not chunks:
            raise ValueError(
                f"No readable text was found in the PDF document: {pdf_path}"
            )

        print(f"Document processed successfully. Generated {len(chunks)} chunks.")
        return chunks
