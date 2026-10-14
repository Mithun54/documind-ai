"""Document ingestion module for DocuMind AI.

This module provides:
1. PDF text extraction preserving 1-based page numbers using PyMuPDF (fitz).
2. Text chunking with configurable chunk size and overlap.
3. Metadata retention (source filename, page numbers) across single and multi-page chunks.
4. Robust error handling for invalid, unopenable, and text-free PDFs.
"""

from dataclasses import dataclass
import os
from typing import List, Dict, Any, Optional

try:
    import pymupdf as fitz
except ImportError:
    import fitz


class PDFProcessingError(ValueError):
    """Raised when a PDF cannot be opened, is corrupted, or has no extractable text."""
    pass


@dataclass
class ExtractedPage:
    """Represents a single extracted page from a PDF document.

    Attributes:
        page_number: 1-based original PDF page index.
        text: Extracted plain text for this page.
    """
    page_number: int
    text: str

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access (e.g., page['page_number'], page['text'])."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Invalid attribute '{key}'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert page object to a dictionary."""
        return {
            "page_number": self.page_number,
            "text": self.text,
        }


@dataclass
class DocumentChunk:
    """Represents a text chunk with source provenance metadata.

    Attributes:
        text: Chunk string content.
        source: Source document filename.
        page_numbers: List of 1-based page numbers that contributed to this chunk.
    """
    text: str
    source: str
    page_numbers: List[int]

    @property
    def page_number(self) -> int:
        """Convenience property returning the primary (first) page number."""
        return self.page_numbers[0] if self.page_numbers else 1

    @property
    def pages(self) -> List[int]:
        """Convenience alias for page_numbers."""
        return self.page_numbers

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access (e.g., chunk['text'], chunk['page_numbers'])."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Invalid attribute '{key}'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk object to a dictionary."""
        return {
            "text": self.text,
            "source": self.source,
            "page_numbers": self.page_numbers,
        }


def extract_pages_from_pdf(pdf_path: str) -> List[ExtractedPage]:
    """Extract text from a PDF file page by page, preserving 1-based page numbers.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        A list of ExtractedPage objects with 1-based page numbers and text.

    Raises:
        FileNotFoundError: If the file does not exist or is not a valid file.
        PDFProcessingError: If the PDF cannot be opened, is invalid, has 0 pages,
            or contains no extractable text.
    """
    if not os.path.exists(pdf_path) or not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: '{pdf_path}'")

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise PDFProcessingError(f"Could not open or parse PDF file '{pdf_path}': {exc}") from exc

    try:
        if not getattr(doc, "is_pdf", True):
            raise PDFProcessingError(f"The file '{pdf_path}' is not a valid PDF document.")

        if len(doc) == 0:
            raise PDFProcessingError(f"The PDF file '{pdf_path}' contains zero pages.")

        pages: List[ExtractedPage] = []
        total_extracted_chars = 0

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_text = page.get_text() or ""
            # Preserving original 1-based page number
            page_number = page_idx + 1
            pages.append(ExtractedPage(page_number=page_number, text=page_text))
            total_extracted_chars += len(page_text.strip())

        if total_extracted_chars == 0:
            raise PDFProcessingError(
                f"No extractable text found in '{pdf_path}'. "
                "The PDF may be empty or contain only scanned/raster images."
            )

        return pages
    finally:
        doc.close()


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract plain text from a PDF file across all pages.

    Args:
        pdf_path: Path to the target PDF document.

    Returns:
        The extracted combined text content.

    Raises:
        FileNotFoundError: If the file does not exist.
        PDFProcessingError: If the PDF cannot be processed or contains no text.
    """
    pages = extract_pages_from_pdf(pdf_path)
    return "\n\n".join(page.text for page in pages if page.text.strip())


def chunk_text(
    text: str,
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> List[str]:
    """Split raw text into overlapping character chunks.

    Args:
        text: Raw text string to divide.
        chunk_size: Target character length per chunk (default: 700).
        chunk_overlap: Number of overlapping characters between chunks (default: 100).

    Returns:
        A list of text chunk strings.

    Raises:
        ValueError: If chunk_size <= 0 or chunk_overlap is invalid.
    """
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly less than chunk_size.")

    step = chunk_size - chunk_overlap
    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_len:
            break
        start += step

    return chunks


def chunk_pages(
    pages: List[ExtractedPage],
    source: str = "",
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> List[DocumentChunk]:
    """Split extracted pages into overlapping DocumentChunk objects.

    Each chunk retains its source filename and all applicable 1-based page numbers,
    accurately tracking spans across multiple pages.

    Args:
        pages: List of ExtractedPage objects.
        source: Source document filename or path.
        chunk_size: Target character length per chunk (default: 700).
        chunk_overlap: Number of overlapping characters between chunks (default: 100).

    Returns:
        A list of DocumentChunk objects with text, source, and page_numbers.
    """
    if not pages:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly less than chunk_size.")

    source_name = os.path.basename(source) if source else ""
    full_text = ""
    char_page_map: List[int] = []

    for page in pages:
        page_text = page.text
        if not page_text:
            continue

        if full_text:
            separator = "\n\n"
            full_text += separator
            char_page_map.extend([page.page_number] * len(separator))

        full_text += page_text
        char_page_map.extend([page.page_number] * len(page_text))

    if not full_text:
        return []

    step = chunk_size - chunk_overlap
    chunks: List[DocumentChunk] = []
    text_len = len(full_text)
    start = 0

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_content = full_text[start:end]

        # Collect unique, ordered page numbers that contribute to this chunk
        pages_in_chunk = sorted(list(dict.fromkeys(char_page_map[start:end])))

        chunks.append(
            DocumentChunk(
                text=chunk_content,
                source=source_name,
                page_numbers=pages_in_chunk,
            )
        )

        if end == text_len:
            break
        start += step

    return chunks


# Helpful alias for chunk_pages
chunk_document = chunk_pages


def ingest_pdf(
    pdf_path: str,
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> List[DocumentChunk]:
    """End-to-end PDF ingestion: extracts pages and produces overlapping chunks with metadata.

    Args:
        pdf_path: Path to the target PDF document.
        chunk_size: Target character length per chunk (default: 700).
        chunk_overlap: Number of overlapping characters between chunks (default: 100).

    Returns:
        A list of DocumentChunk objects with source and page metadata.
    """
    pages = extract_pages_from_pdf(pdf_path)
    return chunk_pages(
        pages=pages,
        source=os.path.basename(pdf_path),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


# Helpful alias for ingest_pdf
process_pdf = ingest_pdf


def index_documents(
    chunks: List[Any],
    collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Store text chunks and their embeddings into ChromaDB.

    Delegates to rag.index_chunks for vector storage.

    Args:
        chunks: List of DocumentChunk objects or text strings.
        collection_name: Target ChromaDB collection name (defaults to active collection).
        persist_directory: Optional custom ChromaDB persistence folder.
        model: Target embedding model (defaults to DEFAULT_EMBEDDING_MODEL).

    Returns:
        Summary dictionary with indexing statistics.
    """
    from rag import index_chunks, DEFAULT_COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL  # Local import
    target_coll = collection_name or DEFAULT_COLLECTION_NAME
    target_model = model or DEFAULT_EMBEDDING_MODEL
    return index_chunks(
        chunks=chunks,
        collection_name=target_coll,
        persist_directory=persist_directory,
        model=target_model,
    )
