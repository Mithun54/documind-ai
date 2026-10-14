"""Unit tests for Phase 2: PDF text extraction and chunking."""

import os
import pathlib
import pytest
try:
    import pymupdf as fitz
except ImportError:
    import fitz

from ingest import (
    extract_pages_from_pdf,
    extract_text_from_pdf,
    chunk_text,
    chunk_pages,
    ingest_pdf,
    ExtractedPage,
    DocumentChunk,
    PDFProcessingError,
)


@pytest.fixture
def sample_pdf(tmp_path: pathlib.Path) -> str:
    """Generate a valid, single-page test PDF with known text."""
    pdf_path = tmp_path / "valid_sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "DocuMind AI makes document intelligence accessible.")
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


@pytest.fixture
def multi_page_pdf(tmp_path: pathlib.Path) -> str:
    """Generate a multi-page PDF with 3 pages of distinct text."""
    pdf_path = tmp_path / "multipage.pdf"
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text((50, 50), "This is the content on page one of the document.")

    p2 = doc.new_page()
    p2.insert_text((50, 50), "This is the content on page two with more information.")

    p3 = doc.new_page()
    p3.insert_text((50, 50), "This is the final content on page three.")

    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


@pytest.fixture
def blank_pdf(tmp_path: pathlib.Path) -> str:
    """Generate a PDF containing pages with no extractable text."""
    pdf_path = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


@pytest.fixture
def invalid_pdf(tmp_path: pathlib.Path) -> str:
    """Create a corrupt, non-PDF file with a .pdf extension."""
    file_path = tmp_path / "corrupt.pdf"
    file_path.write_bytes(b"This is not a real PDF file header.")
    return str(file_path)


def test_extract_text_from_valid_pdf(sample_pdf: str) -> None:
    """Verify text extraction succeeds and contains expected content."""
    text = extract_text_from_pdf(sample_pdf)
    assert "DocuMind AI" in text
    assert "accessible" in text


def test_preserving_page_numbers(multi_page_pdf: str) -> None:
    """Verify that 1-based page numbers are preserved across all pages."""
    pages = extract_pages_from_pdf(multi_page_pdf)
    assert len(pages) == 3

    # Check 1-based page numbering
    assert pages[0].page_number == 1
    assert "page one" in pages[0].text

    assert pages[1].page_number == 2
    assert "page two" in pages[1].text

    assert pages[2].page_number == 3
    assert "page three" in pages[2].text


def test_chunking_long_text() -> None:
    """Verify that chunk_text divides long content into bounded chunks."""
    long_text = "Word " * 500  # 2500 characters
    chunk_size = 700
    chunk_overlap = 100

    chunks = chunk_text(long_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk) <= chunk_size


def test_checking_chunk_overlap() -> None:
    """Verify consecutive chunks share the expected overlap."""
    chunk_size = 700
    chunk_overlap = 100
    # Create test text longer than chunk_size
    text = "A" * 600 + "B" * 200 + "C" * 600

    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    assert len(chunks) >= 2

    # Verify overlap: end of chunk 0 equals start of chunk 1
    tail_chunk0 = chunks[0][-chunk_overlap:]
    head_chunk1 = chunks[1][:chunk_overlap]
    assert tail_chunk0 == head_chunk1


def test_chunk_pages_multi_page_span() -> None:
    """Verify that chunks spanning across pages retain all applicable page numbers."""
    pages = [
        ExtractedPage(page_number=1, text="A" * 400),
        ExtractedPage(page_number=2, text="B" * 400),
    ]

    # chunk_size=500 with overlap=100 will span across page 1 and page 2
    chunks = chunk_pages(pages, source="doc.pdf", chunk_size=500, chunk_overlap=100)
    assert len(chunks) >= 2

    # First chunk covers page 1 and page 2
    spanning_chunk = chunks[0]
    assert 1 in spanning_chunk.page_numbers
    assert 2 in spanning_chunk.page_numbers
    assert spanning_chunk.source == "doc.pdf"


def test_handling_empty_or_text_free_pdf(blank_pdf: str) -> None:
    """Verify that empty PDFs raise a clear PDFProcessingError."""
    with pytest.raises((PDFProcessingError, ValueError)) as excinfo:
        extract_pages_from_pdf(blank_pdf)
    assert "No extractable text" in str(excinfo.value)


def test_handling_invalid_pdf(invalid_pdf: str) -> None:
    """Verify that corrupt or invalid PDFs raise an informative error."""
    with pytest.raises((PDFProcessingError, ValueError)):
        extract_pages_from_pdf(invalid_pdf)


def test_handling_nonexistent_pdf(tmp_path: pathlib.Path) -> None:
    """Verify missing file raises FileNotFoundError."""
    non_existent = str(tmp_path / "does_not_exist.pdf")
    with pytest.raises(FileNotFoundError):
        extract_pages_from_pdf(non_existent)
