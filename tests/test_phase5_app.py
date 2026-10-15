"""Unit tests for Phase 5: Streamlit application helpers and UI/backend integration."""

import io
import os
import pathlib
from unittest.mock import MagicMock
import pytest

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from app import (
    validate_and_save_upload,
    sanitize_filename,
    format_citation_display,
    get_active_filter,
    get_theme_css,
    apply_theme,
)
from rag import (
    index_pdf,
    answer_question,
    get_indexed_documents,
)


class MockUploadedFile:
    """Mock Streamlit UploadedFile object for unit testing."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def read(self) -> bytes:
        return self._data

    def getvalue(self) -> bytes:
        return self._data


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Generate in-memory PDF bytes with valid text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "DocuMind Streamlit UI integration test document.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_sanitize_filename() -> None:
    """Verify path traversal characters and unsafe symbols are sanitized."""
    assert sanitize_filename("../../malicious.pdf") == "malicious.pdf"
    assert sanitize_filename("folder/sub/doc.pdf") == "doc.pdf"
    assert sanitize_filename("my doc (v1).pdf") == "my_doc__v1_.pdf"


def test_validate_and_save_upload_valid(tmp_path: pathlib.Path, sample_pdf_bytes: bytes) -> None:
    """Verify valid PDF upload is stored cleanly in destination folder."""
    upload_dir = str(tmp_path / "uploads")
    mock_file = MockUploadedFile(name="manual.pdf", data=sample_pdf_bytes)

    saved_path = validate_and_save_upload(mock_file, upload_dir=upload_dir, max_size_mb=10)
    assert os.path.exists(saved_path)
    assert os.path.basename(saved_path) == "manual.pdf"


def test_validate_and_save_upload_collision_handling(
    tmp_path: pathlib.Path, sample_pdf_bytes: bytes
) -> None:
    """Verify that uploading different files with the same filename avoids overwriting."""
    upload_dir = str(tmp_path / "uploads")

    # First file
    mock_file1 = MockUploadedFile(name="data.pdf", data=sample_pdf_bytes)
    path1 = validate_and_save_upload(mock_file1, upload_dir=upload_dir)

    # Identical content upload should reuse path
    mock_file_same = MockUploadedFile(name="data.pdf", data=sample_pdf_bytes)
    path_same = validate_and_save_upload(mock_file_same, upload_dir=upload_dir)
    assert path1 == path_same

    # Second file with distinct content
    different_bytes = sample_pdf_bytes + b"%extra-content"
    mock_file2 = MockUploadedFile(name="data.pdf", data=different_bytes)
    path2 = validate_and_save_upload(mock_file2, upload_dir=upload_dir)

    assert path1 != path2
    assert os.path.exists(path1)
    assert os.path.exists(path2)
    assert "data_1.pdf" in path2


def test_validate_and_save_upload_invalid_extension(tmp_path: pathlib.Path) -> None:
    """Verify non-PDF uploads are rejected."""
    upload_dir = str(tmp_path / "uploads")
    mock_txt = MockUploadedFile(name="notes.txt", data=b"text content")

    with pytest.raises(ValueError, match="Only PDF documents are supported"):
        validate_and_save_upload(mock_txt, upload_dir=upload_dir)


def test_validate_and_save_upload_empty_file(tmp_path: pathlib.Path) -> None:
    """Verify 0-byte PDF uploads are rejected."""
    upload_dir = str(tmp_path / "uploads")
    mock_empty = MockUploadedFile(name="empty.pdf", data=b"")

    with pytest.raises(ValueError, match="is empty"):
        validate_and_save_upload(mock_empty, upload_dir=upload_dir)


def test_validate_and_save_upload_size_limit_exceeded(tmp_path: pathlib.Path) -> None:
    """Verify oversized uploads are rejected."""
    upload_dir = str(tmp_path / "uploads")
    large_bytes = b"0" * (2 * 1024 * 1024)  # 2 MB
    mock_large = MockUploadedFile(name="large.pdf", data=large_bytes)

    # Limit to 1 MB
    with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
        validate_and_save_upload(mock_large, upload_dir=upload_dir, max_size_mb=1)


def test_get_active_filter() -> None:
    """Verify dropdown selection mapping to backend document filter."""
    assert get_active_filter("All Documents (Entire Knowledge Base)") is None
    assert get_active_filter("All Documents") is None
    assert get_active_filter("financial_report.pdf") == "financial_report.pdf"
    assert get_active_filter("") is None


def test_format_citation_display() -> None:
    """Verify citation formatting contains source and page numbers."""
    citation = {
        "source": "specification.pdf",
        "page_numbers": [4, 5],
        "page": 4,
    }
    rendered = format_citation_display(citation)
    assert "specification.pdf" in rendered
    assert "4, 5" in rendered


def test_ui_backend_integration_flow(tmp_path: pathlib.Path) -> None:
    """Test full integration from file saving to indexing and question answering with mock AI."""
    persist_dir = str(tmp_path / "app_integration_chroma")
    upload_dir = str(tmp_path / "app_integration_uploads")

    # 1. Create temporary PDF
    pdf_path = tmp_path / "handbook.pdf"
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "The company policy mandates 25 days of annual leave.")
    doc.save(str(pdf_path))
    doc.close()

    # 2. Upload and save
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()
    mock_upload = MockUploadedFile(name="handbook.pdf", data=file_bytes)
    saved_file = validate_and_save_upload(mock_upload, upload_dir=upload_dir)

    # 3. Setup mock client
    mock_client = MagicMock()
    emb_resp = MagicMock()
    emb_val = MagicMock()
    emb_val.values = [0.1, 0.2, 0.3, 0.4]
    emb_resp.embeddings = [emb_val]
    mock_client.models.embed_content.return_value = emb_resp

    gen_resp = MagicMock()
    gen_resp.text = "Employees are entitled to 25 days of annual leave."
    mock_client.models.generate_content.return_value = gen_resp

    # 4. Index PDF
    index_res = index_pdf(
        pdf_path=saved_file,
        collection_name="app_integration_coll",
        persist_directory=persist_dir,
        client=mock_client,
    )
    assert index_res["indexed_count"] > 0

    # 5. Check indexed documents
    indexed_docs = get_indexed_documents(
        collection_name="app_integration_coll",
        persist_directory=persist_dir,
    )
    assert "handbook.pdf" in indexed_docs

    # 6. Ask Question with document filter
    response = answer_question(
        question="How many days of leave do employees get?",
        collection_name="app_integration_coll",
        persist_directory=persist_dir,
        client=mock_client,
        document_filter="handbook.pdf",
    )

    assert "25 days" in response.answer
    assert len(response.citations) > 0
    assert response.citations[0]["source"] == "handbook.pdf"
    assert response.citations[0]["page"] == 1


def test_get_theme_css_dark_mode():
    """Verify dark mode CSS contains dark theme selectors and background colors."""
    css = get_theme_css("Dark")
    assert ".stApp" in css
    assert "#0e1117" in css
    assert '[data-testid="stSidebar"]' in css
    assert '[data-testid="stChatMessage"]' in css


def test_get_theme_css_light_mode():
    """Verify light mode CSS contains light theme selectors and background colors."""
    css = get_theme_css("Light")
    assert ".stApp" in css
    assert "#f8fafc" in css
    assert '[data-testid="stSidebar"]' in css
    assert '[data-testid="stChatMessage"]' in css


def test_apply_theme_calls_st_markdown():
    """Verify apply_theme injects the CSS styles via st.markdown."""
    from unittest.mock import patch
    with patch("streamlit.markdown") as mock_markdown:
        apply_theme("Light")
        assert mock_markdown.called
        call_args = mock_markdown.call_args[0][0]
        assert "<style>" in call_args
        assert "#f8fafc" in call_args
        assert mock_markdown.call_args[1].get("unsafe_allow_html") is True

