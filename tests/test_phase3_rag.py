"""Unit tests for Phase 3: Embeddings, ChromaDB storage, and semantic retrieval."""

import os
import pathlib
from typing import Any, List, Dict
from unittest.mock import MagicMock, patch
import pytest

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from ingest import DocumentChunk, PDFProcessingError
from rag import (
    index_chunks,
    index_pdf,
    retrieve_chunks,
    retrieve_context,
    get_chroma_client,
    get_collection,
    get_api_key,
    get_embedding,
    EmbeddingError,
    DEFAULT_EMBEDDING_MODEL,
)


@pytest.fixture
def mock_genai_client() -> MagicMock:
    """Provide a mock Google GenAI client returning deterministic embeddings.

    Ensures tests require no network calls, no real API keys, and no paid services.
    """
    client = MagicMock()

    def fake_embed_content(model: str, contents: Any, **kwargs: Any) -> MagicMock:
        response = MagicMock()
        items = contents if isinstance(contents, list) else [contents]
        embeddings_list = []

        for item in items:
            text = str(item)
            # Generate deterministic 4-dimensional vector based on text traits
            v1 = float(len(text) % 10) / 10.0
            v2 = 1.0 if "python" in text.lower() else 0.1
            v3 = 1.0 if "finance" in text.lower() else 0.1
            v4 = 0.5
            emb = MagicMock()
            emb.values = [v1, v2, v3, v4]
            embeddings_list.append(emb)

        response.embeddings = embeddings_list
        return response

    client.models.embed_content.side_effect = fake_embed_content
    return client


@pytest.fixture
def sample_pdf_files(tmp_path: pathlib.Path) -> dict:
    """Create two sample PDF files for multi-document testing."""
    pdf_a = tmp_path / "doc_alpha.pdf"
    doc_a = fitz.open()
    p1 = doc_a.new_page()
    p1.insert_text((50, 50), "Python is a popular programming language for AI.")
    p2 = doc_a.new_page()
    p2.insert_text((50, 50), "Python libraries include NumPy and Streamlit.")
    doc_a.save(str(pdf_a))
    doc_a.close()

    pdf_b = tmp_path / "doc_beta.pdf"
    doc_b = fitz.open()
    p_b = doc_b.new_page()
    p_b.insert_text((50, 50), "Finance and quarterly earnings report analysis.")
    doc_b.save(str(pdf_b))
    doc_b.close()

    return {
        "alpha": str(pdf_a),
        "beta": str(pdf_b),
        "dir": str(tmp_path / "chroma_test_db"),
    }


def test_document_metadata(tmp_path: pathlib.Path, mock_genai_client: MagicMock) -> None:
    """Verify that chunks retain source, page_numbers, primary page, and stable chunk_id."""
    persist_dir = str(tmp_path / "meta_chroma")
    chunks = [
        DocumentChunk(
            text="Introductory text spanning multiple pages.",
            source="manual.pdf",
            page_numbers=[1, 2],
        )
    ]

    index_chunks(
        chunks=chunks,
        collection_name="test_meta_coll",
        persist_directory=persist_dir,
        client=mock_genai_client,
    )

    results = retrieve_chunks(
        query="Introductory text",
        collection_name="test_meta_coll",
        persist_directory=persist_dir,
        client=mock_genai_client,
        top_k=1,
    )

    assert len(results) == 1
    chunk = results[0]
    assert chunk.source == "manual.pdf"
    assert chunk.page == 1
    assert chunk.page_numbers == [1, 2]
    assert "manual.pdf_chunk_0" in chunk.chunk_id
    assert "Introductory text" in chunk.text


def test_persistent_storage(tmp_path: pathlib.Path, mock_genai_client: MagicMock) -> None:
    """Verify data persists to the specified local directory and survives client reload."""
    persist_dir = str(tmp_path / "persist_chroma")
    chunks = [
        DocumentChunk(text="Persistent vector data point", source="archive.pdf", page_numbers=[3])
    ]

    # Index using first client
    index_chunks(
        chunks=chunks,
        collection_name="persist_coll",
        persist_directory=persist_dir,
        client=mock_genai_client,
    )

    # Re-open ChromaDB using a completely separate PersistentClient instance
    fresh_chroma = get_chroma_client(persist_directory=persist_dir)
    collection = fresh_chroma.get_collection("persist_coll")
    assert collection.count() == 1

    # Retrieve from re-opened database
    results = retrieve_chunks(
        query="vector data point",
        collection_name="persist_coll",
        persist_directory=persist_dir,
        client=mock_genai_client,
        top_k=1,
    )
    assert len(results) == 1
    assert results[0].text == "Persistent vector data point"


def test_top_k_retrieval_behavior(tmp_path: pathlib.Path, mock_genai_client: MagicMock) -> None:
    """Verify that retrieval limits output to exactly top_k results."""
    persist_dir = str(tmp_path / "topk_chroma")
    chunks = [
        DocumentChunk(text=f"Paragraph content number {i}", source="book.pdf", page_numbers=[i + 1])
        for i in range(6)
    ]

    index_chunks(
        chunks=chunks,
        collection_name="topk_coll",
        persist_directory=persist_dir,
        client=mock_genai_client,
    )

    # Request top_k = 2
    results = retrieve_chunks(
        query="Paragraph content",
        collection_name="topk_coll",
        persist_directory=persist_dir,
        client=mock_genai_client,
        top_k=2,
    )
    assert len(results) == 2

    # Request context strings
    context_strings = retrieve_context(
        query="Paragraph content",
        collection_name="topk_coll",
        persist_directory=persist_dir,
        client=mock_genai_client,
        top_k=3,
    )
    assert len(context_strings) == 3
    assert all(isinstance(c, str) for c in context_strings)


def test_document_level_filtering(tmp_path: pathlib.Path, mock_genai_client: MagicMock) -> None:
    """Verify retrieval can be restricted strictly to a selected document."""
    persist_dir = str(tmp_path / "filter_chroma")
    chunks = [
        DocumentChunk(text="Python deep learning guide", source="python_ai.pdf", page_numbers=[1]),
        DocumentChunk(text="Corporate earnings overview", source="finance_q1.pdf", page_numbers=[1]),
    ]

    index_chunks(
        chunks=chunks,
        collection_name="filter_coll",
        persist_directory=persist_dir,
        client=mock_genai_client,
    )

    # Retrieve with filter for finance_q1.pdf
    filtered_results = retrieve_chunks(
        query="learning and earnings",
        collection_name="filter_coll",
        persist_directory=persist_dir,
        client=mock_genai_client,
        document_filter="finance_q1.pdf",
        top_k=5,
    )

    assert len(filtered_results) == 1
    assert filtered_results[0].source == "finance_q1.pdf"
    assert "Corporate earnings" in filtered_results[0].text


def test_repeated_indexing_idempotent(tmp_path: pathlib.Path, mock_genai_client: MagicMock) -> None:
    """Verify that indexing the same document multiple times does not produce duplicate records."""
    persist_dir = str(tmp_path / "idempotent_chroma")

    chunks_v1 = [
        DocumentChunk(text="First version chunk 1", source="report.pdf", page_numbers=[1]),
        DocumentChunk(text="First version chunk 2", source="report.pdf", page_numbers=[2]),
    ]

    # First indexing
    index_chunks(
        chunks=chunks_v1,
        collection_name="idempotent_coll",
        persist_directory=persist_dir,
        client=mock_genai_client,
    )

    client = get_chroma_client(persist_directory=persist_dir)
    collection = client.get_collection("idempotent_coll")
    assert collection.count() == 2

    # Re-indexing the same document with updated text
    chunks_v2 = [
        DocumentChunk(text="Updated version chunk 1", source="report.pdf", page_numbers=[1]),
        DocumentChunk(text="Updated version chunk 2", source="report.pdf", page_numbers=[2]),
    ]

    index_chunks(
        chunks=chunks_v2,
        collection_name="idempotent_coll",
        persist_directory=persist_dir,
        client=mock_genai_client,
    )

    # Document count must remain 2 (no duplicate accumulation)
    assert collection.count() == 2

    # Verify updated content is present
    retrieved = retrieve_chunks(
        query="Updated version",
        collection_name="idempotent_coll",
        persist_directory=persist_dir,
        client=mock_genai_client,
        top_k=2,
    )
    assert any("Updated version" in c.text for c in retrieved)


def test_empty_input_handling(tmp_path: pathlib.Path, mock_genai_client: MagicMock) -> None:
    """Verify that empty questions and empty PDFs raise clear errors."""
    persist_dir = str(tmp_path / "empty_chroma")

    # Empty question
    with pytest.raises(ValueError, match="Query cannot be empty"):
        retrieve_chunks(
            query="   ",
            collection_name="empty_coll",
            persist_directory=persist_dir,
            client=mock_genai_client,
        )

    # Empty PDF handling via index_pdf
    blank_pdf = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(blank_pdf))
    doc.close()

    with pytest.raises((PDFProcessingError, ValueError)):
        index_pdf(
            pdf_path=str(blank_pdf),
            collection_name="empty_coll",
            persist_directory=persist_dir,
            client=mock_genai_client,
        )


def test_missing_api_key_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that a missing or placeholder API key raises a clear, descriptive ValueError."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Gemini API key is not configured"):
        get_api_key()

    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    with pytest.raises(ValueError, match="Gemini API key is not configured"):
        get_api_key()


def test_embedding_api_error_handling() -> None:
    """Verify that upstream embedding API exceptions are wrapped cleanly."""
    failing_client = MagicMock()
    failing_client.models.embed_content.side_effect = RuntimeError("Service unavailable 503")

    with pytest.raises(EmbeddingError, match="Embedding API error"):
        get_embedding("Test prompt", client=failing_client)


def test_embedding_model_configured_to_gemini_embedding_001() -> None:
    """Verify that DEFAULT_EMBEDDING_MODEL is configured to gemini-embedding-001."""
    assert DEFAULT_EMBEDDING_MODEL == "gemini-embedding-001"


def test_consistent_model_usage_indexing_and_retrieval(tmp_path: pathlib.Path) -> None:
    """Verify that index_chunks and retrieve_chunks use the exact same embedding model."""
    recorded_models = []
    mock_client = MagicMock()

    def capture_embed(model: str, contents: Any, **kwargs: Any) -> MagicMock:
        recorded_models.append(model)
        resp = MagicMock()
        items = contents if isinstance(contents, list) else [contents]
        resp.embeddings = [MagicMock(values=[0.1, 0.2, 0.3]) for _ in items]
        return resp

    mock_client.models.embed_content.side_effect = capture_embed

    persist_dir = str(tmp_path / "model_consistency_chroma")
    chunks = [DocumentChunk(text="Model test text", source="test.pdf", page_numbers=[1])]

    # Index
    index_chunks(
        chunks=chunks,
        collection_name="consistency_coll",
        persist_directory=persist_dir,
        client=mock_client,
    )

    # Retrieve
    retrieve_chunks(
        query="Model test query",
        collection_name="consistency_coll",
        persist_directory=persist_dir,
        client=mock_client,
    )

    # Verify both calls used gemini-embedding-001
    assert len(recorded_models) >= 2
    assert all(m == "gemini-embedding-001" for m in recorded_models)


def test_collection_versioning_preserves_legacy_collection(
    tmp_path: pathlib.Path, mock_genai_client: MagicMock
) -> None:
    """Verify that using versioned collection preserves legacy collection records."""
    from rag import reindex_all_uploaded_documents, get_chroma_client

    persist_dir = str(tmp_path / "versioned_chroma")
    upload_dir = str(tmp_path / "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    # 1. Create a legacy collection with a record
    chroma = get_chroma_client(persist_directory=persist_dir)
    legacy_coll = chroma.get_or_create_collection("documind_docs")
    legacy_coll.add(
        ids=["legacy_1"],
        documents=["Legacy document text"],
        embeddings=[[0.1, 0.2, 0.3, 0.4]],
        metadatas=[{"source": "legacy.pdf", "page": 1}],
    )
    assert legacy_coll.count() == 1

    # 2. Put a new PDF in upload_dir
    pdf_path = os.path.join(upload_dir, "migrated_doc.pdf")
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((50, 50), "Migrated document text for version 2 collection.")
    doc.save(pdf_path)
    doc.close()

    # 3. Re-index into new versioned collection
    reindex_res = reindex_all_uploaded_documents(
        upload_dir=upload_dir,
        collection_name="documind_docs_v2",
        persist_directory=persist_dir,
        client=mock_genai_client,
    )
    assert reindex_res["processed_files"] == 1
    assert reindex_res["total_chunks"] > 0

    # 4. Verify legacy collection was NOT touched or deleted
    assert legacy_coll.count() == 1
    legacy_record = legacy_coll.get()
    assert legacy_record["ids"] == ["legacy_1"]

    # 5. Verify new collection holds the new records
    new_coll = chroma.get_collection("documind_docs_v2")
    assert new_coll.count() > 0
