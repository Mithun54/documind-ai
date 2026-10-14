"""Unit tests for Phase 4: Grounded RAG answer generation and citation synthesis."""

import os
import pathlib
from typing import Any, List, Dict
from unittest.mock import MagicMock, patch
import pytest

from ingest import DocumentChunk
from rag import (
    index_chunks,
    answer_question,
    generate_answer,
    get_api_key,
    GenerationError,
    INSUFFICIENT_EVIDENCE_PHRASE,
    DEFAULT_GENERATION_MODEL,
)


@pytest.fixture
def mock_genai_rag_client() -> MagicMock:
    """Mock Google GenAI client handling both embedding and generation offline.

    Requires no network access, no real API keys, and no paid service calls.
    """
    client = MagicMock()

    # Deterministic mock embeddings
    def fake_embed_content(model: str, contents: Any, **kwargs: Any) -> MagicMock:
        response = MagicMock()
        items = contents if isinstance(contents, list) else [contents]
        embeddings_list = []
        for item in items:
            text = str(item).lower()
            v1 = 1.0 if "python" in text else 0.1
            v2 = 1.0 if "architecture" in text else 0.1
            v3 = 1.0 if "finance" in text else 0.1
            emb = MagicMock()
            emb.values = [v1, v2, v3, 0.5]
            embeddings_list.append(emb)
        response.embeddings = embeddings_list
        return response

    client.models.embed_content.side_effect = fake_embed_content

    # Default mock generation response
    gen_response = MagicMock()
    gen_response.text = "Python is the core programming language used in DocuMind AI."
    client.models.generate_content.return_value = gen_response

    return client


@pytest.fixture
def populated_chroma(tmp_path: pathlib.Path, mock_genai_rag_client: MagicMock) -> Dict[str, str]:
    """Populate a temporary ChromaDB with sample multi-document chunks."""
    persist_dir = str(tmp_path / "rag_chroma_db")
    chunks = [
        DocumentChunk(
            text="Python is the core programming language used in DocuMind AI.",
            source="documind_architecture.pdf",
            page_numbers=[1, 2],
        ),
        DocumentChunk(
            text="ChromaDB provides local vector embeddings storage.",
            source="documind_architecture.pdf",
            page_numbers=[3],
        ),
        DocumentChunk(
            text="The fiscal year revenue grew by 15 percent.",
            source="annual_finance_report.pdf",
            page_numbers=[12],
        ),
    ]

    index_chunks(
        chunks=chunks,
        collection_name="rag_phase4_coll",
        persist_directory=persist_dir,
        client=mock_genai_rag_client,
    )

    return {
        "persist_dir": persist_dir,
        "collection": "rag_phase4_coll",
    }


def test_supported_answer_grounded_in_passages(
    populated_chroma: Dict[str, str], mock_genai_rag_client: MagicMock
) -> None:
    """Verify that a question with relevant passages produces a grounded answer with citations."""
    response = answer_question(
        question="What programming language does DocuMind AI use?",
        collection_name=populated_chroma["collection"],
        persist_directory=populated_chroma["persist_dir"],
        client=mock_genai_rag_client,
        top_k=2,
    )

    assert "Python" in response.answer
    assert len(response.citations) > 0
    assert response.context_chunks_count > 0
    assert response.model == DEFAULT_GENERATION_MODEL


def test_insufficient_evidence(
    populated_chroma: Dict[str, str], mock_genai_rag_client: MagicMock
) -> None:
    """Verify that insufficient evidence returns the required refusal and zero citations."""
    # Configure mock LLM to state insufficient information
    mock_resp = MagicMock()
    mock_resp.text = "The available documents do not contain enough information to answer this question."
    mock_genai_rag_client.models.generate_content.return_value = mock_resp

    response = answer_question(
        question="What is the weather on Mars?",
        collection_name=populated_chroma["collection"],
        persist_directory=populated_chroma["persist_dir"],
        client=mock_genai_rag_client,
        top_k=2,
    )

    assert "not contain enough information" in response.answer.lower()
    # Avoid fabricated citations: citations must be empty
    assert response.citations == []


def test_correct_source_and_page_citations(
    populated_chroma: Dict[str, str], mock_genai_rag_client: MagicMock
) -> None:
    """Verify citations contain original source filename and 1-based page numbers."""
    response = answer_question(
        question="Tell me about Python architecture",
        collection_name=populated_chroma["collection"],
        persist_directory=populated_chroma["persist_dir"],
        client=mock_genai_rag_client,
        top_k=1,
    )

    assert len(response.citations) >= 1
    citation = response.citations[0]
    assert citation["source"] == "documind_architecture.pdf"
    assert citation["page_numbers"] == [1, 2]
    assert citation["page"] == 1
    assert "snippet" in citation


def test_document_level_filtering_generation(
    populated_chroma: Dict[str, str], mock_genai_rag_client: MagicMock
) -> None:
    """Verify that document filtering limits context and citations to the requested document."""
    response = answer_question(
        question="Revenue growth statistics",
        collection_name=populated_chroma["collection"],
        persist_directory=populated_chroma["persist_dir"],
        client=mock_genai_rag_client,
        document_filter="annual_finance_report.pdf",
        top_k=3,
    )

    # All citations must strictly originate from the filtered document
    for citation in response.citations:
        assert citation["source"] == "annual_finance_report.pdf"
        assert citation["source"] != "documind_architecture.pdf"


def test_empty_questions_error(populated_chroma: Dict[str, str], mock_genai_rag_client: MagicMock) -> None:
    """Verify that whitespace or empty questions raise a descriptive ValueError."""
    with pytest.raises(ValueError, match="Question cannot be empty"):
        answer_question(
            question="   ",
            collection_name=populated_chroma["collection"],
            persist_directory=populated_chroma["persist_dir"],
            client=mock_genai_rag_client,
        )


def test_missing_api_key_error(monkeypatch: pytest.MonkeyPatch, populated_chroma: Dict[str, str]) -> None:
    """Verify that missing API key raises ValueError when attempting to generate answers."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Gemini API key is not configured"):
        answer_question(
            question="Valid question",
            collection_name=populated_chroma["collection"],
            persist_directory=populated_chroma["persist_dir"],
            client=None,
        )


def test_gemini_api_failure(populated_chroma: Dict[str, str], mock_genai_rag_client: MagicMock) -> None:
    """Verify upstream Gemini API failures are raised as GenerationError."""
    mock_genai_rag_client.models.generate_content.side_effect = RuntimeError("503 Service Unavailable")

    with pytest.raises((GenerationError, RuntimeError), match="Gemini generation API error"):
        answer_question(
            question="Valid query",
            collection_name=populated_chroma["collection"],
            persist_directory=populated_chroma["persist_dir"],
            client=mock_genai_rag_client,
        )


def test_no_indexed_documents_handling(tmp_path: pathlib.Path, mock_genai_rag_client: MagicMock) -> None:
    """Verify querying an empty collection returns insufficient evidence without error."""
    empty_persist_dir = str(tmp_path / "empty_rag_db")
    response = answer_question(
        question="What is DocuMind?",
        collection_name="empty_coll",
        persist_directory=empty_persist_dir,
        client=mock_genai_rag_client,
    )

    assert "not contain enough information" in response.answer.lower()
    assert response.citations == []
    assert response.context_chunks_count == 0
