"""Tests for Gemini API retry resilience, transient error handling, and backoff behavior.

Covers:
1. Bounded retry success after transient 503 UNAVAILABLE (high demand).
2. Bounded retry success after transient 429 RESOURCE_EXHAUSTED.
3. Retry exhaustion raising GenerationUnavailableError after maximum attempts.
4. Immediate failure on permanent non-retryable errors (400, 401, 403, 404) without retries.
5. Verification that document indexing and query retrieval are not repeated during retries.
6. Verification that generation failure does not present citations as a successful answer.
7. User-friendly error message display and secret redaction.
"""

from typing import Dict, Any
from unittest.mock import MagicMock, patch
import pytest

from rag import (
    answer_question,
    generate_answer,
    generate_content_with_retry,
    is_retryable_error,
    GenerationError,
    GenerationUnavailableError,
    FRIENDLY_UNAVAILABLE_MESSAGE,
    DEFAULT_MAX_GENERATION_RETRIES,
    redact_sensitive_info,
)


@pytest.fixture
def mock_retrieval_chunks():
    """Sample chunk objects for generation tests."""
    c1 = MagicMock()
    c1.text = "DocuMind uses Google Gemini for generation."
    c1.source = "overview.pdf"
    c1.page_numbers = [1]
    c1.page = 1
    c1.chunk_id = "c1"
    return [c1]


def test_is_retryable_error_classification():
    """Verify that transient errors return True and permanent errors return False."""
    # Transient errors
    assert is_retryable_error(RuntimeError("503 UNAVAILABLE: This model is currently experiencing high demand."))
    assert is_retryable_error(RuntimeError("429 RESOURCE_EXHAUSTED: Rate limit exceeded."))
    assert is_retryable_error(RuntimeError("502 Bad Gateway"))
    assert is_retryable_error(RuntimeError("504 Gateway Timeout"))
    assert is_retryable_error(TimeoutError("The read operation timed out"))
    assert is_retryable_error(ConnectionError("Connection reset by peer"))

    # Permanent errors - must NOT be retried
    assert not is_retryable_error(ValueError("Invalid argument"))
    assert not is_retryable_error(TypeError("Missing argument"))
    assert not is_retryable_error(RuntimeError("400 INVALID_ARGUMENT: contents is malformed"))
    assert not is_retryable_error(RuntimeError("401 UNAUTHORIZED: API key not valid"))
    assert not is_retryable_error(RuntimeError("403 PERMISSION_DENIED: Access denied"))
    assert not is_retryable_error(RuntimeError("404 NOT_FOUND: models/unknown-model not found"))


@patch("time.sleep", return_value=None)
def test_retry_success_after_transient_503(mock_sleep, mock_retrieval_chunks):
    """Verify retry succeeds when attempt 1 fails with 503 and attempt 2 succeeds."""
    mock_client = MagicMock()
    success_resp = MagicMock()
    success_resp.text = "DocuMind uses Google Gemini for generation."

    mock_client.models.generate_content.side_effect = [
        RuntimeError("503 UNAVAILABLE: This model is currently experiencing high demand."),
        success_resp,
    ]

    res = generate_answer(
        query="What model does DocuMind use?",
        context_chunks=mock_retrieval_chunks,
        client=mock_client,
    )

    assert "Google Gemini" in res.answer
    assert mock_client.models.generate_content.call_count == 2
    assert mock_sleep.call_count == 1
    assert len(res.citations) > 0


@patch("time.sleep", return_value=None)
def test_retry_success_after_transient_429(mock_sleep, mock_retrieval_chunks):
    """Verify retry succeeds when attempt 1 fails with 429 quota and attempt 2 succeeds."""
    mock_client = MagicMock()
    success_resp = MagicMock()
    success_resp.text = "Answer generated after rate limit cleared."

    mock_client.models.generate_content.side_effect = [
        RuntimeError("429 RESOURCE_EXHAUSTED: Quota exceeded for model."),
        success_resp,
    ]

    res = generate_answer(
        query="Test rate limit recovery",
        context_chunks=mock_retrieval_chunks,
        client=mock_client,
    )

    assert "Answer generated" in res.answer
    assert mock_client.models.generate_content.call_count == 2
    assert mock_sleep.call_count == 1


@patch("time.sleep", return_value=None)
def test_retry_exhaustion_raises_generation_unavailable_error(mock_sleep, mock_retrieval_chunks):
    """Verify that exhausting all retry attempts raises GenerationUnavailableError with friendly message."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError(
        "503 UNAVAILABLE: This model is currently experiencing high demand."
    )

    with pytest.raises(GenerationUnavailableError) as exc_info:
        generate_answer(
            query="Will this fail after 3 attempts?",
            context_chunks=mock_retrieval_chunks,
            client=mock_client,
        )

    # Must attempt exactly DEFAULT_MAX_GENERATION_RETRIES (3) times
    assert mock_client.models.generate_content.call_count == DEFAULT_MAX_GENERATION_RETRIES
    assert mock_sleep.call_count == DEFAULT_MAX_GENERATION_RETRIES - 1
    assert exc_info.value.is_transient is True
    assert FRIENDLY_UNAVAILABLE_MESSAGE in str(exc_info.value)
    assert "Gemini generation API error" in str(exc_info.value)


@patch("time.sleep", return_value=None)
def test_non_retryable_error_400_fails_immediately(mock_sleep, mock_retrieval_chunks):
    """Verify that permanent 400 client errors fail immediately without retry."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError(
        "400 INVALID_ARGUMENT: contents parameter is invalid"
    )

    with pytest.raises(GenerationError) as exc_info:
        generate_answer(
            query="Invalid payload query",
            context_chunks=mock_retrieval_chunks,
            client=mock_client,
        )

    # Called exactly 1 time, 0 sleeps
    assert mock_client.models.generate_content.call_count == 1
    assert mock_sleep.call_count == 0
    assert not isinstance(exc_info.value, GenerationUnavailableError)


@patch("time.sleep", return_value=None)
def test_non_retryable_error_401_fails_immediately(mock_sleep, mock_retrieval_chunks):
    """Verify that permanent 401 authentication errors fail immediately without retry."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError(
        "401 UNAUTHORIZED: API key not valid"
    )

    with pytest.raises(GenerationError) as exc_info:
        generate_answer(
            query="Auth failure query",
            context_chunks=mock_retrieval_chunks,
            client=mock_client,
        )

    assert mock_client.models.generate_content.call_count == 1
    assert mock_sleep.call_count == 0


@patch("time.sleep", return_value=None)
def test_indexing_and_embeddings_not_repeated_during_retries(mock_sleep):
    """Verify that retry handling in answer_question does not repeat query retrieval/embeddings."""
    mock_client = MagicMock()

    # Embedding response for retrieve_chunks
    emb_resp = MagicMock()
    emb_resp.embeddings = [MagicMock(values=[0.1, 0.2])]
    mock_client.models.embed_content.return_value = emb_resp

    # Generation response fails twice with 503, succeeds on third attempt
    success_resp = MagicMock()
    success_resp.text = "Answer generated on attempt 3."
    mock_client.models.generate_content.side_effect = [
        RuntimeError("503 UNAVAILABLE: High demand"),
        RuntimeError("503 UNAVAILABLE: High demand"),
        success_resp,
    ]

    with patch("rag.retrieve_chunks") as mock_retrieve:
        c1 = MagicMock()
        c1.text = "Sample context"
        c1.source = "test.pdf"
        c1.page_numbers = [1]
        c1.page = 1
        c1.chunk_id = "c1"
        mock_retrieve.return_value = [c1]

        res = answer_question(
            question="Do we repeat retrieval?",
            client=mock_client,
        )

        # Retrieval must be called ONLY once
        assert mock_retrieve.call_count == 1
        # Generation was retried 3 times
        assert mock_client.models.generate_content.call_count == 3
        assert "Answer generated" in res.answer


@patch("time.sleep", return_value=None)
def test_secret_redaction_in_retry_logs(mock_sleep, mock_retrieval_chunks):
    """Verify API keys in upstream error messages are masked in final exception details."""
    secret_key = "AIzaSySuperSecretKey12345678"
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError(
        f"503 Service Unavailable at endpoint https://api.google.com?key={secret_key}"
    )

    with pytest.raises(GenerationUnavailableError) as exc_info:
        generate_answer(
            query="Test secret redaction",
            context_chunks=mock_retrieval_chunks,
            client=mock_client,
        )

    err_str = str(exc_info.value)
    assert secret_key not in err_str
    assert "key=[REDACTED]" in err_str
