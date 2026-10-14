"""Phase 1 sanity tests for DocuMind AI."""

import os
import pytest


def test_project_structure_exists() -> None:
    """Verify essential project files exist in the repository root."""
    required_files = [
        "app.py",
        "rag.py",
        "ingest.py",
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "README.md",
    ]
    for filename in required_files:
        assert os.path.exists(filename), f"Missing expected file: {filename}"


def test_env_example_has_no_secret() -> None:
    """Ensure .env.example contains a placeholder and no actual secret keys."""
    with open(".env.example", "r", encoding="utf-8") as f:
        content = f.read()

    assert "GEMINI_API_KEY" in content
    assert "your_gemini_api_key_here" in content
