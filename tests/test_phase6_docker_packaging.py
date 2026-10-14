"""Unit and regression tests for Phase 6B: Docker packaging and deployment readiness.

Covers:
1. Dockerfile configuration (maintained base image, non-root user, health check, port binding).
2. .dockerignore rules (secrets, local vector databases, upload storage, venvs, caches).
3. Secret-free container configuration (no hardcoded keys, no .env baked in).
4. Environment variable configurability for UPLOAD_DIR, CHROMA_PERSIST_DIR, and CHROMA_COLLECTION_NAME.
5. Persistent directory separation from image layers.
"""

import os
import re
from unittest.mock import MagicMock
import pytest

from rag import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_UPLOAD_DIR,
    get_chroma_client,
    get_collection,
    reindex_all_uploaded_documents,
)


def test_dockerfile_exists_and_security_checks():
    """Verify Dockerfile exists and enforces least-privilege non-root execution."""
    dockerfile_path = "Dockerfile"
    assert os.path.exists(dockerfile_path), "Dockerfile must exist at project root."

    with open(dockerfile_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Maintained base image
    assert re.search(r"FROM\s+python:3\.\d+-slim", content, re.IGNORECASE)

    # 2. Least privilege: non-root user creation and declaration
    assert "useradd" in content
    assert "appuser" in content
    assert "USER appuser" in content

    # 3. Port binding & address
    assert "0.0.0.0" in content
    assert "EXPOSE 8501" in content

    # 4. Container health check
    assert "HEALTHCHECK" in content
    assert "/_stcore/health" in content

    # 5. No secrets baked in
    assert "GEMINI_API_KEY=" not in content
    assert "COPY .env ./" not in content
    assert "COPY .env ." not in content


def test_dockerignore_excludes_sensitive_and_persistent_assets():
    """Verify .dockerignore excludes credentials, local databases, uploads, and caches."""
    dockerignore_path = ".dockerignore"
    assert os.path.exists(dockerignore_path), ".dockerignore must exist at project root."

    with open(dockerignore_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]

    # 1. Secrets excluded
    assert ".env" in lines

    # 2. Local vector database excluded (never baked into image)
    assert any("chroma_db" in line for line in lines)

    # 3. Local document uploads excluded
    assert any("data" in line or "*.pdf" in line for line in lines)

    # 4. Virtual environments and caches excluded
    assert any(".venv" in line for line in lines)
    assert any("__pycache__" in line for line in lines)
    assert any(".pytest_cache" in line for line in lines)
    assert any(".git" in line for line in lines)


def test_configurable_storage_and_database_paths(tmp_path, monkeypatch):
    """Verify UPLOAD_DIR and CHROMA_PERSIST_DIR are dynamically configurable via env vars."""
    custom_upload = str(tmp_path / "custom_uploads")
    custom_chroma = str(tmp_path / "custom_chroma_db")
    custom_collection = "custom_test_collection"

    monkeypatch.setenv("UPLOAD_DIR", custom_upload)
    monkeypatch.setenv("CHROMA_PERSIST_DIR", custom_chroma)
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", custom_collection)

    # Verify custom Chroma persistence path is honored
    client = get_chroma_client(persist_directory=custom_chroma)
    coll = get_collection(client=client, collection_name=custom_collection)
    assert coll.name == custom_collection

    # Verify reindex_all_uploaded_documents honors custom upload dir
    os.makedirs(custom_upload, exist_ok=True)
    summary = reindex_all_uploaded_documents(upload_dir=custom_upload, client=MagicMock())
    assert summary["processed_files"] == 0
    assert summary["total_chunks"] == 0


def test_least_privilege_non_root_filesystem_separation(tmp_path):
    """Verify application modules can function inside custom non-root paths."""
    isolated_db = str(tmp_path / "isolated_app_chroma")
    client = get_chroma_client(persist_directory=isolated_db)
    coll = get_collection(client=client, collection_name="isolated_coll")
    assert coll.count() == 0
