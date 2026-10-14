"""Streamlit Web Application for DocuMind AI — Intelligent Document Assistant.

Provides a professional, portfolio-grade user interface for:
1. Validating and securely saving PDF document uploads with collision resistance.
2. Indexing documents with text extraction, chunking, and ChromaDB embeddings.
3. Filtering queries by specific document or searching across all indexed files.
4. Grounded question answering using Google Gemini with exact page citations.
5. Interactive chat session state and history management.
"""

import os
import re
import logging
from typing import Dict, Any, List, Optional
import streamlit as st
from dotenv import load_dotenv

from ingest import PDFProcessingError
from rag import (
    index_pdf,
    answer_question,
    get_indexed_documents,
    reindex_all_uploaded_documents,
    get_api_key,
    redact_sensitive_info,
    EmbeddingError,
    GenerationError,
    GenerationUnavailableError,
    FRIENDLY_UNAVAILABLE_MESSAGE,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_COLLECTION_NAME,
)

logger = logging.getLogger("documind.app")

# Load local environment configuration
load_dotenv()

# Upload storage directory
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join("data", "uploads"))
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "15"))


RESERVED_DEVICE_NAMES = {
    "con", "prn", "aux", "nul",
    "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9",
    "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
}


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal or invalid characters.

    Args:
        filename: Raw uploaded filename.

    Returns:
        Safe, sanitized filename string.
    """
    clean_name = os.path.basename(filename).strip()
    # Replace whitespace and unsafe characters with underscores
    clean_name = re.sub(r"[^\w\.\-_]", "_", clean_name)
    base, ext = os.path.splitext(clean_name)

    # Prevent Windows reserved device names
    if base.lower() in RESERVED_DEVICE_NAMES:
        clean_name = f"doc_{clean_name}"

    return clean_name or "uploaded_document.pdf"


def validate_and_save_upload(
    uploaded_file: Any,
    upload_dir: str = UPLOAD_DIR,
    max_size_mb: int = MAX_UPLOAD_SIZE_MB,
) -> str:
    """Validate and securely write an uploaded PDF file to disk with collision avoidance.

    Args:
        uploaded_file: Streamlit UploadedFile or file-like object with .name and .read().
        upload_dir: Destination folder path for uploaded documents.
        max_size_mb: Maximum allowed file size in megabytes.

    Returns:
        Absolute or relative path to the saved PDF file.

    Raises:
        ValueError: If file is missing, not a PDF, or exceeds size limits.
    """
    if uploaded_file is None:
        raise ValueError("No file was uploaded.")

    filename = getattr(uploaded_file, "name", "document.pdf")
    if not filename.lower().endswith(".pdf"):
        raise ValueError(f"Invalid file type for '{filename}'. Only PDF documents are supported.")

    # Read bytes and validate size
    file_bytes = uploaded_file.read() if hasattr(uploaded_file, "read") else bytes(uploaded_file)
    file_size_mb = len(file_bytes) / (1024 * 1024)

    if file_size_mb > max_size_mb:
        raise ValueError(
            f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed limit of {max_size_mb} MB."
        )

    if len(file_bytes) == 0:
        raise ValueError(f"Uploaded file '{filename}' is empty (0 bytes).")

    os.makedirs(upload_dir, exist_ok=True)
    safe_name = sanitize_filename(filename)
    base, ext = os.path.splitext(safe_name)

    target_path = os.path.join(upload_dir, safe_name)

    # Bulletproof path traversal defense: target must remain strictly inside upload_dir
    target_abs = os.path.abspath(target_path)
    upload_dir_abs = os.path.abspath(upload_dir)
    if os.path.commonpath([target_abs, upload_dir_abs]) != upload_dir_abs:
        raise ValueError(f"Security error: Unsafe file path detected for '{filename}'.")

    counter = 1

    # Collision avoidance: if a file exists with identical content, reuse it;
    # if different content, create a distinct non-colliding filename.
    while os.path.exists(target_path):
        try:
            with open(target_path, "rb") as existing_file:
                if existing_file.read() == file_bytes:
                    return target_path
        except Exception:
            pass

        safe_name = f"{base}_{counter}{ext}"
        target_path = os.path.join(upload_dir, safe_name)
        counter += 1

    with open(target_path, "wb") as f:
        f.write(file_bytes)

    return target_path


def format_citation_display(citation: Dict[str, Any]) -> str:
    """Format a single citation dictionary for clean user interface display."""
    source = citation.get("source", "Unknown Document")
    pages = citation.get("page_numbers", [citation.get("page", 1)])
    pages_str = ", ".join(map(str, pages))
    return f"📄 **{source}** — Page(s): **{pages_str}**"


def get_active_filter(selection: str) -> Optional[str]:
    """Convert UI dropdown selection into a backend document filter string."""
    if not selection or selection.startswith("All Documents"):
        return None
    return selection.strip()


def check_api_key_configured() -> bool:
    """Check if GEMINI_API_KEY is configured in the environment."""
    try:
        get_api_key()
        return True
    except ValueError:
        return False


def main() -> None:
    """Primary entry point for the Streamlit web application."""
    st.set_page_config(
        page_title="DocuMind AI — Intelligent Document Assistant",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize chat session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Sidebar: Document Ingestion and Selection
    with st.sidebar:
        st.title("🧠 DocuMind AI")
        st.caption("Intelligent Document Assistant")

        st.markdown("---")
        st.subheader("📄 Document Ingestion")

        uploaded_pdf = st.file_uploader(
            "Upload a PDF document",
            type=["pdf"],
            help=f"Maximum allowed file size: {MAX_UPLOAD_SIZE_MB} MB",
        )

        if uploaded_pdf is not None:
            size_kb = len(uploaded_pdf.getvalue()) / 1024
            size_display = (
                f"{size_kb / 1024:.2f} MB" if size_kb > 1024 else f"{size_kb:.1f} KB"
            )
            st.info(f"**Selected File:** `{uploaded_pdf.name}` ({size_display})")

            if st.button("📥 Index Document", use_container_width=True, type="primary"):
                try:
                    with st.spinner("Saving and analyzing PDF..."):
                        saved_path = validate_and_save_upload(uploaded_pdf)

                    with st.spinner("Extracting text and indexing into ChromaDB..."):
                        res = index_pdf(saved_path)

                    st.success(
                        f"Indexed **{os.path.basename(saved_path)}** successfully!\n\n"
                        f"Stored **{res['indexed_count']}** text chunks in knowledge base."
                    )
                except PDFProcessingError as exc:
                    st.error(f"Cannot process PDF: {exc}")
                except ValueError as exc:
                    st.error(f"Validation error: {exc}")
                except (EmbeddingError, GenerationError) as exc:
                    sanitized_err = redact_sensitive_info(str(exc))
                    if (
                        isinstance(exc, GenerationUnavailableError)
                        or getattr(exc, "is_transient", False)
                        or "temporarily unavailable" in sanitized_err.lower()
                        or "503" in sanitized_err
                        or "high demand" in sanitized_err.lower()
                    ):
                        logger.warning("Transient AI service error during indexing: %s", sanitized_err)
                        st.warning(f"⚠️ {FRIENDLY_UNAVAILABLE_MESSAGE}")
                    else:
                        logger.error("AI service error during indexing: %s", sanitized_err)
                        st.error(f"AI Service error: {sanitized_err}")
                except Exception as exc:
                    st.error(f"Unexpected indexing error: {exc}")

        st.markdown("---")
        st.subheader("🎯 Search Scope & Filter")

        indexed_docs = get_indexed_documents()
        filter_options = ["All Documents (Entire Knowledge Base)"] + indexed_docs

        selected_option = st.selectbox(
            "Select target document to query:",
            options=filter_options,
            index=0,
            help="Choose a specific document or search across your entire collection.",
        )
        active_filter = get_active_filter(selected_option)

        # Stored PDFs & Re-indexing Migration Helper
        stored_files = [
            f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")
        ] if os.path.exists(UPLOAD_DIR) else []

        if stored_files:
            st.markdown("---")
            with st.expander(f"📁 Stored Uploads ({len(stored_files)})", expanded=False):
                st.caption("Saved documents on server:")
                for sf in stored_files:
                    is_indexed = sf in indexed_docs
                    badge = "✅ Indexed" if is_indexed else "⏳ Needs Indexing"
                    st.text(f"• {sf} [{badge}]")

                if st.button("🔄 Re-index All Stored PDFs", use_container_width=True):
                    with st.spinner("Re-indexing documents into ChromaDB..."):
                        reindex_res = reindex_all_uploaded_documents()
                        if reindex_res["errors"]:
                            st.warning(f"Re-indexed with {len(reindex_res['errors'])} warning(s).")
                        else:
                            st.success(
                                f"Successfully indexed {reindex_res['processed_files']} file(s) "
                                f"({reindex_res['total_chunks']} chunks) into {DEFAULT_COLLECTION_NAME}!"
                            )
                        st.rerun()

        st.markdown("---")
        # Clear chat action
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

        # System Status
        st.markdown("---")
        st.caption("**System Status**")
        has_key = check_api_key_configured()
        if has_key:
            st.caption("🟢 Gemini API: **Connected**")
        else:
            st.warning("⚠️ `GEMINI_API_KEY` missing in `.env`. AI synthesis will be inactive.")

        st.caption(f"🧠 Generation: `{DEFAULT_GENERATION_MODEL}`")
        st.caption(f"🔍 Embeddings: `{DEFAULT_EMBEDDING_MODEL}`")
        st.caption(f"📦 Collection: `{DEFAULT_COLLECTION_NAME}`")
        st.caption(f"📚 Documents in DB: **{len(indexed_docs)}**")

    # Main Area: Chat Interface
    st.title("🧠 DocuMind AI")
    st.caption("Grounded Document Question Answering powered by Google Gemini & ChromaDB")

    # Display active filter scope banner
    if active_filter:
        st.info(
            f"🎯 **Document Filter Active:** Answers will be synthesized strictly from **`{active_filter}`**."
        )
    else:
        st.caption("🌐 **Scope:** Searching across all indexed documents in the knowledge base.")

    # Render conversation history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            citations = msg.get("citations", [])
            if citations:
                with st.expander(f"📚 Verified Sources & Citations ({len(citations)})", expanded=False):
                    for cit in citations:
                        st.markdown(format_citation_display(cit))
                        if cit.get("snippet"):
                            st.caption(f'"{cit["snippet"]}"')

    # Chat input
    user_query = st.chat_input("Ask a question about your indexed documents...")
    if user_query:
        # Render user message immediately
        st.chat_message("user").markdown(user_query)
        st.session_state["messages"].append({"role": "user", "content": user_query})

        # Check if database has any documents
        current_docs = get_indexed_documents()
        if not current_docs:
            warning_msg = (
                "⚠️ No documents have been indexed yet. "
                "Please upload and index a PDF in the sidebar before asking questions."
            )
            with st.chat_message("assistant"):
                st.warning(warning_msg)
            st.session_state["messages"].append({
                "role": "assistant",
                "content": warning_msg,
                "citations": [],
            })
            return

        with st.chat_message("assistant"):
            with st.spinner("DocuMind is retrieving evidence and synthesizing answer..."):
                try:
                    rag_response = answer_question(
                        question=user_query,
                        document_filter=active_filter,
                    )
                    answer_text = rag_response.answer
                    citations = rag_response.citations

                    st.markdown(answer_text)

                    if citations:
                        with st.expander(f"📚 Verified Sources & Citations ({len(citations)})", expanded=True):
                            for cit in citations:
                                st.markdown(format_citation_display(cit))
                                if cit.get("snippet"):
                                    st.caption(f'"{cit["snippet"]}"')

                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": answer_text,
                        "citations": citations,
                    })

                except ValueError as exc:
                    error_msg = f"⚠️ Query error: {exc}"
                    st.error(error_msg)
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": error_msg,
                        "citations": [],
                    })
                except (EmbeddingError, GenerationError) as exc:
                    sanitized_err = redact_sensitive_info(str(exc))
                    if (
                        isinstance(exc, GenerationUnavailableError)
                        or getattr(exc, "is_transient", False)
                        or "temporarily unavailable" in sanitized_err.lower()
                        or "503" in sanitized_err
                        or "high demand" in sanitized_err.lower()
                    ):
                        logger.warning("Transient AI service error during generation: %s", sanitized_err)
                        error_msg = f"⚠️ {FRIENDLY_UNAVAILABLE_MESSAGE}"
                        st.warning(error_msg)
                    else:
                        logger.error("AI service error during generation: %s", sanitized_err)
                        error_msg = f"❌ AI Service error: {sanitized_err}"
                        st.error(error_msg)

                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": error_msg,
                        "citations": [],
                    })
                except Exception as exc:
                    error_msg = f"❌ An unexpected error occurred: {exc}"
                    st.error(error_msg)
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": error_msg,
                        "citations": [],
                    })


if __name__ == "__main__":
    main()
