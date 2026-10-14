# ==============================================================================
# DocuMind AI — Production Dockerfile
# Multi-stage / least-privilege Streamlit container
# ==============================================================================

# Base image: official maintained Python 3.10 slim
FROM python:3.10-slim AS runner

# Set metadata
LABEL maintainer="DocuMind AI Team"
LABEL description="Production container for DocuMind AI Intelligent Document Assistant"

# Python runtime configuration: prevent bytecode writes and force unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8501 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    UPLOAD_DIR=/app/data/uploads \
    CHROMA_PERSIST_DIR=/app/chroma_db \
    CHROMA_COLLECTION_NAME=documind_docs_v2

WORKDIR /app

# Install minimal OS dependencies: curl for container health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root user and group (Least Privilege)
RUN groupadd -r -g 1000 appgroup && \
    useradd -r -u 1000 -g appgroup -d /app -s /sbin/nologin appuser

# Install Python dependencies first to maximize Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source modules (excluding secrets, local dbs, and uploads via .dockerignore)
COPY app.py rag.py ingest.py evaluate.py .env.example ./

# Create mount points for persistent storage and grant non-root ownership
RUN mkdir -p /app/data/uploads /app/chroma_db && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Declare exposed port (defaults to 8501)
EXPOSE 8501

# Container healthcheck querying Streamlit internal health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Launch Streamlit web application on 0.0.0.0 and configured port
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true --browser.gatherUsageStats=false"]
