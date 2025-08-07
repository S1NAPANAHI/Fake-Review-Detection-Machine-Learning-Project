# Multi-stage Dockerfile for Fake Review Detection System
# Stage 1: Builder - Install dependencies and build the application
FROM python:3.11-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Download required NLTK data and spaCy models
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('vader_lexicon')" && \
    python -m spacy download en_core_web_sm

# Stage 2: Runtime - Create the final lightweight image
FROM python:3.11-slim as runtime

# Set runtime arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /bin/bash -c "App User" appuser

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PATH="/opt/venv/bin:$PATH"

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create application directory and set ownership
RUN mkdir -p /app && \
    mkdir -p /app/data/raw \
    /app/data/processed \
    /app/data/external \
    /app/data/interim \
    /app/artifacts/models \
    /app/artifacts/checkpoints \
    /app/artifacts/exports \
    /app/artifacts/features \
    /app/artifacts/feature_store \
    /app/artifacts/reports \
    /app/artifacts/metrics \
    /app/logs \
    /app/config \
    && chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser tests/ ./tests/

# Copy additional configuration files if they exist
# COPY --chown=appuser:appuser pyproject.toml setup.py ./

# Switch to non-root user
USER appuser

# Expose port for the API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - can be overridden
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Labels for metadata
LABEL maintainer="your-email@example.com"
LABEL description="Fake Review Detection System API"
LABEL version="1.0.0"
