# ==============================================================================
# Dockerfile: Optimized for 512MB Free Cloud Tier (Render / Railway / HuggingFace)
# ==============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KERAS_BACKEND=torch \
    MALLOC_TRIM_THRESHOLD_=100000 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

# Install only minimal runtime dependencies and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch and inference libraries (prevents 2GB CUDA bloat)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        transformers \
        "keras>=3.0.0" \
        scikit-learn \
        numpy \
        pandas \
        joblib \
        flask \
        flask-cors

# Copy application files
COPY . /app/

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

CMD ["python", "app.py"]
