# API image: FastAPI + the local models. CPU-only PyTorch keeps the image portable
# (Docker on a Mac can't use the Apple GPU anyway); see docs/decisions.md D39.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    CHROMA_DIR=/data/chroma \
    UPLOAD_DIR=/data/uploads \
    JOBS_DB=/data/jobs.sqlite3

WORKDIR /app

# Dependencies first so code changes don't reinstall them.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

COPY rag ./rag
COPY api ./api
COPY cli.py .

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data /models \
    && chown -R app:app /data /models
USER app

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "--factory", "api.app:create_app", "--host", "0.0.0.0", "--port", "8000"]
