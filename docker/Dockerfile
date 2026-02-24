FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    PyMuPDF \
    Pillow \
    requests \
    fastapi \
    "uvicorn[standard]" \
    python-multipart

COPY src/*.py /app/
COPY src/templates/ /app/templates/
COPY config/ocr_config.json /app/
WORKDIR /app

RUN mkdir -p /data/input /data/output /data/uploads

ENV OCR_WEB_MODE=false
ENV OCR_WEB_PORT=14000
ENV OCR_WEB_USER=admin
ENV OCR_WEB_PASS=changeme
ENV OLLAMA_URL=http://192.168.0.169:11434
ENV OLLAMA_MODEL=qwen3-vl:30b
ENV OLLAMA_TEMPERATURE=0.25
ENV OLLAMA_NUM_CTX=30000
ENV OCR_DPI=200

EXPOSE 14000

ENTRYPOINT ["python3", "/app/entrypoint.py"]
