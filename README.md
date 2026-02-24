# Advanced OCR

PDF to Markdown conversion using [Qwen3-VL-30B](https://ollama.com/library/qwen3-vl) via Ollama.
Pages are sent as images to Ollama, thinking blocks stripped, result assembled as clean Markdown.

## How It Works

```
PDF → page images → Ollama (qwen3-vl:30b) → strip </think> → Markdown
```

- 2 pages processed in parallel per batch
- Diagrams and figures described in place
- Web UI with job queue and download

## Requirements

- Docker
- Ollama instance with `qwen3-vl:30b` loaded

## Deploy

```bash
docker build -f docker/Dockerfile -t ocr-pipeline .

docker run -d \
  --name ocr-pipeline \
  --restart unless-stopped \
  -p 14000:14000 \
  -v /data/ocr:/data \
  -e OCR_WEB_MODE=true \
  -e OCR_WEB_USER=admin \
  -e OCR_WEB_PASS=changeme \
  -e OLLAMA_URL=http://<ollama-host>:11434 \
  -e OLLAMA_MODEL=qwen3-vl:30b \
  -e OLLAMA_TEMPERATURE=0.25 \
  -e OLLAMA_NUM_CTX=30000 \
  ocr-pipeline
```

Web UI available at `http://<host>:14000`

## Settings

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://192.168.0.169:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen3-vl:30b` | Model to use |
| `OLLAMA_TEMPERATURE` | `0.25` | Sampling temperature |
| `OLLAMA_NUM_CTX` | `30000` | Context window |
| `OCR_WEB_USER` | `admin` | Web UI username |
| `OCR_WEB_PASS` | `changeme` | Web UI password |
| `OCR_DPI` | `200` | PDF render resolution |

## Project Structure

```
├── docker/
│   └── Dockerfile
├── src/
│   ├── entrypoint.py
│   ├── qwen_ocr.py
│   └── web_app.py
└── config/
    └── ocr_config.json
```

## License

MIT
