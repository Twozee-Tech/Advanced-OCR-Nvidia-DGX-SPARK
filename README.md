# Advanced OCR LLM

PDF to Markdown conversion using vision LLMs via Ollama.
Pages are sent as images to Ollama, thinking blocks stripped, result assembled as clean Markdown.

Compatible with [obsidian-marker](https://github.com/L3-N0X/obsidian-marker) plugin (self-hosted docker mode).

## How It Works

```
PDF → page images (PyMuPDF) → Ollama (qwen3-vl:30b) → strip </think> → Markdown
```

- 2 pages processed in parallel per batch
- Diagrams and figures described in place
- Web UI with job queue, live page progress, download
- Marker-compatible API (`POST /convert`) for Obsidian integration

## Requirements

- Docker
- Ollama instance with `qwen3-vl:30b` loaded (separate host or local)

---

## Deploy — Standard (any Linux with Docker)

```bash
git clone https://github.com/Twozee-Tech/Advanced-OCR-LLM.git
cd Advanced-OCR-LLM

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

Web UI: `http://<host>:14000`

---

## Deploy — Proxmox LXC

### 1. Create LXC container

In Proxmox UI or via shell:

```bash
# Debian 12, unprivileged
pct create 200 local:vztmpl/debian-12-standard_12.7-1_amd64.tar.zst \
  --hostname ocr-pipeline \
  --memory 2048 \
  --cores 2 \
  --rootfs local-lvm:8 \
  --net0 name=eth0,bridge=vmbr0,ip=dhcp \
  --unprivileged 1 \
  --features nesting=1

pct start 200
pct enter 200
```

> `nesting=1` is required for Docker inside LXC.

### 2. Install Docker inside LXC

```bash
apt update && apt install -y curl
curl -fsSL https://get.docker.com | sh
```

### 3. Clone and build

```bash
apt install -y git
git clone https://github.com/Twozee-Tech/Advanced-OCR-LLM.git
cd Advanced-OCR-LLM
docker build -f docker/Dockerfile -t ocr-pipeline .
```

### 4. Run

```bash
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

Set a static IP for the LXC in Proxmox so the address doesn't change between reboots.

---

## Obsidian Integration

Install [obsidian-marker](https://github.com/L3-N0X/obsidian-marker) plugin, then:

- Mode: `Self-hosted (Docker)`
- URL: `http://<host>:14000`

The `/convert` endpoint accepts `pdf_file` multipart and returns Marker-compatible JSON.

---

## Settings

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://192.168.0.169:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen3-vl:30b` | Model name in Ollama |
| `OLLAMA_TEMPERATURE` | `0.25` | Sampling temperature |
| `OLLAMA_NUM_CTX` | `30000` | Context window (tokens) |
| `OCR_WEB_USER` | `admin` | Web UI username |
| `OCR_WEB_PASS` | `changeme` | Web UI password |
| `OCR_DPI` | `200` | PDF render resolution |

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI |
| `/health` | GET | Health check |
| `/upload` | POST | Queue a PDF job (web UI) |
| `/jobs` | GET | List all jobs |
| `/jobs/{id}/download` | GET | Download result markdown |
| `/convert` | POST | Marker-compatible OCR endpoint |

---

## Project Structure

```
├── docker/
│   └── Dockerfile          # python:3.11-slim, ~400MB, no GPU required
├── src/
│   ├── entrypoint.py
│   ├── qwen_ocr.py         # core OCR pipeline (Ollama API)
│   ├── web_app.py          # FastAPI web UI + /convert endpoint
│   └── templates/
│       └── index.html
└── config/
    └── ocr_config.json
```

## License

MIT
