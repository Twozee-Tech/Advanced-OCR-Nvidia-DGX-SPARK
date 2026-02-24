#!/usr/bin/env python3
"""
Qwen OCR via Ollama — PDF to Markdown pipeline.

Sends each page as a base64 image to Ollama /api/chat.
Strips <think>...</think> blocks from responses (thinking model).
Processes pages in batches of 2 in parallel.
"""

import base64
import io
import os
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz  # PyMuPDF
import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_URL   = os.environ.get("OLLAMA_URL",   "http://192.168.0.169:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-vl:30b")
TEMPERATURE  = float(os.environ.get("OLLAMA_TEMPERATURE", "0.25"))
NUM_CTX      = int(os.environ.get("OLLAMA_NUM_CTX", "30000"))
BATCH_SIZE   = 2

OCR_PROMPT = """You are a document OCR system. Convert this page to markdown.

1. Do OCR of the page - extract all text exactly as it appears.
2. Diagrams and images must be included and described in very detailed manner.
3. Put picture/diagram description in the exact same spot as it was on the page.
4. Do not add any text from yourself that is not on the page."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_path: str, dpi: int = 200) -> list:
    """Convert PDF pages to PIL Images."""
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


def image_to_base64(img: Image.Image) -> str:
    """Encode PIL Image as base64 PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def strip_think(text: str) -> str:
    """Remove everything up to and including </think>."""
    idx = text.find("</think>")
    if idx != -1:
        return text[idx + len("</think>"):].lstrip("\n")
    return text


# ---------------------------------------------------------------------------
# OCR single page
# ---------------------------------------------------------------------------

def ocr_page(img: Image.Image, page_num: int, verbose: bool = False) -> str:
    """Send one page image to Ollama, return cleaned markdown."""
    b64 = image_to_base64(img)

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": OCR_PROMPT,
                "images": [b64],
            }
        ],
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_ctx":     NUM_CTX,
        },
    }

    start = time.time()
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=600,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  Page {page_num}: Ollama request failed — {e}", flush=True)
        return f"*[OCR failed for page {page_num}: {e}]*"

    text = resp.json()["message"]["content"]
    text = strip_think(text).strip()
    elapsed = time.time() - start

    if verbose:
        print(f"  Page {page_num}: {len(text)} chars in {elapsed:.1f}s", flush=True)

    return text


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    pdf_path: str,
    output_path: str = None,
    dpi: int = 200,
    verbose: bool = False,
    **_kwargs,
) -> str:
    """Run OCR pipeline on a PDF file. Returns path to output markdown."""
    pdf_path = Path(pdf_path)
    if output_path is None:
        output_path = pdf_path.with_suffix(".md")
    output_path = Path(output_path)

    print(f"Qwen OCR: {pdf_path.name} → Ollama {OLLAMA_URL} ({OLLAMA_MODEL})", flush=True)
    print(f"Settings: temperature={TEMPERATURE}, num_ctx={NUM_CTX}, batch={BATCH_SIZE}", flush=True)

    images = pdf_to_images(str(pdf_path), dpi=dpi)
    num_pages = len(images)
    print(f"Loaded {num_pages} pages at {dpi} DPI", flush=True)

    results = [None] * num_pages
    total_start = time.time()

    for batch_start in range(0, num_pages, BATCH_SIZE):
        batch_end  = min(batch_start + BATCH_SIZE, num_pages)
        batch_imgs = images[batch_start:batch_end]
        batch_nums = list(range(batch_start + 1, batch_end + 1))

        print(f"Pages {batch_start+1}-{batch_end}/{num_pages}...", flush=True)

        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            futures = {
                executor.submit(ocr_page, img, pnum, verbose): idx
                for idx, (img, pnum) in enumerate(zip(batch_imgs, batch_nums))
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[batch_start + idx] = future.result()

    total_time = time.time() - total_start
    print(f"Done in {total_time:.1f}s ({total_time/num_pages:.1f}s/page)", flush=True)

    # Assemble markdown
    header = f"# {pdf_path.stem}\n\n*Pages: {num_pages} | Model: {OLLAMA_MODEL}*\n\n---\n\n"
    pages  = "\n\n---\n\n".join(
        f"<!-- Page {i+1} -->\n\n{text}" for i, text in enumerate(results)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(header + pages, encoding="utf-8")
    print(f"Output: {output_path}", flush=True)

    return str(output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Qwen OCR via Ollama — PDF to Markdown")
    parser.add_argument("input_pdf")
    parser.add_argument("output_md", nargs="?", default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        pdf_path=args.input_pdf,
        output_path=args.output_md,
        dpi=args.dpi,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
