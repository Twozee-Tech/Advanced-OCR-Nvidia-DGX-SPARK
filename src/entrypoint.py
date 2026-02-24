#!/usr/bin/env python3
"""Docker entrypoint for OCR Pipeline."""
import os
import sys
import glob
from pathlib import Path


def main():
    if os.environ.get("OCR_WEB_MODE", "false").lower() == "true":
        import uvicorn
        uvicorn.run(
            "web_app:app",
            host=os.environ.get("OCR_WEB_HOST", "0.0.0.0"),
            port=int(os.environ.get("OCR_WEB_PORT", "14000")),
            log_level="info",
        )
        return

    input_pdf = os.environ.get("OCR_INPUT_PDF")
    if not input_pdf:
        pdfs = glob.glob("/data/input/*.pdf")
        if len(pdfs) == 1:
            input_pdf = pdfs[0]
        elif len(pdfs) > 1:
            print("ERROR: Multiple PDFs in /data/input. Set OCR_INPUT_PDF env var.")
            sys.exit(1)
        else:
            print("ERROR: No PDF found in /data/input")
            sys.exit(1)

    input_name = Path(input_pdf).stem
    output_md  = os.environ.get("OCR_OUTPUT_PATH", f"/data/output/{input_name}.md")
    dpi        = os.environ.get("OCR_DPI", "200")
    verbose    = os.environ.get("OCR_VERBOSE", "false").lower() == "true"

    sys.argv = ["qwen_ocr.py", input_pdf, output_md, "--dpi", dpi]
    if verbose:
        sys.argv.append("--verbose")

    from qwen_ocr import main as run
    run()


if __name__ == "__main__":
    main()
