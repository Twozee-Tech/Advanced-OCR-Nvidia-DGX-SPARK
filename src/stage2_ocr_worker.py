#!/usr/bin/env python3
"""
DeepSeek-OCR Worker - Runs in venv with transformers 4.46.3

This worker script is called via subprocess from stage2_ocr.py.
It loads the DeepSeek-OCR-2 model using transformers (not vLLM) and
processes pages one by one.

Usage:
    /opt/venv_ocr/bin/python3 stage2_ocr_worker.py input.json output.json [-v]
"""

import sys
import json
import argparse
import time
import os


# =============================================================================
# Prompts (same as in stage2_ocr.py)
# =============================================================================

PROMPTS = {
    'text': "<image>\n<|grounding|>Convert the document to markdown.",
    'document': "<image>\n<|grounding|>Convert the document to markdown.",
    'figure': "<image>\nParse the figure.",
    'diagram': "<image>\nDescribe this image in detail.",
    'flowchart': "<image>\nDescribe this image in detail.",
    'table': "<image>\n<|grounding|>Convert the document to markdown.",
    'mixed': "<image>\n<|grounding|>Convert the document to markdown.",
}


def get_prompt(classification: dict) -> str:
    """Get optimal OCR prompt based on classification."""
    if classification is None:
        return PROMPTS['mixed']

    page_type = classification.get('type', 'mixed')
    confidence = classification.get('confidence', 0.5)

    # Low confidence -> safer mixed prompt
    if confidence < 0.7:
        return PROMPTS['mixed']

    return PROMPTS.get(page_type, PROMPTS['mixed'])


# =============================================================================
# OCR Processing
# =============================================================================

def ocr_single_page(model, tokenizer, image_path: str, classification: dict, temp_dir: str) -> dict:
    """
    OCR a single page using DeepSeek-OCR-2 with transformers.

    Args:
        model: Loaded DeepSeek-OCR model
        tokenizer: Model tokenizer
        image_path: Path to page image
        classification: Classification from Stage 1
        temp_dir: Temp directory for output

    Returns:
        dict: OCR result with 'text', 'classification', 'figures'
    """
    import shutil

    prompt = get_prompt(classification)

    # DeepSeek-OCR-2 infer() API
    # model.infer() prints to stdout and writes to files — the return value is unreliable.
    # We use save_results=True and read output from the filesystem.
    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_path,
        output_path=temp_dir,
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=True,
        test_compress=False,
    )

    # Extract text: check return value first, then fall back to output files
    ocr_text = ""

    if result and isinstance(result, str) and len(result.strip()) > 0:
        ocr_text = result

    # Check result.mmd file
    if not ocr_text:
        mmd_file = os.path.join(temp_dir, 'result.mmd')
        if os.path.exists(mmd_file):
            with open(mmd_file, 'r', encoding='utf-8') as f:
                ocr_text = f.read()
            os.remove(mmd_file)

    # Check 'other' directories
    if not ocr_text:
        for root, dirs, files in os.walk(temp_dir):
            if 'other' in dirs:
                other_dir = os.path.join(root, 'other')
                parts = []
                for fname in sorted(os.listdir(other_dir)):
                    fpath = os.path.join(other_dir, fname)
                    if os.path.isfile(fpath):
                        try:
                            with open(fpath, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                                if content:
                                    parts.append(content)
                        except Exception:
                            pass
                if parts:
                    ocr_text = '\n\n'.join(parts)
                shutil.rmtree(other_dir, ignore_errors=True)
                break

    # Check for .md/.txt files
    if not ocr_text:
        for root, dirs, files in os.walk(temp_dir):
            for fname in files:
                if fname.endswith(('.md', '.txt')):
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if content and len(content) > 10:
                                ocr_text = content
                                os.remove(fpath)
                                break
                    except Exception:
                        pass
            if ocr_text:
                break

    return {
        'text': ocr_text,
        'classification': classification,
        'figures': []
    }


def main():
    parser = argparse.ArgumentParser(description='DeepSeek-OCR Worker')
    parser.add_argument('input_file', help='Input JSON file with task data')
    parser.add_argument('output_file', help='Output JSON file for results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    # Load input data
    with open(args.input_file) as f:
        data = json.load(f)

    model_path = data['model_path']
    image_paths = data['image_paths']
    classifications = data['classifications']

    if args.verbose:
        print(f"DeepSeek-OCR Worker starting...", flush=True)
        print(f"  Model: {model_path}", flush=True)
        print(f"  Pages: {len(image_paths)}", flush=True)

    # Import here so we only load transformers in the venv
    import torch
    from transformers import AutoModel, AutoTokenizer

    # Load model
    load_start = time.time()
    if args.verbose:
        print(f"Loading DeepSeek-OCR model...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load model — use eager attention (flash-attn not installed) and bfloat16
    model = AutoModel.from_pretrained(
        model_path,
        _attn_implementation='eager',
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    ).eval()
    if args.verbose:
        print(f"  Model on CUDA (bfloat16, eager attention)", flush=True)

    if args.verbose:
        load_time = time.time() - load_start
        print(f"  Model loaded in {load_time:.1f}s", flush=True)

    # Create temp dir for model output
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="deepseek_ocr_")

    # Process pages
    results = []
    process_start = time.time()

    for i, (img_path, cls) in enumerate(zip(image_paths, classifications)):
        page_start = time.time()

        if args.verbose:
            page_type = cls.get('type', 'mixed')
            print(f"  Processing page {i+1}/{len(image_paths)} [{page_type}]...", end='', flush=True)

        result = ocr_single_page(model, tokenizer, img_path, cls, temp_dir)
        results.append(result)

        if args.verbose:
            page_time = time.time() - page_start
            chars = len(result.get('text', ''))
            print(f" {chars} chars in {page_time:.1f}s", flush=True)

    # Cleanup temp dir
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False)

    if args.verbose:
        total_time = time.time() - process_start
        total_chars = sum(len(r.get('text', '')) for r in results)
        print(f"DeepSeek-OCR Worker complete:", flush=True)
        print(f"  Total time: {total_time:.1f}s ({total_time/len(image_paths):.1f}s/page)", flush=True)
        print(f"  Total chars: {total_chars}", flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(f"ERROR: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
