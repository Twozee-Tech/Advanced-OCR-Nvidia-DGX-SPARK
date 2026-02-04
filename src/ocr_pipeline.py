#!/usr/bin/env python3
"""
OCR Pipeline Orchestrator

Three-stage PDF to Markdown conversion:
  Stage 1:   Qwen3-VL-8B classifies pages (runs in venv_qwen)
  Stage 1.5: Qwen3-VL-32B describes diagrams (runs in venv_qwen) [optional]
  Stage 2:   DeepSeek-OCR processes pages (runs in system Python)

Usage:
    python3 ocr_pipeline.py input.pdf [output.md] [--classifier qwen3-vl-8b|heuristic]

    # With diagram description (recommended for documents with flowcharts/diagrams)
    python3 ocr_pipeline.py input.pdf --classifier qwen3-vl-8b --describe-diagrams

Setup required:
    ./setup.sh  # Creates venv_qwen for Stage 1 and Stage 1.5
"""

import json
import sys
import os
import time
import argparse
import subprocess
import shutil
from pathlib import Path


# =============================================================================
# Progress Bar
# =============================================================================

class ProgressBar:
    """Simple terminal progress bar."""

    def __init__(self, total: int, width: int = 40, prefix: str = "Progress"):
        self.total = total
        self.width = width
        self.prefix = prefix
        self.current = 0
        self.stage = ""
        self.enabled = True

    def update(self, current: int = None, stage: str = None):
        """Update progress bar."""
        if not self.enabled:
            return

        if current is not None:
            self.current = current
        if stage is not None:
            self.stage = stage

        percent = min(100, int(100 * self.current / self.total)) if self.total > 0 else 0
        filled = int(self.width * percent / 100)
        bar = "█" * filled + "░" * (self.width - filled)

        # Clear line and print progress
        stage_text = f" | {self.stage}" if self.stage else ""
        sys.stdout.write(f"\r{self.prefix}: [{bar}] {percent:3d}%{stage_text}    ")
        sys.stdout.flush()

    def increment(self, amount: int = 1, stage: str = None):
        """Increment progress."""
        self.update(self.current + amount, stage)

    def finish(self, message: str = "Done!"):
        """Complete the progress bar."""
        if not self.enabled:
            return
        self.current = self.total
        self.update()
        print(f"\n{message}")

# Stage 2 imports (system Python)
from stage2_ocr import (
    load_model as load_ocr,
    unload_model as unload_ocr,
    ocr_pages,
    pdf_to_page_images,
    generate_markdown,
)

# Diagram types that trigger Stage 1.5
DIAGRAM_TYPES = {'diagram', 'flowchart'}


# Config file paths
CONFIG_PATHS = [
    './ocr_config.json',
    '/workspace/ocr_config.json',
    '/workspace/data/ocr_config.json',
    os.path.expanduser('~/.ocr_config.json'),
]


def load_config(config_path: str = None) -> dict:
    """Load configuration from JSON file."""
    paths_to_check = [config_path] if config_path else CONFIG_PATHS

    for path in paths_to_check:
        if path and os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    config = json.load(f)
                print(f"Loaded config from: {path}")
                return config
            except Exception as e:
                print(f"Warning: Failed to load config from {path}: {e}")

    return {}


def find_deepseek_model(config: dict, custom_path: str = None) -> str:
    """Find DeepSeek-OCR model path."""
    if custom_path and os.path.exists(custom_path):
        return custom_path

    if config.get('deepseek_model_path') and os.path.exists(config['deepseek_model_path']):
        return config['deepseek_model_path']

    candidates = [
        './DeepSeek-OCR-model',
        '../DeepSeek-OCR-model',
        '/workspace/DeepSeek-OCR-model',
        '/workspace/models/DeepSeek-OCR-model',
        os.path.expanduser('~/DeepSeek-OCR-model'),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def find_venv_python() -> str:
    """Find Python executable in venv_qwen."""
    # Check env var first (for Docker)
    env_venv = os.environ.get('OCR_VENV_PYTHON')
    if env_venv and os.path.exists(env_venv):
        return env_venv

    script_dir = Path(__file__).parent
    venv_python = script_dir / "venv_qwen" / "bin" / "python3"

    if venv_python.exists():
        return str(venv_python)

    # Try relative to cwd
    venv_python = Path("venv_qwen") / "bin" / "python3"
    if venv_python.exists():
        return str(venv_python)

    return None


def run_stage1_subprocess(
    pdf_path: str,
    output_json: str,
    model_path: str = None,
    precision: str = "fp16",
    heuristic: bool = False,
    dpi: int = 200,
    verbose: bool = False
) -> bool:
    """
    Run Stage 1 classifier as subprocess in venv.

    Returns True on success, False on failure.
    """
    script_dir = Path(__file__).parent
    stage1_script = script_dir / "stage1_classifier.py"

    # Find venv Python
    venv_python = find_venv_python()

    if venv_python is None and not heuristic:
        print("WARNING: venv_qwen not found. Run ./setup.sh first.")
        print("         Falling back to heuristic classification.")
        heuristic = True

    # Build command
    if heuristic:
        # Use system Python for heuristic
        python_exe = sys.executable
    else:
        python_exe = venv_python

    cmd = [
        python_exe,
        str(stage1_script),
        pdf_path,
        "-o", output_json,
        "--dpi", str(dpi),
    ]

    if heuristic:
        cmd.append("--heuristic")
    else:
        if model_path:
            cmd.extend(["-m", model_path])
        cmd.extend(["-p", precision])

    if verbose:
        cmd.append("-v")

    # Run subprocess
    print(f"\n--- Running Stage 1 {'(heuristic)' if heuristic else '(Qwen3-VL in venv)'} ---")
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        # Run with real-time output (no capture)
        result = subprocess.run(
            cmd,
            check=True,
        )

        return True

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Stage 1 failed with exit code {e.returncode}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

    except FileNotFoundError as e:
        print(f"ERROR: Could not run Stage 1: {e}")
        return False


def run_stage1_5_subprocess(
    pdf_path: str,
    classifications_json: str,
    output_json: str,
    model_path: str = None,
    precision: str = "fp16",
    dpi: int = 200,
    verbose: bool = False
) -> bool:
    """
    Run Stage 1.5 diagram describer as subprocess in venv.

    Uses Qwen3-VL-32B to generate detailed descriptions with ASCII art
    for diagram and flowchart pages.

    Returns True on success, False on failure.
    """
    script_dir = Path(__file__).parent
    stage1_5_script = script_dir / "stage1_5_diagram.py"

    if not stage1_5_script.exists():
        print(f"WARNING: stage1_5_diagram.py not found at {stage1_5_script}")
        return False

    # Find venv Python
    venv_python = find_venv_python()

    if venv_python is None:
        print("WARNING: venv_qwen not found. Run ./setup.sh first.")
        print("         Skipping Stage 1.5 diagram description.")
        return False

    # Build command
    cmd = [
        venv_python,
        str(stage1_5_script),
        pdf_path,
        "-c", classifications_json,
        "-o", output_json,
        "--dpi", str(dpi),
        "-p", precision,
    ]

    if model_path:
        cmd.extend(["-m", model_path])

    if verbose:
        cmd.append("-v")

    # Run subprocess
    print(f"\n--- Running Stage 1.5 (Qwen3-VL-32B diagram description) ---")
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        # Run with real-time output (no capture)
        result = subprocess.run(
            cmd,
            check=True,
        )

        return True

    except subprocess.CalledProcessError as e:
        print(f"WARNING: Stage 1.5 failed with exit code {e.returncode}")
        print("         Continuing without diagram descriptions.")
        return False

    except FileNotFoundError as e:
        print(f"WARNING: Could not run Stage 1.5: {e}")
        return False


def run_pipeline(
    pdf_path: str,
    output_path: str = None,
    classifier: str = 'heuristic',
    classifier_precision: str = 'fp16',
    deepseek_model_path: str = None,
    qwen_model_path: str = None,
    qwen_describer_path: str = None,
    describe_diagrams: bool = False,
    dpi: int = 200,
    verbose: bool = False,
    keep_assets: bool = False,
    config: dict = None
) -> str:
    """
    Run the full three-stage OCR pipeline.

    Stage 1:   Qwen3-VL-8B classifies pages (subprocess in venv)
    Stage 1.5: Qwen3-VL-32B describes diagrams (subprocess in venv) [optional]
    Stage 2:   DeepSeek-OCR processes pages (current Python)
    """
    config = config or {}
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        sys.exit(1)

    # Default output path
    if output_path is None:
        output_path = pdf_path.with_suffix('.md')
    output_path = Path(output_path)

    # Create output directory
    output_dir = output_path.parent / f"{output_path.stem}_assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Show header
    if verbose:
        print("=" * 60)
        print("OCR Pipeline - Three-Stage Processing")
        print("=" * 60)
        print(f"Input:            {pdf_path}")
        print(f"Output:           {output_path}")
        print(f"Assets:           {output_dir}")
        print(f"Classifier:       {classifier} ({classifier_precision})")
        print(f"Describe diagrams: {describe_diagrams}")
        print()
    else:
        print(f"OCR: {pdf_path.name} → {output_path.name}")

    total_start = time.time()

    # Get page count for progress tracking (quick PDF open)
    import fitz
    with fitz.open(str(pdf_path)) as pdf:
        num_pages = pdf.page_count

    # Calculate total steps: Stage1 + Stage1.5(optional) + Stage2 + Final
    # Weight: Stage1=1x, Stage1.5=1x per diagram, Stage2=2x per page (slower), Final=1
    total_steps = num_pages + (num_pages * 2) + 1  # Stage1 + Stage2 + Final
    if describe_diagrams:
        total_steps += num_pages // 2  # Estimate ~50% might be diagrams

    progress = ProgressBar(total_steps, prefix="OCR")
    progress.enabled = not verbose

    # =========================================================================
    # Stage 1: Classification (subprocess in venv)
    # =========================================================================
    if verbose:
        print("=" * 60)
        print("STAGE 1: Page Classification")
        print("=" * 60)

    progress.update(0, "Stage 1: Classifying pages")

    classifications_file = output_dir / "classifications.json"
    qwen_path = qwen_model_path or config.get('qwen_model_path')

    success = run_stage1_subprocess(
        pdf_path=str(pdf_path),
        output_json=str(classifications_file),
        model_path=qwen_path,
        precision=classifier_precision,
        heuristic=(classifier == 'heuristic'),
        dpi=dpi,
        verbose=verbose
    )

    if not success:
        print("\nERROR: Stage 1 failed. Aborting pipeline.")
        sys.exit(1)

    # Load classifications
    with open(classifications_file, 'r') as f:
        class_data = json.load(f)

    classifications = class_data.get('classifications', [])
    classifier_method = class_data.get('method', 'unknown')

    progress.increment(num_pages, "Stage 1: Complete")

    # Print summary (verbose only)
    if verbose:
        content_types = {}
        for c in classifications:
            ct = c.get('type', 'unknown')
            content_types[ct] = content_types.get(ct, 0) + 1
        print(f"\nContent types: {content_types}")
        print(f"Classifications saved to: {classifications_file}")

    # =========================================================================
    # Stage 1.5: Diagram Description (optional)
    # =========================================================================
    diagram_descriptions = {}
    descriptions_file = output_dir / "diagram_descriptions.json"

    # Check if there are diagram pages and describe_diagrams is enabled
    # Include diagram/flowchart pages AND mixed pages that have diagrams
    diagram_pages = [c for c in classifications
                     if c.get('type', '').lower() in DIAGRAM_TYPES
                     or (c.get('type', '').lower() == 'mixed' and c.get('has_diagrams', False))]

    if describe_diagrams and diagram_pages:
        if verbose:
            print("\n" + "=" * 60)
            print(f"STAGE 1.5: Diagram Description ({len(diagram_pages)} pages)")
            print("=" * 60)

        progress.update(stage=f"Stage 1.5: Describing {len(diagram_pages)} diagrams")
        describer_path = qwen_describer_path or config.get('qwen_describer_path')

        success = run_stage1_5_subprocess(
            pdf_path=str(pdf_path),
            classifications_json=str(classifications_file),
            output_json=str(descriptions_file),
            model_path=describer_path,
            precision=classifier_precision,
            dpi=dpi,
            verbose=verbose
        )

        if success and descriptions_file.exists():
            with open(descriptions_file, 'r') as f:
                desc_data = json.load(f)
            # Convert string keys to int for page numbers
            raw_descriptions = desc_data.get('descriptions', {})
            diagram_descriptions = {int(k): v for k, v in raw_descriptions.items()}
            if verbose:
                print(f"Loaded {len(diagram_descriptions)} diagram descriptions")
        else:
            if verbose:
                print("No diagram descriptions generated (Stage 1.5 skipped or failed)")

        progress.increment(len(diagram_pages), "Stage 1.5: Complete")

    elif describe_diagrams and not diagram_pages:
        if verbose:
            print("\nNo diagram/flowchart pages found - skipping Stage 1.5")

    elif diagram_pages and verbose:
        print(f"\nNote: Found {len(diagram_pages)} diagram pages. Use --describe-diagrams for better output.")

    # =========================================================================
    # Stage 2: OCR Processing (system Python)
    # =========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("STAGE 2: OCR Processing (DeepSeek-OCR)")
        print("=" * 60)

    progress.update(stage="Stage 2: Loading OCR model")

    # Find DeepSeek model
    deepseek_path = find_deepseek_model(config, deepseek_model_path)
    if deepseek_path is None:
        print("\nERROR: DeepSeek-OCR model not found.")
        print("Download: git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR DeepSeek-OCR-model")
        sys.exit(1)

    # Load model
    ocr_model, tokenizer = load_ocr(deepseek_path, verbose=verbose)

    # Convert PDF to images (for Stage 2)
    progress.update(stage="Stage 2: Converting PDF")
    pages = pdf_to_page_images(str(pdf_path), dpi=dpi, verbose=verbose)

    # Validate classification count
    if len(classifications) != len(pages):
        if verbose:
            print(f"WARNING: Classification count mismatch ({len(classifications)} vs {len(pages)} pages)")
        # Pad or truncate
        while len(classifications) < len(pages):
            classifications.append({
                'page': len(classifications) + 1,
                'type': 'mixed',
                'confidence': 0.5,
                'method': 'default'
            })

    # Run OCR with progress callback
    if verbose:
        print(f"\nProcessing {len(pages)} pages...")

    def ocr_progress(page_num, total):
        progress.update(
            num_pages + (page_num * 2),
            f"Stage 2: OCR page {page_num}/{total}"
        )

    results = ocr_pages(
        ocr_model, tokenizer, pages, classifications,
        str(output_dir), verbose,
        progress_callback=None if verbose else ocr_progress
    )

    # Unload model
    unload_ocr(ocr_model, tokenizer, verbose=verbose)
    progress.increment(num_pages * 2, "Stage 2: Complete")

    # =========================================================================
    # Generate Output
    # =========================================================================
    progress.update(stage="Generating output")

    if verbose:
        print("\n" + "=" * 60)
        print("Generating Output")
        print("=" * 60)

    markdown = generate_markdown(
        results, pdf_path.stem, classifier_method,
        diagram_descriptions=diagram_descriptions
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown)

    progress.increment(1)

    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - total_start
    total_chars = sum(len(r.get('text', '')) for r in results)
    total_figures = sum(len(r.get('figures', [])) for r in results)

    # Cleanup assets unless --keep-assets
    if not keep_assets and output_dir.exists():
        shutil.rmtree(output_dir)

    # Final message
    progress.finish(f"Done in {total_time:.1f}s → {output_path}")

    if verbose:
        print(f"\nCompleted in {total_time:.1f}s ({total_time/len(pages):.1f}s per page)")
        print(f"Total characters: {total_chars}")
        print(f"Figures extracted: {total_figures}")
        print(f"Output: {output_path}")
        if not keep_assets:
            print(f"Cleaned up: {output_dir}")
        else:
            print(f"Assets kept: {output_dir}")
        print("=" * 60)

    return str(output_path)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OCR Pipeline - Three-Stage PDF to Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Setup:
  ./setup.sh  # Creates venv_qwen for Qwen3-VL models

Examples:
  # With Qwen3-VL-8B classifier (requires venv)
  python3 ocr_pipeline.py document.pdf --classifier qwen3-vl-8b

  # With diagram description (recommended for diagrams/flowcharts)
  python3 ocr_pipeline.py document.pdf --classifier qwen3-vl-8b --describe-diagrams

  # Heuristic classification (no venv needed)
  python3 ocr_pipeline.py document.pdf --classifier heuristic

  # Full options
  python3 ocr_pipeline.py document.pdf output.md --classifier qwen3-vl-8b --describe-diagrams -v
"""
    )

    parser.add_argument('input_pdf', help='Input PDF file')
    parser.add_argument('output_md', nargs='?', default=None, help='Output markdown file')

    parser.add_argument(
        '--classifier', '-c',
        choices=['qwen3-vl-8b', 'heuristic'],
        default=None,
        help='Page classifier (default from config or heuristic)'
    )

    parser.add_argument(
        '--precision', '-p',
        choices=['fp16', 'fp8'],
        default=None,
        help='Classifier precision (default: fp16)'
    )

    # Diagram description options
    diagram_group = parser.add_mutually_exclusive_group()
    diagram_group.add_argument(
        '--describe-diagrams',
        action='store_true',
        dest='describe_diagrams',
        help='Use Qwen3-VL-32B to describe diagram/flowchart pages with ASCII art'
    )
    diagram_group.add_argument(
        '--no-describe-diagrams',
        action='store_false',
        dest='describe_diagrams',
        help='Skip diagram description (faster, uses DeepSeek only)'
    )
    parser.set_defaults(describe_diagrams=None)

    parser.add_argument('--deepseek-model', default=None, help='Path to DeepSeek-OCR model')
    parser.add_argument('--qwen-model', default=None, help='Path to Qwen3-VL-8B classifier model')
    parser.add_argument('--qwen-describer', default=None, help='Path to Qwen3-VL-32B describer model')
    parser.add_argument('--dpi', type=int, default=200, help='PDF rendering DPI')
    parser.add_argument('--config', default=None, help='Path to config JSON')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--keep-assets', action='store_true', help='Keep intermediate files (classifications, etc.)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Get settings from args or config
    classifier = args.classifier or config.get('classifier', 'heuristic')
    precision = args.precision or config.get('precision', 'fp16')

    # Determine describe_diagrams setting
    # Priority: CLI arg > config > default (False)
    if args.describe_diagrams is not None:
        describe_diagrams = args.describe_diagrams
    else:
        describe_diagrams = config.get('describe_diagrams', False)

    # Run pipeline
    run_pipeline(
        pdf_path=args.input_pdf,
        output_path=args.output_md,
        classifier=classifier,
        classifier_precision=precision,
        deepseek_model_path=args.deepseek_model,
        qwen_model_path=args.qwen_model,
        qwen_describer_path=args.qwen_describer,
        describe_diagrams=describe_diagrams,
        dpi=args.dpi,
        verbose=args.verbose,
        keep_assets=args.keep_assets,
        config=config
    )


if __name__ == "__main__":
    main()
