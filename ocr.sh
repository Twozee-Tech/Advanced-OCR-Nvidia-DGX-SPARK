#!/bin/bash
# OCR Pipeline wrapper script
# Usage: ./ocr.sh input.pdf [--diagrams]

set -e

INPUT_PDF="$1"
shift || true

if [ -z "$INPUT_PDF" ]; then
    echo "Usage: ./ocr.sh input.pdf [--diagrams]"
    echo ""
    echo "Options:"
    echo "  --diagrams    Enable Stage 1.5 diagram description (Qwen3-VL-32B)"
    exit 1
fi

# Defaults
DESCRIBE_DIAGRAMS="false"

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --diagrams) DESCRIBE_DIAGRAMS="true"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

INPUT_DIR=$(dirname "$(realpath "$INPUT_PDF")")
INPUT_NAME=$(basename "$INPUT_PDF")
OUTPUT_DIR="${INPUT_DIR}/output"

mkdir -p "$OUTPUT_DIR"

echo "================================"
echo "OCR Pipeline"
echo "================================"
echo "Input:    $INPUT_PDF"
echo "Output:   $OUTPUT_DIR/${INPUT_NAME%.pdf}.md"
echo "Diagrams: $DESCRIBE_DIAGRAMS"
echo "================================"

docker run --gpus all --rm \
    -e OCR_DESCRIBE_DIAGRAMS="$DESCRIBE_DIAGRAMS" \
    -v /workspace/models:/workspace/models:ro \
    -v "$INPUT_DIR":/data/input:ro \
    -v "$OUTPUT_DIR":/data/output \
    -e OCR_INPUT_PDF="/data/input/$INPUT_NAME" \
    ocr-pipeline

echo "================================"
echo "Done! Output: $OUTPUT_DIR/${INPUT_NAME%.pdf}.md"
echo "================================"
