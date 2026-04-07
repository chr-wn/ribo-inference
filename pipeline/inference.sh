#!/bin/bash
set -e
# Run inference on new data
# bash run_inference.sh --input test.csv --output results/ --model-dir models/inference_ensemble

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_CSV=""
OUTPUT_DIR=""
MODEL_DIR=""

BATCH_SIZE=32
GPU=0
NORM_MEAN=""
NORM_STD=""

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --input)
      INPUT_CSV="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --norm-mean)
      NORM_MEAN="$2"
      shift 2
      ;;
    --norm-std)
      NORM_STD="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

INPUT_CSV=$(readlink -f "$INPUT_CSV")
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
MODEL_DIR=$(readlink -f "$MODEL_DIR")

if [ -z "$INPUT_CSV" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$MODEL_DIR" ]; then
    echo "Usage: bash run_inference.sh --input <csv> --output <dir> --model-dir <dir> [--batch-size <N>]"
    exit 1
fi

echo "=========================================="
echo "RNA-Ligand Inference Pipeline"
echo "=========================================="
echo "Input: $INPUT_CSV"
echo "Output: $OUTPUT_DIR"
echo "Models: $MODEL_DIR"
echo ""

TEMP_FEATURES="$OUTPUT_DIR/features"
TEMP_CONSOLIDATED="$OUTPUT_DIR/consolidated"
mkdir -p "$TEMP_FEATURES"
mkdir -p "$TEMP_CONSOLIDATED"

if [ -z "$CONDA_DIR" ]; then
    if [ -d "$HOME/miniconda3" ]; then
        CONDA_DIR="$HOME/miniconda3"
    elif [ -d "/root/miniconda3" ]; then
        CONDA_DIR="/root/miniconda3"
    elif command -v conda &> /dev/null; then
        CONDA_BIN=$(command -v conda)
        CONDA_DIR=$(dirname $(dirname "$CONDA_BIN"))
    else
        echo "Error: Could not find conda. Please set CONDA_DIR."
        exit 1
    fi
fi
source "$CONDA_DIR/bin/activate"

echo "Step 1: Generating Features..."
conda activate pipeline
python -u "$SCRIPT_DIR/generate_all_features.py" \
    --input "$INPUT_CSV" \
    --output "$TEMP_FEATURES" \
    --gpu "$GPU" \
    --rinalmo-batch-size 16 \
    --unimol-batch-size 32

conda deactivate

echo ""
echo "Step 2: Consolidating Features..."
conda activate training
python -u "$SCRIPT_DIR/consolidate_features.py" \
    --input "$INPUT_CSV" \
    --features "$TEMP_FEATURES" \
    --output "$TEMP_CONSOLIDATED"

cp -r "$TEMP_FEATURES/pssm" "$TEMP_CONSOLIDATED/pssm"

echo ""
echo "Step 3: Running Prediction..."
python -u "$SCRIPT_DIR/predict_inference.py" \
    --data "$TEMP_CONSOLIDATED" \
    --model-dir "$MODEL_DIR" \
    --output "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --gpu "$GPU" \
    ${NORM_MEAN:+--norm-mean "$NORM_MEAN"} \
    ${NORM_STD:+--norm-std "$NORM_STD"}

conda deactivate

echo ""
echo "Inference Completed."
echo "Results: $OUTPUT_DIR/predictions.csv"
