#!/bin/bash
# This script sets up the environment (if needed) and runs the full feature generation pipeline.
# bash run_full_pipeline.sh
# bash run_full_pipeline.sh --skip-setup  # skips env setup if already done

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$INFERENCE_DIR/data"
OUTPUT_DIR="$DATA_DIR/processed"

echo "=========================================="
echo "RNA-Ligand Feature Generation Pipeline"
echo "=========================================="
echo ""
echo "Script directory: $SCRIPT_DIR"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

SKIP_SETUP=false
DEV_MODE=false
for arg in "$@"; do
    case $arg in
        --skip-setup) SKIP_SETUP=true ;;
        --dev) DEV_MODE=true ;;
    esac
done

if [ "$DEV_MODE" = true ]; then
    echo "! RUNNING IN DEVELOPMENT MODE (fast test) !"
    BATCH_SIZE=2
    EPOCHS=1
    NUM_MODELS=1
    RINALMO_BS=2
    UNIMOL_BS=4
else
    BATCH_SIZE=4 
    EPOCHS=20
    NUM_MODELS=5
    RINALMO_BS=4
    UNIMOL_BS=32
fi

if [ "$SKIP_SETUP" = false ]; then
    echo "Step 1: Setting up conda environments..."
    bash "$SCRIPT_DIR/setup.sh"
else
    echo "Step 1: Skipping setup (--skip-setup)"
fi

CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
source "$CONDA_DIR/bin/activate"

echo ""
echo "Step 2: Generating all features..."
echo ""

if [ ! -f "$DATA_DIR/union_dataset_final.csv" ]; then
    echo "ERROR: union_dataset_final.csv not found in $DATA_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

conda activate pipeline
python -u "$SCRIPT_DIR/generate_all_features.py" \
    --input "$DATA_DIR/union_dataset_final.csv" \
    --output "$OUTPUT_DIR" \
    --gpu 0 \
    --rinalmo-batch-size $RINALMO_BS \
    --unimol-batch-size $UNIMOL_BS
conda deactivate

echo ""
echo "Step 3: Consolidating features..."
echo ""

CONSOLIDATED_DIR="$DATA_DIR/consolidated"
mkdir -p "$CONSOLIDATED_DIR"

conda activate training
python -u "$SCRIPT_DIR/consolidate_features.py" \
    --input "$DATA_DIR/union_dataset_final.csv" \
    --features "$OUTPUT_DIR" \
    --output "$CONSOLIDATED_DIR"

echo ""
echo "Step 4: Training Inference Ensemble..."
echo ""

MODEL_DIR="$INFERENCE_DIR/models/inference_ensemble"
if [ "$DEV_MODE" = true ]; then
    MODEL_DIR="$INFERENCE_DIR/models/dev_test"
fi
mkdir -p "$MODEL_DIR"

python -u "$SCRIPT_DIR/train_inference.py" \
    --data "$CONSOLIDATED_DIR" \
    --output "$MODEL_DIR" \
    --num-models $NUM_MODELS \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE

conda deactivate

echo ""
echo "Step 5: Verifying generated files..."
echo ""

echo "Feature file counts:"
for dir in rinalmo mxfold2 pssm rna_onehot unimol mol_onehot mol_graph; do
    if [ -d "$OUTPUT_DIR/$dir" ]; then
        count=$(find "$OUTPUT_DIR/$dir" -type f -name "*.npy" -o -name "*.json" -o -name "*.npz" 2>/dev/null | wc -l)
        echo "  $dir: $count files"
    else
        echo "  $dir: (not found)"
    fi
done

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Features saved to: $OUTPUT_DIR"
echo "Models saved to: $MODEL_DIR"
echo ""
