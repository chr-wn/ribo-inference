#!/bin/bash
set -e
# bash benchmark/run_benchmarks.sh [--debug] [--gpu <id>]

export PYTHONPATH=$PYTHONPATH:.
export PYTHONUNBUFFERED=1

DEBUG=false
GPU=0

for arg in "$@"
do
    case $arg in
        --debug)
        DEBUG=true
        shift
        ;;
        --gpu)
        GPU="$2"
        shift 2
        ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

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

source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate training

if [ "$DEBUG" = true ]; then
    echo "Running in DEBUG mode (fast runs)..."
    EPOCHS=2
    BATCH=4
    FOLDS=2
    REPEATS=1
    K=2
else
    echo "Running in FULL BENCHMARK mode..."
    EPOCHS=100
    BATCH=32
    FOLDS=5
    REPEATS=1
    K=5
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="benchmark/logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"
ABS_LOG_DIR="$(pwd)/$LOG_DIR"

echo "Logs will be saved to $ABS_LOG_DIR"

# echo "========================================================"
# echo "Starting 5-Fold Cross Validation..."
# echo "========================================================"
# python benchmark/train_5fold.py \
#     --epochs $EPOCHS \
#     --batch_size $BATCH \
#     --val_batch_size $BATCH \
#     --folds $FOLDS \
#     --repeats $REPEATS \
#     --k $K \
#     --gpu $GPU \
#     --data-dir "benchmark/data" \
#     --out_dir "benchmark/ckpts/val_5fold" \
#     2>&1 | tee "$LOG_DIR/val_5fold.log"

# echo "5-Fold CV Completed. Check $LOG_DIR/val_5fold.log"
# grep "Weighted Average" "$LOG_DIR/val_5fold.log" -A 2 || echo "No summary found."

# echo "========================================================"
# echo "Starting Semi-Supervised 5-Fold CV..."
# echo "========================================================"
# python benchmark/train_5fold_semi1.py \
#     --ckpt_dir "benchmark/ckpts/val_5fold" \
#     --epochs 80 \
#     --warmup_epochs 3 \
#     --batch_size $BATCH \
#     --val_batch_size $BATCH \
#     --folds $FOLDS \
#     --k $K \
#     --lr 3e-5 \
#     --pseudo_alpha 0.7 \
#     --self_train_rounds 2 \
#     --gpu $GPU \
#     --data-dir "benchmark/data" \
#     --out_dir "benchmark/ckpts/val_5fold_semi1" \
#     2>&1 | tee "$LOG_DIR/val_5fold_semi1.log"

# echo "Semi-Supervised 5-Fold CV Completed. Check $LOG_DIR/val_5fold_semi1.log"
# grep "Weighted Average" "$LOG_DIR/val_5fold_semi1.log" -A 2 || echo "No summary found."

# echo "========================================================"
# echo "Starting Blind RNA Test..."
# echo "========================================================"
# python benchmark/train_blindrna.py \
#     --epochs $EPOCHS \
#     --batch_size $BATCH \
#     --val_batch_size $BATCH \
#     --folds $FOLDS \
#     --k $K \
#     --data-dir "benchmark/data" \
#     --out_dir "benchmark/ckpts/val_blindrna" \
#     2>&1 | tee "$LOG_DIR/val_blindrna.log"

# echo "Blind RNA Test Completed. Check $LOG_DIR/val_blindrna.log"
# grep "BLIND RNA SUMMARY" "$LOG_DIR/val_blindrna.log" -A 2 || echo "No summary found."

# echo "========================================================"
# echo "Starting Blind Disease Test (AIDS)..."
# echo "========================================================"
# python benchmark/train_blinddisease.py \
#     --disease "Acquired immunodeficiency syndrome (AIDS)" \
#     --epochs $EPOCHS \
#     --batch_size $BATCH \
#     --val_batch_size $BATCH \
#     --k $K \
#     --data-dir "benchmark/data" \
#     --out_dir "benchmark/ckpts/val_blinddisease_aids" \
#     2>&1 | tee "$LOG_DIR/val_blinddisease_aids.log"

# echo "Blind Disease Test Completed. Check $LOG_DIR/val_blinddisease_aids.log"
# grep "FINAL .* ENSEMBLE RESULTS" "$LOG_DIR/val_blinddisease_aids.log" -A 5 || echo "No summary found."

python benchmark/eval_ckpts.py --ckpt_dir ckpts/val_5fold --data-dir "benchmark/data" \
2>&1 | tee "$LOG_DIR/eval_ckpts.log"

echo "========================================================"
echo "All Benchmarks Completed."
curl -d "run_benchmarks.sh finished" ntfy.sh/inference-conda-setup
