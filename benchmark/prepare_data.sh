#!/bin/bash
set -e
# Reuses robust inference pipeline to gen features from benchmark/rsim.csv and organizes them into benchmark/data/ for training.

INPUT_CSV="benchmark/rsim.csv"
BENCHMARK_DATA_DIR="benchmark/data"
TEMP_FEATURES="$BENCHMARK_DATA_DIR/features"

mkdir -p "$BENCHMARK_DATA_DIR"
mkdir -p "$TEMP_FEATURES"

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
source "$CONDA_DIR/bin/activate" pipeline


echo "========================================================"
echo "Step 1: Generating Features from benchmark/rsim.csv..."
echo "========================================================"

python pipeline/generate_all_features.py \
    --input benchmark/rsim.csv \
    --output benchmark/data/features \
    --gpu 0 \
    --rinalmo-batch-size 8 \
    --unimol-batch-size 32

echo "========================================================"
echo "Step 2: Consolidating Features..."
echo "========================================================"
conda run -n training python pipeline/consolidate_features.py \
    --input benchmark/rsim.csv \
    --features benchmark/data/features \
    --output benchmark/data

echo "========================================================"
echo "Benchmark Data Preparation Complete."
echo "Data Location: $BENCHMARK_DATA_DIR"
echo "========================================================"
