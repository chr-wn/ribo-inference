# RNA Ligand Interaction Benchmarks

This directory contains the benchmarking suite for the RNA-Ligand interaction prediction model. It is designed to evaluate model performance using 5-fold cross-validation and blind tests on specific datasets.

## 1. Quick Start

### Step 1: Prepare Data
Before running benchmarks, you must prepare the dataset. This script processes the raw CSV, generates features (embeddings, graphs, structures), and consolidates them into the format required for training.

```bash
bash benchmark/prepare_data.sh
```
*   **Input**: `benchmark/rsim.csv`
*   **Output**: `benchmark/data/` (contains pickled features and processed IDs)
*   **Note**: This step may take some time as it generates embeddings using RiNALMo and UniMol, and secondary structures using MXFold2.

### Step 2: Run Benchmarks
Execute the full benchmark suite using the runner script.

```bash
bash benchmark/run_benchmarks.sh [--debug] [--gpu <id>]
```
*   **`--debug`**: Runs a fast, low-epoch version of the benchmarks for verification.
*   **`--gpu <id>`**: Specifies which GPU to use (default: 0).

## 2. Benchmark Components

The suite consists of three main evaluation protocols:

1.  **5-Fold Cross-Validation (`train_5fold.py`)**
    *   Splits the data into 5 folds.
    *   Trains an ensemble of models on 4 folds and validates on the 5th.
    *   Metrics: RMSE, Pearson Correlation.

2.  **Blind RNA Test (`train_blindrna.py`)**
    *   Evaluates generalization to unseen RNA scaffolds.
    *   Splits data such that validation RNAs have low sequence identity to training RNAs.

3.  **Blind Disease Test (`train_blinddisease.py`)**
    *   Evaluates performance on a specific disease area (e.g., AIDS).
    *   Holds out all data related to a specific disease for testing.

## 3. Directory Structure

*   **`data/`**: Processed data and features.
    *   `features/`: Intermediate feature files (RiNALMo, UniMol, etc.).
    *   `processed/`: (Legacy/Alias) pointers to consolidated files.
    *   `pssm_npy_5d/`: PSSM features.
*   **`results/`**: Checkpoints and predictions from benchmark runs.
*   **`logs_<TIMESTAMP>/`**: Detailed logs for each run (stdout/stderr).

## 4. Troubleshooting

*   **Missing Features**: If `prepare_data.sh` fails or is interrupted, simply run it again. It is designed to resume from where it left off.
*   **Conda Errors**: Ensure you have run `pipeline/setup.sh` to create the necessary Conda environments (`rinalmo`, `unimol`, `mxfold2`, `training`).
