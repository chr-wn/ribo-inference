# RiboLead: RNA-Ligand Binding Affinity Prediction

A robust, end-to-end pipeline for predicting the binding affinity (pKd) between RNA sequences and small molecules. Designed for clinical application and virtual screening.

## 1. Full setup & training
```bash
bash pipeline/run_full_pipeline.sh
```
This will:
0. Set up conda environments if needed
1. Generate features for all data in `data/union_dataset_final.csv`.
2. Consolidate features.
3. Train an ensemble of 5 models.
4. Save models to `models/inference_ensemble`.

## 2. Run inference (to screen drug candidates)
To predict affinities for a new set of RNA-Molecule pairs:
```bash
bash pipeline/run_inference.sh \
    --input your_data.csv \
    --output results/output_dir \
    --model-dir models/inference_ensemble \
    --norm-mean 4.678260 \
    --norm-std 2.144659
```
Provide `--norm-mean` and `--norm-std` matching the training data (values above are for the standard trained model). 
Without these, predictions will be raw Z-scores (or zeros if input data has no labels).

Input Format (`your_data.csv`):
Must contain at least these columns:
- `rna_sequence`: The RNA nucleotide sequence in FASTA.
- `smiles`: The molecule SMILES string.
- `rna_canonical_id`, `mol_canonical_id` for tracking (optional but helpful).

Output:
`results/output_dir/predictions.csv`: Contains `pred_pKd` and `uncertainty_sigma`.

---

## Directory Structure

```
### Documentation
> **See [pipeline/README.md](pipeline/README.md) for detailed documentation on scripts, data flow, normalization, and advanced setup.**

### Core Scripts
│   ├── inference.sh        # Entry point for screening/inference
│   ├── run_full_pipeline.sh# Entry point for training
│   ├── predict_inference.py# Inference logic
│   ├── train_inference.py  # Training logic
│   └── ...                 # Feature generation scripts (RNA/Mol)
├── data/                   # Data storage
│   ├── union_dataset_final.csv # Main training data
│   ├── processed/          # Generated features (embeddings, graphs)
│   └── consolidated/       # Packed features for high-speed loading
├── models/                 # Saved model weights
│   ├── inference_ensemble/ # Production models
│   └── dev_test/           # Temporary dev models
├── RMPred.py               # Model Architecture
├── RNAdataset.py           # Data Loading
├── config.py               # Global Configuration
└── archive_v1/             # Archived/Legacy examples
```

## Feature embeddings
The pipeline automatically generates and integrates:
1.  **RNA**:
    *   **RiNALMo Embeddings** (LLM-based)
    *   **MXFold2** (Secondary Structure Graph)
    *   **PSSM** (Evolutionary info via MMseqs2)
2.  **Molecule**:
    *   **UniMol Embeddings** (3D Conformer-based)
    *   **RDKit Graph** (2D Structure)