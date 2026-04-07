"""
Configuration for the RNA-ligand binding affinity prediction pipeline.
Centralized paths and settings for input generation.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Base directory for the inference pipeline
PIPELINE_DIR = Path(__file__).parent
INFERENCE_DIR = PIPELINE_DIR.parent
DATA_DIR = INFERENCE_DIR / "data"

# Input data paths
UNION_DATASET_PATH = DATA_DIR / "union_dataset.csv"
UNIQUE_RNAS_PATH = DATA_DIR / "unique_rnas.csv"
UNIQUE_MOLS_PATH = DATA_DIR / "unique_molecules.csv"

# Output directories for processed features
PROCESSED_DIR = DATA_DIR / "processed"
RNA_FEATURES_DIR = PROCESSED_DIR / "rna"
MOL_FEATURES_DIR = PROCESSED_DIR / "mol"

# Subdirectories for specific feature types
RNA_EMBEDDINGS_DIR = RNA_FEATURES_DIR / "embeddings"     # RiNALMo embeddings
RNA_SECONDARY_DIR = RNA_FEATURES_DIR / "secondary"       # mxfold2 secondary structure
RNA_MSA_DIR = RNA_FEATURES_DIR / "msa"                   # MMseqs2 MSA results
RNA_PSSM_DIR = RNA_FEATURES_DIR / "pssm"                 # PSSM matrices
RNA_ONEHOT_DIR = RNA_FEATURES_DIR / "onehot"             # One-hot encodings

MOL_EMBEDDINGS_DIR = MOL_FEATURES_DIR / "embeddings"     # UniMol embeddings
MOL_ONEHOT_DIR = MOL_FEATURES_DIR / "onehot"             # One-hot encodings
MOL_GRAPH_DIR = MOL_FEATURES_DIR / "graph"               # Graph representations

# Temporary directories
TEMP_DIR = PROCESSED_DIR / "temp"
FASTA_TEMP_DIR = TEMP_DIR / "fasta"

# Conda environment names for each tool
CONDA_ENVS = {
    "rinalmo": "rinalmo_env",
    "mxfold2": "mxfold2_env",
    "mmseqs2": "mmseqs2_env",
    "unimol": "unimol_env",
}

# Default model configurations
@dataclass
class RiNALMoConfig:
    """Configuration for RiNALMo embeddings."""
    model_name: str = "giga-v1"  # Options: giga-v1, mega-v1, micro-v1
    batch_size: int = 4
    max_length: int = 4096  # Maximum sequence length
    device: str = "cuda"  # cuda or cpu


@dataclass
class MxFold2Config:
    """Configuration for mxfold2 secondary structure prediction."""
    # Use default model (no config needed for basic usage)
    pass


@dataclass
class MMseqs2Config:
    """Configuration for MMseqs2 MSA search."""
    database: str = "rnacentral"  # Which database to search against
    sensitivity: float = 7.5  # Search sensitivity (1-7.5)
    num_iterations: int = 2  # Number of search iterations
    max_seqs: int = 1000  # Maximum number of sequences in MSA
    evalue: float = 1e-3  # E-value threshold


@dataclass
class UniMolConfig:
    """Configuration for UniMol embeddings."""
    add_hydrogens: bool = True  # Add explicit hydrogens
    remove_hs: bool = False  # Don't remove hydrogens for embeddings
    data_type: str = "molecule"


@dataclass
class PSSMConfig:
    """Configuration for PSSM generation from MSA."""
    alphabet: str = "ACGU"  # RNA alphabet
    pseudocount: float = 0.5  # Pseudocount for frequency estimation


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    rinalmo: RiNALMoConfig = field(default_factory=RiNALMoConfig)
    mxfold2: MxFold2Config = field(default_factory=MxFold2Config)
    mmseqs2: MMseqs2Config = field(default_factory=MMseqs2Config)
    unimol: UniMolConfig = field(default_factory=UniMolConfig)
    pssm: PSSMConfig = field(default_factory=PSSMConfig)
    
    # Processing options
    overwrite: bool = False  # Overwrite existing features
    num_workers: int = 4  # Number of parallel workers
    device: str = "cuda"  # Default device


def ensure_directories():
    """Create all required directories."""
    dirs = [
        PROCESSED_DIR,
        RNA_FEATURES_DIR,
        MOL_FEATURES_DIR,
        RNA_EMBEDDINGS_DIR,
        RNA_SECONDARY_DIR,
        RNA_MSA_DIR,
        RNA_PSSM_DIR,
        RNA_ONEHOT_DIR,
        MOL_EMBEDDINGS_DIR,
        MOL_ONEHOT_DIR,
        MOL_GRAPH_DIR,
        TEMP_DIR,
        FASTA_TEMP_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print("All directories created successfully.")
    print(f"Pipeline directory: {PIPELINE_DIR}")
    print(f"Data directory: {DATA_DIR}")
