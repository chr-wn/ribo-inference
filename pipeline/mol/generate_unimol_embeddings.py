#!/usr/bin/env python3
"""
Generate UniMol embeddings for molecules.
Run within the unimol_env conda environment.

Usage:
    conda activate unimol_env
    python generate_unimol_embeddings.py --input unique_molecules.csv --output embeddings/
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_unimol_repr(remove_hs: bool = False):
    """Load UniMol representation model."""
    try:
        from unimol_tools import UniMolRepr
    except ImportError:
        raise ImportError(
            "unimol_tools not installed. Please run:\n"
            "  conda activate unimol_env\n"
            "  pip install unimol_tools huggingface_hub"
        )
    
    clf = UniMolRepr(data_type='molecule', remove_hs=remove_hs)
    return clf


def generate_embeddings(
    mol_data: pd.DataFrame,
    output_dir: Path,
    remove_hs: bool = False,
    batch_size: int = 32,
    overwrite: bool = False,
):
    """
    Generate UniMol embeddings for all molecules.
    
    Args:
        mol_data: DataFrame with columns [mol_canonical_id, smiles]
        output_dir: Directory to save embeddings (.npy files)
        remove_hs: Whether to remove hydrogens
        batch_size: Batch size for inference
        overwrite: Whether to overwrite existing embeddings
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate directory for atomic-level embeddings
    cls_dir = output_dir / "cls"
    atomic_dir = output_dir / "atomic"
    cls_dir.mkdir(exist_ok=True)
    atomic_dir.mkdir(exist_ok=True)
    
    # Filter molecules that need processing
    to_process = []
    for _, row in mol_data.iterrows():
        mol_id = row["mol_canonical_id"]
        cls_file = cls_dir / f"{mol_id}.npy"
        if overwrite or not cls_file.exists():
            to_process.append((mol_id, row["smiles"]))
    
    if not to_process:
        print("All embeddings already exist. Use --overwrite to regenerate.")
        return
    
    print(f"Loading UniMol model...")
    clf = load_unimol_repr(remove_hs=remove_hs)
    
    print(f"Processing {len(to_process)} molecules...")
    
    # Process in batches
    for i in tqdm(range(0, len(to_process), batch_size), desc="Generating embeddings"):
        batch = to_process[i:i + batch_size]
        mol_ids = [x[0] for x in batch]
        smiles_list = [x[1] for x in batch]
        
        try:
            # Get representations
            unimol_repr = clf.get_repr(smiles_list, return_atomic_reprs=True)
            
            # Save each molecule's embeddings
            for j, mol_id in enumerate(mol_ids):
                cls_file = cls_dir / f"{mol_id}.npy"
                atomic_file = atomic_dir / f"{mol_id}.npy"
                
                # CLS token (molecule-level) embedding
                cls_repr = np.array(unimol_repr['cls_repr'][j], dtype=np.float16)
                np.save(cls_file, cls_repr)
                
                # Atomic-level embeddings
                atomic_repr = np.array(unimol_repr['atomic_reprs'][j], dtype=np.float16)
                np.save(atomic_file, atomic_repr)
                
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            # Try processing individually
            for mol_id, smiles in batch:
                try:
                    repr_single = clf.get_repr([smiles], return_atomic_reprs=True)
                    
                    cls_file = cls_dir / f"{mol_id}.npy"
                    atomic_file = atomic_dir / f"{mol_id}.npy"
                    
                    cls_repr = np.array(repr_single['cls_repr'][0], dtype=np.float16)
                    np.save(cls_file, cls_repr)
                    
                    atomic_repr = np.array(repr_single['atomic_reprs'][0], dtype=np.float16)
                    np.save(atomic_file, atomic_repr)
                    
                except Exception as e2:
                    print(f"  Failed to process {mol_id}: {e2}")
    
    print(f"CLS embeddings saved to {cls_dir}")
    print(f"Atomic embeddings saved to {atomic_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate UniMol embeddings for molecules")
    parser.add_argument("--input", "-i", required=True, help="Input CSV with mol_canonical_id and smiles")
    parser.add_argument("--output", "-o", required=True, help="Output directory for embeddings")
    parser.add_argument("--remove-hs", action="store_true", help="Remove hydrogens before embedding")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing embeddings")
    
    args = parser.parse_args()
    
    # Load molecule data
    mol_data = pd.read_csv(args.input)
    
    # Validate required columns
    required_cols = ["mol_canonical_id", "smiles"]
    missing = [c for c in required_cols if c not in mol_data.columns]
    if missing:
        print(f"Error: Missing columns in input: {missing}")
        sys.exit(1)
    
    generate_embeddings(
        mol_data=mol_data,
        output_dir=args.output,
        remove_hs=args.remove_hs,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
