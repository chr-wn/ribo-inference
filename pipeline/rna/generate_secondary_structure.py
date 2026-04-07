#!/usr/bin/env python3
"""
Generate RNA secondary structure predictions using mxfold2.
Run within the mxfold2_env conda environment.

Usage:
    conda activate mxfold2_env
    python generate_secondary_structure.py --input unique_rnas.csv --output secondary/
"""

import argparse
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple
import json

import numpy as np
import pandas as pd
from tqdm import tqdm


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe for use as a filename."""
    # Replace problematic characters with underscores
    import re
    # Keep alphanumeric, dots, hyphens, underscores, and spaces
    safe_name = re.sub(r'[^\w\.\-\s]', '_', name)
    # Replace spaces with underscores
    safe_name = safe_name.replace(' ', '_')
    # Remove multiple consecutive underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    # Remove leading/trailing underscores
    safe_name = safe_name.strip('_')
    return safe_name


def parse_dot_bracket(structure: str) -> List[Tuple[int, int]]:
    """
    Parse dot-bracket notation to list of base pairs.
    
    Returns:
        List of (i, j) tuples representing base pairs (0-indexed)
    """
    stack = []
    pairs = []
    
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                pairs.append((j, i))
    
    return pairs


def dot_bracket_to_adjacency(structure: str) -> np.ndarray:
    """
    Convert dot-bracket notation to adjacency matrix.
    
    Args:
        structure: Dot-bracket string
        
    Returns:
        L x L binary adjacency matrix for secondary structure edges
    """
    L = len(structure)
    adj = np.zeros((L, L), dtype=np.int8)
    
    pairs = parse_dot_bracket(structure)
    for i, j in pairs:
        adj[i, j] = 1
        adj[j, i] = 1
    
    return adj


def dot_bracket_to_edges(structure: str) -> np.ndarray:
    """
    Convert dot-bracket notation to edge list.
    
    Returns:
        N x 2 array of edges (pairs of indices, 0-indexed)
    """
    pairs = parse_dot_bracket(structure)
    if not pairs:
        return np.empty((0, 2), dtype=np.int32)
    return np.array(pairs, dtype=np.int32)


def run_mxfold2(sequences: List[Tuple[str, str]], temp_dir: Path) -> Dict[str, str]:
    """
    Run mxfold2 on a list of sequences.
    
    Args:
        sequences: List of (rna_id, sequence) tuples
        temp_dir: Temporary directory for FASTA files
        
    Returns:
        Dict mapping rna_id to dot-bracket structure
    """
    # Write sequences to FASTA file
    fasta_path = temp_dir / "input.fasta"
    with open(fasta_path, "w") as f:
        for rna_id, seq in sequences:
            f.write(f">{rna_id}\n{seq}\n")
    
    # Run mxfold2
    try:
        result = subprocess.run(
            ["mxfold2", "predict", str(fasta_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError("mxfold2 not found. Please install it in mxfold2_env.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"mxfold2 failed: {e.stderr}")
    
    # Parse output
    structures = {}
    lines = result.stdout.strip().split("\n")
    current_id = None
    
    for line in lines:
        if line.startswith(">"):
            current_id = line[1:].strip()
        elif current_id and line and not line[0].isalpha():
            # Structure line (starts with dot or bracket)
            # Extract structure (before any energy annotation)
            parts = line.split()
            if parts:
                struct = parts[0]
                # Only keep valid characters
                struct = ''.join(c for c in struct if c in '.()[]{}')
                structures[current_id] = struct
                current_id = None
        elif current_id and line.startswith(('A', 'C', 'G', 'U', 'T')):
            # This is the sequence line, skip it
            pass
    
    return structures


def generate_secondary_structures(
    rna_data: pd.DataFrame,
    output_dir: Path,
    batch_size: int = 32,
    overwrite: bool = False,
):
    """
    Generate secondary structure predictions for all RNA sequences.
    
    Args:
        rna_data: DataFrame with columns [rna_canonical_id, rna_sequence]
        output_dir: Directory to save outputs
        batch_size: Number of sequences to process at once
        overwrite: Whether to overwrite existing predictions
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter sequences that need processing
    to_process = []
    rna_id_to_filename = {}  # Map original rna_id to safe filename
    for _, row in rna_data.iterrows():
        rna_id = row["rna_canonical_id"]
        safe_filename = sanitize_filename(rna_id)
        rna_id_to_filename[rna_id] = safe_filename
        out_file = output_dir / f"{safe_filename}.json"
        if overwrite or not out_file.exists():
            to_process.append((rna_id, row["rna_sequence"]))
    
    # Save mapping for later use
    mapping_file = output_dir / "rna_id_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(rna_id_to_filename, f, indent=2)
    
    if not to_process:
        print("All secondary structures already exist. Use --overwrite to regenerate.")
        return
    
    print(f"Processing {len(to_process)} RNA sequences with mxfold2...")
    
    # Process in batches
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        for i in tqdm(range(0, len(to_process), batch_size), desc="Predicting structures"):
            batch = to_process[i:i + batch_size]
            
            try:
                structures = run_mxfold2(batch, temp_path)
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                raise e
            
            # Save results
            for rna_id, seq in batch:
                safe_filename = rna_id_to_filename[rna_id]
                out_file = output_dir / f"{safe_filename}.json"
                
                if rna_id in structures:
                    struct = structures[rna_id]
                    
                    # Validate length matches
                    if len(struct) != len(seq):
                        print(f"  Warning: {rna_id} structure length mismatch: {len(struct)} vs {len(seq)}")
                        # Pad or truncate
                        if len(struct) < len(seq):
                            struct = struct + '.' * (len(seq) - len(struct))
                        else:
                            struct = struct[:len(seq)]
                    
                    # Convert to edges
                    edges = dot_bracket_to_edges(struct)
                    
                    result = {
                        "rna_id": rna_id,
                        "length": len(seq),
                        "dot_bracket": struct,
                        "base_pairs": edges.tolist(),
                        "num_pairs": len(edges),
                    }
                    
                    with open(out_file, "w") as f:
                        json.dump(result, f, indent=2)
                else:
                    print(f"  Warning: No structure returned for {rna_id}")
    
    print(f"Secondary structures saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate RNA secondary structures using mxfold2")
    parser.add_argument("--input", "-i", required=True, help="Input CSV with rna_canonical_id and rna_sequence")
    parser.add_argument("--output", "-o", required=True, help="Output directory for structures")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for mxfold2")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing structures")
    
    args = parser.parse_args()
    
    # Load RNA data
    rna_data = pd.read_csv(args.input)
    
    # Validate required columns
    required_cols = ["rna_canonical_id", "rna_sequence"]
    missing = [c for c in required_cols if c not in rna_data.columns]
    if missing:
        print(f"Error: Missing columns in input: {missing}")
        sys.exit(1)
    
    generate_secondary_structures(
        rna_data=rna_data,
        output_dir=args.output,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
