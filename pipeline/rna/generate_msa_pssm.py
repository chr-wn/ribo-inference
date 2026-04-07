#!/usr/bin/env python3
"""
Generate MSA (Multiple Sequence Alignment) for RNA sequences using MMseqs2.
Then convert MSA to PSSM (Position-Specific Scoring Matrix).
Run within the mmseqs2_env conda environment.

Usage:
    conda activate mmseqs2_env
    python generate_msa_pssm.py --input unique_rnas.csv --output-msa msa/ --output-pssm pssm/
    
Note: This script requires an RNAcentral database or similar nucleotide database.
If the database is not available, it will skip MSA generation and create basic PSSM from sequence only.
"""

import argparse
import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
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


# RNA alphabet
RNA_ALPHABET = "ACGU"
ALPHABET_SIZE = len(RNA_ALPHABET)
CHAR_TO_IDX = {c: i for i, c in enumerate(RNA_ALPHABET)}


def sequence_to_onehot(seq: str) -> np.ndarray:
    """
    Convert RNA sequence to one-hot encoding.
    
    Args:
        seq: RNA sequence string
        
    Returns:
        L x 4 one-hot matrix (A, C, G, U)
    """
    L = len(seq)
    onehot = np.zeros((L, ALPHABET_SIZE), dtype=np.float32)
    
    for i, char in enumerate(seq.upper()):
        # Handle T as U
        if char == 'T':
            char = 'U'
        if char in CHAR_TO_IDX:
            onehot[i, CHAR_TO_IDX[char]] = 1.0
        else:
            # Unknown character - uniform distribution
            onehot[i, :] = 0.25
    
    return onehot


def msa_to_pssm(msa_sequences: List[str], query_seq: str, pseudocount: float = 0.5) -> np.ndarray:
    """
    Convert MSA to PSSM (Position-Specific Scoring Matrix).
    
    Args:
        msa_sequences: List of aligned sequences (including query)
        query_seq: Original query sequence
        pseudocount: Pseudocount for frequency estimation
        
    Returns:
        L x 4 PSSM matrix (log-odds scores)
    """
    if not msa_sequences:
        # No MSA available, return one-hot encoding
        return sequence_to_onehot(query_seq)
    
    L = len(msa_sequences[0])
    N = len(msa_sequences)
    
    # Count frequencies at each position
    counts = np.zeros((L, ALPHABET_SIZE), dtype=np.float32)
    
    for seq in msa_sequences:
        for i, char in enumerate(seq.upper()):
            if char == 'T':
                char = 'U'
            if char in CHAR_TO_IDX:
                counts[i, CHAR_TO_IDX[char]] += 1.0
            elif char not in ['-', '.', 'N']:
                # Unknown but not gap
                counts[i, :] += 0.25
    
    # Add pseudocounts and normalize to get frequencies
    counts += pseudocount
    freqs = counts / counts.sum(axis=1, keepdims=True)
    
    # Background frequency (uniform)
    background = np.full(ALPHABET_SIZE, 1.0 / ALPHABET_SIZE)
    
    # Log-odds scoring (PSSM)
    pssm = np.log2(freqs / background + 1e-10)
    
    # Alternatively, just return frequencies (often more useful for neural nets)
    return freqs


def parse_a3m(a3m_path: Path) -> List[str]:
    """Parse A3M format MSA file."""
    sequences = []
    current_seq = ""
    
    with open(a3m_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    # Remove lowercase (insertions) for standard alignment
                    clean_seq = ''.join(c for c in current_seq if c.isupper() or c in '.-')
                    sequences.append(clean_seq)
                current_seq = ""
            else:
                current_seq += line
    
    if current_seq:
        clean_seq = ''.join(c for c in current_seq if c.isupper() or c in '.-')
        sequences.append(clean_seq)
    
    return sequences


def run_mmseqs2_search(
    query_fasta: Path,
    database_path: Path,
    output_dir: Path,
    sensitivity: float = 7.5,
    num_iterations: int = 2,
    max_seqs: int = 1000,
    evalue: float = 1e-3,
) -> Optional[Path]:
    """
    Run MMseqs2 to find homologous sequences and create MSA.
    
    Returns:
        Path to A3M output file, or None if search failed
    """
    try:
        # Create query database
        query_db = output_dir / "query_db"
        result_db = output_dir / "result_db"
        result_a3m = output_dir / "result.a3m"
        
        # mmseqs createdb
        subprocess.run(
            ["mmseqs", "createdb", str(query_fasta), str(query_db)],
            check=True, capture_output=True
        )
        
        # mmseqs search
        subprocess.run([
            "mmseqs", "search",
            str(query_db), str(database_path), str(result_db), str(output_dir / "tmp"),
            "-s", str(sensitivity),
            "--num-iterations", str(num_iterations),
            "-e", str(evalue),
            "--max-seqs", str(max_seqs),
        ], check=True, capture_output=True)
        
        # Convert to A3M
        subprocess.run([
            "mmseqs", "result2msa",
            str(query_db), str(database_path), str(result_db), str(result_a3m),
            "--msa-format-mode", "6",  # A3M format
        ], check=True, capture_output=True)
        
        return result_a3m
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return None


def check_database_exists(db_path: str) -> bool:
    """Check if MMseqs2 database exists."""
    if not db_path:
        return False
    path = Path(db_path)
    # MMseqs2 databases consist of multiple files
    return path.exists() or (path.parent / (path.name + ".index")).exists()


def generate_pssm_basic(seq: str, pseudocount: float = 0.5) -> np.ndarray:
    """
    Generate a basic PSSM from sequence alone (no MSA).
    This is essentially just the one-hot encoding with slight smoothing.
    """
    L = len(seq)
    pssm = np.zeros((L, ALPHABET_SIZE), dtype=np.float32)
    
    for i, char in enumerate(seq.upper()):
        if char == 'T':
            char = 'U'
        if char in CHAR_TO_IDX:
            # High probability for observed base, low for others
            pssm[i, :] = pseudocount / (ALPHABET_SIZE - 1 + pseudocount)
            pssm[i, CHAR_TO_IDX[char]] = 1.0 - pseudocount
        else:
            # Unknown - uniform distribution
            pssm[i, :] = 0.25
    
    return pssm


def generate_msa_and_pssm(
    rna_data: pd.DataFrame,
    output_msa_dir: Path,
    output_pssm_dir: Path,
    database_path: Optional[str] = None,
    sensitivity: float = 7.5,
    num_iterations: int = 2,
    max_seqs: int = 1000,
    pseudocount: float = 0.5,
    overwrite: bool = False,
):
    """
    Generate MSA and PSSM for all RNA sequences.
    
    Args:
        rna_data: DataFrame with columns [rna_canonical_id, rna_sequence]
        output_msa_dir: Directory to save MSA files
        output_pssm_dir: Directory to save PSSM files
        database_path: Path to MMseqs2 database (e.g., RNAcentral)
        sensitivity: MMseqs2 search sensitivity
        num_iterations: Number of search iterations
        max_seqs: Maximum MSA sequences
        pseudocount: PSSM pseudocount
        overwrite: Whether to overwrite existing files
    """
    output_msa_dir = Path(output_msa_dir)
    output_pssm_dir = Path(output_pssm_dir)
    output_msa_dir.mkdir(parents=True, exist_ok=True)
    output_pssm_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if database is available
    has_database = check_database_exists(database_path) if database_path else False
    
    if not has_database:
        print("=" * 60)
        print("WARNING: No MMseqs2 database available.")
        print("MSA search will be skipped. PSSM will be generated from sequence only.")
        print("To enable MSA search, provide an RNAcentral database path.")
        print("=" * 60)
    
    # Filter sequences that need processing
    to_process = []
    rna_id_to_filename = {}  # Map original rna_id to safe filename
    for _, row in rna_data.iterrows():
        rna_id = row["rna_canonical_id"]
        safe_filename = sanitize_filename(rna_id)
        rna_id_to_filename[rna_id] = safe_filename
        pssm_file = output_pssm_dir / f"{safe_filename}.npy"
        if overwrite or not pssm_file.exists():
            to_process.append((rna_id, row["rna_sequence"]))
    
    # Save mapping for later use
    mapping_file = output_pssm_dir / "rna_id_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(rna_id_to_filename, f, indent=2)
    
    if not to_process:
        print("All PSSMs already exist. Use --overwrite to regenerate.")
        return
    
    print(f"Processing {len(to_process)} RNA sequences...")
    
    for rna_id, seq in tqdm(to_process, desc="Generating PSSM"):
        safe_filename = rna_id_to_filename[rna_id]
        pssm_file = output_pssm_dir / f"{safe_filename}.npy"
        msa_file = output_msa_dir / f"{safe_filename}.a3m"
        
        if has_database:
            # Try to run MMseqs2 search
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Write query FASTA
                query_fasta = temp_path / "query.fasta"
                with open(query_fasta, 'w') as f:
                    f.write(f">{rna_id}\n{seq}\n")
                
                # Run search
                result_a3m = run_mmseqs2_search(
                    query_fasta=query_fasta,
                    database_path=Path(database_path),
                    output_dir=temp_path,
                    sensitivity=sensitivity,
                    num_iterations=num_iterations,
                    max_seqs=max_seqs,
                )
                
                if result_a3m and result_a3m.exists():
                    # Parse MSA and generate PSSM
                    msa_sequences = parse_a3m(result_a3m)
                    pssm = msa_to_pssm(msa_sequences, seq, pseudocount)
                    
                    # Copy MSA file
                    shutil.copy(result_a3m, msa_file)
                else:
                    # Fall back to basic PSSM
                    pssm = generate_pssm_basic(seq, pseudocount)
        else:
            # No database - generate basic PSSM
            pssm = generate_pssm_basic(seq, pseudocount)
        
        # Save PSSM
        np.save(pssm_file, pssm.astype(np.float32))
    
    print(f"PSSM files saved to {output_pssm_dir}")
    if has_database:
        print(f"MSA files saved to {output_msa_dir}")


def generate_onehot(
    rna_data: pd.DataFrame,
    output_dir: Path,
    overwrite: bool = False,
):
    """
    Generate one-hot encodings for all RNA sequences.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    to_process = []
    for _, row in rna_data.iterrows():
        rna_id = row["rna_canonical_id"]
        out_file = output_dir / f"{rna_id}.npy"
        if overwrite or not out_file.exists():
            to_process.append((rna_id, row["rna_sequence"]))
    
    if not to_process:
        print("All one-hot encodings already exist. Use --overwrite to regenerate.")
        return
    
    print(f"Generating one-hot encodings for {len(to_process)} sequences...")
    
    for rna_id, seq in tqdm(to_process, desc="Generating one-hot"):
        out_file = output_dir / f"{rna_id}.npy"
        onehot = sequence_to_onehot(seq)
        np.save(out_file, onehot.astype(np.float32))
    
    print(f"One-hot encodings saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate MSA and PSSM for RNA sequences")
    parser.add_argument("--input", "-i", required=True, help="Input CSV with rna_canonical_id and rna_sequence")
    parser.add_argument("--output-msa", required=True, help="Output directory for MSA files")
    parser.add_argument("--output-pssm", required=True, help="Output directory for PSSM files")
    parser.add_argument("--output-onehot", help="Output directory for one-hot encodings")
    parser.add_argument("--database", help="Path to MMseqs2 database (e.g., RNAcentral)")
    parser.add_argument("--sensitivity", type=float, default=7.5, help="MMseqs2 search sensitivity")
    parser.add_argument("--num-iterations", type=int, default=2, help="Number of search iterations")
    parser.add_argument("--max-seqs", type=int, default=1000, help="Maximum MSA sequences")
    parser.add_argument("--pseudocount", type=float, default=0.5, help="PSSM pseudocount")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    
    args = parser.parse_args()
    
    # Load RNA data
    rna_data = pd.read_csv(args.input)
    
    # Validate required columns
    required_cols = ["rna_canonical_id", "rna_sequence"]
    missing = [c for c in required_cols if c not in rna_data.columns]
    if missing:
        print(f"Error: Missing columns in input: {missing}")
        sys.exit(1)
    
    # Generate MSA and PSSM
    generate_msa_and_pssm(
        rna_data=rna_data,
        output_msa_dir=args.output_msa,
        output_pssm_dir=args.output_pssm,
        database_path=args.database,
        sensitivity=args.sensitivity,
        num_iterations=args.num_iterations,
        max_seqs=args.max_seqs,
        pseudocount=args.pseudocount,
        overwrite=args.overwrite,
    )
    
    # Generate one-hot if requested
    if args.output_onehot:
        generate_onehot(
            rna_data=rna_data,
            output_dir=args.output_onehot,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
