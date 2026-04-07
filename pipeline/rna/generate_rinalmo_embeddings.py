#!/usr/bin/env python3
"""
Generate RiNALMo embeddings for RNA sequences.
Run within the rinalmo_env conda environment.

Usage:
    conda activate rinalmo_env
    python generate_rinalmo_embeddings.py --input unique_rnas.csv --output embeddings/
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import json

import numpy as np
import pandas as pd
import torch
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


def load_rinalmo_model(model_name: str = "giga-v1", device: str = "cuda"):
    """Load RiNALMo pretrained model."""
    try:
        from rinalmo.pretrained import get_pretrained_model
    except ImportError:
        raise ImportError(
            "RiNALMo not installed. Please run:\n"
            "  conda activate rinalmo_env\n"
            "  cd rinalmo && pip install ."
        )
    
    model, alphabet = get_pretrained_model(model_name=model_name)
    model = model.to(device=device)
    model.eval()
    return model, alphabet


def chunk_sequences(sequences: List[Tuple[str, str]], max_length: int = 4096) -> List[List[Tuple[str, str, int, int]]]:
    """
    Chunk sequences that exceed max_length into overlapping segments.
    Returns list of (rna_id, sequence_chunk, start_idx, end_idx).
    """
    chunked = []
    overlap = 256  # Overlap for merging
    
    for rna_id, seq in sequences:
        if len(seq) <= max_length:
            chunked.append([(rna_id, seq, 0, len(seq))])
        else:
            # Split into overlapping chunks
            chunks = []
            start = 0
            while start < len(seq):
                end = min(start + max_length, len(seq))
                chunks.append((rna_id, seq[start:end], start, end))
                if end >= len(seq):
                    break
                start = end - overlap
            chunked.append(chunks)
    
    return chunked


def generate_embeddings(
    rna_data: pd.DataFrame,
    output_dir: Path,
    model_name: str = "giga-v1",
    batch_size: int = 4,
    max_length: int = 4096,
    device: str = "cuda",
    overwrite: bool = False,
):
    """
    Generate RiNALMo embeddings for all RNA sequences.
    
    Args:
        rna_data: DataFrame with columns [rna_canonical_id, rna_sequence]
        output_dir: Directory to save embeddings (.npy files)
        model_name: RiNALMo model variant
        batch_size: Batch size for inference
        max_length: Maximum sequence length before chunking
        device: Device to run on
        overwrite: Whether to overwrite existing embeddings
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
        out_file = output_dir / f"{safe_filename}.npy"
        if overwrite or not out_file.exists():
            to_process.append((rna_id, row["rna_sequence"]))
    
    # Save mapping for later use
    mapping_file = output_dir / "rna_id_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(rna_id_to_filename, f, indent=2)
    
    if not to_process:
        print("All embeddings already exist. Use --overwrite to regenerate.")
        return
    
    print(f"Loading RiNALMo model ({model_name})...")
    model, alphabet = load_rinalmo_model(model_name, device)
    
    print(f"Processing {len(to_process)} RNA sequences...")
    
    # Process in batches
    for i in tqdm(range(0, len(to_process), batch_size), desc="Generating embeddings"):
        batch = to_process[i:i + batch_size]
        
        for rna_id, seq in batch:
            safe_filename = rna_id_to_filename[rna_id]
            out_file = output_dir / f"{safe_filename}.npy"
            
            try:
                # Handle long sequences by processing in chunks
                if len(seq) > max_length:
                    print(f"  Warning: {rna_id} has length {len(seq)}, processing in chunks...")
                    embeddings = process_long_sequence(model, alphabet, seq, max_length, device)
                else:
                    # Tokenize
                    tokens = torch.tensor(
                        alphabet.batch_tokenize([seq]), 
                        dtype=torch.int64, 
                        device=device
                    )
                    
                    # Generate embeddings
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        outputs = model(tokens)
                    
                    # Extract representation (L x D)
                    embeddings = outputs["representation"][0].cpu().numpy()
                    
                    # Remove special tokens if present (first and last)
                    # RiNALMo adds BOS and EOS tokens
                    if embeddings.shape[0] == len(seq) + 2:
                        embeddings = embeddings[1:-1]
                
                # Save as numpy array
                np.save(out_file, embeddings.astype(np.float16))  # Use float16 to save space
                
            except Exception as e:
                print(f"  Error processing {rna_id}: {e}")
                # Raise error to fail the batch/script if critical
                # For now, let's stop on error since we don't want partial results
                raise e
    
    print(f"Embeddings saved to {output_dir}")


def process_long_sequence(model, alphabet, seq: str, max_length: int, device: str) -> np.ndarray:
    """Process a sequence longer than max_length by chunking and averaging overlaps."""
    overlap = 256
    chunks = []
    positions = []
    
    start = 0
    while start < len(seq):
        end = min(start + max_length, len(seq))
        chunks.append(seq[start:end])
        positions.append((start, end))
        if end >= len(seq):
            break
        start = end - overlap
    
    # Get embeddings for each chunk
    all_embeddings = []
    for chunk in chunks:
        tokens = torch.tensor(
            alphabet.batch_tokenize([chunk]), 
            dtype=torch.int64, 
            device=device
        )
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model(tokens)
        
        emb = outputs["representation"][0].cpu().numpy()
        # Remove special tokens
        if emb.shape[0] == len(chunk) + 2:
            emb = emb[1:-1]
        all_embeddings.append(emb)
    
    # Merge overlapping regions by averaging
    d = all_embeddings[0].shape[-1]
    full_embeddings = np.zeros((len(seq), d), dtype=np.float32)
    counts = np.zeros(len(seq), dtype=np.float32)
    
    for emb, (start, end) in zip(all_embeddings, positions):
        full_embeddings[start:end] += emb
        counts[start:end] += 1
    
    full_embeddings /= counts[:, np.newaxis]
    return full_embeddings


def main():
    parser = argparse.ArgumentParser(description="Generate RiNALMo embeddings for RNA sequences")
    parser.add_argument("--input", "-i", required=True, help="Input CSV with rna_canonical_id and rna_sequence")
    parser.add_argument("--output", "-o", required=True, help="Output directory for embeddings")
    parser.add_argument("--model", default="giga-v1", choices=["giga-v1", "mega-v1", "micro-v1"],
                        help="RiNALMo model variant")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-length", type=int, default=4096, help="Max sequence length before chunking")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing embeddings")
    
    args = parser.parse_args()
    
    # Load RNA data
    rna_data = pd.read_csv(args.input)
    
    # Validate required columns
    required_cols = ["rna_canonical_id", "rna_sequence"]
    missing = [c for c in required_cols if c not in rna_data.columns]
    if missing:
        print(f"Error: Missing columns in input: {missing}")
        sys.exit(1)
    
    generate_embeddings(
        rna_data=rna_data,
        output_dir=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
