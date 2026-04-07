#!/usr/bin/env python3
"""
Generates all features for RNA-ligand binding prediction:
- RiNALMo embeddings (RNA)
- mxfold2 secondary structure (RNA)
- UniMol embeddings (Mol)
- Molecular graph features (Mol)

python generate_all_features.py --input data/union_dataset.csv --output features/
optional flags: --rna-only, --mol-only, --skip-[modelname] (see argparse for details)
"""

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd
import os


def run_command(cmd: list, env: dict = None, check: bool = False):
    """Run a command and stream output. Raises exception if check=True and command fails."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    
    print(f"Running command: {' '.join(cmd)}")
    sys.stdout.flush()
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=full_env,
        bufsize=1, # Line buffered
        universal_newlines=True 
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    if check and process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)
        
    return process.returncode


def generate_rinalmo_embeddings(df: pd.DataFrame, output_dir: Path, gpu_id: int = 0, batch_size: int = 4, overwrite: bool = False):
    """Generate RiNALMo embeddings using the rinalmo conda environment."""
    print("\n" + "="*60)
    print("Generating RiNALMo Embeddings")
    print("="*60)
    
    # Get unique RNAs
    rna_df = df[['rna_canonical_id', 'rna_sequence']].drop_duplicates()
    
    embeddings_dir = output_dir / "rinalmo"
    
    # Check if already generated
    # Save to temp file
    temp_csv = output_dir / "temp_rna_data.csv"
    rna_df.to_csv(temp_csv, index=False)
    
    # Run the generation script
    script_path = Path(__file__).parent / "rna" / "generate_rinalmo_embeddings.py"
    cmd = [
        "conda", "run", "-n", "rinalmo", "python", "-u", str(script_path),
        "--input", str(temp_csv),
        "--output", str(embeddings_dir),
        "--batch-size", str(batch_size),
    ]
    
    env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    result = run_command(cmd, env)
    
    # Clean up temp file
    temp_csv.unlink(missing_ok=True)
    
    if result == 0:
        print(f"✅ RiNALMo embeddings saved to {embeddings_dir}")
    else:
        print(f"❌ RiNALMo embedding generation failed")
    
    return result


def generate_secondary_structure(df: pd.DataFrame, output_dir: Path, overwrite: bool = False):
    """Generate mxfold2 secondary structures using the mxfold2 conda environment."""
    print("\n" + "="*60)
    print("Generating mxfold2 Secondary Structures")
    print("="*60)
    
    # Get unique RNAs
    rna_df = df[['rna_canonical_id', 'rna_sequence']].drop_duplicates()
    
    ss_dir = output_dir / "mxfold2"
    
    # Check if already generated

    # Save to temp file
    temp_csv = output_dir / "temp_rna_data.csv"
    rna_df.to_csv(temp_csv, index=False)
    
    # Run the generation script
    script_path = Path(__file__).parent / "rna" / "generate_secondary_structure.py"
    cmd = [
        "conda", "run", "-n", "mxfold2", "python", "-u", str(script_path),
        "--input", str(temp_csv),
        "--output", str(ss_dir),
    ]
    
    result = run_command(cmd)
    
    # Clean up temp file
    temp_csv.unlink(missing_ok=True)
    
    if result == 0:
        print(f"✅ Secondary structures saved to {ss_dir}")
    else:
        print(f"❌ Secondary structure generation failed")
    
    return result


def generate_unimol_embeddings(df: pd.DataFrame, output_dir: Path, gpu_id: int = 0, batch_size: int = 32, overwrite: bool = False):
    """Generate UniMol embeddings using the unimol conda environment."""
    print("\n" + "="*60)
    print("Generating UniMol Embeddings")
    print("="*60)
    
    # Get unique molecules
    mol_df = df[['mol_canonical_id', 'smiles']].drop_duplicates()
    
    # Save to temp file
    temp_csv = output_dir / "temp_mol_data.csv"
    mol_df.to_csv(temp_csv, index=False)
    
    embeddings_dir = output_dir / "unimol"
    
    # Check if already generated

    # Run the generation script
    script_path = Path(__file__).parent / "mol" / "generate_unimol_embeddings.py"
    cmd = [
        "conda", "run", "-n", "unimol", "python", "-u", str(script_path),
        "--input", str(temp_csv),
        "--output", str(embeddings_dir),
        "--batch-size", str(batch_size),
    ]
    
    env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    result = run_command(cmd, env)
    
    # Clean up temp file
    temp_csv.unlink(missing_ok=True)
    
    if result == 0:
        print(f"✅ UniMol embeddings saved to {embeddings_dir}")
    else:
        print(f"❌ UniMol embedding generation failed")
    
    return result


def generate_rna_onehot(df: pd.DataFrame, output_dir: Path, overwrite: bool = False):
    """Generate RNA one-hot encodings."""
    print("\n" + "="*60)
    print("Generating RNA One-Hot Encodings")
    print("="*60)
    
    import numpy as np
    
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
    
    RNA_ALPHABET = "ACGU"
    CHAR_TO_IDX = {c: i for i, c in enumerate(RNA_ALPHABET)}
    
    def sequence_to_onehot(seq: str) -> np.ndarray:
        L = len(seq)
        onehot = np.zeros((L, 4), dtype=np.float32)
        for i, char in enumerate(seq.upper()):
            if char == 'T':
                char = 'U'
            if char in CHAR_TO_IDX:
                onehot[i, CHAR_TO_IDX[char]] = 1.0
            else:
                onehot[i, :] = 0.25  # Unknown
        return onehot
    
    # Get unique RNAs
    rna_df = df[['rna_canonical_id', 'rna_sequence']].drop_duplicates()
    
    onehot_dir = output_dir / "rna_onehot"
    onehot_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mapping for filename sanitization
    rna_id_to_filename = {}
    for rna_id in rna_df['rna_canonical_id'].unique():
        rna_id_to_filename[rna_id] = sanitize_filename(rna_id)
    
    # Save mapping for later use
    mapping_file = onehot_dir / "rna_id_mapping.json"
    import json
    with open(mapping_file, 'w') as f:
        json.dump(rna_id_to_filename, f, indent=2)
    
    success_count = 0
    for _, row in rna_df.iterrows():
        rna_id = row['rna_canonical_id']
        seq = row['rna_sequence']
        safe_filename = rna_id_to_filename[rna_id]
        out_file = onehot_dir / f"{safe_filename}.npy"
        
        try:
            onehot = sequence_to_onehot(seq)
            np.save(out_file, onehot)
            success_count += 1
        except Exception as e:
            print(f"  Error processing {rna_id}: {e}")
    
    print(f"RNA one-hot encodings saved to {onehot_dir} ({success_count}/{len(rna_df)} files)")
    return 0 if success_count == len(rna_df) else 1


def generate_mol_features(df: pd.DataFrame, output_dir: Path, overwrite: bool = False):
    """Generate molecular one-hot and graph features."""
    print("\n" + "="*60)
    print("Generating Molecular Graph Features")
    print("="*60)
    
    # Get unique molecules
    mol_df = df[['mol_canonical_id', 'smiles']].drop_duplicates()
    
    # Save to temp file
    temp_csv = output_dir / "temp_mol_data.csv"
    mol_df.to_csv(temp_csv, index=False)
    
    onehot_dir = output_dir / "mol_onehot"
    graph_dir = output_dir / "mol_graph"
    
    
    script_path = Path(__file__).parent / "mol" / "generate_mol_features.py"
    cmd = [
        "conda", "run", "-n", "unimol", "python", "-u", str(script_path),
        "--input", str(temp_csv),
        "--output-onehot", str(onehot_dir),
        "--output-graph", str(graph_dir),
    ]
    
    result = run_command(cmd)
    
    temp_csv.unlink(missing_ok=True)
    
    if result == 0:
        print(f"Molecular features saved to {onehot_dir} and {graph_dir}")
    else:
        print(f"Molecular feature generation failed")
    
    return result


def generate_pssm(df: pd.DataFrame, output_dir: Path, database_path: str = None):
    """Generate PSSM (Position-Specific Scoring Matrix) using MMseqs2 and RNAcentral."""
    print("\n" + "="*60)
    print("Generating PSSM (MSA-based)")
    print("="*60)
    
    if database_path is None:
        default_db = Path(__file__).parent.parent / "databases" / "rnacentral_db"
        if default_db.exists() or (default_db.parent / "rnacentral_db.index").exists():
            database_path = str(default_db)
        else:
            print("⚠️  RNAcentral database not found. Generating basic PSSM from sequence only.")
            print(f"   Expected path: {default_db}")
            print("   Run: databases/download_rnacentral.sh to set up the database")
            database_path = None
    
    rna_df = df[['rna_canonical_id', 'rna_sequence']].drop_duplicates()
    
    temp_csv = output_dir / "temp_rna_pssm.csv"
    rna_df.to_csv(temp_csv, index=False)
    
    pssm_dir = output_dir / "pssm"
    msa_dir = output_dir / "msa"
    
    script_path = Path(__file__).parent / "rna" / "generate_msa_pssm.py"
    cmd = [
        "conda", "run", "-n", "mmseqs2", "python", "-u", str(script_path),
        "--input", str(temp_csv),
        "--output-pssm", str(pssm_dir),
        "--output-msa", str(msa_dir),
    ]
    
    if database_path:
        cmd.extend(["--database", database_path])
    
    result = run_command(cmd)
    
    temp_csv.unlink(missing_ok=True)
    
    if result == 0:
        print(f"✅ PSSM saved to {pssm_dir}")
    else:
        print(f"❌ PSSM generation failed")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate all features for RNA-ligand binding")
    parser.add_argument("--input", "-i", required=True, help="Input CSV with union dataset")
    parser.add_argument("--output", "-o", required=True, help="Output directory for all features")
    parser.add_argument("--gpu", type=int, default=1, help="GPU ID to use (default: 1)")
    parser.add_argument("--rinalmo-batch-size", type=int, default=4, help="RiNALMo batch size")
    parser.add_argument("--unimol-batch-size", type=int, default=32, help="UniMol batch size")
    parser.add_argument("--rna-only", action="store_true", help="Generate only RNA features")
    parser.add_argument("--mol-only", action="store_true", help="Generate only molecule features")
    parser.add_argument("--skip-rinalmo", action="store_true", help="Skip RiNALMo embeddings")
    parser.add_argument("--skip-mxfold2", action="store_true", help="Skip mxfold2 secondary structure")
    parser.add_argument("--skip-pssm", action="store_true", help="Skip PSSM generation")
    parser.add_argument("--database", type=str, default=None, help="Path to RNAcentral MMseqs2 database")
    parser.add_argument("--skip-unimol", action="store_true", help="Skip UniMol embeddings")
    parser.add_argument("--skip-mol-features", action="store_true", help="Skip molecular graph features")
    parser.add_argument("--force", action="store_true", help="Force regeneration of features")
    args = parser.parse_args()
    
    print(f"Loading dataset from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"  Total entries: {len(df)}")
    print(f"  Unique RNAs: {df['rna_canonical_id'].nunique()}")
    print(f"  Unique molecules: {df['mol_canonical_id'].nunique()}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    if not args.mol_only:
        if not args.skip_rinalmo:
            results['rinalmo'] = generate_rinalmo_embeddings(
                df, output_dir, args.gpu, args.rinalmo_batch_size, args.force
            )
        
        if not args.skip_mxfold2:
            results['mxfold2'] = generate_secondary_structure(df, output_dir, args.force)
        
        if not args.skip_pssm:
            results['pssm'] = generate_pssm(df, output_dir, args.database)
        
        results['rna_onehot'] = generate_rna_onehot(df, output_dir, args.force)
    
    if not args.rna_only:
        if not args.skip_unimol:
            results['unimol'] = generate_unimol_embeddings(
                df, output_dir, args.gpu, args.unimol_batch_size, args.force
            )
        
        if not args.skip_mol_features:
            results['mol_features'] = generate_mol_features(df, output_dir, args.force)
    
    print("\n" + "="*60)
    print("FEATURE GENERATION SUMMARY")
    print("="*60)
    
    for name, result in results.items():
        status = "SUCCESS" if result == 0 else "FAILED"
        print(f"  {name}: {status}")
    
    all_success = all(r == 0 for r in results.values())
    
    if all_success:
        print("\nAll features generated successfully!")
        print(f"\nFeature directories:")
        for name in results.keys():
            feature_dir = output_dir / name
            if feature_dir.exists():
                file_count = len(list(feature_dir.rglob('*.npy')))
                print(f"  {feature_dir}: {file_count} files")
    else:
        print("\nSome features failed to generate. Check errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
