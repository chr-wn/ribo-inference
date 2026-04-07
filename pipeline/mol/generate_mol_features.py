#!/usr/bin/env python3
"""
Generate one-hot encodings and graph representations for molecules.
This can run in either unimol_env or a basic Python environment with RDKit.

Usage:
    python generate_mol_features.py --input unique_molecules.csv --output-onehot onehot/ --output-graph graph/
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

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    raise ImportError("RDKit is required. Install with: pip install rdkit")


# Atom vocabulary for one-hot encoding
ATOM_VOCAB = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'H', 'B', 'Si', 'Se', 'Other']
ATOM_TO_IDX = {a: i for i, a in enumerate(ATOM_VOCAB)}
NUM_ATOM_TYPES = len(ATOM_VOCAB)

# Bond types for edge features
BOND_TYPES = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}
NUM_BOND_TYPES = len(BOND_TYPES)


def smiles_to_mol(smiles: str, add_hs: bool = True) -> Optional[Chem.Mol]:
    """
    Convert SMILES to RDKit molecule.
    
    Args:
        smiles: SMILES string
        add_hs: Whether to add explicit hydrogens
        
    Returns:
        RDKit Mol object or None if invalid
    """
    if not smiles or not isinstance(smiles, str) or smiles.strip() in ['', '-', 'nan', 'NA', 'N/A']:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    if add_hs:
        mol = Chem.AddHs(mol)
    
    return mol


def mol_to_onehot(mol: Chem.Mol) -> np.ndarray:
    """
    Convert molecule to atom one-hot encoding.
    
    Args:
        mol: RDKit molecule
        
    Returns:
        N x NUM_ATOM_TYPES one-hot matrix
    """
    num_atoms = mol.GetNumAtoms()
    onehot = np.zeros((num_atoms, NUM_ATOM_TYPES), dtype=np.float32)
    
    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        idx = ATOM_TO_IDX.get(symbol, ATOM_TO_IDX['Other'])
        onehot[i, idx] = 1.0
    
    return onehot


def mol_to_edges(mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract edges (bonds) from molecule.
    
    Args:
        mol: RDKit molecule
        
    Returns:
        Tuple of:
            - edges: E x 2 array of atom indices
            - edge_types: E array of bond type indices
    """
    edges = []
    edge_types = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = BOND_TYPES.get(bond.GetBondType(), 0)
        
        # Add both directions for undirected graph
        edges.append([i, j])
        edges.append([j, i])
        edge_types.append(bond_type)
        edge_types.append(bond_type)
    
    if not edges:
        return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.int32)
    
    return np.array(edges, dtype=np.int32), np.array(edge_types, dtype=np.int32)


def mol_to_atom_features(mol: Chem.Mol) -> np.ndarray:
    """
    Extract extended atom features.
    
    Features per atom:
        - One-hot encoding (14 dims)
        - Degree (1 dim)
        - Formal charge (1 dim)
        - Num Hs (1 dim)
        - Is aromatic (1 dim)
        - Hybridization (4 dims: sp, sp2, sp3, other)
        
    Total: 22 dims
    """
    num_atoms = mol.GetNumAtoms()
    features = []
    
    for atom in mol.GetAtoms():
        feat = []
        
        # One-hot encoding
        symbol = atom.GetSymbol()
        onehot = [0.0] * NUM_ATOM_TYPES
        idx = ATOM_TO_IDX.get(symbol, ATOM_TO_IDX['Other'])
        onehot[idx] = 1.0
        feat.extend(onehot)
        
        # Degree (normalized)
        feat.append(min(atom.GetDegree() / 4.0, 1.0))
        
        # Formal charge (normalized)
        feat.append((atom.GetFormalCharge() + 2) / 4.0)
        
        # Number of Hs (normalized)
        feat.append(min(atom.GetTotalNumHs() / 4.0, 1.0))
        
        # Is aromatic
        feat.append(1.0 if atom.GetIsAromatic() else 0.0)
        
        # Hybridization
        hyb = atom.GetHybridization()
        hyb_onehot = [0.0, 0.0, 0.0, 0.0]
        if hyb == Chem.rdchem.HybridizationType.SP:
            hyb_onehot[0] = 1.0
        elif hyb == Chem.rdchem.HybridizationType.SP2:
            hyb_onehot[1] = 1.0
        elif hyb == Chem.rdchem.HybridizationType.SP3:
            hyb_onehot[2] = 1.0
        else:
            hyb_onehot[3] = 1.0
        feat.extend(hyb_onehot)
        
        features.append(feat)
    
    return np.array(features, dtype=np.float32)


def generate_mol_features(
    mol_data: pd.DataFrame,
    output_onehot_dir: Path,
    output_graph_dir: Path,
    add_hs: bool = True,
    overwrite: bool = False,
):
    """
    Generate one-hot encodings and graph representations for all molecules.
    
    Args:
        mol_data: DataFrame with columns [mol_canonical_id, smiles]
        output_onehot_dir: Directory to save one-hot encodings
        output_graph_dir: Directory to save graph data
        add_hs: Whether to add explicit hydrogens
        overwrite: Whether to overwrite existing files
    """
    output_onehot_dir = Path(output_onehot_dir)
    output_graph_dir = Path(output_graph_dir)
    output_onehot_dir.mkdir(parents=True, exist_ok=True)
    output_graph_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter molecules that need processing
    to_process = []
    for _, row in mol_data.iterrows():
        mol_id = row["mol_canonical_id"]
        onehot_file = output_onehot_dir / f"{mol_id}.npy"
        if overwrite or not onehot_file.exists():
            to_process.append((mol_id, row["smiles"]))
    
    if not to_process:
        print("All features already exist. Use --overwrite to regenerate.")
        return
    
    print(f"Processing {len(to_process)} molecules...")
    
    # Track statistics
    stats = {
        "processed": 0,
        "failed": 0,
        "total_atoms": 0,
        "total_bonds": 0,
    }
    
    for mol_id, smiles in tqdm(to_process, desc="Generating features"):
        onehot_file = output_onehot_dir / f"{mol_id}.npy"
        graph_file = output_graph_dir / f"{mol_id}.npz"
        
        try:
            mol = smiles_to_mol(smiles, add_hs=add_hs)
            if mol is None:
                print(f"  Warning: Invalid SMILES for {mol_id}")
                stats["failed"] += 1
                continue
            
            # Generate features
            onehot = mol_to_onehot(mol)
            atom_features = mol_to_atom_features(mol)
            edges, edge_types = mol_to_edges(mol)
            
            # Save one-hot
            np.save(onehot_file, onehot)
            
            # Save graph data (compressed)
            np.savez_compressed(
                graph_file,
                atom_features=atom_features,
                edges=edges,
                edge_types=edge_types,
                num_atoms=mol.GetNumAtoms(),
            )
            
            stats["processed"] += 1
            stats["total_atoms"] += mol.GetNumAtoms()
            stats["total_bonds"] += mol.GetNumBonds()
            
        except Exception as e:
            print(f"  Error processing {mol_id}: {e}")
            stats["failed"] += 1
    
    print(f"\nProcessing complete:")
    print(f"  Processed: {stats['processed']}")
    print(f"  Failed: {stats['failed']}")
    if stats["processed"] > 0:
        print(f"  Avg atoms per molecule: {stats['total_atoms'] / stats['processed']:.1f}")
        print(f"  Avg bonds per molecule: {stats['total_bonds'] / stats['processed']:.1f}")
    print(f"\nOne-hot files saved to {output_onehot_dir}")
    print(f"Graph files saved to {output_graph_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate molecule features (one-hot, graph)")
    parser.add_argument("--input", "-i", required=True, help="Input CSV with mol_canonical_id and smiles")
    parser.add_argument("--output-onehot", required=True, help="Output directory for one-hot encodings")
    parser.add_argument("--output-graph", required=True, help="Output directory for graph data")
    parser.add_argument("--no-hydrogens", action="store_true", help="Don't add explicit hydrogens")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    
    args = parser.parse_args()
    
    # Load molecule data
    mol_data = pd.read_csv(args.input)
    
    # Validate required columns
    required_cols = ["mol_canonical_id", "smiles"]
    missing = [c for c in required_cols if c not in mol_data.columns]
    if missing:
        print(f"Error: Missing columns in input: {missing}")
        sys.exit(1)
    
    generate_mol_features(
        mol_data=mol_data,
        output_onehot_dir=args.output_onehot,
        output_graph_dir=args.output_graph,
        add_hs=not args.no_hydrogens,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
