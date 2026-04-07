#!/usr/bin/env python3
"""
Consolidate generated features into dictionary pickles for efficient loading.
This bridges the gap between the file-based feature generation and the 
dictionary-based RNAdataset loader.
"""

import argparse
import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def load_rna_features(rna_data, features_dir):
    """Load RNA features into dictionaries."""
    print("Loading RNA features...")
    
    rna_embed = {}
    rna_graph = {}
    
    rinalmo_dir = features_dir / "rinalmo"
    mxfold2_dir = features_dir / "mxfold2"
    
    # RiNALMo
    if rinalmo_dir.exists():
        print("  Loading RiNALMo embeddings...")
        
        # Load filename mapping if it exists
        mapping_file = rinalmo_dir / "rna_id_mapping.json"
        rna_id_to_filename = {}
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                rna_id_to_filename = json.load(f)
        
        for rna_id in tqdm(rna_data['rna_canonical_id'].unique()):
            # Use safe filename if mapping exists, otherwise use original
            filename = rna_id_to_filename.get(rna_id, rna_id)
            fpath = rinalmo_dir / f"{filename}.npy"
            if fpath.exists():
                rna_embed[rna_id] = np.load(fpath)
            else:
                # Try finding without extension if generated differently
                pass
    
    # MXFold2
    if mxfold2_dir.exists():
        print("  Loading MXFold2 structures...")
        
        # Load filename mapping if it exists
        mapping_file = mxfold2_dir / "rna_id_mapping.json"
        rna_id_to_filename = {}
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                rna_id_to_filename = json.load(f)
        
        for rna_id in tqdm(rna_data['rna_canonical_id'].unique()):
            # Use safe filename if mapping exists, otherwise use original
            filename = rna_id_to_filename.get(rna_id, rna_id)
            fpath = mxfold2_dir / f"{filename}.json"
            if fpath.exists():
                with open(fpath, 'r') as f:
                    data = json.load(f)
                    # RNAdataset expects edge list (N x 2)
                    if 'base_pairs' in data:
                        rna_graph[rna_id] = np.array(data['base_pairs'], dtype=np.int64)
    
    return rna_embed, rna_graph

def load_mol_features(mol_data, features_dir):
    """Load Molecule features into dictionaries."""
    print("Loading Molecule features...")
    
    mole_embed = {}
    mole_graph = {}
    
    unimol_dir = features_dir / "unimol" / "atomic" # Use atomic embeddings
    mol_graph_dir = features_dir / "mol_graph"
    
    # UniMol
    if unimol_dir.exists():
        print("  Loading UniMol atomic embeddings...")
        for mol_id in tqdm(mol_data['mol_canonical_id'].unique()):
            fpath = unimol_dir / f"{mol_id}.npy"
            if fpath.exists():
                mole_embed[mol_id] = np.load(fpath)
    

    # Mol Graph
    # Helper to count atoms
    from rdkit import Chem
    
    if mol_graph_dir.exists():
        print("  Loading Molecular Graphs and Sanitizing Embeddings...")
        
        # We need smiles to check counts
        # mol_data has 'mol_canonical_id' and 'smiles'
        mol_smiles_map = dict(zip(mol_data['mol_canonical_id'], mol_data['smiles']))

        for mol_id in tqdm(mol_data['mol_canonical_id'].unique(), desc="Processing Molecules"):
            # Load Graph
            fpath_graph = mol_graph_dir / f"{mol_id}.npz"
            if fpath_graph.exists():
                try:
                    data = np.load(fpath_graph)
                    edges = data['edges']   # E x 2
                    types = data['edge_types'] # E
                    
                    if len(edges) > 0:
                        if types.ndim == 1:
                            types = types[:, np.newaxis]
                        combined = np.hstack([edges, types])
                        mole_graph[mol_id] = combined.astype(np.int64)
                    else:
                        mole_graph[mol_id] = np.empty((0, 3), dtype=np.int64)
                except Exception as e:
                    print(f"    Error graph {mol_id}: {e}")

            # Load Embedding & Sanitize
            fpath_emb = unimol_dir / f"{mol_id}.npy"
            if fpath_emb.exists():
                emb = np.load(fpath_emb)
                
                # Check dimensions against RDKit
                smiles = mol_smiles_map.get(mol_id)
                if smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        # UniMol uses Explicit Hs usually if add_hs=True
                        # Our pipeline default generate_mol_features uses add_hs=True
                        # And RNAdataset uses AddHs checks.
                        # We should try to match Explicit H count.
                        mol_h = Chem.AddHs(mol)
                        n_explicit = mol_h.GetNumAtoms()
                        
                        L = emb.shape[0]
                        if L == n_explicit:
                            pass # All good
                        elif L == n_explicit + 2:
                            # Remove CLS and SEP
                            emb = emb[1:-1]
                        elif L == n_explicit + 1:
                            # Remove CLS?
                            emb = emb[1:]
                        else:
                            # Check for truncation (Unimol max len)
                            # Unimol embedding might be [CLS] + atoms[:max_len] + [SEP] (if room)
                            # or just truncated.
                            # If L is significantly smaller than n_explicit, assume truncation.
                            # Standard Unimol max is often 512, but here we see 257.
                            # 257 could be [CLS] + 256 atoms.
                            if L < n_explicit:
                                print(f"    Warning: {mol_id} embedding len {L} < Explicit ({n_explicit}). Assuming Truncation.")
                                # Assume [CLS] + atoms
                                # Or [CLS] + atoms + [SEP]
                                # If L = N + 1, assume CLS + N atoms.
                                # If we assume the first (L-1) atoms are kept:
                                num_kept = L - 1 # Remove CLS
                                # Check if [SEP] is at end? 
                                # Unimol usually puts [SEP] at end if it fits. 
                                # If truncated, maybe not?
                                # Let's assume [CLS] + atoms[:num_kept]
                                # But if L=257, num_kept=256. 
                                emb = emb[1:] # Remove CLS. Now (N_kept, D)
                                
                                # We must truncate the graph accordingly!
                                # Graph edges refer to indices. We must filter edges where u,v < num_kept.
                                if mol_id in mole_graph:
                                    g = mole_graph[mol_id] # (E, 3)
                                    u, v = g[:, 0], g[:, 1]
                                    mask = (u < num_kept) & (v < num_kept)
                                    mole_graph[mol_id] = g[mask]
                                    
                            else:
                                # Try Implicit?
                                n_implicit = mol.GetNumAtoms()
                                if L == n_implicit:
                                    pass # Weird but ok
                                elif L == n_implicit + 2:
                                    emb = emb[1:-1]
                                elif L == n_implicit + 1:
                                    emb = emb[1:]
                                else:
                                    print(f"    Warning: {mol_id} embedding len {L} matches neither Explicit ({n_explicit}) nor Implicit ({n_implicit}) atoms. Skipping.")
                                    continue
                                
                        mole_embed[mol_id] = emb

    
    return mole_embed, mole_graph

def main():
    parser = argparse.ArgumentParser(description="Consolidate features into pickles")
    parser.add_argument("--input", "-i", required=True, help="Input CSV (union dataset)")
    parser.add_argument("--features", "-f", required=True, help="Directory containing generated features")
    parser.add_argument("--output", "-o", required=True, help="Output directory for pickle files")
    
    args = parser.parse_args()
    
    print(f"Reading dataset: {args.input}")
    df = pd.read_csv(args.input)
    
    features_dir = Path(args.features)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Structural/Map Data (ids.pkl)
    print("Constructing metadata...")
    entry_binding_dict = {}
    rna_id_to_name = {}
    mol_id_to_name = {}
    rna_seqs = {}
    mole_smiles = {}
    
    # Normalize labels if present
    labels = []
    
    for idx, row in df.iterrows():
        # Use simple index or existing ID if available as entry_id
        entry_id = row.get('entry_id', str(idx))
        
        rna_id = row['rna_canonical_id']
        mol_id = row['mol_canonical_id']
        
        # Try to find label (affinity or pkd or label or pKd)
        label = row.get('affinity')
        if pd.isna(label): label = row.get('pkd')
        if pd.isna(label): label = row.get('pKd')
        if pd.isna(label): label = row.get('label')
        if pd.notna(label):
             labels.append(float(label))
        
        entry_binding_dict[entry_id] = {
            'rna_id': rna_id,
            'mol_id': mol_id, # Handle key mismatch in dataset loader
            'mole_id': mol_id,
            'pkd': float(label) if pd.notna(label) else None,
            'affinity': float(label) if pd.notna(label) else None
        }
        
        # Maps
        rna_name = row.get('rna_name', rna_id)
        mol_name = row.get('mol_name', mol_id)
        
        rna_id_to_name[rna_id] = rna_name
        mol_id_to_name[mol_id] = mol_name
        
        rna_seqs[rna_id] = row['rna_sequence']
        mole_smiles[mol_id] = row['smiles']
    
    # Calc normalizer
    if labels:
        mean = np.mean(labels)
        std = np.std(labels)
        pkd_norm = {'mean': float(mean), 'std': float(std)}
    else:
        pkd_norm = {'mean': 0.0, 'std': 1.0}
    
    ids_data = {
        "entry_binding_dict": entry_binding_dict,
        "rna_id_to_name_dict": rna_id_to_name,
        "mol_id_to_name_dict": mol_id_to_name,
        "rna_seq_dict": rna_seqs,
        "mole_smiles_dict": mole_smiles,
        "pkd_normalizer": pkd_norm
    }
    
    # 2. Load Features
    rna_embed, rna_graph = load_rna_features(df, features_dir)
    mole_embed, mole_graph = load_mol_features(df, features_dir)
    
    # 3. Save Pickles
    print("Saving pickle files...")
    
    with open(output_dir / "ids.pkl", "wb") as f:
        pickle.dump(ids_data, f)
        
    with open(output_dir / "rna_embed.pkl", "wb") as f:
        pickle.dump(rna_embed, f)
        
    with open(output_dir / "rna_graph.pkl", "wb") as f:
        pickle.dump(rna_graph, f)
        
    with open(output_dir / "mole_embed.pkl", "wb") as f:
        pickle.dump(mole_embed, f)
        
    with open(output_dir / "mole_graph.pkl", "wb") as f:
        pickle.dump(mole_graph, f)
        
    print("✅ Consolidation complete!")
    print(f"  Entries: {len(entry_binding_dict)}")
    print(f"  RNAs: {len(rna_embed)} embeddings, {len(rna_graph)} graphs")
    print(f"  Molecules: {len(mole_embed)} embeddings, {len(mole_graph)} graphs")

if __name__ == "__main__":
    main()
