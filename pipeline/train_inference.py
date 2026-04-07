#!/usr/bin/env python3
# Trains N ensemble of models with different seeds for inference

import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from RMPred import RMPred
from RNAdataset import load_global_stores, make_dataloader, GlobalStores

def get_dimensions(stores: GlobalStores):
    rna_ex = next(iter(stores.rna_embed.values()))
    d_llm_rna = rna_ex.shape[1]
    
    pssm_path = Path(stores.pssm_dir)
    pssm_file = next(pssm_path.glob("*.npy"))
    pssm_ex = np.load(pssm_file)
    if pssm_ex.ndim == 1:
        if pssm_ex.shape[0] % d_llm_rna == 0:
             pass 
    d_pssm_rna = 21
    if pssm_ex.ndim == 2:
        d_pssm_rna = pssm_ex.shape[1]
    elif pssm_ex.ndim == 1:
          pass
    
    mol_ex = next(iter(stores.mole_embed.values()))
    d_llm_mole = mol_ex.shape[1]
    
    return {
        "d_llm_rna": d_llm_rna,
        "c_onehot_rna": 5,
        "d_pssm_rna": 21,
        "d_llm_mole": d_llm_mole,
        "c_onehot_mole": 14,
    }

def train_one_model(
    model_idx: int,
    stores: GlobalStores,
    config: dict,
    output_dir: Path,
    device: str = "cuda"
):
    """Train a single model instance."""
    print(f"\nTraining Model {model_idx + 1}...")
    
    seed = 42 + model_idx
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    full_loader = make_dataloader(
        stores, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        label_key='pkd',
        num_workers=2
    )
    
    batch = next(iter(full_loader))
    d_pssm_rna = batch['rna_pssm'].shape[2]
    d_llm_rna = batch['rna_llm'].shape[2]
    d_llm_mole = batch['mole_llm'].shape[2]
    
    print(f"  Input Dims: RNA_LLM={d_llm_rna}, RNA_PSSM={d_pssm_rna}, MOL_LLM={d_llm_mole}")
    
    model = RMPred(
        d_llm_rna=d_llm_rna,
        c_onehot_rna=5,
        d_pssm_rna=d_pssm_rna,
        d_llm_mole=d_llm_mole,
        c_onehot_mole=14,
        d_model_inner=256,
        d_model_fusion=512,
        dropout=config['dropout'],
        fusion_layers=2,
        rna_max_len=8192
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    model_path = output_dir / f"model_{model_idx}.pt"
    
    epochs = config['epochs']
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'], 
                                             steps_per_epoch=len(full_loader), 
                                             epochs=epochs)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        count = 0
        
        pbar = tqdm(full_loader, desc=f"  Epoch {epoch+1}/{epochs}", leave=False)
        for batch in pbar:
            if batch is None: continue
            
            rna_llm = batch['rna_llm'].to(device)
            rna_oh = batch['rna_onehot'].to(device)
            rna_pssm = batch['rna_pssm'].to(device)
            rna_mask = batch['rna_mask'].to(device)
            
            mole_llm = batch['mole_llm'].to(device)
            mole_oh = batch['mole_onehot'].to(device)
            mole_mask = batch['mole_mask'].to(device)
            
            labels = batch['pkd'].to(device)
            
            rna_edges = [e.to(device) for e in batch['rna_edges']]
            mole_edges = [e.to(device) for e in batch['mole_edges']]
            
            optimizer.zero_grad()
            
            pred = model(
                rna_llm, rna_oh, rna_edges, rna_pssm, rna_mask,
                mole_llm, mole_oh, mole_edges, mole_mask
            )
            
            loss = criterion(pred, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item() * len(labels)
            count += len(labels)
            
            pbar.set_postfix({"loss": total_loss/count})
        
        avg_loss = total_loss / count
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)
            
    print(f"  Finished. Best Loss: {best_loss:.4f}. Saved to {model_path}")
    return best_loss

def main():
    parser = argparse.ArgumentParser(description="Train Inference Ensemble")
    parser.add_argument("--data", "-d", required=True, help="Directory containing consolidated pickles")
    parser.add_argument("--output", "-o", required=True, help="Output directory for models")
    parser.add_argument("--num-models", "-n", type=int, default=5, help="Number of models in ensemble")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per model")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading Global Stores...")
    stores = load_global_stores(
        ids_path=os.path.join(args.data, "ids.pkl"),
        rna_embed_path=os.path.join(args.data, "rna_embed.pkl"),
        rna_graph_path=os.path.join(args.data, "rna_graph.pkl"),
        mole_embed_path=os.path.join(args.data, "mole_embed.pkl"),
        mole_edge_path=os.path.join(args.data, "mole_graph.pkl"),
        pssm_dir=os.path.join(args.data, "pssm"),
    )
    
    if not os.path.exists(stores.pssm_dir):
        alt_pssm = os.path.join(args.data, "pssm")
        if os.path.exists(alt_pssm):
            stores.pssm_dir = alt_pssm
            print(f"  Updated PSSM dir to {stores.pssm_dir}")
    
    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'dropout': 0.1
    }
    
    print(f"Starting Training of {args.num_models} models...")
    for i in range(args.num_models):
        train_one_model(i, stores, config, out_dir)
        
    print("\nEnsemble Training Complete!")

if __name__ == "__main__":
    main()
