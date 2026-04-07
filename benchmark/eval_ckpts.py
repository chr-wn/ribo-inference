import os
import sys
import argparse
from typing import Dict, Any, List
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
from collections import defaultdict
from tqdm import tqdm

from RNAdataset import load_global_stores, RMPredDataset, collate_rmpred_batch
from benchmark.train_5fold import repeated_stratified_kfold_indices, load_rna_type_map
from config import BASE_DIR, PSSM_DIR, DICT_PATH, MODEL_CONFIG
from RMPred import RMPred
from utils import move_batch_to_device, compute_metrics
from torch.cuda.amp import autocast

def make_subset_loader(dataset, indices, batch_size, num_workers=0):
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_rmpred_batch)

def find_optimal_threshold(preds: np.ndarray, targets: np.ndarray, optimization_metric='bacc'):
    """
    Finds the optimal threshold for continuous predictions to cast to binary classes (1 for >= 4.0 target).
    Optimization metric can be 'bacc' (Balanced Accuracy) or 'acc' (Accuracy).
    """
    y_true_bin = (targets >= 4.0).astype(int)
    
    if len(np.unique(y_true_bin)) < 2:
        return 4.0, {"acc": 1.0, "bacc": 1.0, "spec": 1.0, "sens": 1.0}

    min_p, max_p = np.min(preds), np.max(preds)
    thresholds = np.linspace(min_p, max_p, max(10, min(100, int((max_p - min_p) * 10))))
    if 4.0 not in thresholds:
        thresholds = np.append(thresholds, 4.0)

    best_thresh = 4.0
    best_score = -1.0
    best_metrics = {}

    for t in sorted(thresholds):
        y_pred_bin = (preds >= t).astype(int)
        acc = accuracy_score(y_true_bin, y_pred_bin)
        bacc = balanced_accuracy_score(y_true_bin, y_pred_bin)
        
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        score = bacc if optimization_metric == 'bacc' else acc
        if score > best_score:
            best_score = score
            best_thresh = t
            best_metrics = {"acc": acc, "bacc": bacc, "spec": spec, "sens": sens}
            
    return best_thresh, best_metrics

def evaluate_ensemble_and_find_thresholds(models, loader, device, optimize_for='bacc'):
    for m in models: m.eval()
    all_preds_list = []
    all_ys = []

    for batch in tqdm(loader, desc="Evaluating Ensemble"):
        if batch is None: continue
        batch = move_batch_to_device(batch, device)
        y = batch.get("pkd", batch.get("labels"))
        if y is None: continue
        y = y.to(device)
        keep = torch.isfinite(y)
        if keep.sum().item() == 0: continue

        member_preds = []
        for m in models:
            with autocast():
                mu = m(
                    rna_llm=batch["rna_llm"], rna_onehot=batch["rna_onehot"], rna_edges=batch["rna_edges"],
                    rna_pssm=batch["rna_pssm"], rna_mask=batch["rna_mask"],
                    mole_llm=batch["mole_llm"], mole_onehot=batch["mole_onehot"], mole_edges=batch["mole_edges"],
                    mole_mask=batch["mole_mask"],
                )
            member_preds.append(mu.view(-1)[keep].detach().cpu())
        
        # Save per-member predictions instead of pre-averaging them so we can weight them
        all_preds_list.extend([torch.tensor(p) for p in zip(*member_preds)])
        valid_ys = y[keep].detach().cpu()
        all_ys.append(valid_ys)

    if not all_preds_list:
        return None

    # Simple Mean
    preds = torch.stack(all_preds_list, dim=0).numpy() # (N, num_models)
    targets = torch.cat(all_ys, dim=0).numpy()
    
    mu_mean = preds.mean(axis=1)
    
    # Base regression metrics
    base_metrics = compute_metrics(torch.tensor(mu_mean), torch.tensor(targets))

    # Existing 4.0 threshold performance
    thresh_4, metrics_4 = find_optimal_threshold(mu_mean, targets, optimization_metric='acc')
    
    # Calculate 4.0 specifically
    y_true_bin = (targets >= 4.0).astype(int)
    y_pred_bin_4 = (mu_mean >= 4.0).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin_4, labels=[0, 1]).ravel()
    spec_4 = tn / (tn + fp) if (tn + fp) > 0 else 0
    bacc_4 = balanced_accuracy_score(y_true_bin, y_pred_bin_4)
    acc_4 = accuracy_score(y_true_bin, y_pred_bin_4)
    
    metrics_dict = {
        "rmse": base_metrics["rmse"],
        "pearson": base_metrics["pearson"],
        "auc_roc": roc_auc_score(y_true_bin, mu_mean) if len(np.unique(y_true_bin)) > 1 else 0.5,
        "default_4.0": {
            "acc": acc_4, "bacc": bacc_4, "spec": spec_4,
        }
    }

    # Find optimal threshold to max BACC for simple mean
    opt_thresh_bacc, opt_metrics_bacc = find_optimal_threshold(mu_mean, targets, optimization_metric='bacc')
    metrics_dict["optimized_bacc"] = {
        "threshold": opt_thresh_bacc,
        "acc": opt_metrics_bacc["acc"],
        "bacc": opt_metrics_bacc["bacc"],
        "spec": opt_metrics_bacc["spec"],
    }
    
    # Find optimal threshold to max ACC for simple mean
    opt_thresh_acc, opt_metrics_acc = find_optimal_threshold(mu_mean, targets, optimization_metric='acc')
    metrics_dict["optimized_acc"] = {
        "threshold": opt_thresh_acc,
        "acc": opt_metrics_acc["acc"],
        "bacc": opt_metrics_acc["bacc"],
        "spec": opt_metrics_acc["spec"],
    }
    
    # --- ENSEMBLE WEIGHT OPTIMIZATION ---
    # Fast Random Search for best weights
    num_models = preds.shape[1]
    best_weighted_auc = metrics_dict["auc_roc"]
    best_weighted_bacc = metrics_dict["optimized_bacc"]["bacc"]
    
    if num_models > 1 and len(np.unique(y_true_bin)) > 1:
        np.random.seed(42)
        # Generate 2000 random weight combinations
        random_weights = np.random.dirichlet(np.ones(num_models), size=2000)
        
        for w in random_weights:
            w_preds = np.average(preds, axis=1, weights=w)
            
            # Check AUC
            auc = roc_auc_score(y_true_bin, w_preds)
            if auc > best_weighted_auc:
                best_weighted_auc = auc
            
            # Sub-sample thresholds for speed inside loop
            min_p, max_p = np.min(w_preds), np.max(w_preds)
            thresholds = np.linspace(min_p, max_p, 20)
            
            for t in thresholds:
                w_pred_bin = (w_preds >= t).astype(int)
                bacc = balanced_accuracy_score(y_true_bin, w_pred_bin)
                if bacc > best_weighted_bacc:
                    best_weighted_bacc = bacc
                    
    metrics_dict["weighted_best_auc"] = best_weighted_auc
    metrics_dict["weighted_best_bacc"] = best_weighted_bacc

    return metrics_dict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=str, default="ckpts/val_5fold", help="Directory containing fold_xx folders with models")
    ap.add_argument("--data-dir", type=str, default=None, help="Root data directory (e.g. benchmark/data)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    code_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_base = os.path.join(code_dir, args.ckpt_dir)
    if not os.path.exists(ckpt_base):
        print(f"Checkpoint directory {ckpt_base} does not exist. Make sure you've trained models first.")
        return

    if args.data_dir:
        DATA_BASE = args.data_dir
        IDS_PATH = os.path.join(DATA_BASE, "ids.pkl")
        RNA_EMBED_PATH = os.path.join(DATA_BASE, "rna_embed.pkl")
        RNA_GRAPH_PATH = os.path.join(DATA_BASE, "rna_graph.pkl")
        MOLE_EMBED_PATH = os.path.join(DATA_BASE, "mole_embed.pkl")
        MOLE_EDGE_PATH = os.path.join(DATA_BASE, "mole_graph.pkl")
        MY_PSSM_DIR = os.path.join(DATA_BASE, "features", "pssm")
    else:
        IDS_PATH = os.path.join(BASE_DIR, "ids.pkl")
        RNA_EMBED_PATH = os.path.join(BASE_DIR, "rna_embed.pkl")
        RNA_GRAPH_PATH = os.path.join(BASE_DIR, "rna_graph.pkl")
        MOLE_EMBED_PATH = os.path.join(BASE_DIR, "mole_embed.pkl")
        MOLE_EDGE_PATH = os.path.join(BASE_DIR, "mole_graph.pkl")
        MY_PSSM_DIR = PSSM_DIR

    print("Loading global stores... (this may take a minute)")
    stores = load_global_stores(
        ids_path=IDS_PATH,
        rna_embed_path=RNA_EMBED_PATH,
        rna_graph_path=RNA_GRAPH_PATH,
        mole_embed_path=MOLE_EMBED_PATH,
        mole_edge_path=MOLE_EDGE_PATH,
        pssm_dir=MY_PSSM_DIR,
    )
    
    print("Loading Dataset...")
    dataset = RMPredDataset(
        stores, strict=True, max_rna_len=1024, max_mole_len=2048, truncate_if_exceed=False, label_key="pkd"
    )
    entry_ids = list(stores.entry_binding.keys())
    
    folds, _ = repeated_stratified_kfold_indices(
        stores=stores, entry_ids=entry_ids, label_key="pkd", n_splits=args.folds, n_repeats=1, group_count=5, strategy="uniform", seed=args.seed
    )

    all_fold_results = []

    for fold_id, (_, val_idx) in enumerate(folds):
        fold_dir = os.path.join(ckpt_base, f"fold_{fold_id:02d}")
        if not os.path.exists(fold_dir):
            print(f"Fold {fold_id} directory not found: {fold_dir}")
            continue

        print(f"\n================ FOLD {fold_id} ================")
        val_loader = make_subset_loader(dataset, val_idx.tolist(), batch_size=args.batch_size)
        
        # Load models
        models = []
        for ckpt_file in sorted(os.listdir(fold_dir)):
            if ckpt_file.endswith(".pt") and "member" in ckpt_file:
                ckpt_path = os.path.join(fold_dir, ckpt_file)
                print(f"Loading {ckpt_file}...")
                sd = torch.load(ckpt_path, map_location=device)
                
                dims = sd.get("dims", {})
                model = RMPred(
                    d_llm_rna=dims.get("dim_rna_llm", 1280), c_onehot_rna=dims.get("c_onehot_rna", 8), d_pssm_rna=dims.get("d_pssm", 5),
                    d_llm_mole=dims.get("dim_mole_llm", 512), c_onehot_mole=dims.get("c_onehot_mole", 6),
                    d_model_inner=MODEL_CONFIG["d_model_inner"], d_model_fusion=MODEL_CONFIG["d_model_fusion"], dropout=0.0,
                    fusion_layers=MODEL_CONFIG["fusion_layers"], fusion_heads=MODEL_CONFIG["fusion_heads"],
                    rna_gnn_layers=MODEL_CONFIG["rna_gnn_layers"], rna_gnn_heads=MODEL_CONFIG["rna_gnn_heads"],
                    mole_gnn_layers=MODEL_CONFIG["mole_gnn_layers"], mole_gnn_heads=MODEL_CONFIG["mole_gnn_heads"],
                    mole_num_edge_types=MODEL_CONFIG["mole_num_edge_types"], rna_max_len=1024,
                ).to(device)
                model.load_state_dict(sd["model_state"], strict=True)
                models.append(model)
                
        if not models:
            print("No models found for this fold.")
            continue
            
        fold_metrics = evaluate_ensemble_and_find_thresholds(models, val_loader, device)
        all_fold_results.append((fold_id, fold_metrics))
        
        print("\nFold Results:")
        print(f"  RMSE: {fold_metrics['rmse']:.4f} | Pearson: {fold_metrics['pearson']:.4f} | AUC-ROC: {fold_metrics['auc_roc']:.4f}")
        print(f"  [Default 4.0 Threshold]    Acc: {fold_metrics['default_4.0']['acc']:.4f} | BACC: {fold_metrics['default_4.0']['bacc']:.4f} | Spec: {fold_metrics['default_4.0']['spec']:.4f}")
        
        ob = fold_metrics['optimized_bacc']
        print(f"  [Optimized for BACC]       Threshold: {ob['threshold']:.4f} | Acc: {ob['acc']:.4f} | BACC: {ob['bacc']:.4f} | Spec: {ob['spec']:.4f}")
        
        oa = fold_metrics['optimized_acc']
        print(f"  [Optimized for Accuracy]   Threshold: {oa['threshold']:.4f} | Acc: {oa['acc']:.4f} | BACC: {oa['bacc']:.4f} | Spec: {oa['spec']:.4f}")
        
        print(f"  [Ensemble Weight Opt]      Best Possible AUC: {fold_metrics['weighted_best_auc']:.4f} | Best Possible BACC: {fold_metrics['weighted_best_bacc']:.4f}")

    if all_fold_results:
        print("\n================ FINAL SUMMARY ================")
        avg_rmse = np.mean([m['rmse'] for _, m in all_fold_results])
        avg_pearson = np.mean([m['pearson'] for _, m in all_fold_results])
        avg_auc = np.mean([m['auc_roc'] for _, m in all_fold_results])
        print(f"Mean RMSE: {avg_rmse:.4f} | Mean Pearson: {avg_pearson:.4f} | Mean AUC-ROC: {avg_auc:.4f}")
        
        # Mean default 4.0
        avg_acc_4 = np.mean([m['default_4.0']['acc'] for _, m in all_fold_results])
        avg_bacc_4 = np.mean([m['default_4.0']['bacc'] for _, m in all_fold_results])
        print(f"Threshold = 4.0:                  Acc={avg_acc_4:.4f} BACC={avg_bacc_4:.4f}")
        
        # Mean optimized BACC
        avg_acc_ob = np.mean([m['optimized_bacc']['acc'] for _, m in all_fold_results])
        avg_bacc_ob = np.mean([m['optimized_bacc']['bacc'] for _, m in all_fold_results])
        avg_thresh_ob = np.mean([m['optimized_bacc']['threshold'] for _, m in all_fold_results])
        print(f"Optimized for BACC (avg Th={avg_thresh_ob:.2f}): Acc={avg_acc_ob:.4f} BACC={avg_bacc_ob:.4f}")

        # Mean optimized AAC
        avg_acc_oa = np.mean([m['optimized_acc']['acc'] for _, m in all_fold_results])
        avg_bacc_oa = np.mean([m['optimized_acc']['bacc'] for _, m in all_fold_results])
        avg_thresh_oa = np.mean([m['optimized_acc']['threshold'] for _, m in all_fold_results])
        print(f"Optimized for ACC (avg Th={avg_thresh_oa:.2f}):  Acc={avg_acc_oa:.4f} BACC={avg_bacc_oa:.4f}")
        
        # Mean weighted opt
        avg_w_auc = np.mean([m['weighted_best_auc'] for _, m in all_fold_results])
        avg_w_bacc = np.mean([m['weighted_best_bacc'] for _, m in all_fold_results])
        print(f"Ensemble Weight Opt (Max Potential): AUC={avg_w_auc:.4f} BACC={avg_w_bacc:.4f}")

if __name__ == "__main__":
    main()
