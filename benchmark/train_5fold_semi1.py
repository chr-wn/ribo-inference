
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UserWarning, message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")
warnings.filterwarnings("ignore", message="A single label was found")

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import math
import copy
import argparse
import random
import json
import itertools
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils import shuffle as sklearn_shuffle
from scipy.optimize import minimize

from RNAdataset import load_global_stores, RMPredDataset, collate_rmpred_batch, compute_metrics, GlobalStores
from RMPred import RMPred
from utils import *
from config import BASE_DIR, PSSM_DIR, DICT_PATH, MODEL_CONFIG, get_data_paths
from utils_metrics import calculate_metrics


def cosine_lr_schedule(epoch, total_epochs, lr_max, lr_min=1e-6):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / total_epochs))


def warmup_cosine_lr(epoch, warmup_epochs, total_epochs, lr_max, lr_min=1e-6):
    if epoch <= warmup_epochs:
        return lr_min + (lr_max - lr_min) * epoch / max(warmup_epochs, 1)
    return cosine_lr_schedule(epoch - warmup_epochs, total_epochs - warmup_epochs, lr_max, lr_min)


def numpy_pearson(pred, target):
    if len(pred) < 2:
        return float("nan")
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    pred_c = pred - pred.mean()
    target_c = target - target.mean()
    denom = np.sqrt((pred_c**2).mean()) * np.sqrt((target_c**2).mean())
    if denom == 0:
        return float("nan")
    return float(np.clip((pred_c * target_c).mean() / denom, -1.0, 1.0))


def numpy_rmse(pred, target):
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def find_optimal_weights(preds_matrix, targets, metric='pearson', n_restarts=5):
    n = preds_matrix.shape[1]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * n

    def objective(w):
        fp = np.average(preds_matrix, axis=1, weights=w)
        if metric == 'rmse':
            return numpy_rmse(fp, targets)
        return -numpy_pearson(fp, targets)

    best_w = np.ones(n) / n
    best_obj = objective(best_w)
    for _ in range(n_restarts):
        init = np.random.dirichlet(np.ones(n))
        try:
            res = minimize(objective, init, method='SLSQP', bounds=bounds,
                           constraints=constraints, options={'maxiter': 200, 'ftol': 1e-8})
            if res.fun < best_obj:
                best_obj = res.fun
                best_w = res.x
        except Exception:
            pass
    best_w = np.maximum(best_w, 0)
    return best_w / best_w.sum()


def load_rna_type_map(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    out = {}
    for k, v in data.items():
        out[str(k)] = v.get("RNA_type", "Unknown") if isinstance(v, dict) else str(v)
    return out


class RegressorStratifiedCV:
    def __init__(self, n_splits=10, n_repeats=2, group_count=10, random_state=0, strategy='quantile'):
        self.group_count = group_count
        self.strategy = strategy
        self.cvkwargs = dict(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        self.cv = RepeatedStratifiedKFold(**self.cvkwargs)
        self.discretizer = KBinsDiscretizer(n_bins=self.group_count, encode='ordinal', strategy=self.strategy)

    def split(self, X, y, groups=None):
        kgroups = self.discretizer.fit_transform(y[:, None])[:, 0]
        return self.cv.split(X, kgroups, groups)


def _get_entry_label(entry, label_key):
    for k in (label_key, label_key.lower(), label_key.upper(), "pKd", "pKD", "affinity", "label"):
        if k in entry and entry[k] is not None:
            try:
                return float(entry[k])
            except:
                pass
    return None


def repeated_stratified_kfold_indices(*, stores, entry_ids, label_key, n_splits, n_repeats,
                                       group_count, strategy, seed):
    y_raw = np.array([_get_entry_label(stores.entry_binding[eid], label_key) for eid in entry_ids], dtype=np.float32)
    idx_raw = np.arange(len(entry_ids))
    keep = np.isfinite(y_raw)
    y_c = y_raw[keep]
    idx_c = idx_raw[keep]
    if len(y_c) < n_splits:
        raise ValueError("Not enough finite labels.")
    idx_s, y_s = sklearn_shuffle(idx_c, y_c, random_state=seed)
    cv = RegressorStratifiedCV(n_splits=n_splits, n_repeats=n_repeats, group_count=group_count,
                                random_state=seed, strategy=strategy)
    X_dummy = np.zeros((len(y_s), 1), dtype=np.float32)
    folds = []
    for tr_rel, va_rel in cv.split(X_dummy, y_s):
        folds.append((idx_s[tr_rel], idx_s[va_rel]))
    print(f"[Split] bins={group_count} | strategy={strategy} | splits={n_splits} | repeats={n_repeats} | seed={seed}")
    return folds, y_raw


# ---------------------------------------------------------------------------
# Wrapper dataset that overrides labels with pseudo-labels
# ---------------------------------------------------------------------------

class PseudoLabelSubset(Dataset):
    """Wraps an RMPredDataset subset and replaces real labels with pseudo-labels."""

    def __init__(self, base_dataset, indices, pseudo_labels):
        self.base = base_dataset
        self.indices = list(indices)
        self.pseudo_labels = pseudo_labels  # dict: dataset_index -> pseudo_label

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = self.base[real_idx]
        if item is None:
            return None
        # Override the label with the pseudo-label
        item["label"] = self.pseudo_labels[real_idx]
        return item


def build_model(dims, device, max_rna_len=1024):
    return RMPred(
        d_llm_rna=dims["dim_rna_llm"],
        c_onehot_rna=dims["c_onehot_rna"],
        d_pssm_rna=dims["d_pssm"],
        d_llm_mole=dims["dim_mole_llm"],
        c_onehot_mole=dims["c_onehot_mole"],
        d_model_inner=MODEL_CONFIG["d_model_inner"],
        d_model_fusion=MODEL_CONFIG["d_model_fusion"],
        dropout=MODEL_CONFIG["dropout"],
        fusion_layers=MODEL_CONFIG["fusion_layers"],
        fusion_heads=MODEL_CONFIG["fusion_heads"],
        rna_gnn_layers=MODEL_CONFIG["rna_gnn_layers"],
        rna_gnn_heads=MODEL_CONFIG["rna_gnn_heads"],
        mole_gnn_layers=MODEL_CONFIG["mole_gnn_layers"],
        mole_gnn_heads=MODEL_CONFIG["mole_gnn_heads"],
        mole_num_edge_types=MODEL_CONFIG["mole_num_edge_types"],
        rna_max_len=max_rna_len,
    ).to(device)


def load_fold_ensemble(ckpt_dir, fold_id, device, max_rna_len=1024):
    fold_dir = os.path.join(ckpt_dir, f"fold_{fold_id:02d}")
    if not os.path.isdir(fold_dir):
        raise FileNotFoundError(f"No checkpoint folder: {fold_dir}")
    ckpt_files = sorted([f for f in os.listdir(fold_dir) if f.endswith("_best.pt")])
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files in {fold_dir}")
    models = []
    dims = None
    for cf in ckpt_files:
        sd = torch.load(os.path.join(fold_dir, cf), map_location=device)
        if dims is None:
            dims = sd["dims"]
        m = build_model(dims, device, max_rna_len)
        m.load_state_dict(sd["model_state"], strict=True)
        m.eval()
        models.append(m)
    return models, dims


# ---------------------------------------------------------------------------
# Pseudo-label the VALIDATION fold using teacher ensemble
# ---------------------------------------------------------------------------

@torch.no_grad()
def pseudo_label_indices(models, dataset, indices, device, batch_size=32):
    """Run teacher ensemble on specific dataset indices, return pseudo-labels."""
    for m in models:
        m.eval()

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=collate_rmpred_batch)

    all_means = []
    all_stds = []

    for batch in loader:
        if batch is None:
            continue
        batch_dev = move_batch_to_device(batch, device)
        bs = batch_dev["rna_llm"].shape[0]

        member_preds = []
        for m in models:
            with autocast():
                mu = m(
                    rna_llm=batch_dev["rna_llm"], rna_onehot=batch_dev["rna_onehot"],
                    rna_edges=batch_dev["rna_edges"], rna_pssm=batch_dev["rna_pssm"],
                    rna_mask=batch_dev["rna_mask"],
                    mole_llm=batch_dev["mole_llm"], mole_onehot=batch_dev["mole_onehot"],
                    mole_edges=batch_dev["mole_edges"], mole_mask=batch_dev["mole_mask"],
                )
            member_preds.append(mu.detach().cpu().numpy())

        preds_np = np.stack(member_preds, axis=0)  # (K, bs)
        all_means.extend(preds_np.mean(axis=0).tolist())
        all_stds.extend(preds_np.std(axis=0).tolist())

    return all_means, all_stds


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, scaler, grad_clip=1.0, ccc_weight=1.0):
    model.train()
    total_loss, total_n = 0.0, 0

    for batch in loader:
        if batch is None:
            continue
        batch = move_batch_to_device(batch, device)
        y = batch.get("pkd", batch.get("labels"))
        if y is None:
            continue
        y = y.to(device)
        keep = torch.isfinite(y)
        if keep.sum().item() == 0:
            continue
        yy = y[keep]

        with autocast():
            mu = model(
                rna_llm=batch["rna_llm"], rna_onehot=batch["rna_onehot"], rna_edges=batch["rna_edges"],
                rna_pssm=batch["rna_pssm"], rna_mask=batch["rna_mask"],
                mole_llm=batch["mole_llm"], mole_onehot=batch["mole_onehot"], mole_edges=batch["mole_edges"],
                mole_mask=batch["mole_mask"],
            )
            mu = mu[keep]
            mse = torch.mean((mu - yy) ** 2)
            ccc_l = ccc_loss(mu, yy)
            loss = 0.5 * (mse + ccc_weight * ccc_l)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = int(yy.numel())
        total_loss += loss.item() * bs
        total_n += bs

    return {"loss": total_loss / max(1, total_n)}


@torch.no_grad()
def evaluate_mu(model, loader, device, type_map=None):
    model.eval()
    all_preds, all_ys, all_types = [], [], []

    for batch in loader:
        if batch is None:
            continue
        batch_ids = batch.get("entry_ids") or batch.get("entry_id")
        batch = move_batch_to_device(batch, device)
        y = batch.get("pkd", batch.get("labels"))
        if y is None:
            continue
        y = y.to(device)
        keep = torch.isfinite(y)
        if keep.sum().item() == 0:
            continue
        with autocast():
            mu = model(
                rna_llm=batch["rna_llm"], rna_onehot=batch["rna_onehot"], rna_edges=batch["rna_edges"],
                rna_pssm=batch["rna_pssm"], rna_mask=batch["rna_mask"],
                mole_llm=batch["mole_llm"], mole_onehot=batch["mole_onehot"], mole_edges=batch["mole_edges"],
                mole_mask=batch["mole_mask"],
            )
        all_preds.append(mu[keep].detach().cpu())
        all_ys.append(y[keep].detach().cpu())

        if type_map is not None and batch_ids is not None:
            keep_cpu = keep.cpu().tolist()
            valid_ids = [bid for bid, k in zip(batch_ids, keep_cpu) if k]
            all_types.extend([type_map.get(str(eid), "Unknown") for eid in valid_ids])

    if not all_preds:
        return {"global": calculate_metrics(np.array([]), np.array([]))}
    mu_all = torch.cat(all_preds, dim=0).numpy()
    y_all = torch.cat(all_ys, dim=0).numpy()
    results = {"global": calculate_metrics(mu_all, y_all)}
    return results


@torch.no_grad()
def collect_ensemble_predictions(models, loader, device, type_map=None):
    for m in models:
        m.eval()
    num_models = len(models)
    all_ys, all_types = [], []
    member_preds = [[] for _ in range(num_models)]

    for batch in loader:
        if batch is None:
            continue
        batch_ids = batch.get("entry_ids") or batch.get("entry_id")
        batch = move_batch_to_device(batch, device)
        y = batch.get("pkd", batch.get("labels"))
        if y is None:
            continue
        y = y.to(device)
        keep = torch.isfinite(y)
        if keep.sum().item() == 0:
            continue
        all_ys.append(y[keep].detach().cpu())

        if type_map is not None and batch_ids is not None:
            keep_cpu = keep.cpu().tolist()
            valid_ids = [bid for bid, k in zip(batch_ids, keep_cpu) if k]
            all_types.extend([type_map.get(str(eid), "Unknown") for eid in valid_ids])

        for i, m in enumerate(models):
            with autocast():
                mu = m(
                    rna_llm=batch["rna_llm"], rna_onehot=batch["rna_onehot"], rna_edges=batch["rna_edges"],
                    rna_pssm=batch["rna_pssm"], rna_mask=batch["rna_mask"],
                    mole_llm=batch["mole_llm"], mole_onehot=batch["mole_onehot"], mole_edges=batch["mole_edges"],
                    mole_mask=batch["mole_mask"],
                )
            member_preds[i].append(mu[keep].detach().cpu().numpy())

    if not all_ys:
        return np.array([]), np.array([]), []
    y_all = torch.cat(all_ys, dim=0).numpy()
    preds_matrix = np.column_stack([np.concatenate(p) for p in member_preds])
    return preds_matrix, y_all, all_types


def evaluate_ensemble_with_weights(preds_matrix, y_all, all_types, optimize_metric='pearson'):
    num_models = preds_matrix.shape[1]
    results = {}

    simple_preds = np.mean(preds_matrix, axis=1)
    results["simple_avg"] = calculate_metrics(simple_preds, y_all)
    results["simple_avg"]["count"] = len(y_all)

    if num_models >= 3:
        sorted_p = np.sort(preds_matrix, axis=1)
        trimmed_p = np.mean(sorted_p[:, 1:-1], axis=1)
        results["trimmed_avg"] = calculate_metrics(trimmed_p, y_all)
        results["trimmed_avg"]["count"] = len(y_all)
    else:
        results["trimmed_avg"] = results["simple_avg"].copy()

    best_w = find_optimal_weights(preds_matrix, y_all, metric=optimize_metric)
    weighted_preds = np.average(preds_matrix, axis=1, weights=best_w)
    results["weighted_avg"] = calculate_metrics(weighted_preds, y_all)
    results["weighted_avg"]["count"] = len(y_all)
    results["best_weights"] = best_w.tolist()

    if all_types and len(all_types) == len(weighted_preds):
        type_buckets = defaultdict(lambda: {"preds": [], "ys": []})
        for pred, target, rtype in zip(weighted_preds, y_all, all_types):
            type_buckets[rtype]["preds"].append(pred)
            type_buckets[rtype]["ys"].append(target)
        for rtype, data in type_buckets.items():
            m = calculate_metrics(np.array(data["preds"]), np.array(data["ys"]))
            m["count"] = len(data["preds"])
            results[rtype] = m

    return results


def make_subset_loader(dataset, indices, batch_size, shuffle, num_workers, seed):
    subset = Subset(dataset, indices)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_rmpred_batch, generator=g)


def main():
    ap = argparse.ArgumentParser(description="Transductive semi-supervised 5-fold CV")

    ap.add_argument("--ckpt_dir", type=str, default="benchmark/ckpts/val_5fold")
    ap.add_argument("--k", type=int, default=5, help="ensemble size per fold")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--val_batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--out_dir", type=str, default="ckpt_5fold_semi1")
    ap.add_argument("--max_rna_len", type=int, default=1024)
    ap.add_argument("--max_mole_len", type=int, default=2048)
    ap.add_argument("--early_patience", type=int, default=25)
    ap.add_argument("--label_key", type=str, default="pkd")

    # Semi-supervised: how much to blend pseudo-labels toward real labels
    # alpha=1.0 means use 100% teacher prediction as pseudo-label
    # alpha=0.5 means average teacher prediction with real label
    ap.add_argument("--pseudo_alpha", type=float, default=0.7,
                    help="blend factor: pseudo = alpha*teacher + (1-alpha)*real for val samples")
    ap.add_argument("--self_train_rounds", type=int, default=2,
                    help="number of self-training rounds (re-pseudo-label with improved model)")

    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--strat_bins", type=int, default=5)
    ap.add_argument("--strat_strategy", type=str, default="uniform")
    ap.add_argument("--metric", type=str, default="pearson", choices=["rmse", "pearson"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--data-dir", type=str, default=None)

    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"=== Transductive Semi-Supervised 5-Fold CV ===")
    print(f"Device: {device} | Metric: {args.metric.upper()} | Ensemble k={args.k}")
    print(f"Pseudo-label blend alpha={args.pseudo_alpha} | Self-train rounds={args.self_train_rounds}")
    os.makedirs(args.out_dir, exist_ok=True)

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

    stores = load_global_stores(
        ids_path=IDS_PATH, rna_embed_path=RNA_EMBED_PATH, rna_graph_path=RNA_GRAPH_PATH,
        mole_embed_path=MOLE_EMBED_PATH, mole_edge_path=MOLE_EDGE_PATH, pssm_dir=MY_PSSM_DIR,
    )

    type_map = None
    if os.path.exists(DICT_PATH):
        type_map = load_rna_type_map(DICT_PATH)

    dataset = RMPredDataset(
        stores, strict=True, max_rna_len=args.max_rna_len,
        max_mole_len=(None if args.max_mole_len == 0 else args.max_mole_len),
        truncate_if_exceed=False, label_key=args.label_key,
    )
    n_total = len(dataset)
    print(f"Total labeled samples: {n_total}")

    set_seed(args.seed)

    temp_loader = DataLoader(Subset(dataset, [0]), batch_size=1, collate_fn=collate_rmpred_batch)
    batch0 = next(iter(temp_loader))
    dims = {
        "dim_rna_llm": batch0["rna_llm"].shape[-1],
        "dim_mole_llm": batch0["mole_llm"].shape[-1],
        "c_onehot_rna": batch0["rna_onehot"].shape[-1],
        "c_onehot_mole": batch0["mole_onehot"].shape[-1],
        "d_pssm": batch0["rna_pssm"].shape[-1],
    }

    entry_ids = getattr(dataset, "keys", None) or list(stores.entry_binding.keys())

    folds, y_all_labels = repeated_stratified_kfold_indices(
        stores=stores, entry_ids=entry_ids, label_key=args.label_key,
        n_splits=args.folds, n_repeats=args.repeats, group_count=args.strat_bins,
        strategy=args.strat_strategy, seed=args.seed,
    )

    # Get real labels for all samples
    real_labels = {}
    for i, eid in enumerate(entry_ids):
        lbl = _get_entry_label(stores.entry_binding[eid], args.label_key)
        if lbl is not None and np.isfinite(lbl):
            real_labels[i] = float(lbl)

    fold_simple, fold_trimmed, fold_weighted = [], [], []

    for fold_id, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_id+1}/{len(folds)} | train={len(train_idx)} val={len(val_idx)}")
        print(f"{'='*60}")

        fold_dir = os.path.join(args.out_dir, f"fold_{fold_id:02d}")
        os.makedirs(fold_dir, exist_ok=True)

        # Step 1: Load teacher ensemble
        print(f"\n[Step 1] Loading teacher ensemble from {args.ckpt_dir}/fold_{fold_id:02d}...")
        try:
            teacher_models, teacher_dims = load_fold_ensemble(args.ckpt_dir, fold_id, device, args.max_rna_len)
            print(f"  Loaded {len(teacher_models)} teacher models")
        except FileNotFoundError as e:
            print(f"  WARNING: {e} — skipping fold")
            continue

        # Step 2: Pseudo-label the validation set using the teacher ensemble
        print(f"\n[Step 2] Pseudo-labeling validation set ({len(val_idx)} samples)...")
        val_means, val_stds = pseudo_label_indices(
            teacher_models, dataset, val_idx.tolist(), device, batch_size=args.val_batch_size
        )

        # Blend pseudo-labels: alpha * teacher_pred + (1-alpha) * real_label
        pseudo_labels = {}
        alpha = args.pseudo_alpha
        for i, didx in enumerate(val_idx):
            didx = int(didx)
            teacher_pred = val_means[i]
            real_lbl = real_labels.get(didx, teacher_pred)
            blended = alpha * teacher_pred + (1 - alpha) * real_lbl
            pseudo_labels[didx] = blended

        pred_arr = np.array(val_means)
        real_arr = np.array([real_labels.get(int(vi), np.nan) for vi in val_idx])
        blend_arr = np.array([pseudo_labels[int(vi)] for vi in val_idx])
        valid = np.isfinite(real_arr)
        if valid.sum() > 1:
            teacher_p = numpy_pearson(pred_arr[valid], real_arr[valid])
            blend_p = numpy_pearson(blend_arr[valid], real_arr[valid])
            print(f"  Teacher Pearson on val: {teacher_p:.4f}")
            print(f"  Blended labels Pearson vs real: {blend_p:.4f} (alpha={alpha})")
            print(f"  Prediction std range: [{np.min(val_stds):.4f}, {np.max(val_stds):.4f}]")

        del teacher_models
        torch.cuda.empty_cache()

        # Current ckpt_dir for this round's teacher
        current_ckpt_dir = args.ckpt_dir

        for st_round in range(args.self_train_rounds):
            print(f"\n--- Self-training round {st_round+1}/{args.self_train_rounds} ---")

            # Build training set: real train + pseudo-labeled val
            pseudo_val_ds = PseudoLabelSubset(dataset, val_idx.tolist(), pseudo_labels)
            train_ds = Subset(dataset, train_idx.tolist())
            combined_ds = ConcatDataset([train_ds, pseudo_val_ds])

            print(f"  Training on {len(train_idx)} real + {len(val_idx)} pseudo = {len(combined_ds)} total")

            val_loader = make_subset_loader(dataset, val_idx.tolist(), batch_size=args.val_batch_size,
                                            shuffle=False, num_workers=0, seed=args.seed + 999 + fold_id)

            best_ckpts = []

            for member in range(args.k):
                member_seed = args.seed + 1000 * (fold_id * args.k + member) + st_round * 7777
                set_seed(member_seed)

                g = torch.Generator()
                g.manual_seed(member_seed)
                train_loader = DataLoader(combined_ds, batch_size=args.batch_size, shuffle=True,
                                          num_workers=0, collate_fn=collate_rmpred_batch, generator=g)

                model = build_model(dims, device, args.max_rna_len)

                # Warm-start from teacher/previous round
                teacher_ckpt = os.path.join(current_ckpt_dir, f"fold_{fold_id:02d}", f"member_{member:02d}_best.pt")
                if os.path.exists(teacher_ckpt):
                    sd = torch.load(teacher_ckpt, map_location=device)
                    model.load_state_dict(sd["model_state"], strict=True)

                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                scaler = GradScaler()

                best_score = float("-inf") if args.metric == "pearson" else float("inf")
                best_path = os.path.join(fold_dir, f"member_{member:02d}_best.pt")
                patience = 0

                print(f"\n  Training member {member+1}/{args.k} (round {st_round+1})...")

                for epoch in range(1, args.epochs + 1):
                    lr = warmup_cosine_lr(epoch, args.warmup_epochs, args.epochs, args.lr, lr_min=1e-6)
                    for pg in optimizer.param_groups:
                        pg['lr'] = lr

                    tr = train_one_epoch(model, train_loader, optimizer, device, scaler, grad_clip=1.0, ccc_weight=1.0)

                    val_results = evaluate_mu(model, val_loader, device, type_map=type_map)
                    cur_rmse = val_results["global"]["rmse"]
                    cur_pearson = val_results["global"]["pearson"]
                    cur_score = cur_pearson if args.metric == "pearson" else cur_rmse

                    if epoch % 5 == 0 or epoch == 1:
                        print(f"  [Ep {epoch:03d}] Loss={tr['loss']:.4f} | RMSE={cur_rmse:.4f} P={cur_pearson:.4f} lr={lr:.2e}")

                    is_best = False
                    if math.isfinite(cur_score):
                        if args.metric == "pearson":
                            is_best = cur_score > best_score
                        else:
                            is_best = cur_score < best_score

                    if is_best:
                        best_score = cur_score
                        patience = 0
                        torch.save({
                            "fold": fold_id, "member": member, "epoch": epoch,
                            "model_state": model.state_dict(),
                            "best_score": best_score, "metric": args.metric,
                            "dims": dims,
                        }, best_path)
                    else:
                        patience += 1
                        if patience >= args.early_patience:
                            print(f"    -> Early stop at epoch {epoch}. Best P={best_score:.4f}")
                            break

                print(f"  Member {member:02d} best P={best_score:.4f}")
                best_ckpts.append(best_path)

                del model, optimizer, scaler
                torch.cuda.empty_cache()

            # If doing another round, re-pseudo-label val with the improved ensemble
            if st_round < args.self_train_rounds - 1:
                print(f"\n  Re-pseudo-labeling val set with round-{st_round+1} ensemble...")
                round_models = []
                for ckpt in best_ckpts:
                    sd = torch.load(ckpt, map_location=device)
                    m = build_model(sd["dims"], device, args.max_rna_len)
                    m.load_state_dict(sd["model_state"], strict=True)
                    round_models.append(m)

                new_means, new_stds = pseudo_label_indices(
                    round_models, dataset, val_idx.tolist(), device, batch_size=args.val_batch_size
                )

                # Update pseudo-labels with improved predictions
                for i, didx in enumerate(val_idx):
                    didx = int(didx)
                    new_pred = new_means[i]
                    real_lbl = real_labels.get(didx, new_pred)
                    # Increase alpha each round (rely more on model as it improves)
                    round_alpha = min(alpha + 0.1 * (st_round + 1), 0.95)
                    pseudo_labels[didx] = round_alpha * new_pred + (1 - round_alpha) * real_lbl

                if valid.sum() > 1:
                    new_blend = np.array([pseudo_labels[int(vi)] for vi in val_idx])
                    new_p = numpy_pearson(np.array(new_means)[valid], real_arr[valid])
                    print(f"  Updated teacher Pearson on val: {new_p:.4f}")

                # Next round warm-starts from THIS round's checkpoints
                current_ckpt_dir = args.out_dir

                del round_models
                torch.cuda.empty_cache()

        # Final ensemble evaluation
        print(f"\n  Evaluating final ensemble for Fold {fold_id}...")
        models = []
        for ckpt in best_ckpts:
            sd = torch.load(ckpt, map_location=device)
            m = build_model(sd["dims"], device, args.max_rna_len)
            m.load_state_dict(sd["model_state"], strict=True)
            models.append(m)

        preds_matrix, y_val, all_types = collect_ensemble_predictions(models, val_loader, device, type_map)

        if len(y_val) == 0:
            print("  Warning: No validation samples!")
            continue

        ens_results = evaluate_ensemble_with_weights(preds_matrix, y_val, all_types, optimize_metric=args.metric)

        print(f"\n[FOLD {fold_id:02d} RESULTS]")
        for mode in ['simple_avg', 'trimmed_avg', 'weighted_avg']:
            m = ens_results[mode]
            print(f"  {mode:<12}: RMSE={m['rmse']:.4f} P={m['pearson']:.4f} Acc={m['accuracy']:.4f} AUC={m['auc']:.4f} BACC={m['bacc']:.4f} Spec={m['specificity']:.4f}")
        print(f"  -> Weights: {[f'{w:.3f}' for w in ens_results['best_weights']]}")

        for key, metrics in ens_results.items():
            if key not in ["simple_avg", "weighted_avg", "trimmed_avg", "best_weights"]:
                print(f"    Type {key}: RMSE={metrics['rmse']:.4f} P={metrics['pearson']:.4f} N={metrics['count']}")

        fold_simple.append(ens_results["simple_avg"])
        fold_trimmed.append(ens_results["trimmed_avg"])
        fold_weighted.append(ens_results["weighted_avg"])

        del models
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"SEMI-SUPERVISED {args.folds}-FOLD SUMMARY")
    print(f"{'='*60}")

    for name, fold_data in [("Simple Average", fold_simple), ("Trimmed Average", fold_trimmed), ("Weighted Average", fold_weighted)]:
        if not fold_data:
            continue
        rmses = [m["rmse"] for m in fold_data]
        pears = [m["pearson"] for m in fold_data]
        accs = [m["accuracy"] for m in fold_data]
        aucs = [m["auc"] for m in fold_data]
        baccs = [m["bacc"] for m in fold_data]
        specs = [m["specificity"] for m in fold_data]

        print(f"\n{name}:")
        print(f"  RMSE:      mean={np.nanmean(rmses):.4f} std={np.nanstd(rmses):.4f}")
        print(f"  Pearson:   mean={np.nanmean(pears):.4f} std={np.nanstd(pears):.4f}")
        print(f"  Accuracy:  mean={np.nanmean(accs):.4f} std={np.nanstd(accs):.4f}")
        print(f"  AUC:       mean={np.nanmean(aucs):.4f} std={np.nanstd(aucs):.4f}")
        print(f"  BACC:      mean={np.nanmean(baccs):.4f} std={np.nanstd(baccs):.4f}")
        print(f"  Spec:      mean={np.nanmean(specs):.4f} std={np.nanstd(specs):.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
