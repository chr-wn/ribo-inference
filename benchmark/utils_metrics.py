
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UserWarning, message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, balanced_accuracy_score, mean_squared_error
from scipy.stats import pearsonr

def calculate_metrics(preds, targets, threshold=4.0):
    mask = np.isfinite(preds) & np.isfinite(targets)
    preds = preds[mask]
    targets = targets[mask]
    
    if len(preds) < 2:
        return {
            "rmse": float("nan"), "pearson": float("nan"),
            "accuracy": float("nan"), "auc": float("nan"),
            "specificity": float("nan"), "bacc": float("nan")
        }

    rmse = np.sqrt(mean_squared_error(targets, preds))
    pearson, _ = pearsonr(preds, targets)
    
    y_true_bin = (targets >= threshold).astype(int)
    y_pred_bin = (preds >= threshold).astype(int)
    
    accuracy = accuracy_score(y_true_bin, y_pred_bin)
    bacc = balanced_accuracy_score(y_true_bin, y_pred_bin)
    
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    
    try:
        auc = roc_auc_score(y_true_bin, preds)
    except ValueError:
        auc = float("nan")

    return {
        "rmse": rmse,
        "pearson": pearson,
        "accuracy": accuracy,
        "auc": auc,
        "specificity": specificity,
        "bacc": bacc
    }
