from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(labels, probs, threshold: float = 0.5) -> Dict[str, float]:
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs).astype(float)
    pred = (p >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    out = {
        "AUROC": roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
        "AUPRC": average_precision_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
        "MCC": matthews_corrcoef(y, pred),
        "F1": f1_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "Precision": precision_score(y, pred, zero_division=0),
        "Specificity": specificity,
        "Accuracy": accuracy_score(y, pred),
        "Threshold": threshold,
    }
    return {k: float(v) for k, v in out.items()}


def metric_is_better(metric_name: str) -> str:
    return "min" if metric_name.lower() in {"loss", "val_loss"} else "max"


def normalize_metric_name(name: str) -> str:
    aliases = {
        "auc": "AUROC",
        "roc_auc": "AUROC",
        "auroc": "AUROC",
        "pr_auc": "AUPRC",
        "prauc": "AUPRC",
        "auprc": "AUPRC",
        "mcc": "MCC",
        "f1": "F1",
        "recall": "Recall",
        "precision": "Precision",
        "specificity": "Specificity",
        "spe": "Specificity",
        "accuracy": "Accuracy",
        "acc": "Accuracy",
        "loss": "Val_Loss",
        "val_loss": "Val_Loss",
    }
    return aliases.get(name.lower(), name)
