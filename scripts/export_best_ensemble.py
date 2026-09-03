#!/usr/bin/env python3
"""Export the ablation-selected best GB ensemble for Human/Mouse.

Reads pre-computed val predictions from the benchmark outputs, trains a
GradientBoosting classifier on the selected model subset, and saves it as the
package's default 'ensemble' meta-learner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
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

# Best subsets from greedy-forward selection with Gradient Boosting (07_01_18, v2):
# anchored at DeepPASTA (best single model by test AUPRC), members and subset size
# selected by validation AUPRC. Listed in order of selection; the order defines the
# feature order of the exported GB meta-learner and must match best_ensemble.json.
BEST_ENSEMBLE: dict[str, list[str]] = {
    "Human": [
        "deeppasta",
        "DNABERT2",
        "pasnet",
        "GENA-LM-base",
        "polyaid",
        "UTRLM",
        "DNABERT1",
    ],
    "Mouse": [
        "deeppasta",
        "pasnet",
        "DNABERT2",
        "polyaid",
        "DNABERT1",
        "UTRLM",
        "deeppolya",
        "deepgsr",
    ],
}

# Mapping from model name to the val predictions path in benchmark outputs.
# Traditional models use test_predictions.csv under eval_for_stacking;
# LLMs use val_predictions.csv.
VAL_PRED_TEMPLATES: dict[str, str] = {
    "deeppolya": "{root}/outputs/eval_for_stacking/{species}/train_1/val/deeppolya/test_predictions.csv",
    "deerect": "{root}/outputs/eval_for_stacking/{species}/train_1/val/deerect/test_predictions.csv",
    "polyaid": "{root}/outputs/eval_for_stacking/{species}/train_1/val/polyaid/test_predictions.csv",
    "deepgsr": "{root}/outputs/eval_for_stacking/{species}/train_1/val/deepgsr/test_predictions.csv",
    "pasnet": "{root}/outputs/eval_for_stacking/{species}/train_1/val/pasnet/test_predictions.csv",
    "deeppasta": "{root}/outputs/eval_for_stacking/{species}/train_1/val/deeppasta/test_predictions.csv",
    "UTRLM": "{root}/outputs/eval_for_stacking/{species}/train_1/val/UTRLM/val_predictions.csv",
    "RNABERT": "{root}/outputs/eval_for_stacking/{species}/train_1/val/RNABERT/val_predictions.csv",
    "GENA-LM-base": "{root}/outputs/eval_for_stacking/{species}/train_1/val/GENA-LM-base/val_predictions.csv",
    "DNABERT1": "{root}/outputs/eval_for_stacking/{species}/train_1/val/DNABERT1/val_predictions.csv",
    "DNABERT2": "{root}/outputs/eval_for_stacking/{species}/train_1/val/DNABERT2/val_predictions.csv",
}


def _prob_col(df: pd.DataFrame) -> str:
    for col in ("prob", "prob_positive"):
        if col in df.columns:
            return col
    raise ValueError(f"No probability column found; columns={list(df.columns)}")


def load_val_predictions(benchmark_root: str, species: str, models: list[str]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for name in models:
        path = VAL_PRED_TEMPLATES[name].format(root=benchmark_root, species=species)
        df = pd.read_csv(path)
        prob_col = _prob_col(df)
        sub = df[["header", "label", prob_col]].copy()
        sub.columns = ["header", "label", f"{name}_prob"]
        if merged is None:
            merged = sub
        else:
            merged = merged.merge(sub[["header", f"{name}_prob"]], on="header", how="inner")
    if merged is None:
        raise ValueError("No models provided")
    return merged


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "MCC") -> tuple[float, float]:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t = 0.5
    best_v = -1e9
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "MCC":
            v = matthews_corrcoef(y_true, y_pred)
        elif metric == "F1":
            v = f1_score(y_true, y_pred, zero_division=0)
        else:
            v = (y_pred == y_true).mean()
        if v > best_v:
            best_v = float(v)
            best_t = float(t)
    return best_t, best_v


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "AUROC": float(roc_auc_score(y_true, y_prob)),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(specificity),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Threshold": float(threshold),
    }


def train_and_save(
    benchmark_root: str,
    species: str,
    output_dir: str,
    n_estimators: int = 150,
    learning_rate: float = 0.03,
    max_depth: int = 2,
    random_state: int = 2024,
) -> dict:
    models = BEST_ENSEMBLE[species]
    print(f"[{species}] Training best ensemble on {len(models)} models: {models}")

    df = load_val_predictions(benchmark_root, species, models)
    prob_cols = [f"{m}_prob" for m in models]
    X = df[prob_cols].values.astype(float)
    y = df["label"].values.astype(int)

    gb = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )
    gb.fit(X, y)

    val_prob = gb.predict_proba(X)[:, 1]
    best_t, best_v = find_best_threshold(y, val_prob, "MCC")
    metrics = compute_metrics(y, val_prob, best_t)

    print(f"[{species}] Trained on {len(y)} samples")
    print(f"[{species}] Best threshold (MCC): {best_t:.4f} (MCC={best_v:.4f})")
    print(f"[{species}] Metrics: {metrics}")

    out_path = Path(output_dir) / species / "gb_ensemble_best.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": gb,
            "model_names": prob_cols,
            "threshold": best_t,
            "threshold_metric": "MCC",
            "metrics": metrics,
            "species": species,
            "base_models": models,
        },
        out_path,
    )
    print(f"[{species}] Saved meta-learner to {out_path}")
    return {"species": species, "models": models, "metrics": metrics, "path": str(out_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export best ablated ensemble GB model")
    parser.add_argument("--benchmark_root", default="/mnt/raid61/Personal_data/yangqingxin/software/apa-polyA-benchmark", help="Path to apa-polyA-benchmark repository")
    parser.add_argument("--species", default=None, choices=["Human", "Mouse"], help="Species to export. If omitted, both Human and Mouse are exported.")
    parser.add_argument("--output_dir", default="/mnt/raid61/Personal_data/yangqingxin/software/apa-polya-predictor/weights", help="Directory to save the meta-learner")
    parser.add_argument("--n_estimators", type=int, default=150)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--max_depth", type=int, default=2)
    parser.add_argument("--random_state", type=int, default=2024)
    parser.add_argument("--write_config", default=None, help="Optional JSON path to write the best-ensemble config")
    args = parser.parse_args()

    species_list = [args.species] if args.species else ["Human", "Mouse"]
    results = []
    for sp in species_list:
        results.append(
            train_and_save(
                benchmark_root=args.benchmark_root,
                species=sp,
                output_dir=args.output_dir,
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                max_depth=args.max_depth,
                random_state=args.random_state,
            )
        )

    if args.write_config:
        config = {sp: BEST_ENSEMBLE[sp] for sp in species_list}
        Path(args.write_config).parent.mkdir(parents=True, exist_ok=True)
        with open(args.write_config, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Wrote best-ensemble config to {args.write_config}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
