#!/usr/bin/env python3
"""Command-line interface for apa-polya-predictor."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from apa_polya_predictor.predictor import DEFAULT_MODELS, ENSEMBLE_KEYWORD, SUPPORTED_MODELS, Predictor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict APA/polyA sites from DNA/RNA sequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Best greedy-forward-selected ensemble (recommended)
  apa-polya-predict -i examples/example.fa -s Human -m ensemble -o predictions.txt

  # Default: 5 traditional models, per-model probabilities and binary labels
  apa-polya-predict -i examples/example.fa -s Human -o predictions.txt

  # Use a specific model
  apa-polya-predict -i examples/example.fa -s Mouse -m deeppolya -o predictions.txt

  # Use all 11 models
  apa-polya-predict -i examples/example.fa -s Human -m all+llm -o predictions.txt

  # Include deeppasta (RNAshapes must be installed and in PATH, or use --rnashapes_bin)
  apa-polya-predict -i examples/example.fa -s Human -m all+llm,deeppasta -o predictions.txt

  # Force CPU / GPU
  apa-polya-predict -i examples/example.fa -s Human -m ensemble -o predictions.txt --device cpu
        """.strip(),
    )
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file.")
    parser.add_argument("-s", "--species", required=True, choices=["Human", "Mouse"], help="Species model to use.")
    parser.add_argument(
        "-m",
        "--models",
        default="all",
        help=(
            "Comma-separated model names, 'all' for default one-hot models, "
            "'all+llm' for all 11 models, or 'ensemble' for the greedy-forward-selected "
            "best ensemble. "
            f"Supported: {', '.join(SUPPORTED_MODELS)}. "
            "Default: all (deeppolya,deerect,polyaid,deepgsr,pasnet)."
        ),
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Decision threshold for binary prediction. Used for *_pred columns in "
            "single/multi-model runs (default: 0.5). For '-m ensemble', this overrides "
            "the validation-tuned threshold stored in the meta-learner."
        ),
    )
    parser.add_argument("-o", "--output", required=True, help="Output TXT/TSV file path.")
    parser.add_argument(
        "--weights_dir",
        default=None,
        help="Directory containing weights/Human and weights/Mouse. Default: package weights/ directory.",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Inference batch size for traditional models.")
    parser.add_argument("--llm_batch_size", type=int, default=32, help="Inference batch size for LLM models.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--device", default=None, help="Device: cpu/cuda. Auto-detected if omitted.")
    parser.add_argument(
        "--rnashapes_bin",
        default=None,
        help="Path to RNAshapes binary. Required when using deeppasta. Searches PATH if omitted.",
    )
    parser.add_argument("--keep_cache", action="store_true", help="Keep the intermediate uint8 cache.")
    parser.add_argument("--cache_dir", default=".apa_polya_cache", help="Directory for intermediate cache.")
    parser.add_argument(
        "--output_format",
        default="tsv",
        choices=["tsv", "csv"],
        help="Output file format. Default: tsv.",
    )
    return parser.parse_args(argv)


def resolve_weights_dir(weights_dir: str | None) -> str:
    if weights_dir:
        return weights_dir
    # package-relative weights/
    import apa_polya_predictor

    return str(Path(apa_polya_predictor.__file__).resolve().parent.parent / "weights")


def apply_best_ensemble(df, meta_bundle: dict, threshold: float = 0.5):
    """Apply the shipped greedy-forward gradient-boosting ensemble."""
    meta_learner = meta_bundle["model"]
    expected_cols = list(meta_bundle.get("model_names", []))
    prob_cols = [c for c in df.columns if c.endswith("_prob")]

    if expected_cols:
        missing = [c for c in expected_cols if c not in prob_cols]
        extra = [c for c in prob_cols if c not in expected_cols]
        if missing or extra:
            raise ValueError(
                f"Feature mismatch: best ensemble expects {sorted(expected_cols)}, "
                f"but current predictions have {sorted(prob_cols)}."
            )
        # Align columns to the training-time feature order of the meta-learner.
        prob_cols = expected_cols

    X = df[prob_cols].values.astype(float)
    ensemble_prob = meta_learner.predict_proba(X)[:, 1]

    # Use validation-tuned threshold unless user overrides.
    if threshold == 0.5 and "threshold" in meta_bundle:
        threshold = float(meta_bundle["threshold"])

    out = df.copy()
    out["ensemble_prob"] = ensemble_prob
    out["ensemble_pred"] = (ensemble_prob >= threshold).astype(int)
    return out, threshold


def add_per_model_predictions(df, threshold: float = 0.5):
    """Add binary *_pred columns for every *_prob column."""
    out = df.copy()
    for c in [c for c in out.columns if c.endswith("_prob") and c != "ensemble_prob"]:
        model_name = c[:-5]  # strip trailing '_prob'
        out[f"{model_name}_pred"] = (out[c] >= threshold).astype(int)
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    models_arg = args.models.strip()
    lower = models_arg.lower()
    using_best_ensemble = lower == ENSEMBLE_KEYWORD
    if using_best_ensemble:
        models = [ENSEMBLE_KEYWORD]
    elif lower == "all":
        models = list(DEFAULT_MODELS)
    elif lower == "all+llm":
        models = list(DEFAULT_MODELS) + ["DNABERT1", "DNABERT2", "RNABERT", "UTRLM", "GENA-LM-base"]
    elif lower == "all+llm,deeppasta" or lower == "deeppasta,all+llm":
        models = list(DEFAULT_MODELS) + ["deeppasta"] + ["DNABERT1", "DNABERT2", "RNABERT", "UTRLM", "GENA-LM-base"]
    else:
        # preserve original case for LLM names, lowercase for traditional
        models = [m.strip() for m in models_arg.split(",") if m.strip()]

    weights_dir = resolve_weights_dir(args.weights_dir)

    predictor = Predictor(
        weights_dir=weights_dir,
        species=args.species,
        models=models,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        llm_batch_size=args.llm_batch_size,
        rnashapes_bin=args.rnashapes_bin,
    )

    df = predictor.predict_fasta(
        fasta_path=args.input,
        cache_dir=args.cache_dir,
        keep_cache=args.keep_cache,
    )

    if using_best_ensemble:
        meta_path = Path(weights_dir) / args.species.capitalize() / "gb_ensemble_best.joblib"
        if not meta_path.exists():
            print(
                f"Error: best ensemble meta-learner not found at {meta_path}",
                file=sys.stderr,
            )
            return 1
        meta_bundle = joblib.load(meta_path)
        df, args.threshold = apply_best_ensemble(df, meta_bundle, threshold=args.threshold)
    else:
        # For single / multi-model runs, add a binary label for each model.
        df = add_per_model_predictions(df, threshold=args.threshold)

    # reorder columns: header, per-model prob/pred pairs, ensemble cols
    cols = ["header"]
    prob_cols = [c for c in df.columns if c.endswith("_prob") and c != "ensemble_prob"]
    for c in prob_cols:
        cols.append(c)
        pred_c = f"{c[:-5]}_pred"
        if pred_c in df.columns:
            cols.append(pred_c)
    if "ensemble_prob" in df.columns:
        cols.extend(["ensemble_prob", "ensemble_pred"])
    df = df[cols]

    sep = "\t" if args.output_format == "tsv" else ","
    df.to_csv(args.output, sep=sep, index=False)
    print(f"Saved predictions for {len(df)} sequences to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
