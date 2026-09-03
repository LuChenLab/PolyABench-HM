from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .apa_fasta_dataset import APAFastaDataset
from .models import AVAILABLE_LLM_MODELS, build_tokenizer_and_model, canonical_method, get_model_defaults
from .trainer import APALLMTrainer, LLMTrainConfig


def str2list(v):
    if isinstance(v, list):
        return v
    return [x.strip() for x in str(v).split(",") if x.strip()]


def build_parser():
    p = argparse.ArgumentParser("APA LLM benchmark")
    p.add_argument("--method", required=True, choices=sorted(AVAILABLE_LLM_MODELS))
    p.add_argument("--dataset_train", required=True)
    p.add_argument("--dataset_val", required=True)
    p.add_argument("--dataset_test", required=True)
    p.add_argument("--model_path", default=None)
    p.add_argument("--model_config", default=None)
    p.add_argument("--vocab_path", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--best_checkpoint_dir", default=None)

    p.add_argument("--max_seq_len", type=int, default=None)
    p.add_argument("--kmer", type=int, default=None)
    p.add_argument("--pad_token_id", type=int, default=None)
    p.add_argument("--freeze_backbone", action="store_true")

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_train_epochs", type=int, default=10)
    p.add_argument("--early_stopping_patience", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--selection_metric", default="MCC")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--scheduler", choices=["none", "linear", "cosine"], default="none")
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2024)
    return p


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one(args):
    args.method = canonical_method(args.method)
    set_seed(args.seed)
    tokenizer, model, runtime = build_tokenizer_and_model(args)

    train_ds = APAFastaDataset(args.dataset_train, seed=args.seed, shuffle=True)
    val_ds = APAFastaDataset(args.dataset_val, seed=args.seed, shuffle=False)
    test_ds = APAFastaDataset(args.dataset_test, seed=args.seed, shuffle=False)

    cfg = LLMTrainConfig(
        method=args.method,
        output_dir=args.output_dir,
        best_checkpoint_dir=args.best_checkpoint_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_epochs=args.num_train_epochs,
        early_stopping_patience=args.early_stopping_patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        selection_metric=args.selection_metric,
        mixed_precision=args.mixed_precision,
        device=args.device,
        seed=args.seed,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        threshold=args.threshold,
        scheduler=args.scheduler,
        warmup_ratio=args.warmup_ratio,
    )
    trainer = APALLMTrainer(cfg, model, tokenizer, runtime, train_ds, val_ds, test_ds)
    return trainer.fit()


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    res = run_one(args)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
