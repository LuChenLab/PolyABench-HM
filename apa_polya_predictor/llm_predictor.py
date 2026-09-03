"""LLM-based APA/polyA site prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from apa_polya_predictor.llm.collators import APALLMCollator
from apa_polya_predictor.llm.models import build_tokenizer_and_model, canonical_method


class SequenceDataset(Dataset):
    def __init__(self, headers: List[str], sequences: List[str]):
        self.headers = headers
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.headers)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {
            "header": self.headers[idx],
            "seq": self.sequences[idx],
            "label": "0",
        }


def _args_for_method(
    method: str,
    weights_dir: Path,
    species: str,
    checkpoint_path: Path,
) -> argparse.Namespace:
    method = canonical_method(method)
    pretrained_root = weights_dir / "pretrained" / "llm"

    if method == "DNABERT1":
        model_path = pretrained_root / "dnabert1"
    elif method == "DNABERT2":
        model_path = pretrained_root / "dnabert2"
    elif method == "GENA-LM-base":
        model_path = pretrained_root / "gena_lm_base"
    elif method == "RNABERT":
        model_path = None
    elif method == "UTRLM":
        model_path = pretrained_root / "utrlm" / "model.pt"
    else:
        raise ValueError(method)

    if method == "RNABERT":
        vocab_path = pretrained_root / "rnabert" / "vocab.txt"
        model_config = pretrained_root / "rnabert" / "RNABERT.json"
    elif method == "UTRLM":
        vocab_path = pretrained_root / "utrlm" / "UTRLM.txt"
        model_config = None
    else:
        vocab_path = None
        model_config = None

    return argparse.Namespace(
        method=method,
        model_path=str(model_path) if model_path else None,
        model_config=str(model_config) if model_config else None,
        vocab_path=str(vocab_path) if vocab_path else None,
        max_seq_len=None,
        kmer=None,
        pad_token_id=None,
        freeze_backbone=True,
    )


class LLMPredictor:
    """Run inference for one or more LLM models."""

    def __init__(
        self,
        weights_dir: str,
        species: str,
        models: List[str],
        device: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        self.weights_dir = Path(weights_dir)
        self.species = species.capitalize()
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.models = [canonical_method(m) for m in models]
        self._tokenizers: Dict[str, Any] = {}
        self._models: Dict[str, torch.nn.Module] = {}
        self._runtimes: Dict[str, Dict[str, Any]] = {}

        llm_ckpt_dir = self.weights_dir / self.species / "llm"
        for method in self.models:
            ckpt = llm_ckpt_dir / f"{method}_best_model.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"LLM checkpoint not found: {ckpt}")

            args = _args_for_method(method, self.weights_dir, self.species, ckpt)
            tokenizer, model, runtime = build_tokenizer_and_model(args)
            ckpt_state = torch.load(str(ckpt), map_location="cpu")
            # Trainer checkpoints are wrapped dicts; extract the actual weights.
            if isinstance(ckpt_state, dict) and "model_state_dict" in ckpt_state:
                ckpt_state = ckpt_state["model_state_dict"]
            ckpt_state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ckpt_state.items()}
            model.load_state_dict(ckpt_state, strict=True)
            model.to(self.device)
            model.eval()

            self._tokenizers[method] = tokenizer
            self._models[method] = model
            self._runtimes[method] = runtime

    def predict_fasta(self, fasta_path: str) -> pd.DataFrame:
        from apa_polya_predictor.fasta_utils import read_fasta, normalize_sequence

        headers, seqs = read_fasta(fasta_path)
        normalized = [normalize_sequence(s, 240) for s in seqs]
        ds = SequenceDataset(headers, normalized)

        records: List[Dict[str, Any]] = []
        for method in self.models:
            rt = self._runtimes[method]
            collator = APALLMCollator(
                tokenizer=self._tokenizers[method],
                max_seq_len=rt["max_seq_len"],
                kmer=rt["kmer"],
                replace_t_with_u=rt["replace_t_with_u"],
                replace_u_with_t=rt["replace_u_with_t"],
                pad_token_id=rt["pad_token_id"],
                add_cls_prefix=rt["add_cls_prefix"],
            )
            loader = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collator,
                pin_memory=self.device.type == "cuda",
            )

            probs_all: List[float] = []
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                logits = self._models[method](input_ids)
                probs = torch.softmax(logits.detach(), dim=-1)[:, 1].cpu().numpy()
                probs_all.extend(probs.astype(float).tolist())

            if not records:
                records = [{"header": h, f"{method}_prob": p} for h, p in zip(headers, probs_all)]
            else:
                for rec, p in zip(records, probs_all):
                    rec[f"{method}_prob"] = p

        return pd.DataFrame(records)
