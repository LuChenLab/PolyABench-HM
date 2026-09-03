from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


LABEL2ID = {"0": 0, "1": 1}


def seq2kmer(seq: str, kmer: int) -> str:
    if kmer <= 0:
        return seq
    if len(seq) < kmer:
        return seq
    return " ".join(seq[i : i + kmer] for i in range(0, len(seq) - kmer + 1))


class APALLMCollator:
    """Collator compatible with RNABERT/UTR-LM/DNABERT/GENA style tokenizers."""

    def __init__(
        self,
        tokenizer,
        max_seq_len: int,
        kmer: int = 1,
        replace_t_with_u: bool = False,
        replace_u_with_t: bool = True,
        pad_token_id: int = 0,
        add_cls_prefix: bool = True,
    ):
        assert not (replace_t_with_u and replace_u_with_t), "Only one replacement direction is allowed."
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.kmer = int(kmer)
        self.replace_t_with_u = bool(replace_t_with_u)
        self.replace_u_with_t = bool(replace_u_with_t)
        self.pad_token_id = int(pad_token_id)
        self.add_cls_prefix = bool(add_cls_prefix)

    def _prepare_seq(self, seq: str) -> str:
        seq = seq.upper()
        if self.replace_t_with_u:
            seq = seq.replace("T", "U")
        if self.replace_u_with_t:
            seq = seq.replace("U", "T")
        text = seq2kmer(seq, self.kmer) if self.kmer > 0 else seq
        if self.add_cls_prefix:
            text = "[CLS] " + text
        return text

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        input_ids_stack: List[List[int]] = []
        labels_stack: List[int] = []

        for item in batch:
            text = self._prepare_seq(item["seq"])
            tokenized = self.tokenizer(text)
            input_ids = tokenized["input_ids"]
            input_ids = [0 if x is None else int(x) for x in input_ids]

            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[: self.max_seq_len]
            if len(input_ids) < self.max_seq_len:
                input_ids = input_ids + [self.pad_token_id] * (self.max_seq_len - len(input_ids))

            input_ids_stack.append(input_ids)
            labels_stack.append(LABEL2ID[str(item["label"])])

        return {
            "input_ids": torch.from_numpy(np.asarray(input_ids_stack, dtype=np.int64)),
            "labels": torch.from_numpy(np.asarray(labels_stack, dtype=np.int64)),
        }
