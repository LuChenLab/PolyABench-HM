import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


NUC_TO_ID = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "U": 3,
    "N": 4,
}


@dataclass
class OneHotCacheMeta:
    num_samples: int
    seq_len: int
    vocab_size: int = 5
    encoding: str = "A=0,C=1,G=2,T/U=3,N/other=4"
    source_fasta: Optional[str] = None



def parse_binary_label_from_header(header: str) -> int:
    """Parse a 0/1 label from common FASTA header styles.

    Supported examples:
      >seq1:1
      >seq1 label=0
      >seq1|label=1
      >seq1 y=0
      >seq1 class=1
      >seq1 0
    """
    patterns = [
        r"(?:^|[\s|;,_-])label\s*[=:]\s*([01])(?:$|[\s|;,_-])",
        r"(?:^|[\s|;,_-])y\s*[=:]\s*([01])(?:$|[\s|;,_-])",
        r"(?:^|[\s|;,_-])class\s*[=:]\s*([01])(?:$|[\s|;,_-])",
        r"[:|\s]([01])$",
    ]
    for pat in patterns:
        m = re.search(pat, header, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    raise ValueError(f"Cannot parse label from FASTA header: {header}")


def read_fasta_with_labels(path: str) -> Tuple[List[str], np.ndarray, List[str]]:
    seqs: List[str] = []
    labels: List[int] = []
    headers: List[str] = []

    header: Optional[str] = None
    seq_chunks: List[str] = []

    def flush() -> None:
        nonlocal header, seq_chunks
        if header is None:
            return
        seq = "".join(seq_chunks).strip().upper()
        if not seq:
            raise ValueError(f"Empty sequence found for header: {header}")
        label = parse_binary_label_from_header(header)
        if label not in (0, 1):
            raise ValueError(f"Label must be 0/1, got {label} in header: {header}")
        seqs.append(seq)
        labels.append(label)
        headers.append(header)
        header = None
        seq_chunks = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:]
            else:
                seq_chunks.append(line)
        flush()

    return seqs, np.asarray(labels, dtype=np.int64), headers



def encode_sequence_to_uint8(seq: str, seq_len: int) -> np.ndarray:
    if len(seq) != seq_len:
        raise ValueError(f"Expected sequence length {seq_len}, got {len(seq)}")
    arr = np.empty(seq_len, dtype=np.uint8)
    for i, ch in enumerate(seq.upper()):
        arr[i] = NUC_TO_ID.get(ch, 4)
    return arr



def build_uint8_cache_from_fasta(
    fasta_path: str,
    out_prefix: str,
    seq_len: Optional[int] = None,
) -> Dict[str, str]:
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    seqs, labels, headers = read_fasta_with_labels(fasta_path)
    if len(seqs) == 0:
        raise ValueError(f"No sequences found in FASTA: {fasta_path}")

    observed_lengths = sorted({len(x) for x in seqs})
    if seq_len is None:
        if len(observed_lengths) != 1:
            raise ValueError(
                f"FASTA contains mixed sequence lengths: {observed_lengths[:10]}. "
                "Please pass seq_len explicitly."
            )
        seq_len = observed_lengths[0]

    num_samples = len(seqs)

    x_path = f"{out_prefix}.x.uint8.npy"
    y_path = f"{out_prefix}.y.npy"
    h_path = f"{out_prefix}.headers.npy"
    m_path = f"{out_prefix}.meta.json"

    x_mem = np.lib.format.open_memmap(
        x_path,
        mode="w+",
        dtype=np.uint8,
        shape=(num_samples, seq_len),
    )

    for i, seq in enumerate(seqs):
        x_mem[i] = encode_sequence_to_uint8(seq, seq_len)

    np.save(y_path, labels.astype(np.int64))
    np.save(h_path, np.asarray(headers, dtype=object))

    meta = OneHotCacheMeta(
        num_samples=num_samples,
        seq_len=seq_len,
        source_fasta=fasta_path,
    )
    with open(m_path, "w", encoding="utf-8") as f:
        json.dump(meta.__dict__, f, indent=2)

    return {
        "x_path": x_path,
        "y_path": y_path,
        "headers_path": h_path,
        "meta_path": m_path,
    }


class UInt8SequenceDataset(Dataset):
    def __init__(
        self,
        x_path: str,
        y_path: str,
        headers_path: Optional[str] = None,
        mmap_mode: str = "r",
    ):
        self.x = np.load(x_path, mmap_mode=mmap_mode)
        self.y = np.load(y_path, mmap_mode=mmap_mode)
        self.headers = np.load(headers_path, allow_pickle=True) if headers_path else None

        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"x/y size mismatch: x has {self.x.shape[0]} rows, y has {self.y.shape[0]} rows"
            )

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        item = {
            "seq_idx": np.asarray(self.x[idx], dtype=np.uint8),
            "label": int(self.y[idx]),
        }
        if self.headers is not None:
            item["header"] = str(self.headers[idx])
        return item


class UInt8SequenceCollator:
    def __call__(self, batch: Sequence[Dict]):
        seq_idx = torch.as_tensor(
            np.stack([b["seq_idx"] for b in batch], axis=0),
            dtype=torch.long,
        )
        labels = torch.as_tensor([b["label"] for b in batch], dtype=torch.long)
        out = {
            "seq_idx": seq_idx,
            "labels": labels,
        }
        if "header" in batch[0]:
            out["headers"] = [b["header"] for b in batch]
        return out


class GPUOneHotEncoder(torch.nn.Module):
    def __init__(self, unknown_fill: float = 0.25):
        super().__init__()
        table = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [unknown_fill, unknown_fill, unknown_fill, unknown_fill],
            ],
            dtype=torch.float32,
        )
        self.embedding = torch.nn.Embedding.from_pretrained(table, freeze=True)

    def forward(self, seq_idx: torch.Tensor) -> torch.Tensor:
        # seq_idx: [B, L] -> onehot: [B, 4, L]
        return self.embedding(seq_idx).transpose(1, 2).contiguous()


class ConstantPad1dChannels(torch.nn.Module):
    def __init__(self, pad_left: int, pad_right: int, value: float = 0.25):
        super().__init__()
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        if self.pad_left == 0 and self.pad_right == 0:
            return x
        return torch.nn.functional.pad(x, (self.pad_left, self.pad_right), value=self.value)
