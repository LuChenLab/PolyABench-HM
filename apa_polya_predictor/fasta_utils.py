"""FASTA preprocessing for APA/polyA site prediction.

Input sequences are normalized to exactly 240 nt centered on the candidate
poly(A) site:
- longer than 240 nt  -> extract a 240-nt window centered on the sequence
- shorter than 240 nt -> pad with 'N' symmetrically around the sequence
- exactly 240 nt      -> keep as-is
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


NUC_TO_ID = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "U": 3,
    "N": 4,
}


def read_fasta(path: str) -> Tuple[List[str], List[str]]:
    """Read a FASTA file and return (headers, sequences)."""
    headers: List[str] = []
    seqs: List[str] = []

    current_header: str | None = None
    current_chunks: List[str] = []

    def flush() -> None:
        nonlocal current_header, current_chunks
        if current_header is None:
            return
        seq = "".join(current_chunks).strip().upper()
        if not seq:
            raise ValueError(f"Empty sequence found for header: {current_header}")
        headers.append(current_header)
        seqs.append(seq)
        current_header = None
        current_chunks = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                current_header = line[1:]
            else:
                current_chunks.append(line)
        flush()

    if not headers:
        raise ValueError(f"No sequences found in FASTA: {path}")
    return headers, seqs


def normalize_sequence(seq: str, target_len: int = 240) -> str:
    """Crop or pad a sequence to target_len around its center."""
    seq = seq.upper()
    n = len(seq)
    if n == target_len:
        return seq
    if n > target_len:
        # extract a window centered on the sequence
        start = (n - target_len) // 2
        return seq[start : start + target_len]

    # pad with N symmetrically
    pad_total = target_len - n
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return "N" * pad_left + seq + "N" * pad_right


def encode_sequence_to_uint8(seq: str, seq_len: int = 240) -> np.ndarray:
    """Encode ACGTUN to uint8 ids; any other character becomes 4 (N)."""
    if len(seq) != seq_len:
        raise ValueError(f"Expected sequence length {seq_len}, got {len(seq)}")
    arr = np.empty(seq_len, dtype=np.uint8)
    for i, ch in enumerate(seq):
        arr[i] = NUC_TO_ID.get(ch, 4)
    return arr


ID_TO_NUC = ["A", "C", "G", "T", "N"]


def _decode_uint8_to_seq(arr: np.ndarray) -> str:
    return "".join(ID_TO_NUC[int(x)] for x in arr)


def preprocess_fasta(
    fasta_path: str,
    out_prefix: str,
    target_len: int = 240,
    write_normalized_fasta: bool = False,
) -> Tuple[str, List[str]]:
    """Read FASTA, normalize lengths, write uint8 cache, return prefix and headers."""
    headers, seqs = read_fasta(fasta_path)
    normalized = [normalize_sequence(s, target_len) for s in seqs]

    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    n = len(normalized)
    x_path = f"{out_prefix}.x.uint8.npy"
    h_path = f"{out_prefix}.headers.npy"

    x = np.empty((n, target_len), dtype=np.uint8)
    for i, seq in enumerate(normalized):
        x[i] = encode_sequence_to_uint8(seq, target_len)

    np.save(x_path, x)
    np.save(h_path, np.asarray(headers, dtype=object))

    normalized_fasta: Optional[str] = None
    if write_normalized_fasta:
        normalized_fasta = f"{out_prefix}.240.fa"
        with open(normalized_fasta, "w", encoding="utf-8") as f:
            for h, seq in zip(headers, normalized):
                f.write(f">{h}\n{seq}\n")

    if write_normalized_fasta:
        return out_prefix, headers, normalized_fasta
    return out_prefix, headers
