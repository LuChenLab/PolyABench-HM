from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from Bio import SeqIO
from torch.utils.data import Dataset


@dataclass
class APARecord:
    header: str
    seq: str
    label: str


def parse_label(description: str) -> str:
    """Parse binary label from FASTA description.

    Supported examples:
      >id:1
      >id label=1
      >id|label=0
      >id 0

    The returned label is always string "0" or "1".
    """
    desc = description.strip()
    candidates: List[str] = []

    for sep in ["label=", "label:", "y=", "y:"]:
        if sep in desc:
            tail = desc.split(sep)[-1]
            candidates.append(tail.replace("|", " ").replace(":", " ").split()[0])

    # m6A/old cache style: description.split(":")[-1]
    if ":" in desc:
        candidates.append(desc.split(":")[-1].strip().split()[0])
    if "|" in desc:
        candidates.append(desc.split("|")[-1].strip().split()[0])
    if desc.split():
        candidates.append(desc.split()[-1])

    for c in candidates:
        c = str(c).strip()
        if c in {"0", "1"}:
            return c
        if c.lower() in {"neg", "negative", "false"}:
            return "0"
        if c.lower() in {"pos", "positive", "true"}:
            return "1"

    raise ValueError(
        f"Could not parse binary label from FASTA header: {description!r}. "
        "Use headers like '>seq_id:1' or '>seq_id label=0'."
    )


class APAFastaDataset(Dataset):
    def __init__(self, fasta_path: str | Path, seed: int | None = None, shuffle: bool = False):
        self.fasta_path = Path(fasta_path)
        if not self.fasta_path.exists():
            raise FileNotFoundError(f"FASTA not found: {self.fasta_path}")

        records: List[APARecord] = []
        for rec in SeqIO.parse(str(self.fasta_path), "fasta"):
            seq = str(rec.seq).upper().replace(" ", "").replace("\n", "")
            label = parse_label(rec.description)
            records.append(APARecord(header=rec.description, seq=seq, label=label))

        if shuffle:
            import numpy as np
            rng = np.random.RandomState(seed if seed is not None else 0)
            rng.shuffle(records)

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        r = self.records[idx]
        return {"header": r.header, "seq": r.seq, "label": r.label}
