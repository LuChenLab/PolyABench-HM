"""Build DeepPASTA RNA structure cache on-the-fly using RNAshapes."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from apa_polya_predictor.models.deeppasta.build_deeppasta_structure_cache import (
    process_one_fasta,
)


def find_rnashapes() -> Optional[str]:
    """Return path to RNAshapes binary.

    Search order:
      1. bundled binary inside apa_polya_predictor/models/deeppasta
      2. PATH
    """
    import apa_polya_predictor.models.deeppasta as _dp

    bundled = Path(_dp.__file__).resolve().parent / "RNAshapes"
    if bundled.exists() and bundled.is_file():
        return str(bundled)
    return shutil.which("RNAshapes") or shutil.which("RNAshapes2") or shutil.which("rnashapes")


def build_structure_cache(
    fasta_path: str,
    out_prefix: str,
    rnashapes_bin: Optional[str] = None,
    tmp_dir: Optional[str] = None,
    seq_len: int = 240,
    half_len: int = 120,
    n_structures: int = 3,
    rnashapes_c: int = 5,
    rnashapes_t: int = 1,
) -> dict:
    """Build a DeepPASTA structure cache for a 240-nt FASTA file."""
    if rnashapes_bin is None:
        rnashapes_bin = find_rnashapes()
    if rnashapes_bin is None:
        raise RuntimeError(
            "RNAshapes binary not found. Please install RNAshapes or provide "
            "its path with --rnashapes_bin."
        )
    rnashapes_bin = str(Path(rnashapes_bin).expanduser().resolve())

    if tmp_dir is None:
        tmp_dir = str(Path(out_prefix).parent / ".apa_struct_tmp")

    if seq_len != half_len * 2:
        raise ValueError(f"seq_len must equal 2*half_len. Got {seq_len}, {half_len}")

    # verify binary is callable
    try:
        result = subprocess.run(
            [rnashapes_bin, "--help"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"Cannot execute RNAshapes binary: {rnashapes_bin}") from e

    return process_one_fasta(
        fasta=fasta_path,
        out_prefix=out_prefix,
        tmp_dir=tmp_dir,
        rnashapes_bin=rnashapes_bin,
        seq_len=seq_len,
        half_len=half_len,
        n_structures=n_structures,
        rnashapes_c=rnashapes_c,
        rnashapes_t=rnashapes_t,
    )
