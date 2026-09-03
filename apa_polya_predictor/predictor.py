"""APA/polyA site prediction using trained benchmark models."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from apa_polya_predictor.core.trainer import GPUOneHotEncoder
from apa_polya_predictor.dataset import UInt8SequenceCollator, UInt8SequenceDataset
from apa_polya_predictor.fasta_utils import preprocess_fasta
from apa_polya_predictor.models.deepgsr.model import DeepGSR240RegularizedClassifier, Trinucleotide2DEncoder
from apa_polya_predictor.models.pasnet.model import KmerIndexEncoder, PASNetClassifier
from apa_polya_predictor.models.deeppolya.model import DeepPolyAOneHotClassifier
from apa_polya_predictor.models.deerect.model import DeeReCTPolyAOneHot
from apa_polya_predictor.models.polyaid.model import PolyaIDOneHotClassifier

from apa_polya_predictor.models.deeppasta.model import (
    DeepPASTAFullClassifier,
    DeepPASTAStructCacheDataset,
    DeepPASTAStructCollator,
)

from apa_polya_predictor.llm_predictor import LLMPredictor, canonical_method as canonical_llm


TRADITIONAL_MODELS = ["deeppolya", "deerect", "polyaid", "deepgsr", "pasnet", "deeppasta"]
LLM_MODELS = ["DNABERT1", "DNABERT2", "RNABERT", "UTRLM", "GENA-LM-base"]
SUPPORTED_MODELS = TRADITIONAL_MODELS + LLM_MODELS
DEFAULT_MODELS = ["deeppolya", "deerect", "polyaid", "deepgsr", "pasnet"]


def _load_best_ensemble() -> Dict[str, List[str]]:
    """Load the greedy-forward-selected best model subsets for each species."""
    import json

    here = Path(__file__).resolve().parent
    path = here / "best_ensemble.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


BEST_ENSEMBLE: Dict[str, List[str]] = _load_best_ensemble()
ENSEMBLE_KEYWORD = "ensemble"


def _normalize_model_name(name: str) -> str:
    """Return canonical model name (lowercase for traditional, original case for LLM)."""
    n = name.strip()
    lower = n.lower()
    if lower in TRADITIONAL_MODELS:
        return lower
    return canonical_llm(n)


@dataclass
class BuiltModel:
    model: nn.Module
    encoder: nn.Module


def _args(**kwargs) -> argparse.Namespace:
    """Build a namespace with the defaults expected by runner-style builders."""
    defaults = dict(
        input_len=240,
        dna_only=False,
        pretrained_embedding=None,
        freeze_embedding=False,
        n_structures=3,
        structure_vocab_size=7,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def build_deeppolya() -> BuiltModel:
    args = _args()
    model = DeepPolyAOneHotClassifier(
        input_len=args.input_len,
        conv1_filters=16,
        conv1_kernel=8,
        conv2_filters=64,
        conv2_kernel=6,
        pool_size=2,
        pool_stride=2,
        hidden_dim=64,
        dropout=0.4,
        use_batchnorm=True,
        num_classes=2,
    )
    encoder = GPUOneHotEncoder(dna_only=args.dna_only, dtype=torch.float32)
    return BuiltModel(model, encoder)


def build_deerect() -> BuiltModel:
    args = _args()
    model = DeeReCTPolyAOneHot(
        input_len=args.input_len,
        patch_size=10,
        depth=16,
        num_hidden=64,
        num_groups=4,
        keep_prob=0.5,
    )
    encoder = GPUOneHotEncoder(dna_only=args.dna_only, dtype=torch.float32)
    return BuiltModel(model, encoder)


def build_polyaid() -> BuiltModel:
    args = _args()
    model = PolyaIDOneHotClassifier(
        input_len=args.input_len,
        conv_filters=512,
        kernel_size=8,
        pool_size=3,
        pool_stride=3,
        lstm_hidden=128,
        dense_dims=[256, 256, 128, 64],
        dropout=0.7,
        num_classes=2,
    )
    encoder = GPUOneHotEncoder(dna_only=args.dna_only, dtype=torch.float32)
    return BuiltModel(model, encoder)


def build_deepgsr() -> BuiltModel:
    args = _args()
    model = DeepGSR240RegularizedClassifier(
        input_len=args.input_len,
        conv1_filters=16,
        conv1_kernel_h=12,
        conv1_kernel_w=16,
        conv2_filters=32,
        conv2_kernel_h=5,
        conv2_kernel_w=8,
        pool1_h=1,
        pool1_w=2,
        pool2_h=1,
        pool2_w=2,
        adaptive_h=12,
        adaptive_w=8,
        hidden_dim=64,
        dropout=0.35,
        conv_dropout=0.05,
        fc_activation="tanh",
        conv1_init="xavier",
        use_batch_norm=True,
        num_classes=2,
    )
    encoder = Trinucleotide2DEncoder(dna_only=args.dna_only, dtype=torch.float32)
    return BuiltModel(model, encoder)


def build_pasnet() -> BuiltModel:
    args = _args()
    kmer = 5
    embed_dim = 90
    model = PASNetClassifier(
        input_len=args.input_len,
        kmer=kmer,
        embed_dim=embed_dim,
        filters=32,
        kernel_size=21,
        attention_heads=8,
        attention_head_dim=16,
        hidden_dim=64,
        dropout=0.5,
        num_classes=2,
        trainable_embedding=True,
        vocab_size=4 ** kmer + 1,
    )
    encoder = KmerIndexEncoder(kmer=kmer)
    return BuiltModel(model, encoder)


def build_deeppasta() -> BuiltModel:
    args = _args()
    model = DeepPASTAFullClassifier(
        input_len=args.input_len,
        n_structures=args.n_structures,
        structure_vocab_size=args.structure_vocab_size,
        conv_filters=128,
        kernel_size=8,
        pool_size=3,
        lstm_hidden=64,
        lstm_layers=1,
        fusion_hidden=256,
        dropout=0.3,
        num_classes=2,
    )
    encoder = nn.Identity()
    return BuiltModel(model, encoder)


TRADITIONAL_BUILDERS: Dict[str, Callable[[], BuiltModel]] = {
    "deeppolya": build_deeppolya,
    "deerect": build_deerect,
    "polyaid": build_polyaid,
    "deepgsr": build_deepgsr,
    "pasnet": build_pasnet,
    "deeppasta": build_deeppasta,
}


def _strip_compile_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("_orig_mod.") for k in state_dict):
        return state_dict
    return {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}


def load_checkpoint(built: BuiltModel, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    model_sd = ckpt.get("model_state_dict", ckpt)
    enc_sd = ckpt.get("encoder_state_dict", None)
    try:
        built.model.load_state_dict(model_sd)
    except RuntimeError:
        built.model.load_state_dict(_strip_compile_prefix(model_sd))
    if enc_sd is not None:
        try:
            built.encoder.load_state_dict(enc_sd)
        except RuntimeError:
            built.encoder.load_state_dict(_strip_compile_prefix(enc_sd))
    built.model.to(device)
    built.encoder.to(device)


class Predictor:
    """Run inference for one or more APA/polyA classification models."""

    def __init__(
        self,
        weights_dir: str,
        species: str,
        models: Optional[List[str]] = None,
        device: Optional[str] = None,
        batch_size: int = 256,
        num_workers: int = 0,
        llm_batch_size: int = 32,
        rnashapes_bin: Optional[str] = None,
    ):
        self.weights_dir = Path(weights_dir)
        self.species = species.lower()
        if self.species not in {"human", "mouse"}:
            raise ValueError(f"species must be Human or Mouse, got {species}")
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.llm_batch_size = llm_batch_size
        self.rnashapes_bin = rnashapes_bin

        if models is None:
            models = list(DEFAULT_MODELS)

        # Expand the 'ensemble' keyword to the greedy-forward-selected best subset.
        self._using_best_ensemble = False
        if len(models) == 1 and models[0].lower() == ENSEMBLE_KEYWORD:
            species_cap = species.capitalize()
            if species_cap not in BEST_ENSEMBLE:
                raise ValueError(
                    f"No best ensemble configured for {species}. "
                    f"Supported species: {list(BEST_ENSEMBLE.keys())}"
                )
            models = list(BEST_ENSEMBLE[species_cap])
            self._using_best_ensemble = True

        models = [_normalize_model_name(m) for m in models]
        bad = [m for m in models if m not in SUPPORTED_MODELS]
        if bad:
            raise ValueError(f"Unsupported models: {bad}. Supported: {SUPPORTED_MODELS}")
        self.models = models

        self.traditional_models = [m for m in models if m in TRADITIONAL_MODELS]
        self.llm_models = [m for m in models if m in LLM_MODELS]
        self._has_deeppasta = "deeppasta" in self.traditional_models
        self._non_pasta_models = [m for m in self.traditional_models if m != "deeppasta"]

        self._built: Dict[str, BuiltModel] = {}
        for name in self.traditional_models:
            built = TRADITIONAL_BUILDERS[name]()
            ckpt = self.weights_dir / self.species.capitalize() / f"{name}_best_model.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
            load_checkpoint(built, str(ckpt), self.device)
            built.model.eval()
            built.encoder.eval()
            self._built[name] = built

        self._llm_predictor: Optional[LLMPredictor] = None
        if self.llm_models:
            self._llm_predictor = LLMPredictor(
                weights_dir=str(self.weights_dir),
                species=self.species,
                models=self.llm_models,
                device=str(self.device),
                batch_size=llm_batch_size,
                num_workers=num_workers,
            )

    def _make_non_pasta_loader(self, cache_prefix: str) -> DataLoader:
        ds = UInt8SequenceDataset(
            x_path=f"{cache_prefix}.x.uint8.npy",
            y_path=f"{cache_prefix}.y.npy",
            headers_path=f"{cache_prefix}.headers.npy",
            mmap_mode="r",
        )
        if np.any(ds.y != 0):
            warnings.warn("Input cache contains non-zero labels; they will be ignored during prediction.")
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=UInt8SequenceCollator(),
            pin_memory=self.device.type == "cuda",
        )

    def _make_deeppasta_loader(self, seq_cache_prefix: str, struct_cache_prefix: str) -> DataLoader:
        ds = DeepPASTAStructCacheDataset(
            seq_cache_prefix=seq_cache_prefix,
            struct_cache_prefix=struct_cache_prefix,
            input_len=240,
            n_structures=3,
            check_headers=False,
            mmap=True,
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=DeepPASTAStructCollator(),
            pin_memory=self.device.type == "cuda",
        )

    @torch.no_grad()
    def _predict_non_pasta(self, cache_prefix: str) -> pd.DataFrame:
        if not self._non_pasta_models:
            return pd.DataFrame()
        loader = self._make_non_pasta_loader(cache_prefix)

        all_probs: Dict[str, List[float]] = {name: [] for name in self._non_pasta_models}
        headers_all: List[str] = []

        for batch in loader:
            seq_idx = batch["seq_idx"].to(self.device)
            headers = batch["headers"]
            headers_all.extend(headers)
            for name in self._non_pasta_models:
                built = self._built[name]
                x = built.encoder(seq_idx)
                logits = built.model(x)
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                all_probs[name].extend(probs.astype(float).tolist())

        df = pd.DataFrame({"header": headers_all})
        for name in self._non_pasta_models:
            df[f"{name}_prob"] = all_probs[name]
        return df

    @torch.no_grad()
    def _predict_deeppasta(self, seq_cache_prefix: str, struct_cache_prefix: str) -> pd.DataFrame:
        loader = self._make_deeppasta_loader(seq_cache_prefix, struct_cache_prefix)
        probs_all: List[float] = []
        headers_all: List[str] = []
        built = self._built["deeppasta"]

        for batch in loader:
            x, _ = batch
            x = x.to(self.device)
            logits = built.model(x)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs_all.extend(probs.astype(float).tolist())

        # headers from seq cache (same order)
        headers = np.load(f"{seq_cache_prefix}.headers.npy", allow_pickle=True)
        headers_all.extend([str(h) for h in headers])

        return pd.DataFrame({"header": headers_all, "deeppasta_prob": probs_all})

    def predict_fasta(
        self,
        fasta_path: str,
        cache_dir: Optional[str] = None,
        keep_cache: bool = False,
    ) -> pd.DataFrame:
        """Predict from a FASTA file."""
        fasta_path = Path(fasta_path)

        df_llm: Optional[pd.DataFrame] = None
        if self._llm_predictor is not None:
            df_llm = self._llm_predictor.predict_fasta(str(fasta_path))

        df_trad: Optional[pd.DataFrame] = None
        if self.traditional_models:
            if cache_dir is None:
                cache_dir = Path(".apa_polya_cache")
            else:
                cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            prefix = str(cache_dir / fasta_path.stem)

            _, headers, normalized_fasta = preprocess_fasta(
                str(fasta_path), prefix, target_len=240, write_normalized_fasta=True
            )
            np.save(f"{prefix}.y.npy", np.zeros(len(headers), dtype=np.int64))

            struct_prefix: Optional[str] = None
            try:
                if self._has_deeppasta:
                    from apa_polya_predictor.structure_cache import build_structure_cache

                    struct_dir = cache_dir / f"{fasta_path.stem}_struct"
                    struct_prefix = str(struct_dir / fasta_path.stem)
                    build_structure_cache(
                        fasta_path=normalized_fasta,
                        out_prefix=struct_prefix,
                        rnashapes_bin=self.rnashapes_bin,
                        tmp_dir=str(struct_dir / "tmp"),
                        seq_len=240,
                        half_len=120,
                        n_structures=3,
                    )

                df_non_pasta = self._predict_non_pasta(prefix)

                if self._has_deeppasta:
                    df_pasta = self._predict_deeppasta(prefix, struct_prefix)
                    if df_non_pasta.empty:
                        df_trad = df_pasta
                    else:
                        df_trad = pd.merge(df_non_pasta, df_pasta, on="header", how="inner")
                else:
                    df_trad = df_non_pasta

            finally:
                if not keep_cache:
                    for suffix in [".x.uint8.npy", ".y.npy", ".headers.npy", ".240.fa"]:
                        p = Path(f"{prefix}{suffix}")
                        if p.exists():
                            p.unlink()
                    if struct_prefix is not None:
                        for suffix in [".ss.uint8.npy", ".ss.headers.npy", ".ss.meta.json"]:
                            p = Path(f"{struct_prefix}{suffix}")
                            if p.exists():
                                p.unlink()
                        struct_dir = Path(struct_prefix).parent
                        if struct_dir.exists():
                            import shutil
                            shutil.rmtree(struct_dir, ignore_errors=True)

        if df_trad is not None and df_llm is not None:
            df = pd.merge(df_trad, df_llm, on="header", how="inner")
        elif df_trad is not None:
            df = df_trad
        elif df_llm is not None:
            df = df_llm
        else:
            raise ValueError("No models configured")
        return df


