from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .external.rna_tokenizer import RNATokenizer
from .external import wrappers


class _TritonImportBlocker:
    """Blocks triton imports while loading DNABERT2.

    DNABERT2's flash-attention Triton kernel (tl.dot(trans_b=...)) is
    incompatible with several released Triton versions and only supports CUDA.
    Blocking the triton import makes the remote code fall back to standard
    (eager) attention, which is numerically equivalent.
    """

    def find_module(self, name, path=None):
        if name == "triton" or name.startswith("triton."):
            return self
        return None

    def load_module(self, name):
        raise ImportError("triton import blocked by apa_polya_predictor")


def _force_dnabert2_eager_attention():
    """Disable DNABERT2 flash attention if its module was already imported with it."""
    for mod in list(sys.modules.values()):
        if mod is not None and "dnabert2" in getattr(mod, "__name__", "").lower():
            if hasattr(mod, "flash_attn_qkvpacked_func"):
                mod.flash_attn_qkvpacked_func = None


MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # max_seq_len is token length after tokenizer/kmer conversion, not raw bp length.
    "UTRLM": {
        "max_seq_len": 256,
        "kmer": 1,
        "replace_t_with_u": False,
        "replace_u_with_t": True,
        "pad_token_id": 0,
        "add_cls_prefix": True,
        "vocab_path": "pretrained/llm/utrlm/UTRLM.txt",
    },
    "DNABERT1": {
        "max_seq_len": 256,
        "kmer": 3,
        "replace_t_with_u": False,
        "replace_u_with_t": True,
        "pad_token_id": 0,
        "add_cls_prefix": True,
    },
    "DNABERT2": {
        "max_seq_len": 128,
        "kmer": 0,
        "replace_t_with_u": False,
        "replace_u_with_t": True,
        "pad_token_id": 3,
        "add_cls_prefix": False,
    },
    "RNABERT": {
        "max_seq_len": 256,
        "kmer": 1,
        "replace_t_with_u": True,
        "replace_u_with_t": False,
        "pad_token_id": 0,
        "add_cls_prefix": True,
        "vocab_path": "pretrained/llm/rnabert/vocab.txt",
        "model_config": "pretrained/llm/rnabert/RNABERT.json",
    },
    "GENA-LM-base": {
        "max_seq_len": 128,
        "kmer": 0,
        "replace_t_with_u": False,
        "replace_u_with_t": True,
        "pad_token_id": 3,
        "add_cls_prefix": False,
    },
}

AVAILABLE_LLM_MODELS = tuple(MODEL_DEFAULTS.keys())


def canonical_method(method: str) -> str:
    m = method.strip()
    aliases = {
        "dnabert": "DNABERT1",
        "dnabert1": "DNABERT1",
        "dnabert2": "DNABERT2",
        "rnabert": "RNABERT",
        "utrlm": "UTRLM",
        "utr-lm": "UTRLM",
        "gena": "GENA-LM-base",
        "gena-lm-base": "GENA-LM-base",
    }
    return aliases.get(m.lower(), m)


def get_model_defaults(method: str) -> Dict[str, Any]:
    method = canonical_method(method)
    if method not in MODEL_DEFAULTS:
        raise ValueError(f"Unknown LLM method: {method}. Available: {sorted(AVAILABLE_LLM_MODELS)}")
    return dict(MODEL_DEFAULTS[method])


def build_tokenizer_and_model(args) -> Tuple[Any, torch.nn.Module, Dict[str, Any]]:
    method = canonical_method(args.method)
    defaults = get_model_defaults(method)

    # CLI overrides
    max_seq_len = args.max_seq_len if args.max_seq_len is not None else defaults["max_seq_len"]
    kmer = args.kmer if args.kmer is not None else defaults["kmer"]
    pad_token_id = args.pad_token_id if args.pad_token_id is not None else defaults["pad_token_id"]

    model_path = Path(args.model_path) if args.model_path else None
    if method not in {"RNABERT"} and (model_path is None or not model_path.exists()):
        raise FileNotFoundError(f"--model_path is required for {method}: {model_path}")

    freeze = bool(args.freeze_backbone)

    if method == "RNABERT":
        from .external.RNABERT.bert import get_config
        from .external.RNABERT.rnabert import BertModel

        config_path = Path(args.model_config or defaults["model_config"])
        vocab_path = Path(args.vocab_path or defaults["vocab_path"])
        tokenizer = RNATokenizer(str(vocab_path))
        config = get_config(str(config_path))
        base = BertModel(config)
        model = wrappers.RNABertForSeqCls(base, hidden_size=getattr(config, "hidden_size", 120), class_num=2, freeze_backbone=freeze)
        if args.model_path:
            state = torch.load(args.model_path, map_location="cpu")
            model.load_state_dict(state, strict=False)

    elif method == "UTRLM":
        from .external.UTRLM.utrlm import UTRLM

        vocab_path = Path(args.vocab_path or defaults["vocab_path"])
        tokenizer = RNATokenizer(str(vocab_path))
        base = UTRLM()
        state = torch.load(str(model_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        base.load_state_dict(state, strict=True)
        model = wrappers.UTRLMForSeqCls(base, hidden_size=128, class_num=2, freeze_backbone=freeze)

    elif method == "DNABERT1":
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        base = AutoModelForSequenceClassification.from_pretrained(str(model_path), num_labels=2)
        model = wrappers.DNABERTForSeqCls(base, pad_token_id=pad_token_id, freeze_backbone=freeze)

    elif method == "DNABERT2":
        # Block triton while loading so DNABERT2 uses eager attention (see
        # _TritonImportBlocker); also force the fallback if the module was
        # already imported with flash attention earlier in this process.
        blocker = _TritonImportBlocker()
        sys.meta_path.insert(0, blocker)
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            base = AutoModelForSequenceClassification.from_pretrained(str(model_path), num_labels=2, trust_remote_code=True)
        finally:
            sys.meta_path.remove(blocker)
        _force_dnabert2_eager_attention()
        model = wrappers.DNABERT2ForSeqCls(base, pad_token_id=pad_token_id, freeze_backbone=freeze)

    elif method == "GENA-LM-base":
        from .external.GENA import modeling_bert as GENA

        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        base = GENA.BertForSequenceClassification.from_pretrained(str(model_path), num_labels=2)
        model = wrappers.GENAForSeqCls(base, pad_token_id=pad_token_id, freeze_backbone=freeze)

    else:
        raise ValueError(method)

    runtime = dict(defaults)
    runtime.update({"method": method, "max_seq_len": max_seq_len, "kmer": kmer, "pad_token_id": pad_token_id})
    return tokenizer, model, runtime
