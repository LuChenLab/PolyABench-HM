#!/usr/bin/env python3
import argparse
import json
import os
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from apa_polya_predictor.core.trainer import (
    OneHotBinaryTrainer,
    TrainerConfig,
    UInt8CacheDataset,
    UInt8SequenceCollator,
    set_seed,
)


class Trinucleotide2DEncoder(nn.Module):
    """Convert uint8 nucleotide index sequence [B, L] to trinucleotide 2D map [B, 1, L-2, 64]."""

    def __init__(self, dna_only: bool = True, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.dna_only = bool(dna_only)
        self.dtype = dtype

    def forward(self, x_idx: torch.Tensor) -> torch.Tensor:
        if x_idx.ndim != 2:
            raise ValueError(f"x_idx must have shape [B, L], got {tuple(x_idx.shape)}")

        a = x_idx[:, :-2]
        b = x_idx[:, 1:-1]
        c = x_idx[:, 2:]

        valid = (a < 4) & (b < 4) & (c < 4)
        tri_idx = (a.clamp(0, 3) * 16 + b.clamp(0, 3) * 4 + c.clamp(0, 3)).long()

        bsz, seqlen = tri_idx.shape
        out = torch.zeros((bsz, seqlen, 64), device=x_idx.device, dtype=self.dtype)
        out.scatter_(2, tri_idx.unsqueeze(-1), 1.0)
        out = out * valid.unsqueeze(-1).to(out.dtype)
        return out.unsqueeze(1).contiguous()


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: tuple[int, int],
        pool_size: tuple[int, int],
        use_batch_norm: bool = True,
        conv_dropout: float = 0.0,
        same_pad: bool = False,
    ):
        super().__init__()
        self.same_pad = bool(same_pad)
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=0, bias=not use_batch_norm)
        self.bn = nn.BatchNorm2d(out_ch) if use_batch_norm else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
        self.drop = nn.Dropout2d(conv_dropout) if conv_dropout > 0 else nn.Identity()

    def _same_pad(self, x: torch.Tensor) -> torch.Tensor:
        kh, kw = self.kernel_size
        total_h = kh - 1
        total_w = kw - 1
        pad_top = total_h // 2
        pad_bottom = total_h - pad_top
        pad_left = total_w // 2
        pad_right = total_w - pad_left
        return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.same_pad:
            x = self._same_pad(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.drop(x)
        return x


class DeepGSR240RegularizedClassifier(nn.Module):
    """
    DeepGSR-style 2D-CNN adjusted for 240-bp APA windows.

    Important default design:
    - pool_h = 1: do not downsample sequence length too early.
    - pool_w = 2: close to DeepGSR's 1x2 pooling direction.
    - adaptive_h/adaptive_w are smaller to reduce FC parameters.
    """

    def __init__(
        self,
        input_len: int = 240,
        conv1_filters: int = 16,
        conv1_kernel_h: int = 12,
        conv1_kernel_w: int = 16,
        conv2_filters: int = 32,
        conv2_kernel_h: int = 5,
        conv2_kernel_w: int = 8,
        pool1_h: int = 1,
        pool1_w: int = 2,
        pool2_h: int = 1,
        pool2_w: int = 2,
        adaptive_h: int = 12,
        adaptive_w: int = 8,
        hidden_dim: int = 64,
        dropout: float = 0.35,
        conv_dropout: float = 0.05,
        fc_activation: str = "tanh",
        conv1_init: str = "xavier",
        use_batch_norm: bool = True,
        num_classes: int = 2,
    ):
        super().__init__()
        self.input_len = int(input_len)
        self.tri_len = self.input_len - 2
        self.conv1_filters = int(conv1_filters)
        self.conv1_kernel_h = int(conv1_kernel_h)
        self.conv1_kernel_w = int(conv1_kernel_w)
        self.conv2_filters = int(conv2_filters)
        self.conv2_kernel_h = int(conv2_kernel_h)
        self.conv2_kernel_w = int(conv2_kernel_w)
        self.pool1_h = int(pool1_h)
        self.pool1_w = int(pool1_w)
        self.pool2_h = int(pool2_h)
        self.pool2_w = int(pool2_w)
        self.adaptive_h = int(adaptive_h)
        self.adaptive_w = int(adaptive_w)
        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout)
        self.conv_dropout = float(conv_dropout)
        self.fc_activation_name = str(fc_activation)
        self.conv1_init = str(conv1_init)
        self.use_batch_norm = bool(use_batch_norm)
        self.num_classes = int(num_classes)

        self.block1 = ConvBlock(
            in_ch=1,
            out_ch=self.conv1_filters,
            kernel_size=(self.conv1_kernel_h, self.conv1_kernel_w),
            pool_size=(self.pool1_h, self.pool1_w),
            use_batch_norm=self.use_batch_norm,
            conv_dropout=self.conv_dropout,
            same_pad=True,
        )
        self.block2 = ConvBlock(
            in_ch=self.conv1_filters,
            out_ch=self.conv2_filters,
            kernel_size=(self.conv2_kernel_h, self.conv2_kernel_w),
            pool_size=(self.pool2_h, self.pool2_w),
            use_batch_norm=self.use_batch_norm,
            conv_dropout=self.conv_dropout,
            same_pad=False,
        )

        self.adaptive_pool = nn.AdaptiveMaxPool2d((self.adaptive_h, self.adaptive_w))
        flatten_dim = self.conv2_filters * self.adaptive_h * self.adaptive_w

        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_p),
            nn.Linear(flatten_dim, self.hidden_dim),
            nn.Tanh() if self.fc_activation_name == "tanh" else nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

        self.model_info = {
            "input_len": self.input_len,
            "tri_len": self.tri_len,
            "conv1_filters": self.conv1_filters,
            "conv1_kernel_h": self.conv1_kernel_h,
            "conv1_kernel_w": self.conv1_kernel_w,
            "conv2_filters": self.conv2_filters,
            "conv2_kernel_h": self.conv2_kernel_h,
            "conv2_kernel_w": self.conv2_kernel_w,
            "pool1_h": self.pool1_h,
            "pool1_w": self.pool1_w,
            "pool2_h": self.pool2_h,
            "pool2_w": self.pool2_w,
            "adaptive_h": self.adaptive_h,
            "adaptive_w": self.adaptive_w,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout_p,
            "conv_dropout": self.conv_dropout,
            "fc_activation": self.fc_activation_name,
            "conv1_init": self.conv1_init,
            "use_batch_norm": self.use_batch_norm,
            "flatten_dim": flatten_dim,
            "num_classes": self.num_classes,
        }

        self._init_weights()

    def _init_weights(self):
        # Conv1 can be zero-init for closer DeepGSR search-space testing, but xavier is safer by default.
        conv1 = self.block1.conv
        if self.conv1_init == "zero":
            nn.init.zeros_(conv1.weight)
        else:
            nn.init.xavier_uniform_(conv1.weight)
        if conv1.bias is not None:
            nn.init.zeros_(conv1.bias)

        nn.init.xavier_uniform_(self.block2.conv.weight)
        if self.block2.conv.bias is not None:
            nn.init.zeros_(self.block2.conv.bias)

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = self.block2(h)
        h = self.adaptive_pool(h)
        h = h.flatten(start_dim=1)
        return self.classifier(h)


def parse_class_weights(text: Optional[str], device: torch.device) -> Optional[torch.Tensor]:
    if text is None or str(text).lower() in {"none", "no", "false", "0"}:
        return None
    vals = [float(v.strip()) for v in str(text).split(",")]
    if len(vals) != 2:
        raise ValueError("--class_weights must contain two comma-separated numbers, e.g. 1,3")
    return torch.tensor(vals, dtype=torch.float32, device=device)


def make_ce_loss(class_weights: Optional[torch.Tensor], label_smoothing: float):
    """Create CE loss while staying compatible with older PyTorch versions."""
    try:
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(label_smoothing))
    except TypeError:
        if label_smoothing and label_smoothing > 0:
            raise RuntimeError("Your PyTorch version does not support CrossEntropyLoss(label_smoothing=...). Set --label_smoothing 0")
        return nn.CrossEntropyLoss(weight=class_weights)


def main():
    parser = argparse.ArgumentParser(description="Train regularized 240-bp DeepGSR-style benchmark model on cached APA data")
    parser.add_argument("--train_cache", required=True, type=str)
    parser.add_argument("--val_cache", required=True, type=str)
    parser.add_argument("--test_cache", default=None, type=str)
    parser.add_argument("--output_dir", required=True, type=str)

    # Model defaults: intentionally smaller and more regularized than the earlier lite version.
    parser.add_argument("--input_len", default=240, type=int)
    parser.add_argument("--conv1_filters", default=16, type=int)
    parser.add_argument("--conv1_kernel_h", default=12, type=int)
    parser.add_argument("--conv1_kernel_w", default=16, type=int)
    parser.add_argument("--conv2_filters", default=32, type=int)
    parser.add_argument("--conv2_kernel_h", default=5, type=int)
    parser.add_argument("--conv2_kernel_w", default=8, type=int)
    parser.add_argument("--pool1_h", default=1, type=int)
    parser.add_argument("--pool1_w", default=2, type=int)
    parser.add_argument("--pool2_h", default=1, type=int)
    parser.add_argument("--pool2_w", default=2, type=int)
    parser.add_argument("--adaptive_h", default=12, type=int)
    parser.add_argument("--adaptive_w", default=8, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--dropout", default=0.35, type=float)
    parser.add_argument("--conv_dropout", default=0.05, type=float)
    parser.add_argument("--fc_activation", default="tanh", choices=["tanh", "relu"])
    parser.add_argument("--conv1_init", default="xavier", choices=["zero", "xavier"])
    parser.add_argument("--no_batch_norm", action="store_true")

    # Training defaults: more conservative than Adam lr=1e-3 with no weight decay.
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--optimizer", default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--nesterov", action="store_true")

    parser.add_argument("--num_train_epochs", default=50, type=int)
    parser.add_argument("--selection_metric", default="pr_auc", type=str)
    parser.add_argument("--early_stopping_patience", default=4, type=int)
    parser.add_argument("--seed", default=2024, type=int)

    # For pos:neg = 1:3 with label 1 = positive and label 0 = negative, use weight[positive] = 3.
    parser.add_argument("--class_weights", default="1,3", type=str, help="Class weights for CrossEntropyLoss. Default assumes label0=negative, label1=positive with pos:neg=1:3. Use 'none' to disable.")
    parser.add_argument("--label_smoothing", default=0.02, type=float)

    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dna_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_ds = UInt8CacheDataset(args.train_cache, mmap=True)
    val_ds = UInt8CacheDataset(args.val_cache, mmap=True)
    test_ds = UInt8CacheDataset(args.test_cache, mmap=True) if args.test_cache else None

    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        if ds is not None and ds.seq_len != args.input_len:
            raise ValueError(f"{name} cache seq_len={ds.seq_len}, but --input_len={args.input_len}")

    collator = UInt8SequenceCollator()
    common_loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
    )
    if args.num_workers > 0:
        common_loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **common_loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **common_loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **common_loader_kwargs) if test_ds is not None else None

    encoder = Trinucleotide2DEncoder(dna_only=args.dna_only, dtype=torch.float32)
    raw_model = DeepGSR240RegularizedClassifier(
        input_len=args.input_len,
        conv1_filters=args.conv1_filters,
        conv1_kernel_h=args.conv1_kernel_h,
        conv1_kernel_w=args.conv1_kernel_w,
        conv2_filters=args.conv2_filters,
        conv2_kernel_h=args.conv2_kernel_h,
        conv2_kernel_w=args.conv2_kernel_w,
        pool1_h=args.pool1_h,
        pool1_w=args.pool1_w,
        pool2_h=args.pool2_h,
        pool2_w=args.pool2_w,
        adaptive_h=args.adaptive_h,
        adaptive_w=args.adaptive_w,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        conv_dropout=args.conv_dropout,
        fc_activation=args.fc_activation,
        conv1_init=args.conv1_init,
        use_batch_norm=(not args.no_batch_norm),
        num_classes=2,
    )
    model_info = dict(raw_model.model_info)
    model = raw_model

    if args.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )

    class_weights = parse_class_weights(args.class_weights, device)
    train_loss = make_ce_loss(class_weights=class_weights, label_smoothing=args.label_smoothing)
    eval_loss = make_ce_loss(class_weights=class_weights, label_smoothing=0.0)

    trainer_cfg = TrainerConfig(
        output_dir=args.output_dir,
        selection_metric=args.selection_metric,
        early_stopping_patience=args.early_stopping_patience,
        num_train_epochs=args.num_train_epochs,
        mixed_precision=args.mixed_precision,
    )

    run_config = {
        "device": str(device),
        "train_cache": args.train_cache,
        "val_cache": args.val_cache,
        "test_cache": args.test_cache,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "nesterov": args.nesterov,
        "class_weights": None if class_weights is None else [float(x) for x in class_weights.detach().cpu().tolist()],
        "label_smoothing": args.label_smoothing,
        "mixed_precision": args.mixed_precision,
        "compile_model": args.compile_model,
        "dna_only": args.dna_only,
        "model_info": model_info,
    }
    with open(os.path.join(args.output_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    print(json.dumps(run_config, indent=2))

    trainer = OneHotBinaryTrainer(
        model=model,
        encoder=encoder,
        optimizer=optimizer,
        scheduler=None,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_dataset_len=len(train_ds),
        val_dataset_len=len(val_ds),
        test_dataset_len=(len(test_ds) if test_ds is not None else None),
        device=device,
        trainer_cfg=trainer_cfg,
        train_loss_fn=lambda model_obj, logits, y: train_loss(logits, y),
        eval_loss_fn=eval_loss,
    )

    final = trainer.fit()
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
