#!/usr/bin/env python3
import argparse
import json
import math
import os

import numpy as np
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


class KmerIndexEncoder(nn.Module):
    """
    Convert cached uint8 sequence [B, L] to PASNet k-mer index [B, L-k+1].

    Base encoding from existing cache:
      A=0, C=1, G=2, T=3, N=4

    k=5:
      A/C/G/T-only 5-mer -> 0~1023
      N-containing 5-mer -> 1024
    """

    def __init__(self, kmer: int = 5):
        super().__init__()
        self.kmer = int(kmer)
        self.vocab_size = 4 ** self.kmer + 1
        self.unk_idx = 4 ** self.kmer

    def forward(self, x_idx: torch.Tensor) -> torch.Tensor:
        if x_idx.ndim != 2:
            raise ValueError(f"x_idx must have shape [B, L], got {tuple(x_idx.shape)}")

        bsz, seq_len = x_idx.shape
        out_len = seq_len - self.kmer + 1

        if out_len <= 0:
            raise ValueError(f"seq_len={seq_len} is shorter than kmer={self.kmer}")

        idx = torch.zeros(
            (bsz, out_len),
            device=x_idx.device,
            dtype=torch.long,
        )

        valid = torch.ones(
            (bsz, out_len),
            device=x_idx.device,
            dtype=torch.bool,
        )

        for j in range(self.kmer):
            token = x_idx[:, j:j + out_len].long()
            valid = valid & (token < 4)
            idx = idx * 4 + token.clamp(0, 3)

        idx = torch.where(
            valid,
            idx,
            torch.full_like(idx, self.unk_idx),
        )

        return idx


def lecun_normal_(tensor: torch.Tensor):
    fan_in = nn.init._calculate_correct_fan(tensor, mode="fan_in")
    std = math.sqrt(1.0 / fan_in)
    with torch.no_grad():
        return tensor.normal_(0.0, std)


class GatedConv1dHighway(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.kernel_size = int(kernel_size)
        self.dropout_p = float(dropout)

        if self.kernel_size % 2 == 0:
            raise ValueError("PASNet original kernel sizes are odd; use odd kernel_size.")

        padding = self.kernel_size // 2

        self.conv = nn.Conv1d(
            in_channels=self.dim,
            out_channels=self.dim * 2,
            kernel_size=self.kernel_size,
            padding=padding,
            bias=True,
        )

        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        lecun_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)

        gate_raw = h[:, :self.dim, :]
        transform_raw = h[:, self.dim:, :]

        gate = torch.sigmoid(gate_raw)
        transform = F.selu(transform_raw)

        out = (1.0 - gate) * x + gate * transform
        out = self.dropout(out)

        return out


class MultiScaleGatedConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filters: int,
        kernel_size: int,
        dropout: float = 0.1,
        apply_dropout: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.dropout_p = float(dropout)
        self.apply_dropout = bool(apply_dropout)

        ks1 = self.kernel_size
        ks2 = self.kernel_size + 6
        ks3 = self.kernel_size + 12

        for ks in [ks1, ks2, ks3]:
            if ks % 2 == 0:
                raise ValueError(f"kernel size must be odd, got {ks}")

        self.conv0 = nn.Conv1d(
            self.in_channels,
            self.filters,
            kernel_size=ks1,
            padding=ks1 // 2,
            bias=True,
        )
        self.conv1 = nn.Conv1d(
            self.in_channels,
            self.filters,
            kernel_size=ks2,
            padding=ks2 // 2,
            bias=True,
        )
        self.conv2 = nn.Conv1d(
            self.in_channels,
            self.filters,
            kernel_size=ks3,
            padding=ks3 // 2,
            bias=True,
        )

        self.gated = GatedConv1dHighway(
            dim=self.filters,
            kernel_size=self.kernel_size,
            dropout=0.0,
        )

        self.dropout = nn.Dropout(self.dropout_p) if self.apply_dropout else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        for conv in [self.conv0, self.conv1, self.conv2]:
            lecun_normal_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = F.selu(self.conv0(x))
        h1 = F.selu(self.conv1(x))
        h2 = F.selu(self.conv2(x))

        h = h0 + h1 + h2
        h = self.gated(h)
        h = self.dropout(h)

        return h


class PASNetSelfAttention(nn.Module):
    """
    Replacement for missing original Keras Attention(8, 16).

    Input:
      x: [B, L, embed_dim]

    Output:
      [B, L, num_heads * head_dim]
    """

    def __init__(
        self,
        input_dim: int = 90,
        num_heads: int = 8,
        head_dim: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.dropout_p = float(dropout)

        self.q_proj = nn.Linear(self.input_dim, self.inner_dim)
        self.k_proj = nn.Linear(self.input_dim, self.inner_dim)
        self.v_proj = nn.Linear(self.input_dim, self.inner_dim)

        self.dropout = nn.Dropout(self.dropout_p)
        self._init_weights()

    def _init_weights(self):
        for layer in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.inner_dim)

        return out


class PASNetClassifier(nn.Module):
    def __init__(
        self,
        input_len: int = 240,
        kmer: int = 5,
        vocab_size: int = 1025,
        embed_dim: int = 90,
        filters: int = 32,
        kernel_size: int = 21,
        attention_heads: int = 8,
        attention_head_dim: int = 16,
        hidden_dim: int = 64,
        dropout: float = 0.5,
        num_classes: int = 2,
        trainable_embedding: bool = True,
    ):
        super().__init__()

        self.input_len = int(input_len)
        self.kmer = int(kmer)
        self.kmer_len = self.input_len - self.kmer + 1
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.attention_heads = int(attention_heads)
        self.attention_head_dim = int(attention_head_dim)
        self.attention_dim = self.attention_heads * self.attention_head_dim
        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout)
        self.num_classes = int(num_classes)
        self.trainable_embedding = bool(trainable_embedding)

        if self.kmer_len <= 0:
            raise ValueError(f"input_len={self.input_len} is shorter than kmer={self.kmer}")

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embed_dim,
        )
        self.embedding.weight.requires_grad = self.trainable_embedding

        self.embed_dropout = nn.Dropout(self.dropout_p)

        self.block1 = MultiScaleGatedConvBlock(
            in_channels=self.embed_dim,
            filters=self.filters,
            kernel_size=self.kernel_size,
            dropout=self.dropout_p,
            apply_dropout=True,
        )

        self.block2 = MultiScaleGatedConvBlock(
            in_channels=self.filters,
            filters=self.filters,
            kernel_size=self.kernel_size,
            dropout=self.dropout_p,
            apply_dropout=True,
        )

        self.block3 = MultiScaleGatedConvBlock(
            in_channels=self.filters,
            filters=self.filters,
            kernel_size=self.kernel_size,
            dropout=self.dropout_p,
            apply_dropout=False,
        )

        self.attention = PASNetSelfAttention(
            input_dim=self.embed_dim,
            num_heads=self.attention_heads,
            head_dim=self.attention_head_dim,
            dropout=0.0,
        )

        fusion_dim_per_pos = self.attention_dim + self.filters
        flatten_dim = self.kmer_len * fusion_dim_per_pos

        self.flatten_dropout = nn.Dropout(self.dropout_p)
        self.fc1 = nn.Linear(flatten_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.num_classes)

        self.model_info = {
            "model_name": "PASNetClassifier",
            "input_len": self.input_len,
            "kmer": self.kmer,
            "kmer_len": self.kmer_len,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "conv_kernel_sizes": [
                self.kernel_size,
                self.kernel_size + 6,
                self.kernel_size + 12,
            ],
            "attention_heads": self.attention_heads,
            "attention_head_dim": self.attention_head_dim,
            "attention_dim": self.attention_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout_p,
            "flatten_dim": flatten_dim,
            "fusion_dim_per_pos": fusion_dim_per_pos,
            "num_classes": self.num_classes,
            "trainable_embedding": self.trainable_embedding,
        }

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        lecun_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x_kmer_idx: torch.Tensor) -> torch.Tensor:
        if x_kmer_idx.ndim != 2:
            raise ValueError(f"x_kmer_idx must have shape [B, L_kmer], got {tuple(x_kmer_idx.shape)}")

        if x_kmer_idx.shape[1] != self.kmer_len:
            raise ValueError(
                f"kmer length mismatch: got {x_kmer_idx.shape[1]}, expected {self.kmer_len}"
            )

        x = self.embedding(x_kmer_idx.long())
        x = self.embed_dropout(x)

        attn_features = self.attention(x)

        h = x.transpose(1, 2).contiguous()
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = h.transpose(1, 2).contiguous()

        features = torch.cat([attn_features, h], dim=-1)

        features = features.flatten(start_dim=1)
        features = self.flatten_dropout(features)

        features = self.fc1(features)
        features = F.selu(features)

        logits = self.out(features)

        return logits


def load_pretrained_embedding(model: PASNetClassifier, embedding_path: str):
    embedding = np.load(embedding_path).astype(np.float32)

    expected_shape = (model.vocab_size, model.embed_dim)

    if embedding.shape != expected_shape:
        raise ValueError(
            f"Pretrained embedding shape mismatch: got {embedding.shape}, "
            f"expected {expected_shape}"
        )

    if not np.all(np.isfinite(embedding)):
        bad_n = int(np.sum(~np.isfinite(embedding)))
        raise ValueError(f"Pretrained embedding contains non-finite values: {bad_n}")

    with torch.no_grad():
        model.embedding.weight.copy_(torch.from_numpy(embedding))

    print(f"Loaded pretrained embedding: {embedding_path}")
    print(f"  shape: {embedding.shape}")
    print(f"  mean/std: {float(embedding.mean()):.6f} / {float(embedding.std()):.6f}")
    print(f"  min/max: {float(embedding.min()):.6f} / {float(embedding.max()):.6f}")


def build_optimizer(args, model):
    if args.embedding_lr_scale != 1.0 and not args.freeze_embedding:
        embedding_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if name.startswith("embedding."):
                embedding_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {
                "params": other_params,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": embedding_params,
                "lr": args.lr * args.embedding_lr_scale,
                "weight_decay": args.weight_decay,
            },
        ]

        print(f"Using separate embedding lr scale: {args.embedding_lr_scale}")
        print(f"  main lr: {args.lr}")
        print(f"  embedding lr: {args.lr * args.embedding_lr_scale}")

    else:
        param_groups = [p for p in model.parameters() if p.requires_grad]

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            param_groups,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )

    return optimizer


def main():
    parser = argparse.ArgumentParser(
        description="Train PASNet benchmark model on cached APA uint8 data"
    )

    parser.add_argument("--train_cache", required=True, type=str)
    parser.add_argument("--val_cache", required=True, type=str)
    parser.add_argument("--test_cache", default=None, type=str)
    parser.add_argument("--output_dir", required=True, type=str)

    parser.add_argument("--input_len", default=240, type=int)
    parser.add_argument("--kmer", default=5, type=int)
    parser.add_argument("--embed_dim", default=90, type=int)
    parser.add_argument("--filters", default=32, type=int)
    parser.add_argument("--kernel_size", default=21, type=int)
    parser.add_argument("--attention_heads", default=8, type=int)
    parser.add_argument("--attention_head_dim", default=16, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)

    parser.add_argument("--pretrained_embedding", default=None, type=str)
    parser.add_argument("--freeze_embedding", action="store_true")
    parser.add_argument("--embedding_lr_scale", default=1.0, type=float)

    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=8, type=int)

    parser.add_argument("--optimizer", default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--lr", default=3e-6, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--nesterov", action="store_true")

    parser.add_argument("--num_train_epochs", default=100, type=int)
    parser.add_argument("--selection_metric", default="pr_auc", type=str)
    parser.add_argument("--early_stopping_patience", default=10, type=int)
    parser.add_argument("--seed", default=2024, type=int)

    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

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
            raise ValueError(
                f"{name} cache seq_len={ds.seq_len}, but --input_len={args.input_len}"
            )

    collator = UInt8SequenceCollator()

    common_loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
    )

    if args.num_workers > 0:
        common_loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **common_loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **common_loader_kwargs,
    )

    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        **common_loader_kwargs,
    ) if test_ds is not None else None

    vocab_size = 4 ** args.kmer + 1

    encoder = KmerIndexEncoder(
        kmer=args.kmer,
    )

    model = PASNetClassifier(
        input_len=args.input_len,
        kmer=args.kmer,
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        filters=args.filters,
        kernel_size=args.kernel_size,
        attention_heads=args.attention_heads,
        attention_head_dim=args.attention_head_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_classes=2,
        trainable_embedding=True,
    )

    if args.pretrained_embedding is not None:
        load_pretrained_embedding(model, args.pretrained_embedding)

    if args.freeze_embedding:
        model.embedding.weight.requires_grad = False

    model.model_info["trainable_embedding"] = bool(model.embedding.weight.requires_grad)
    model.model_info["pretrained_embedding"] = args.pretrained_embedding
    model.model_info["freeze_embedding"] = bool(args.freeze_embedding)
    model.model_info["embedding_lr_scale"] = float(args.embedding_lr_scale)

    model_info = dict(model.model_info)

    if args.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = build_optimizer(args, model)

    trainer_cfg = TrainerConfig(
        output_dir=args.output_dir,
        selection_metric=args.selection_metric,
        early_stopping_patience=args.early_stopping_patience,
        num_train_epochs=args.num_train_epochs,
        mixed_precision=args.mixed_precision,
    )

    with open(os.path.join(args.output_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
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
                "mixed_precision": args.mixed_precision,
                "compile_model": args.compile_model,
                "seed": args.seed,
                "pretrained_embedding": args.pretrained_embedding,
                "freeze_embedding": args.freeze_embedding,
                "embedding_lr_scale": args.embedding_lr_scale,
                "encoder_info": {
                    "encoder": "KmerIndexEncoder",
                    "kmer": args.kmer,
                    "vocab_size": vocab_size,
                    "unk_idx_for_N_containing_kmer": 4 ** args.kmer,
                    "base_encoding": "A=0,C=1,G=2,T=3,N=4",
                },
                "model_info": model_info,
            },
            f,
            indent=2,
        )

    train_loss = nn.CrossEntropyLoss()
    eval_loss = nn.CrossEntropyLoss()

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
