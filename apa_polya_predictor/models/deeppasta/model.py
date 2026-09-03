#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from apa_polya_predictor.core.trainer import (
    OneHotBinaryTrainer,
    TrainerConfig,
    set_seed,
)


class DeepPASTAStructCacheDataset(Dataset):
    """
    Sequence cache:
      prefix.x.uint8.npy      [N, L]
      prefix.y.npy            [N]
      prefix.headers.npy      [N]

    Structure cache:
      prefix.ss.uint8.npy     [N, S, L]
      prefix.ss.headers.npy   [N]

    Return:
      packed_x [1 + S, L]
        packed_x[0]     = sequence index, A=0,C=1,G=2,T=3,N=4
        packed_x[1:S+1] = RNAshape structure index, U/E/L/R/H/M/I=0-6
      y
    """

    def __init__(
        self,
        seq_cache_prefix: str,
        struct_cache_prefix: str,
        input_len: int = 240,
        n_structures: int = 3,
        check_headers: bool = False,
        mmap: bool = True,
    ):
        self.seq_cache_prefix = str(seq_cache_prefix)
        self.struct_cache_prefix = str(struct_cache_prefix)
        self.input_len = int(input_len)
        self.n_structures = int(n_structures)

        mmap_mode = "r" if mmap else None

        self.x_path = self.seq_cache_prefix + ".x.uint8.npy"
        self.y_path = self.seq_cache_prefix + ".y.npy"
        self.seq_headers_path = self.seq_cache_prefix + ".headers.npy"

        self.ss_path = self.struct_cache_prefix + ".ss.uint8.npy"
        self.ss_headers_path = self.struct_cache_prefix + ".ss.headers.npy"

        self.x = np.load(self.x_path, mmap_mode=mmap_mode)
        self.y = np.load(self.y_path, mmap_mode=mmap_mode)
        self.ss = np.load(self.ss_path, mmap_mode=mmap_mode)

        if self.x.ndim != 2:
            raise ValueError(f"x must be [N, L], got {self.x.shape}")

        if self.ss.ndim != 3:
            raise ValueError(f"ss must be [N, S, L], got {self.ss.shape}")

        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(f"x/y sample mismatch: {self.x.shape[0]} vs {self.y.shape[0]}")

        if self.x.shape[0] != self.ss.shape[0]:
            raise ValueError(f"x/ss sample mismatch: {self.x.shape[0]} vs {self.ss.shape[0]}")

        if self.x.shape[1] != self.input_len:
            raise ValueError(f"x seq_len={self.x.shape[1]}, expected {self.input_len}")

        if self.ss.shape[1] != self.n_structures:
            raise ValueError(f"ss n_structures={self.ss.shape[1]}, expected {self.n_structures}")

        if self.ss.shape[2] != self.input_len:
            raise ValueError(f"ss seq_len={self.ss.shape[2]}, expected {self.input_len}")

        self.n = int(self.x.shape[0])
        self.seq_len = int(self.x.shape[1])

        if check_headers:
            seq_headers = np.load(self.seq_headers_path, allow_pickle=True)
            ss_headers = np.load(self.ss_headers_path, allow_pickle=True)

            if len(seq_headers) != self.n or len(ss_headers) != self.n:
                raise ValueError("Header length mismatch.")

            if list(seq_headers) != list(ss_headers):
                raise ValueError(
                    f"Header order mismatch between sequence cache and structure cache:\n"
                    f"  seq_cache={self.seq_cache_prefix}\n"
                    f"  struct_cache={self.struct_cache_prefix}"
                )

            print(f"Header check passed: {self.seq_cache_prefix}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        seq = self.x[idx]          # [L]
        ss = self.ss[idx]          # [S, L]
        y = int(self.y[idx])

        packed = np.concatenate(
            [seq[None, :], ss],
            axis=0,
        ).astype(np.uint8, copy=False)

        return packed, y


class DeepPASTAStructCollator:
    def __call__(self, batch):
        xs, ys = zip(*batch)

        x = torch.from_numpy(np.stack(xs, axis=0).astype(np.uint8, copy=False))
        y = torch.as_tensor(ys, dtype=torch.long)

        return x, y


class DeepPASTABranch(nn.Module):
    """
    DeepPASTA-like branch:
      Conv1D
      BatchNorm
      ReLU
      MaxPool
      Dropout
      BiLSTM
      mean-pool + max-pool
    """

    def __init__(
        self,
        in_channels: int,
        conv_filters: int = 128,
        kernel_size: int = 8,
        pool_size: int = 3,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.conv_filters = int(conv_filters)
        self.kernel_size = int(kernel_size)
        self.pool_size = int(pool_size)
        self.lstm_hidden = int(lstm_hidden)
        self.lstm_layers = int(lstm_layers)
        self.dropout_p = float(dropout)

        self.conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.conv_filters,
            kernel_size=self.kernel_size,
            padding=0,
            bias=True,
        )

        self.bn = nn.BatchNorm1d(self.conv_filters)
        self.pool = nn.MaxPool1d(kernel_size=self.pool_size)
        self.dropout = nn.Dropout(self.dropout_p)

        self.lstm = nn.LSTM(
            input_size=self.conv_filters,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_p if self.lstm_layers > 1 else 0.0,
        )

        self.out_dim = self.lstm_hidden * 2 * 2

    def forward(self, x):
        """
        x: [B, C, L]
        """
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = x.transpose(1, 2).contiguous()
        x, _ = self.lstm(x)

        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values

        feat = torch.cat([mean_pool, max_pool], dim=1)
        return feat


class DeepPASTAFullClassifier(nn.Module):
    """
    Input:
      packed_x [B, 1 + n_structures, L]

    packed_x[:, 0, :]      sequence index, A=0,C=1,G=2,T=3,N=4
    packed_x[:, 1:, :]     RNAshape index, U/E/L/R/H/M/I=0-6

    Sequence branch:
      one-hot [B, 4, L] -> CNN + BiLSTM

    Structure branch:
      one-hot [B, S*7, L] -> CNN + BiLSTM

    Fusion:
      concat -> FC -> logits
    """

    def __init__(
        self,
        input_len: int = 240,
        n_structures: int = 3,
        structure_vocab_size: int = 7,
        seq_channels: int = 4,
        conv_filters: int = 128,
        kernel_size: int = 8,
        pool_size: int = 3,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        fusion_hidden: int = 256,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()

        self.input_len = int(input_len)
        self.n_structures = int(n_structures)
        self.structure_vocab_size = int(structure_vocab_size)
        self.seq_channels = int(seq_channels)
        self.struct_channels = self.n_structures * self.structure_vocab_size

        self.seq_branch = DeepPASTABranch(
            in_channels=self.seq_channels,
            conv_filters=conv_filters,
            kernel_size=kernel_size,
            pool_size=pool_size,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
        )

        self.struct_branch = DeepPASTABranch(
            in_channels=self.struct_channels,
            conv_filters=conv_filters,
            kernel_size=kernel_size,
            pool_size=pool_size,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
        )

        fusion_dim = self.seq_branch.out_dim + self.struct_branch.out_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

        self.model_info = {
            "model_name": "DeepPASTAFullClassifier",
            "input_len": self.input_len,
            "n_structures": self.n_structures,
            "structure_vocab_size": self.structure_vocab_size,
            "seq_channels": self.seq_channels,
            "struct_channels": self.struct_channels,
            "conv_filters": int(conv_filters),
            "kernel_size": int(kernel_size),
            "pool_size": int(pool_size),
            "lstm_hidden": int(lstm_hidden),
            "lstm_layers": int(lstm_layers),
            "fusion_hidden": int(fusion_hidden),
            "dropout": float(dropout),
            "fusion_dim": int(fusion_dim),
            "num_classes": int(num_classes),
        }

    def encode_sequence(self, x_seq):
        """
        x_seq: [B, L], A=0,C=1,G=2,T=3,N=4
        output: [B, 4, L]
        """
        x_seq = x_seq.long()
        x_oh = F.one_hot(x_seq.clamp(0, 4), num_classes=5).float()
        x_oh = x_oh[:, :, :4]
        x_oh = x_oh.permute(0, 2, 1).contiguous()
        return x_oh

    def encode_structure(self, x_ss):
        """
        x_ss: [B, S, L], U/E/L/R/H/M/I=0-6
        output: [B, S*7, L]
        """
        x_ss = x_ss.long()
        x_oh = F.one_hot(
            x_ss.clamp(0, self.structure_vocab_size - 1),
            num_classes=self.structure_vocab_size,
        ).float()

        x_oh = x_oh.permute(0, 1, 3, 2).contiguous()
        b, s, c, l = x_oh.shape
        x_oh = x_oh.view(b, s * c, l)
        return x_oh

    def forward(self, packed_x):
        if packed_x.ndim != 3:
            raise ValueError(f"packed_x must be [B, 1+S, L], got {tuple(packed_x.shape)}")

        if packed_x.shape[1] != self.n_structures + 1:
            raise ValueError(
                f"packed_x channel mismatch: got {packed_x.shape[1]}, "
                f"expected {self.n_structures + 1}"
            )

        if packed_x.shape[2] != self.input_len:
            raise ValueError(
                f"packed_x length mismatch: got {packed_x.shape[2]}, expected {self.input_len}"
            )

        x_seq = packed_x[:, 0, :]
        x_ss = packed_x[:, 1:, :]

        seq_input = self.encode_sequence(x_seq)
        ss_input = self.encode_structure(x_ss)

        seq_feat = self.seq_branch(seq_input)
        ss_feat = self.struct_branch(ss_input)

        feat = torch.cat([seq_feat, ss_feat], dim=1)
        logits = self.classifier(feat)

        return logits


def build_optimizer(args, model):
    params = [p for p in model.parameters() if p.requires_grad]

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )

    return optimizer


def main():
    parser = argparse.ArgumentParser(
        description="Train DeepPASTA full benchmark model on sequence + RNAshape caches"
    )

    parser.add_argument("--train_cache", required=True, type=str)
    parser.add_argument("--train_struct_cache", required=True, type=str)

    parser.add_argument("--val_cache", required=True, type=str)
    parser.add_argument("--val_struct_cache", required=True, type=str)

    parser.add_argument("--test_cache", default=None, type=str)
    parser.add_argument("--test_struct_cache", default=None, type=str)

    parser.add_argument("--output_dir", required=True, type=str)

    parser.add_argument("--input_len", default=240, type=int)
    parser.add_argument("--n_structures", default=3, type=int)
    parser.add_argument("--structure_vocab_size", default=7, type=int)

    parser.add_argument("--conv_filters", default=128, type=int)
    parser.add_argument("--kernel_size", default=8, type=int)
    parser.add_argument("--pool_size", default=3, type=int)
    parser.add_argument("--lstm_hidden", default=64, type=int)
    parser.add_argument("--lstm_layers", default=1, type=int)
    parser.add_argument("--fusion_hidden", default=256, type=int)
    parser.add_argument("--dropout", default=0.3, type=float)

    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=8, type=int)

    parser.add_argument("--optimizer", default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--nesterov", action="store_true")

    parser.add_argument("--num_train_epochs", default=50, type=int)
    parser.add_argument("--selection_metric", default="pr_auc", type=str)
    parser.add_argument("--early_stopping_patience", default=10, type=int)
    parser.add_argument("--seed", default=2024, type=int)

    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    parser.add_argument("--check_headers", action="store_true")

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

    train_ds = DeepPASTAStructCacheDataset(
        seq_cache_prefix=args.train_cache,
        struct_cache_prefix=args.train_struct_cache,
        input_len=args.input_len,
        n_structures=args.n_structures,
        check_headers=args.check_headers,
        mmap=True,
    )

    val_ds = DeepPASTAStructCacheDataset(
        seq_cache_prefix=args.val_cache,
        struct_cache_prefix=args.val_struct_cache,
        input_len=args.input_len,
        n_structures=args.n_structures,
        check_headers=args.check_headers,
        mmap=True,
    )

    if args.test_cache is not None and args.test_struct_cache is not None:
        test_ds = DeepPASTAStructCacheDataset(
            seq_cache_prefix=args.test_cache,
            struct_cache_prefix=args.test_struct_cache,
            input_len=args.input_len,
            n_structures=args.n_structures,
            check_headers=args.check_headers,
            mmap=True,
        )
    else:
        test_ds = None

    collator = DeepPASTAStructCollator()

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

    model = DeepPASTAFullClassifier(
        input_len=args.input_len,
        n_structures=args.n_structures,
        structure_vocab_size=args.structure_vocab_size,
        conv_filters=args.conv_filters,
        kernel_size=args.kernel_size,
        pool_size=args.pool_size,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        fusion_hidden=args.fusion_hidden,
        dropout=args.dropout,
        num_classes=2,
    )

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
                "train_struct_cache": args.train_struct_cache,
                "val_cache": args.val_cache,
                "val_struct_cache": args.val_struct_cache,
                "test_cache": args.test_cache,
                "test_struct_cache": args.test_struct_cache,
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
                "check_headers": args.check_headers,
                "encoder_info": {
                    "encoder": "Identity",
                    "packed_input": "[B, 1+n_structures, input_len]",
                    "packed_channel_0": "sequence index: A=0,C=1,G=2,T=3,N=4",
                    "packed_channel_1_to_S": "RNAshape structure index: U=0,E=1,L=2,R=3,H=4,M=5,I=6",
                },
                "model_info": model_info,
                "train_n": len(train_ds),
                "val_n": len(val_ds),
                "test_n": len(test_ds) if test_ds is not None else None,
            },
            f,
            indent=2,
        )

    train_loss = nn.CrossEntropyLoss()
    eval_loss = nn.CrossEntropyLoss()

    trainer = OneHotBinaryTrainer(
        model=model,
        encoder=nn.Identity(),
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
