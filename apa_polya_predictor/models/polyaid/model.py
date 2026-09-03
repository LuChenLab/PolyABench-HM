import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from apa_polya_predictor.core.trainer import (
    GPUOneHotEncoder,
    OneHotBinaryTrainer,
    TrainerConfig,
    UInt8CacheDataset,
    UInt8SequenceCollator,
    set_seed,
)


class PolyaIDOneHotClassifier(nn.Module):
    """
    PolyaID-style classification-only model for cached uint8 sequences.

    Adapted from the published PolyaID backbone:
      input (240, 4)
      -> Conv1D(512, kernel_size=8)
      -> ReLU
      -> MaxPool1D(pool=3, stride=3)
      -> Dropout(0.7)
      -> BiLSTM(128, return_sequences=True)
      -> Dropout(0.7)
      -> Flatten
      -> Dense 256 -> Dense 256 -> Dense 128 -> Dense 64
      -> classification output

    In the original PolyaID, there are two output heads:
      1) classification probability
      2) cleavage probability vector (50 nt)

    Here we keep only the classification branch because the current APA
    benchmark data provide binary 0/1 labels but no cleavage-vector targets.
    """

    def __init__(
        self,
        input_len: int = 240,
        conv_filters: int = 512,
        kernel_size: int = 8,
        pool_size: int = 3,
        pool_stride: int = 3,
        lstm_hidden: int = 128,
        dense_dims = (256, 256, 128, 64),
        dropout: float = 0.7,
        num_classes: int = 2,
    ):
        super().__init__()
        self.input_len = int(input_len)
        self.conv_filters = int(conv_filters)
        self.kernel_size = int(kernel_size)
        self.pool_size = int(pool_size)
        self.pool_stride = int(pool_stride)
        self.lstm_hidden = int(lstm_hidden)
        self.dense_dims = tuple(int(x) for x in dense_dims)
        self.dropout_p = float(dropout)
        self.num_classes = int(num_classes)

        self.conv = nn.Conv1d(
            in_channels=4,
            out_channels=self.conv_filters,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
            bias=True,
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=self.pool_size, stride=self.pool_stride)
        self.dropout1 = nn.Dropout(p=self.dropout_p)

        self.bilstm = nn.LSTM(
            input_size=self.conv_filters,
            hidden_size=self.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout2 = nn.Dropout(p=self.dropout_p)

        conv_out_len = self.input_len - self.kernel_size + 1
        pooled_len = (conv_out_len - self.pool_size) // self.pool_stride + 1
        flatten_dim = pooled_len * (2 * self.lstm_hidden)

        layers = []
        prev_dim = flatten_dim
        for i, dim in enumerate(self.dense_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            if i < len(self.dense_dims) - 1:
                layers.append(nn.Dropout(p=self.dropout_p))
            prev_dim = dim
        self.classifier = nn.Sequential(*layers)
        self.out = nn.Linear(prev_dim, self.num_classes)

        self.model_info = {
            "input_len": self.input_len,
            "conv_filters": self.conv_filters,
            "kernel_size": self.kernel_size,
            "pool_size": self.pool_size,
            "pool_stride": self.pool_stride,
            "lstm_hidden": self.lstm_hidden,
            "dense_dims": list(self.dense_dims),
            "dropout": self.dropout_p,
            "conv_out_len": int(conv_out_len),
            "pooled_len": int(pooled_len),
            "flatten_dim": int(flatten_dim),
            "num_classes": self.num_classes,
        }

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        for name, param in self.bilstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 4, L]
        h = self.conv(x)           # [B, 512, 233]
        h = self.relu(h)
        h = self.pool(h)           # [B, 512, 77]
        h = self.dropout1(h)

        h = h.transpose(1, 2)      # [B, 77, 512]
        h, _ = self.bilstm(h)      # [B, 77, 256]
        h = self.dropout2(h)

        h = h.flatten(start_dim=1) # [B, 77*256]
        h = self.classifier(h)
        logits = self.out(h)
        return logits


def build_scheduler(optimizer, decay_gamma: float, decay_steps: int):
    def lr_lambda(step: int):
        return decay_gamma ** (step / max(decay_steps, 1))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def main():
    parser = argparse.ArgumentParser(description="Train PolyaID-style classification model on cached APA data")
    parser.add_argument("--train_cache", required=True, type=str)
    parser.add_argument("--val_cache", required=True, type=str)
    parser.add_argument("--test_cache", default=None, type=str)
    parser.add_argument("--output_dir", required=True, type=str)

    parser.add_argument("--input_len", default=240, type=int)
    parser.add_argument("--conv_filters", default=512, type=int)
    parser.add_argument("--kernel_size", default=8, type=int)
    parser.add_argument("--pool_size", default=3, type=int)
    parser.add_argument("--pool_stride", default=3, type=int)
    parser.add_argument("--lstm_hidden", default=128, type=int)
    parser.add_argument("--dense_dims", nargs="+", default=[256, 256, 128, 64], type=int)
    parser.add_argument("--dropout", default=0.7, type=float)

    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)

    parser.add_argument("--use_exponential_decay", action="store_true")
    parser.add_argument("--lr_decay_gamma", default=0.96, type=float)
    parser.add_argument("--lr_decay_steps", default=3000, type=int)

    parser.add_argument("--num_train_epochs", default=50, type=int)
    parser.add_argument("--selection_metric", default="pr_auc", type=str)
    parser.add_argument("--early_stopping_patience", default=10, type=int)
    parser.add_argument("--seed", default=2024, type=int)

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

    encoder = GPUOneHotEncoder(dna_only=args.dna_only, dtype=torch.float32)
    model = PolyaIDOneHotClassifier(
        input_len=args.input_len,
        conv_filters=args.conv_filters,
        kernel_size=args.kernel_size,
        pool_size=args.pool_size,
        pool_stride=args.pool_stride,
        lstm_hidden=args.lstm_hidden,
        dense_dims=args.dense_dims,
        dropout=args.dropout,
        num_classes=2,
    )

    if args.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    scheduler = None
    if args.use_exponential_decay:
        scheduler = build_scheduler(optimizer, args.lr_decay_gamma, args.lr_decay_steps)

    train_loss_fn = nn.CrossEntropyLoss()
    eval_loss_fn = nn.CrossEntropyLoss()

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
                "use_exponential_decay": args.use_exponential_decay,
                "lr_decay_gamma": args.lr_decay_gamma,
                "lr_decay_steps": args.lr_decay_steps,
                "mixed_precision": args.mixed_precision,
                "compile_model": args.compile_model,
                "dna_only": args.dna_only,
                "model_info": model.model_info,
            },
            f,
            indent=2,
        )

    trainer = OneHotBinaryTrainer(
        model=model,
        encoder=encoder,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_dataset_len=len(train_ds),
        val_dataset_len=len(val_ds),
        test_dataset_len=(len(test_ds) if test_ds is not None else None),
        device=device,
        trainer_cfg=trainer_cfg,
        train_loss_fn=lambda model_obj, logits, y: train_loss_fn(logits, y),
        eval_loss_fn=eval_loss_fn,
    )

    final = trainer.fit()
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
