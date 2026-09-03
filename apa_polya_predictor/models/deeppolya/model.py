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


class DeepPolyAOneHotClassifier(nn.Module):
    """
    DeepPolyA-style adapted classifier for cached uint8 DNA sequences.

    Original DeepPolyA uses:
      - one-hot DNA input
      - two convolutional layers
      - one max-pooling layer
      - one fully connected hidden layer
      - sigmoid output for binary classification

    This adapted version keeps the same backbone style, but uses the
    user's existing 240-nt cache instead of the original 162-nt input.
    """

    def __init__(
        self,
        input_len: int = 240,
        conv1_filters: int = 16,
        conv1_kernel: int = 8,
        conv2_filters: int = 64,
        conv2_kernel: int = 6,
        pool_size: int = 2,
        pool_stride: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.4,
        use_batchnorm: bool = True,
        num_classes: int = 2,
    ):
        super().__init__()
        self.input_len = int(input_len)
        self.conv1_filters = int(conv1_filters)
        self.conv1_kernel = int(conv1_kernel)
        self.conv2_filters = int(conv2_filters)
        self.conv2_kernel = int(conv2_kernel)
        self.pool_size = int(pool_size)
        self.pool_stride = int(pool_stride)
        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)
        self.num_classes = int(num_classes)

        self.conv1 = nn.Conv1d(4, self.conv1_filters, kernel_size=self.conv1_kernel, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(self.conv1_filters) if self.use_batchnorm else nn.Identity()
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(self.dropout_p)

        self.conv2 = nn.Conv1d(self.conv1_filters, self.conv2_filters, kernel_size=self.conv2_kernel, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(self.conv2_filters) if self.use_batchnorm else nn.Identity()
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=self.pool_size, stride=self.pool_stride)
        self.drop2 = nn.Dropout(self.dropout_p)

        conv1_out_len = self.input_len - self.conv1_kernel + 1
        conv2_out_len = conv1_out_len - self.conv2_kernel + 1
        pooled_len = (conv2_out_len - self.pool_size) // self.pool_stride + 1
        flatten_dim = self.conv2_filters * pooled_len

        self.fc1 = nn.Linear(flatten_dim, self.hidden_dim)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(self.dropout_p)
        self.out = nn.Linear(self.hidden_dim, self.num_classes)

        self.model_info = {
            "input_len": self.input_len,
            "conv1_filters": self.conv1_filters,
            "conv1_kernel": self.conv1_kernel,
            "conv2_filters": self.conv2_filters,
            "conv2_kernel": self.conv2_kernel,
            "pool_size": self.pool_size,
            "pool_stride": self.pool_stride,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout_p,
            "use_batchnorm": self.use_batchnorm,
            "conv1_out_len": int(conv1_out_len),
            "conv2_out_len": int(conv2_out_len),
            "pooled_len": int(pooled_len),
            "flatten_dim": int(flatten_dim),
            "num_classes": self.num_classes,
        }

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.drop1(h)

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = self.pool(h)
        h = self.drop2(h)

        h = h.flatten(start_dim=1)
        h = self.fc1(h)
        h = self.relu3(h)
        h = self.drop3(h)
        logits = self.out(h)
        return logits


def main():
    parser = argparse.ArgumentParser(description="Train adapted DeepPolyA benchmark model on cached APA data")
    parser.add_argument("--train_cache", required=True, type=str)
    parser.add_argument("--val_cache", required=True, type=str)
    parser.add_argument("--test_cache", default=None, type=str)
    parser.add_argument("--output_dir", required=True, type=str)

    parser.add_argument("--input_len", default=240, type=int)
    parser.add_argument("--conv1_filters", default=16, type=int)
    parser.add_argument("--conv1_kernel", default=8, type=int)
    parser.add_argument("--conv2_filters", default=64, type=int)
    parser.add_argument("--conv2_kernel", default=6, type=int)
    parser.add_argument("--pool_size", default=2, type=int)
    parser.add_argument("--pool_stride", default=2, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--dropout", default=0.4, type=float)
    parser.add_argument("--disable_batchnorm", action="store_true")

    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--nesterov", action="store_true")

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
    model = DeepPolyAOneHotClassifier(
        input_len=args.input_len,
        conv1_filters=args.conv1_filters,
        conv1_kernel=args.conv1_kernel,
        conv2_filters=args.conv2_filters,
        conv2_kernel=args.conv2_kernel,
        pool_size=args.pool_size,
        pool_stride=args.pool_stride,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_batchnorm=(not args.disable_batchnorm),
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
            nesterov=args.nesterov,
        )

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
        scheduler=None,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_dataset_len=len(train_ds),
        val_dataset_len=len(val_ds),
        test_dataset_len=(len(test_ds) if test_ds is not None else None),
        device=device,
        trainer_cfg=trainer_cfg,
        train_loss_fn=lambda model_obj, logits, y: nn.CrossEntropyLoss()(logits, y),
        eval_loss_fn=nn.CrossEntropyLoss(),
    )

    final = trainer.fit()
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
