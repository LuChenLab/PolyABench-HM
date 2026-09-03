import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from apa_polya_predictor.core.trainer import (
    ConstantPad1dChannels,
    GPUOneHotEncoder,
    OneHotBinaryTrainer,
    TrainerConfig,
    UInt8CacheDataset,
    UInt8SequenceCollator,
    set_seed,
)


class DeeReCTPolyAOneHot(nn.Module):
    """
    Structural reproduction of DeeReCT-PolyA on cached uint8 sequences.

    Original idea:
      one-hot -> constant padding(0.25) -> conv(k=10, d=16) -> group norm ->
      ReLU -> maxpool(k=10,s=10) -> FC64 -> binary classifier
    """
    def __init__(
        self,
        input_len: int = 240,
        patch_size: int = 10,
        depth: int = 16,
        num_hidden: int = 64,
        num_groups: int = 4,
        keep_prob: float = 0.5,
    ):
        super().__init__()
        self.input_len = input_len
        self.patch_size = patch_size
        self.depth = depth
        self.num_hidden = num_hidden
        self.num_groups = num_groups
        self.keep_prob = keep_prob

        self.pad = ConstantPad1dChannels(patch_size - 1, patch_size - 1, value=0.25)
        self.conv = nn.Conv1d(
            in_channels=4,
            out_channels=depth,
            kernel_size=patch_size,
            stride=1,
            padding=0,
            bias=False,
        )
        self.group_norm = nn.GroupNorm(num_groups, depth, affine=False)
        self.post_norm_bias = nn.Parameter(torch.zeros(depth))
        self.pool = nn.MaxPool1d(kernel_size=patch_size, stride=patch_size)

        padded_len = input_len + 2 * patch_size - 2
        conv_out_len = padded_len - patch_size + 1
        pooled_len = (conv_out_len - patch_size) // patch_size + 1
        flatten_dim = pooled_len * depth

        self.fc1 = nn.Linear(flatten_dim, num_hidden)
        self.fc2 = nn.Linear(num_hidden, 2)
        self.dropout = nn.Dropout(p=1.0 - keep_prob)

        self._init_weights()

        self.model_info = {
            "input_len": input_len,
            "patch_size": patch_size,
            "depth": depth,
            "num_hidden": num_hidden,
            "num_groups": num_groups,
            "keep_prob": keep_prob,
            "padded_len": padded_len,
            "conv_out_len": conv_out_len,
            "pooled_len": pooled_len,
            "flatten_dim": flatten_dim,
        }

    def _init_weights(self):
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.05)
        nn.init.constant_(self.fc1.bias, 1.0)
        nn.init.constant_(self.fc2.bias, 1.0)

    def forward(self, x):
        # x: [B, 4, L]
        h = self.pad(x)
        h = self.conv(h)
        h = self.group_norm(h)
        h = h + self.post_norm_bias.view(1, -1, 1)
        h = torch.relu(h)
        h = self.pool(h)
        h = h.flatten(start_dim=1)
        h = torch.relu(self.fc1(h))
        h = self.dropout(h)
        logits = self.fc2(h)
        return logits


class DeeReCTLoss(nn.Module):
    def __init__(self, motif_weight_decay: float, fc_weight_decay: float):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.motif_weight_decay = motif_weight_decay
        self.fc_weight_decay = fc_weight_decay

    def forward(self, model: DeeReCTPolyAOneHot, logits: torch.Tensor, labels: torch.Tensor):
        loss = self.ce(logits, labels)
        loss = loss + self.motif_weight_decay * torch.sum(model.conv.weight ** 2) / 2.0
        loss = loss + self.fc_weight_decay * (
            torch.sum(model.fc1.weight ** 2) / 2.0 + torch.sum(model.fc2.weight ** 2) / 2.0
        )
        return loss


def build_scheduler(optimizer, decay_gamma: float, decay_steps: int):
    def lr_lambda(step: int):
        return decay_gamma ** (step / max(decay_steps, 1))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def main():
    parser = argparse.ArgumentParser(description="Train cached DeeReCT-PolyA-style CNN")
    parser.add_argument("--train_cache", required=True, type=str)
    parser.add_argument("--val_cache", required=True, type=str)
    parser.add_argument("--test_cache", default=None, type=str)
    parser.add_argument("--output_dir", required=True, type=str)

    parser.add_argument("--input_len", default=240, type=int)
    parser.add_argument("--patch_size", default=10, type=int)
    parser.add_argument("--depth", default=16, type=int)
    parser.add_argument("--num_hidden", default=64, type=int)
    parser.add_argument("--num_groups", default=4, type=int)
    parser.add_argument("--keep_prob", default=0.5, type=float)

    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--lr", default=5e-3, type=float)
    parser.add_argument("--momentum", default=0.97, type=float)
    parser.add_argument("--lr_decay_gamma", default=0.96, type=float)
    parser.add_argument("--lr_decay_steps", default=3000, type=int)
    parser.add_argument("--motif_weight_decay", default=1e-4, type=float)
    parser.add_argument("--fc_weight_decay", default=1e-4, type=float)

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

    if train_ds.seq_len != args.input_len:
        raise ValueError(
            f"train cache seq_len={train_ds.seq_len}, but --input_len={args.input_len}"
        )
    if val_ds.seq_len != args.input_len:
        raise ValueError(
            f"val cache seq_len={val_ds.seq_len}, but --input_len={args.input_len}"
        )
    if test_ds is not None and test_ds.seq_len != args.input_len:
        raise ValueError(
            f"test cache seq_len={test_ds.seq_len}, but --input_len={args.input_len}"
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

    train_loader = DataLoader(train_ds, shuffle=True, **common_loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **common_loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **common_loader_kwargs) if test_ds is not None else None

    encoder = GPUOneHotEncoder(dna_only=args.dna_only, dtype=torch.float32)
    model = DeeReCTPolyAOneHot(
        input_len=args.input_len,
        patch_size=args.patch_size,
        depth=args.depth,
        num_hidden=args.num_hidden,
        num_groups=args.num_groups,
        keep_prob=args.keep_prob,
    )

    if args.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = build_scheduler(optimizer, args.lr_decay_gamma, args.lr_decay_steps)

    train_loss_fn = DeeReCTLoss(args.motif_weight_decay, args.fc_weight_decay)
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
                "lr": args.lr,
                "momentum": args.momentum,
                "lr_decay_gamma": args.lr_decay_gamma,
                "lr_decay_steps": args.lr_decay_steps,
                "motif_weight_decay": args.motif_weight_decay,
                "fc_weight_decay": args.fc_weight_decay,
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
        train_loss_fn=train_loss_fn,
        eval_loss_fn=eval_loss_fn,
    )

    final = trainer.fit()
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
