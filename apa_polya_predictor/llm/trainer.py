from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .collators import APALLMCollator
from .metrics import binary_metrics, metric_is_better, normalize_metric_name


@dataclass
class LLMTrainConfig:
    method: str
    output_dir: str
    best_checkpoint_dir: Optional[str]
    batch_size: int = 8
    num_workers: int = 4
    num_train_epochs: int = 10
    early_stopping_patience: int = 3
    lr: float = 1e-5
    weight_decay: float = 1e-3
    selection_metric: str = "MCC"
    mixed_precision: bool = False
    device: str = "cuda"
    seed: int = 2024
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    threshold: float = 0.5
    scheduler: str = "none"  # none, linear, cosine
    warmup_ratio: float = 0.05


class APALLMTrainer:
    def __init__(self, cfg: LLMTrainConfig, model, tokenizer, runtime: Dict, train_dataset, val_dataset, test_dataset):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.runtime = runtime
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.model.to(self.device)
        self.use_amp = bool(cfg.mixed_precision and self.device.type == "cuda")
        print(f"Using AMP: {self.use_amp}")
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable}")

        self.collator = APALLMCollator(
            tokenizer=self.tokenizer,
            max_seq_len=int(runtime["max_seq_len"]),
            kmer=int(runtime["kmer"]),
            replace_t_with_u=bool(runtime["replace_t_with_u"]),
            replace_u_with_t=bool(runtime["replace_u_with_t"]),
            pad_token_id=int(runtime["pad_token_id"]),
            add_cls_prefix=bool(runtime["add_cls_prefix"]),
        )
        self.train_loader = self._loader(train_dataset, shuffle=True)
        self.val_loader = self._loader(val_dataset, shuffle=False)
        self.test_loader = self._loader(test_dataset, shuffle=False)

        self.optimizer = AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = self._build_scheduler()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.selection_metric = normalize_metric_name(cfg.selection_metric)
        self.best_value = math.inf if metric_is_better(self.selection_metric) == "min" else -math.inf
        self.best_epoch = -1
        self.bad_epochs = 0
        self.history: List[Dict] = []

    def _loader(self, dataset, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collator,
            pin_memory=(self.device.type == "cuda"),
        )

    def _build_scheduler(self):
        scheduler = self.cfg.scheduler.lower()
        if scheduler == "none":
            return None
        total_update_steps = math.ceil(len(self.train_loader) / max(1, self.cfg.gradient_accumulation_steps)) * self.cfg.num_train_epochs
        warmup_steps = int(total_update_steps * self.cfg.warmup_ratio)
        try:
            from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
        except Exception as exc:
            raise RuntimeError("transformers is required for LLM scheduler support") from exc
        if scheduler == "linear":
            return get_linear_schedule_with_warmup(self.optimizer, warmup_steps, total_update_steps)
        if scheduler == "cosine":
            return get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, total_update_steps)
        raise ValueError(f"Unknown scheduler: {self.cfg.scheduler}")

    def _step_logits(self, batch):
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        logits = self.model(input_ids)
        loss = self.loss_fn(logits, labels)
        return loss, logits, labels

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        losses = []
        self.optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.cfg.num_train_epochs}")
        for step, batch in enumerate(pbar):
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                loss, _, _ = self._step_logits(batch)
                loss_to_backward = loss / max(1, self.cfg.gradient_accumulation_steps)
            self.scaler.scale(loss_to_backward).backward()

            if (step + 1) % self.cfg.gradient_accumulation_steps == 0:
                if self.cfg.max_grad_norm and self.cfg.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()

            losses.append(float(loss.detach().cpu().item()))
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")
        return float(np.mean(losses)) if losses else float("nan")

    @torch.no_grad()
    def evaluate(self, loader, split: str, save_predictions: bool = False):
        self.model.eval()
        losses, labels_all, probs_all, preds_all = [], [], [], []
        for batch in tqdm(loader, desc=f"Evaluate {split}", leave=False):
            loss, logits, labels = self._step_logits(batch)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = (probs >= self.cfg.threshold).long()
            losses.append(float(loss.detach().cpu().item()))
            labels_all.extend(labels.detach().cpu().numpy().tolist())
            probs_all.extend(probs.detach().cpu().numpy().tolist())
            preds_all.extend(preds.detach().cpu().numpy().tolist())
        metrics = binary_metrics(labels_all, probs_all, threshold=self.cfg.threshold)
        metrics[f"{split}_Loss"] = float(np.mean(losses)) if losses else float("nan")
        if save_predictions:
            pd.DataFrame({"label": labels_all, "prob": probs_all, "pred": preds_all}).to_csv(
                self.output_dir / f"{split.lower()}_predictions.csv", index=False
            )
        return metrics

    def _is_better(self, value: float) -> bool:
        if metric_is_better(self.selection_metric) == "min":
            return value < self.best_value
        return value > self.best_value

    def save_best(self):
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "config": asdict(self.cfg),
            "runtime": self.runtime,
            "best_epoch": self.best_epoch,
            "best_metric": self.selection_metric,
            "best_value": self.best_value,
        }
        path = self.output_dir / "best_model.pt"
        torch.save(ckpt, path)
        if self.cfg.best_checkpoint_dir:
            best_dir = Path(self.cfg.best_checkpoint_dir)
            best_dir.mkdir(parents=True, exist_ok=True)
            dst = best_dir / f"{self.runtime['method']}_best_model.pt"
            shutil.copy2(path, dst)

    def fit(self):
        start = time.time()
        for epoch in range(self.cfg.num_train_epochs):
            train_loss = self.train_one_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader, "Val", save_predictions=False)
            row = {"epoch": epoch, "Train_Loss": train_loss, **{f"Val_{k}": v for k, v in val_metrics.items() if not k.startswith("Val_")}, "Val_Loss": val_metrics["Val_Loss"], "lr": self.optimizer.param_groups[0]["lr"]}
            self.history.append(row)

            # Prefer Val_<metric>; Val_Loss is stored exactly.
            if self.selection_metric == "Val_Loss":
                monitor_value = row["Val_Loss"]
            else:
                monitor_value = row[f"Val_{self.selection_metric}"]

            print(
                f"Epoch {epoch} | Train_Loss={train_loss:.6f} | Val_Loss={row['Val_Loss']:.6f} | "
                f"Val_AUROC={row['Val_AUROC']:.4f} | Val_AUPRC={row['Val_AUPRC']:.4f} | "
                f"Val_MCC={row['Val_MCC']:.4f} | Val_F1={row['Val_F1']:.4f} | "
                f"Val_Recall={row['Val_Recall']:.4f} | Val_Precision={row['Val_Precision']:.4f} | "
                f"Val_Specificity={row['Val_Specificity']:.4f} | Val_Accuracy={row['Val_Accuracy']:.4f}"
            )

            if self._is_better(float(monitor_value)):
                self.best_value = float(monitor_value)
                self.best_epoch = epoch
                self.bad_epochs = 0
                self.save_best()
            else:
                self.bad_epochs += 1
                if self.bad_epochs >= self.cfg.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}; best epoch={self.best_epoch}, {self.selection_metric}={self.best_value:.6f}")
                    break

            pd.DataFrame(self.history).to_csv(self.output_dir / "metrics.csv", index=False)

        total_time = time.time() - start
        # reload best checkpoint before test
        best_path = self.output_dir / "best_model.pt"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"], strict=True)

        test_metrics = self.evaluate(self.test_loader, "Test", save_predictions=True)
        test_out = {k: v for k, v in test_metrics.items() if not k.startswith("Test_")}
        test_out["Test_Loss"] = test_metrics["Test_Loss"]
        test_out["best_epoch"] = self.best_epoch
        test_out["best_selection_metric"] = self.selection_metric
        test_out["best_selection_value"] = self.best_value
        test_out["total_train_time_sec"] = total_time
        with open(self.output_dir / "final_test_metrics.json", "w") as f:
            json.dump(test_out, f, indent=2)
        with open(self.output_dir / "model_config.json", "w") as f:
            json.dump({"train_config": asdict(self.cfg), "runtime": self.runtime}, f, indent=2)
        pd.DataFrame(self.history).to_csv(self.output_dir / "metrics.csv", index=False)
        return test_out
