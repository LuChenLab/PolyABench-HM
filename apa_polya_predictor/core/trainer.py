import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# =========================================================
# Cache dataset
# Expects files from build_onehot_cache.py:
#   <prefix>.x.uint8.npy
#   <prefix>.y.npy
#   <prefix>.headers.npy
#   <prefix>.meta.json
# =========================================================

class UInt8CacheDataset(Dataset):
    def __init__(self, cache_prefix: str, mmap: bool = True):
        self.cache_prefix = cache_prefix
        self.x_path = f"{cache_prefix}.x.uint8.npy"
        self.y_path = f"{cache_prefix}.y.npy"
        self.h_path = f"{cache_prefix}.headers.npy"
        self.m_path = f"{cache_prefix}.meta.json"

        mmap_mode = "r" if mmap else None

        self.x = np.load(self.x_path, mmap_mode=mmap_mode)
        self.y = np.load(self.y_path, mmap_mode=mmap_mode)
        self.headers = np.load(self.h_path, allow_pickle=True)

        with open(self.m_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        if "seq_len" in self.meta:
            self.seq_len = int(self.meta["seq_len"])
        else:
            self.seq_len = int(self.x.shape[1])

        if "num_samples" in self.meta:
            self.n_samples = int(self.meta["num_samples"])
        elif "n_samples" in self.meta:
            self.n_samples = int(self.meta["n_samples"])
        else:
            self.n_samples = int(self.x.shape[0])

        if len(self.x) != len(self.y):
            raise ValueError("x and y have inconsistent lengths")
        if len(self.x) != len(self.headers):
            raise ValueError("x and headers have inconsistent lengths")
        if self.x.shape[1] != self.seq_len:
            raise ValueError("x shape and seq_len are inconsistent")
        if self.x.shape[0] != self.n_samples:
            raise ValueError("x shape and n_samples are inconsistent")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return {
            "x_idx": np.asarray(self.x[idx], dtype=np.uint8),
            "y": int(self.y[idx]),
            "header": str(self.headers[idx]),
        }


class UInt8SequenceCollator:
    def __call__(self, batch: List[Dict]):
        x_idx = torch.tensor(
            np.stack([item["x_idx"] for item in batch], axis=0),
            dtype=torch.long,
        )
        y = torch.tensor([item["y"] for item in batch], dtype=torch.long)
        headers = [item["header"] for item in batch]
        return {"x_idx": x_idx, "y": y, "header": headers}


# =========================================================
# GPU one-hot encoder and channel padding
# =========================================================

class GPUOneHotEncoder(nn.Module):
    """
    Encodes integer tokens to channel-first one-hot-like tensors on GPU.

    Token mapping:
      0:A, 1:C, 2:G, 3:T/U, 4:N/unknown
    """
    def __init__(self, dna_only: bool = False, dtype: torch.dtype = torch.float32):
        super().__init__()
        if dna_only:
            table = torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],  # A
                    [0.0, 1.0, 0.0, 0.0],  # C
                    [0.0, 0.0, 1.0, 0.0],  # G
                    [0.0, 0.0, 0.0, 1.0],  # T
                    [0.0, 0.0, 0.0, 0.0],  # unexpected in strict DNA mode
                ],
                dtype=dtype,
            )
        else:
            table = torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],      # A
                    [0.0, 1.0, 0.0, 0.0],      # C
                    [0.0, 0.0, 1.0, 0.0],      # G
                    [0.0, 0.0, 0.0, 1.0],      # T/U
                    [0.25, 0.25, 0.25, 0.25],  # N/unknown
                ],
                dtype=dtype,
            )
        self.register_buffer("table", table, persistent=False)

    def forward(self, x_idx: torch.Tensor) -> torch.Tensor:
        # x_idx: [B, L]
        x = self.table[x_idx]      # [B, L, 4]
        x = x.transpose(1, 2)      # [B, 4, L]
        return x.contiguous()


class ConstantPad1dChannels(nn.Module):
    def __init__(self, pad_left: int, pad_right: int, value: float = 0.25):
        super().__init__()
        self.pad_left = int(pad_left)
        self.pad_right = int(pad_right)
        self.value = float(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        if self.pad_left == 0 and self.pad_right == 0:
            return x
        bsz, ch, seqlen = x.shape
        out = torch.full(
            (bsz, ch, seqlen + self.pad_left + self.pad_right),
            fill_value=self.value,
            dtype=x.dtype,
            device=x.device,
        )
        out[:, :, self.pad_left:self.pad_left + seqlen] = x
        return out


# =========================================================
# Metrics and helpers
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Return unified binary classification metrics for benchmark reporting.

    Metric keys are intentionally stable across all models:
      AUROC, AUPRC, MCC, F1, Recall, Precision, Specificity, Accuracy
    """
    probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
    preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    y = labels.detach().cpu().numpy()

    if len(np.unique(y)) > 1:
        auroc = float(roc_auc_score(y, probs))
        auprc = float(average_precision_score(y, probs))
    else:
        auroc = 0.0
        auprc = 0.0

    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "MCC": float(matthews_corrcoef(y, preds)) if len(np.unique(y)) > 1 else 0.0,
        "F1": float(f1_score(y, preds, zero_division=0)),
        "Recall": float(recall_score(y, preds, zero_division=0)),
        "Precision": float(precision_score(y, preds, zero_division=0)),
        "Specificity": specificity,
        "Accuracy": float(accuracy_score(y, preds)),
    }


def _lower_metric_dict(metrics: Dict[str, float]) -> Dict[str, float]:
    lowered = {k.lower(): v for k, v in metrics.items()}
    # Backward-compatible aliases for older scripts / configs.
    if "auprc" in lowered:
        lowered["pr_auc"] = lowered["auprc"]
        lowered["average_precision"] = lowered["auprc"]
    if "auroc" in lowered:
        lowered["auc"] = lowered["auroc"]
        lowered["roc_auc"] = lowered["auroc"]
    if "mcc" in lowered:
        lowered["mcc"] = lowered["mcc"]
    return lowered


def _batch_to_xy(batch, device: torch.device):
    """Support both dict batches and tuple/list batches.

    Dict formats supported:
      {"x_idx", "y"} from apa_onehot_trainer.UInt8CacheDataset
      {"seq_idx", "labels"} from apa_onehot_dataset.UInt8SequenceDataset
    Tuple format supported:
      (x, y) used by DeepPASTA structure-cache dataset.
    """
    if isinstance(batch, dict):
        if "x_idx" in batch:
            x_raw = batch["x_idx"]
        elif "seq_idx" in batch:
            x_raw = batch["seq_idx"]
        else:
            raise KeyError("Batch dict must contain 'x_idx' or 'seq_idx'.")

        if "y" in batch:
            y = batch["y"]
        elif "labels" in batch:
            y = batch["labels"]
        else:
            raise KeyError("Batch dict must contain 'y' or 'labels'.")

        headers = batch.get("header", batch.get("headers", None))

    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        x_raw, y = batch[0], batch[1]
        headers = None
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    x_raw = x_raw.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    return x_raw, y, headers


@dataclass
class TrainerConfig:
    output_dir: str
    selection_metric: str = "pr_auc"
    early_stopping_patience: int = 10
    num_train_epochs: int = 50
    mixed_precision: bool = True
    log_every: int = 1


class OneHotBinaryTrainer:
    """
    Generic trainer for one-hot-input binary classification models.

    Model contract:
      model(x_onehot) -> logits of shape [B, 2]
    """
    def __init__(
        self,
        model: nn.Module,
        encoder: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader],
        train_dataset_len: int,
        val_dataset_len: int,
        test_dataset_len: Optional[int],
        device: torch.device,
        trainer_cfg: TrainerConfig,
        train_loss_fn,
        eval_loss_fn,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        self.model = model.to(device)
        self.encoder = encoder.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_dataset_len = train_dataset_len
        self.val_dataset_len = val_dataset_len
        self.test_dataset_len = test_dataset_len
        self.device = device
        self.cfg = trainer_cfg
        self.train_loss_fn = train_loss_fn
        self.eval_loss_fn = eval_loss_fn

        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self.metrics_csv = os.path.join(self.cfg.output_dir, "metrics.csv")
        self.best_ckpt = os.path.join(self.cfg.output_dir, "best_model.pt")
        self.final_json = os.path.join(self.cfg.output_dir, "final_test_metrics.json")

        self.use_amp = bool(self.cfg.mixed_precision and device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _move_encode(self, x_raw: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_raw)

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        dataset_len: int,
        desc: str,
        return_predictions: bool = False,
    ):
        self.model.eval()
        self.encoder.eval()

        total_loss = 0.0
        total_n = 0
        logits_all = []
        labels_all = []
        headers_all = []

        pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        for batch in pbar:
            x_raw, y, headers = _batch_to_xy(batch, self.device)
            x = self._move_encode(x_raw)

            if self.use_amp:
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.model(x)
                    loss = self.eval_loss_fn(logits, y)
            else:
                logits = self.model(x)
                loss = self.eval_loss_fn(logits, y)

            bsz = y.size(0)
            total_loss += float(loss.item()) * bsz
            total_n += bsz
            logits_all.append(logits.float())
            labels_all.append(y)
            if headers is not None:
                headers_all.extend(headers)

            pbar.set_postfix(loss=f"{total_loss / max(total_n, 1):.4f}")

        logits_cat = torch.cat(logits_all, dim=0)
        labels_cat = torch.cat(labels_all, dim=0)
        metrics = binary_classification_metrics(logits_cat, labels_cat)
        avg_loss = total_loss / max(total_n, 1)

        if not return_predictions:
            return avg_loss, metrics

        probs = torch.softmax(logits_cat, dim=-1)[:, 1].detach().cpu().numpy()
        preds = torch.argmax(logits_cat, dim=-1).detach().cpu().numpy()
        labels_np = labels_cat.detach().cpu().numpy()
        pred_df = pd.DataFrame({
            "label": labels_np,
            "prob_positive": probs,
            "pred": preds,
        })
        if len(headers_all) == len(pred_df):
            pred_df.insert(0, "header", headers_all)

        return avg_loss, metrics, pred_df

    def fit(self) -> Dict[str, float]:
        selection_metric = self.cfg.selection_metric.lower()
        best_metric = float("-inf")
        best_epoch = -1
        bad_epochs = 0
        history = []

        run_start = time.perf_counter()

        print(f"Trainable parameters: {count_trainable_parameters(self.model)}")
        print(f"Using AMP: {self.use_amp}")

        for epoch in range(self.cfg.num_train_epochs):
            self.model.train()
            self.encoder.train()

            total_loss = 0.0
            total_n = 0

            train_pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.cfg.num_train_epochs}",
                leave=True,
                dynamic_ncols=True,
            )

            for batch in train_pbar:
                x_raw, y, _ = _batch_to_xy(batch, self.device)
                x = self._move_encode(x_raw)

                self.optimizer.zero_grad(set_to_none=True)

                if self.use_amp:
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        logits = self.model(x)
                        loss = self.train_loss_fn(self.model, logits, y)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits = self.model(x)
                    loss = self.train_loss_fn(self.model, logits, y)
                    loss.backward()
                    self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                bsz = y.size(0)
                total_loss += float(loss.item()) * bsz
                total_n += bsz

                train_pbar.set_postfix(
                    loss=f"{total_loss / max(total_n, 1):.4f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                )

            train_loss = total_loss / max(total_n, 1)

            val_loss, val_metrics = self.evaluate(
                self.val_loader,
                self.val_dataset_len,
                desc=f"Val {epoch + 1}/{self.cfg.num_train_epochs}",
            )

            current_metric = _lower_metric_dict(val_metrics)[selection_metric]
            is_best = current_metric > best_metric

            if is_best:
                best_metric = current_metric
                best_epoch = epoch
                bad_epochs = 0
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "encoder_state_dict": self.encoder.state_dict(),
                        "best_metric": best_metric,
                        "best_epoch": best_epoch,
                        "selection_metric": selection_metric,
                    },
                    self.best_ckpt,
                )
            else:
                bad_epochs += 1

            row = {
                "Epoch": epoch,
                "Train_Loss": train_loss,
                "Val_Loss": val_loss,
                "LR": self.optimizer.param_groups[0]["lr"],
                f"Val_{selection_metric}": current_metric,
                f"Best_{selection_metric}": best_metric,
                "Best_Epoch_So_Far": best_epoch,
                "Is_Best": int(is_best),
                "Epochs_Without_Improvement": bad_epochs,
            }
            for k, v in val_metrics.items():
                row[f"Val_{k}"] = v

            history.append(row)
            pd.DataFrame(history).to_csv(self.metrics_csv, index=False)

            print(
                f"Epoch {epoch} | Train_Loss={train_loss:.6f} | Val_Loss={val_loss:.6f} | "
                + " | ".join([f"Val_{k}={v:.4f}" for k, v in val_metrics.items()])
            )

            if bad_epochs >= self.cfg.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best epoch={best_epoch}, best {selection_metric}={best_metric:.6f}"
                )
                break

        total_train_time = time.perf_counter() - run_start

        if os.path.exists(self.best_ckpt):
            ckpt = torch.load(self.best_ckpt, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.encoder.load_state_dict(ckpt["encoder_state_dict"])
            print(
                f"Loaded best checkpoint from epoch {ckpt['best_epoch']} "
                f"with best {selection_metric}={ckpt['best_metric']:.6f}"
            )

        final = {
            "best_epoch": best_epoch,
            "best_selection_metric": selection_metric,
            "best_selection_metric_value": best_metric,
            "total_train_time_sec": total_train_time,
        }

        if self.test_loader is not None and self.test_dataset_len is not None:
            test_loss, test_metrics, test_pred_df = self.evaluate(
                self.test_loader,
                self.test_dataset_len,
                desc="Test",
                return_predictions=True,
            )
            test_pred_df.to_csv(os.path.join(self.cfg.output_dir, "test_predictions.csv"), index=False)
            final.update(
                {
                    "final_test_loss": test_loss,
                    "final_test_metrics": test_metrics,
                    "test_predictions_csv": os.path.join(self.cfg.output_dir, "test_predictions.csv"),
                }
            )
            print("Final test metrics:")
            print(json.dumps(final, indent=2))

        with open(self.final_json, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2)

        return final
