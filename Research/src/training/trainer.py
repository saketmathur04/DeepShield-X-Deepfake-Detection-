"""
DeepShield-X — Trainer

Reusable training loop with mixed precision, early stopping, checkpointing,
and detailed logging. Model-agnostic — works with any model from the factory.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
try:
    from torch.amp import GradScaler as _GradScaler, autocast as _autocast
    def create_grad_scaler(device_type="cuda"):
        return _GradScaler(device_type)
    def get_autocast(device_type="cuda"):
        return _autocast(device_type)
except (ImportError, AttributeError):
    from torch.cuda.amp import GradScaler as _GradScaler, autocast as _autocast
    def create_grad_scaler(device_type="cuda"):
        return _GradScaler()
    def get_autocast(device_type="cuda"):
        return _autocast()
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.losses import create_loss
from src.training.optimizer import create_optimizer
from src.training.scheduler import create_scheduler
from src.evaluation.metrics import compute_all_metrics
from src.utils.checkpoint import save_checkpoint, save_training_history

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to halt training when monitored metric stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False


METRIC_MODES: Dict[str, str] = {
    "val_loss": "min",
    "loss": "min",
    "val_acc": "max",
    "accuracy": "max",
    "val_precision": "max",
    "precision": "max",
    "val_recall": "max",
    "recall": "max",
    "val_f1": "max",
    "f1": "max",
    "val_specificity": "max",
    "specificity": "max",
    "val_balanced_accuracy": "max",
    "balanced_accuracy": "max",
    "val_roc_auc": "max",
    "roc_auc": "max",
    "val_pr_auc": "max",
    "pr_auc": "max",
}


def resolve_monitor_mode(monitor: str, configured_mode: Optional[str] = None) -> Tuple[str, str]:
    """
    Resolve canonical validation monitor name and optimization direction ('min' or 'max').

    Supported monitors include at minimum:
        val_loss -> min
        val_roc_auc -> max
        val_pr_auc -> max
        val_f1 -> max
        val_balanced_accuracy -> max
    """
    canonical = monitor if monitor.startswith("val_") else f"val_{monitor}"
    if configured_mode and str(configured_mode).lower() in ("min", "max"):
        return canonical, str(configured_mode).lower()

    if canonical in METRIC_MODES:
        return canonical, METRIC_MODES[canonical]
    if monitor in METRIC_MODES:
        return canonical, METRIC_MODES[monitor]

    if "loss" in canonical.lower():
        return canonical, "min"
    return canonical, "max"


class Trainer:
    """
    Model-agnostic training framework.

    Supports:
    - GPU / CPU / MPS training
    - Mixed precision (automatic)
    - Early stopping
    - Learning rate scheduling
    - Checkpointing (best + last)
    - Training history logging
    - Resume from checkpoint
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Dict[str, Any],
    ):
        """
        Args:
            model: Model to train (any model from the factory).
            device: Target device (cuda, mps, cpu).
            config: Training configuration dict.
        """
        self.model = model.to(device)
        self.device = device
        self.config = config

        training_cfg = config.get("training", config)

        # Loss function
        pos_weight_val = training_cfg.get("pos_weight", None)
        pos_weight_tensor = None
        if pos_weight_val is not None:
            pos_weight_tensor = torch.tensor(float(pos_weight_val), device=device)
            logger.info(f"Using pos_weight={pos_weight_val} for class-balanced BCE")

        self.criterion = create_loss(
            loss_name=training_cfg.get("loss", "bce"),
            label_smoothing=training_cfg.get("label_smoothing", 0.0),
            focal_alpha=training_cfg.get("focal_alpha", 0.25),
            focal_gamma=training_cfg.get("focal_gamma", 2.0),
            pos_weight=pos_weight_tensor,
        )

        # Optimizer
        self.optimizer = create_optimizer(
            model=self.model,
            optimizer_name=training_cfg.get("optimizer", "adamw"),
            learning_rate=training_cfg.get("learning_rate", 1e-4),
            weight_decay=training_cfg.get("weight_decay", 1e-4),
        )

        # Scheduler
        self.scheduler = create_scheduler(
            optimizer=self.optimizer,
            scheduler_name=training_cfg.get("scheduler", "cosine"),
            epochs=training_cfg.get("epochs", 20),
            scheduler_params=training_cfg.get("scheduler_params", {}),
        )

        # Mixed precision
        self.use_amp = training_cfg.get("mixed_precision", True) and device.type == "cuda"
        self.scaler = create_grad_scaler(device.type) if self.use_amp else None

        # Checkpoint monitor & early stopping
        es_cfg = training_cfg.get("early_stopping", {})
        raw_monitor = training_cfg.get("checkpoint_monitor") or es_cfg.get("monitor", "val_loss")
        configured_mode = es_cfg.get("mode") or training_cfg.get("checkpoint_mode")
        self.monitor, self.mode = resolve_monitor_mode(raw_monitor, configured_mode)

        self.early_stopping = None
        if es_cfg.get("enabled", False):
            self.early_stopping = EarlyStopping(
                patience=es_cfg.get("patience", 5),
                min_delta=es_cfg.get("min_delta", 0.001),
                mode=self.mode,
            )
            logger.info(
                f"Early stopping enabled: monitor={self.monitor}, mode={self.mode}, "
                f"patience={es_cfg.get('patience', 5)}, min_delta={es_cfg.get('min_delta', 0.001)}"
            )
        else:
            logger.info(f"Checkpoint monitor configured: {self.monitor} (mode={self.mode})")

        # Gradient clipping
        self.gradient_clip = training_cfg.get("gradient_clip", 0.0)

        # Training params
        self.epochs = training_cfg.get("epochs", 20)
        self.checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "./experiments/checkpoints"))

        # History
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rate": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "val_specificity": [],
            "val_balanced_accuracy": [],
            "val_roc_auc": [],
            "val_pr_auc": [],
            "epoch_seconds": [],
            "samples_per_second": [],
            "gpu_peak_memory_gb": [],
        }
        self.last_val_metrics: Dict[str, Any] = {}

        self.best_val_loss = float("inf")
        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self.start_epoch = 0

    def train_epoch(self, dataloader: DataLoader, epoch: int = 0) -> Tuple[float, float]:
        """Run one training epoch with live epoch-level progress. Returns (loss, accuracy)."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        desc = f"Epoch [{epoch+1}/{self.epochs}] Train"
        pbar = tqdm(dataloader, desc=desc, leave=True)
        for images, labels, _ in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.use_amp:
                with get_autocast(self.device.type):
                    outputs = self.model(images).squeeze(-1)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images).squeeze(-1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                if self.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate_epoch(self, dataloader: DataLoader, epoch: int = 0) -> Tuple[float, float]:
        """Run one validation epoch with live progress. Returns (loss, accuracy)."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels: List[np.ndarray] = []
        all_probs: List[np.ndarray] = []

        desc = f"Epoch [{epoch+1}/{self.epochs}] Val  "
        pbar = tqdm(dataloader, desc=desc, leave=False)
        for images, labels, _ in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.use_amp:
                with get_autocast(self.device.type):
                    outputs = self.model(images).squeeze(-1)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images).squeeze(-1)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_labels.append(labels.detach().cpu().numpy())
            all_probs.append(torch.sigmoid(outputs).detach().cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(val_loss=f"{loss.item():.4f}", val_acc=f"{correct / total:.4f}")

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        self.last_val_metrics = compute_all_metrics(
            np.concatenate(all_labels), None, np.concatenate(all_probs), threshold=0.5
        )
        return epoch_loss, epoch_acc

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_name: str = "model",
        experiment_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            model_name: Model name for checkpoint naming.
            experiment_id: Experiment identifier.

        Returns:
            Dict with final metrics and training history.
        """
        logger.info(
            f"Starting training: model={model_name}, epochs={self.epochs}, "
            f"device={self.device}, amp={self.use_amp}"
        )

        ckpt_dir = self.checkpoint_dir / experiment_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        start_time = time.time()

        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch=epoch)

            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader, epoch=epoch)

            # Get current LR
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rate"].append(current_lr)
            for key in ("precision", "recall", "f1", "specificity", "balanced_accuracy", "roc_auc", "pr_auc"):
                self.history[f"val_{key}"].append(self.last_val_metrics.get(key))

            epoch_time = time.time() - epoch_start
            self.history["epoch_seconds"].append(epoch_time)
            self.history["samples_per_second"].append((len(train_loader.dataset) + len(val_loader.dataset)) / epoch_time if epoch_time else 0.0)
            self.history["gpu_peak_memory_gb"].append(
                round(torch.cuda.max_memory_allocated(self.device) / 1e9, 4) if self.device.type == "cuda" else None
            )

            # Resolve current monitored metric score
            current_metrics = {
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_loss": train_loss,
                "train_acc": train_acc,
            }
            for k, v in self.last_val_metrics.items():
                current_metrics[f"val_{k}"] = v
                current_metrics[k] = v

            monitored_score = current_metrics.get(self.monitor)
            if monitored_score is None and self.monitor.startswith("val_"):
                monitored_score = current_metrics.get(self.monitor[4:])

            score_str = f"{monitored_score:.4f}" if isinstance(monitored_score, (int, float)) else "N/A"
            logger.info(
                f"Epoch [{epoch+1}/{self.epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"{self.monitor}={score_str} "
                f"lr={current_lr:.2e} time={epoch_time:.1f}s"
            )

            # Determine whether this epoch is the best model so far
            if monitored_score is not None:
                if self.best_score is None:
                    is_best = True
                elif self.mode == "min":
                    is_best = monitored_score < self.best_score
                else:
                    is_best = monitored_score > self.best_score
            else:
                # Fallback to val_loss minimization if monitor score is unavailable
                is_best = val_loss < self.best_val_loss

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

            checkpoint_metrics = {
                "val_loss": val_loss,
                "val_acc": val_acc,
                "monitored_metric": self.monitor,
                "monitored_score": monitored_score,
                "monitored_mode": self.mode,
            }
            for k in ("roc_auc", "pr_auc", "f1", "balanced_accuracy", "precision", "recall", "specificity"):
                val_k = self.last_val_metrics.get(k)
                if val_k is not None:
                    checkpoint_metrics[f"val_{k}"] = val_k

            # Save best model
            if is_best:
                if monitored_score is not None:
                    self.best_score = monitored_score
                self.best_epoch = epoch + 1
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=checkpoint_metrics,
                    config=self.config,
                    path=str(ckpt_dir / f"{model_name}_best.pth"),
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    is_best=True,
                )

            # Save last model
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics=checkpoint_metrics,
                config=self.config,
                path=str(ckpt_dir / f"{model_name}_last.pth"),
                scheduler=self.scheduler,
                scaler=self.scaler,
                is_best=False,
            )

            # Early stopping check
            if self.early_stopping is not None:
                es_score = monitored_score if monitored_score is not None else val_loss
                if self.early_stopping(es_score):
                    logger.info(
                        f"Early stopping triggered at epoch {epoch+1}: "
                        f"{self.monitor}={es_score:.4f} stopped improving"
                    )
                    break

        total_time = time.time() - start_time

        # Save training history
        history_path = ckpt_dir / f"{model_name}_history.json"
        save_training_history(self.history, str(history_path))

        result = {
            "model_name": model_name,
            "experiment_id": experiment_id,
            "epochs_trained": epoch + 1,
            "best_epoch": self.best_epoch,
            "checkpoint_monitor": self.monitor,
            "monitor_mode": self.mode,
            "best_score": self.best_score,
            "best_val_loss": self.best_val_loss,
            "final_train_loss": self.history["train_loss"][-1],
            "final_train_acc": self.history["train_acc"][-1],
            "final_val_loss": self.history["val_loss"][-1],
            "final_val_acc": self.history["val_acc"][-1],
            "total_time_seconds": round(total_time, 2),
            "history": self.history,
            "checkpoint_dir": str(ckpt_dir),
            "best_checkpoint": str(ckpt_dir / f"{model_name}_best.pth"),
        }

        best_score_str = f"{self.best_score:.4f}" if self.best_score is not None else "N/A"
        logger.info(
            f"Training complete: {epoch+1} epochs in {total_time:.1f}s | "
            f"best_{self.monitor}={best_score_str} (epoch {self.best_epoch}) | "
            f"best_val_loss={self.best_val_loss:.4f}"
        )

        return result
