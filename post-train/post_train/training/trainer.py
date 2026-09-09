"""Training orchestration for leakage-free S1 next-token post-training.

Accepted objective contract (D-016): compute ordinary mean cross-entropy from
the final-position S1 logits and the single target S1 token.  No class weights,
label smoothing, focal terms, or auxiliary losses.

Accepted optimization contract (D-017): AdamW with LR 4e-5, betas (0.9, 0.95),
weight decay 0.1, FP32, per-GPU batch size 50, no gradient accumulation,
30 epochs, OneCycleLR with cosine annealing (pct_start=0.03, div_factor=10,
final_div_factor=10000), gradient clipping at 3.0, seed 100, 2 workers.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.dataset import S1NextTokenDataset, build_stamp
from ..evaluation.token_metrics import compute_s1_metrics
from ..modeling.s1_model import S1HeadOnlyWrapper
from ..utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


class S1Trainer:
    """Trainer for head-only S1 next-token post-training."""

    def __init__(
        self,
        model: S1HeadOnlyWrapper,
        config: dict,
    ):
        self.model = model
        self.config = config
        self.device = model.device

        self.batch_size = config["training"]["batch_size"]
        self.epochs = config["training"]["epochs"]
        self.grad_clip = config["training"]["max_grad_norm"]
        self.log_interval = config["training"]["log_interval_steps"]
        self.output_dir = Path(config["outputs"]["root_dir"])
        self.save_best_by = config["outputs"]["save_best_by"]

        # Optimizer (D-017)
        self.optimizer = torch.optim.AdamW(
            model.trainable_parameters(),
            lr=config["training"]["head_learning_rate"],
            betas=(config["training"]["adam_beta1"], config["training"]["adam_beta2"]),
            weight_decay=config["training"]["weight_decay"],
        )

        # Scheduler will be created after we know total_steps
        self.scheduler = None
        self.best_metric = float("inf")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_dataset: S1NextTokenDataset,
        val_dataset: S1NextTokenDataset,
    ) -> None:
        """Run the full training pipeline."""
        pin_memory = self.config["training"]["pin_memory"] and self.device.type == "cuda"

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # dataset handles randomness internally
            num_workers=self.config["training"]["num_workers"],
            pin_memory=pin_memory,
            drop_last=self.config["training"]["train_drop_last"],
            collate_fn=_collate_batch,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config["training"]["num_workers"],
            pin_memory=pin_memory,
            drop_last=self.config["training"]["validation_drop_last"],
            collate_fn=_collate_batch,
        )

        total_steps = self.epochs * len(train_loader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config["training"]["head_learning_rate"],
            total_steps=total_steps,
            pct_start=self.config["training"]["one_cycle_pct_start"],
            div_factor=self.config["training"]["one_cycle_div_factor"],
            final_div_factor=self.config["training"]["one_cycle_final_div_factor"],
            anneal_strategy=self.config["training"]["one_cycle_anneal_strategy"],
        )

        for epoch in range(1, self.epochs + 1):
            train_dataset.set_epoch_seed(epoch)
            train_metrics = self._train_epoch(train_loader, epoch)
            val_metrics = self._validate_epoch(val_loader)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_ce=%.4f val_top1=%.4f",
                epoch, self.epochs,
                train_metrics.get("loss", float("nan")),
                val_metrics.get("s1_cross_entropy", float("nan")),
                val_metrics.get("s1_top1_accuracy", float("nan")),
            )

            self._save_checkpoint(epoch, val_metrics)

    def _train_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(loader):
            x = batch["x"].to(self.device)              # [B, 256, 6]
            target_x = batch["target_x"].to(self.device) # [B, 1, 6]
            batch_size = x.size(0)

            # Tokenize full sequence (context + target) to get the s1 label
            full_x = torch.cat([x, target_x], dim=1)     # [B, 257, 6]
            with torch.no_grad():
                s1_full, _ = self.model.tokenizer.encode(full_x, half=True)
                # s1_full: [B, 257]
            targets = s1_full[:, -1].to(self.device)     # [B]

            # Forward: context → last-position s1 logits
            stamp = torch.from_numpy(
                build_stamp(x.size(1), batch_size)
            ).to(self.device)
            s1_logits_last = self.model.forward(x, stamp)  # [B, vocab_s1]

            # Loss (D-016)
            loss = F.cross_entropy(s1_logits_last, targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.trainable_parameters(), self.grad_clip
            )
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            if step > 0 and step % self.log_interval == 0:
                elapsed = time.time() - start_time
                logger.info(
                    "  Epoch %d step %d/%d | loss=%.4f | lr=%.2e | %.1fs",
                    epoch, step, len(loader), loss.item(),
                    self.scheduler.get_last_lr()[0], elapsed,
                )

        return {"loss": total_loss / max(len(loader), 1)}

    @torch.no_grad()
    def _validate_epoch(self, loader: DataLoader) -> dict:
        self.model.eval()
        all_logits = []
        all_targets = []

        for batch in loader:
            x = batch["x"].to(self.device)
            target_x = batch["target_x"].to(self.device)
            batch_size = x.size(0)

            full_x = torch.cat([x, target_x], dim=1)
            s1_full, _ = self.model.tokenizer.encode(full_x, half=True)
            targets = s1_full[:, -1].to(self.device)

            stamp = torch.from_numpy(
                build_stamp(x.size(1), batch_size)
            ).to(self.device)
            s1_logits_last = self.model.forward(x, stamp)

            all_logits.append(s1_logits_last.cpu())
            all_targets.append(targets.cpu())

        logits = torch.cat(all_logits, dim=0)    # [total_N, vocab_s1]
        targets = torch.cat(all_targets, dim=0)   # [total_N]

        return compute_s1_metrics(logits, targets, self.model.predictor.s1_bits)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, metrics: dict) -> None:
        current = metrics.get(self.save_best_by, float("inf"))
        direction = -1 if self.save_best_by.endswith("cross_entropy") or self.save_best_by.endswith("loss") else 1

        is_best = (current * direction) < (self.best_metric * direction)
        if is_best:
            self.best_metric = current
            ckpt_path = self.output_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.predictor.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "metrics": metrics,
                },
                ckpt_path,
            )
            logger.info("  → Best checkpoint saved (%s=%.4f)", self.save_best_by, current)

        # Also save latest and metrics log
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.predictor.state_dict()},
            self.output_dir / "last_model.pt",
        )
        metrics_path = self.output_dir / "metrics.jsonl"
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"epoch": epoch, **metrics}) + "\n")


# ------------------------------------------------------------------
# Collation
# ------------------------------------------------------------------

def _collate_batch(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Stack a list of dataset dicts into a batched dict."""
    x = torch.stack([item["x"] for item in batch])           # [B, 256, 6]
    target_x = torch.stack([item["target_x"] for item in batch])  # [B, 1, 6]
    return {"x": x, "target_x": target_x}
