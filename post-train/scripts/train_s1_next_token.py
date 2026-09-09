"""Command-line entry point for S1 next-token post-training."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml
import torch

# Ensure post_train is importable
_Kronos_ROOT = Path(__file__).resolve().parents[2]
if str(_Kronos_ROOT / "post-train") not in sys.path:
    sys.path.insert(0, str(_Kronos_ROOT / "post-train"))
if str(_Kronos_ROOT) not in sys.path:
    sys.path.insert(0, str(_Kronos_ROOT))

from post_train.data.dataset import S1NextTokenDataset
from post_train.data.splits import SplitConfig, build_split_manifest
from post_train.modeling.s1_model import S1HeadOnlyWrapper
from post_train.training.trainer import S1Trainer
from post_train.utils.reproducibility import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_s1")


def main() -> None:
    parser = argparse.ArgumentParser(description="S1 next-token post-training")
    parser.add_argument(
        "--config",
        type=str,
        default="post-train/configs/s1_next_token_baseline.yaml",
        help="Path to experiment YAML config",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["experiment"]["seed"])

    # --- Data splits ---
    split_manifest_path = Path(config["data"]["manifest_path"].replace(
        "manifests/dataset_manifest.json", "manifests/split_manifest.json"
    ))
    if split_manifest_path.exists():
        logger.info("Loading existing split manifest from %s", split_manifest_path)
        with open(split_manifest_path) as f:
            manifest = json.load(f)
    else:
        logger.info("Building split manifest...")
        split_cfg = SplitConfig(
            train_start=config["data"]["split"]["train"][0],
            train_end=config["data"]["split"]["train"][1],
            val_start=config["data"]["split"]["validation"][0],
            val_end=config["data"]["split"]["validation"][1],
            test_start=config["data"]["split"]["test"][0],
            test_end=config["data"]["split"]["test"][1]
            if len(config["data"]["split"]["test"]) > 1
            else None,
        )

        import pandas as pd

        calendar_df = pd.read_parquet(
            os.path.join(config["data"]["raw_root"], "calendar", "calendar.parquet"),
            columns=["cal_date", "is_open"],
        )

        manifest = build_split_manifest(
            config["data"]["source_path"],
            calendar_df,
            split_cfg,
            str(split_manifest_path),
        )

    # --- Datasets ---
    logger.info("Creating datasets...")
    train_dataset = S1NextTokenDataset(
        processed_dir=config["data"]["source_path"],
        split_manifest=manifest["train"],
        lookback=config["data"]["lookback"],
        n_samples=config["data"]["sampling"]["samples_per_epoch"],
        seed=config["experiment"]["seed"],
    )
    val_dataset = S1NextTokenDataset(
        processed_dir=config["data"]["source_path"],
        split_manifest=manifest["val"],
        lookback=config["data"]["lookback"],
        n_samples=config["data"]["sampling"]["validation_samples"],
        seed=config["data"]["sampling"]["validation_seed"],
    )

    # --- Model ---
    logger.info("Loading model...")
    device = config["training"]["device"]
    # Auto-detect best available device
    if device == "cuda:0" and not torch.cuda.is_available():
        device = "auto"
    model = S1HeadOnlyWrapper(
        tokenizer_path=config["model"]["tokenizer"],
        predictor_path=config["model"]["predictor"],
        device=device,
    )

    # --- Train ---
    logger.info("Starting training...")
    trainer = S1Trainer(model, config)
    trainer.train(train_dataset, val_dataset)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
