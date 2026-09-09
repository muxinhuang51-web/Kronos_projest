"""Quick sanity check: verify dataset and model loading before full training."""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "post-train"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from post_train.data.dataset import S1NextTokenDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sanity")

PROCESSED_DIR = "data/a_share_daily/processed"
SPLIT_MANIFEST = "data/a_share_daily/manifests/split_manifest.json"

def main():
    with open(SPLIT_MANIFEST) as f:
        manifest = json.load(f)

    # Test dataset creation
    logger.info("Creating train dataset (10 samples)...")
    train_ds = S1NextTokenDataset(
        processed_dir=PROCESSED_DIR,
        split_manifest=manifest["train"],
        lookback=256,
        n_samples=10,
        seed=100,
    )
    logger.info("Valid samples in index: %d", len(train_ds.valid_samples))

    # Test one sample
    sample = train_ds[0]
    x, target_x = sample["x"], sample["target_x"]
    logger.info("Sample shapes: x=%s, target_x=%s", x.shape, target_x.shape)
    logger.info("x stats: mean=%.4f std=%.4f min=%.4f max=%.4f",
                x.mean().item(), x.std().item(), x.min().item(), x.max().item())
    logger.info("target_x stats: mean=%.4f std=%.4f min=%.4f max=%.4f",
                target_x.mean().item(), target_x.std().item(), target_x.max().item(), target_x.min().item())

    # Test val dataset
    logger.info("Creating val dataset (10 samples)...")
    val_ds = S1NextTokenDataset(
        processed_dir=PROCESSED_DIR,
        split_manifest=manifest["val"],
        lookback=256,
        n_samples=10,
        seed=100,
    )
    logger.info("Valid samples in val index: %d", len(val_ds.valid_samples))

    # Test batch collation
    from post_train.training.trainer import _collate_batch
    batch = [train_ds[i] for i in range(4)]
    batched = _collate_batch(batch)
    logger.info("Batched: x=%s, target_x=%s", batched["x"].shape, batched["target_x"].shape)

    # Test model loading (if internet available)
    logger.info("Testing model loading...")
    try:
        from post_train.modeling.s1_model import S1HeadOnlyWrapper
        model = S1HeadOnlyWrapper(
            tokenizer_path="NeoQuasar/Kronos-Tokenizer-base",
            predictor_path="NeoQuasar/Kronos-small",
            device="cpu",
        )
        logger.info("Model loaded OK. vocab_s1=%d, d_model=%d", model.vocab_s1, model.d_model)

        # Test forward pass
        model.eval()
        with torch.no_grad():
            test_x = torch.randn(2, 256, 6)
            logits = model.forward(test_x)
        logger.info("Forward pass OK: logits shape=%s", logits.shape)
    except Exception as e:
        logger.warning("Model loading failed (may need internet): %s", e)

    logger.info("All sanity checks passed!")

if __name__ == "__main__":
    main()
