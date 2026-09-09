"""Dataset for grouped, leakage-free S1 next-token samples.

Accepted sampling contract (D-015): every valid ``(ts_code, target_date)`` sample
has equal probability.  Each epoch makes 100,000 independent uniform training draws
with replacement, while validation replays a deterministic 20,000-draw sequence
generated from seed 100.

The dataset does not rebalance by asset, calendar date, target token, market
regime, or current model difficulty.

Normalization (D-006): OHLCVA six dimensions use context-only mean/std (ddof=0,
eps=1e-5), clipped to [-5, 5].  The target row is normalized with the same
statistics.

Suspension handling (D-010): no forward fill.  Context consists of the 256 most
recent actual trading days (not calendar days) for the given stock.  The target
must be the immediate next trading day after the context.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

FEATURE_COLS = ["open", "high", "low", "close", "vol", "amount"]


class S1NextTokenDataset(Dataset):
    """Dataset yielding (context_x, target_x) pairs for S1 next-token training.

    Each sample:
      - context_x: [256, 6] normalized OHLCVA of 256 trading days.
      - target_x:  [1, 6]   normalized OHLCVA of the next trading day.

    Tokenization is deferred to the training loop so that encoding can be
    batched across the full 257-row sequence.
    """

    def __init__(
        self,
        processed_dir: str,
        split_manifest: list[list[str]],  # [[ts_code, trade_date], ...]
        lookback: int = 256,
        n_samples: int = 100000,
        seed: int | None = None,
    ):
        """
        Args:
            processed_dir: Directory containing ``{ts_code}.parquet`` files.
            split_manifest: List of ``[ts_code, trade_date]`` pairs belonging
                to this split.
            lookback: Number of context trading days (256).
            n_samples: ``__len__`` return value; used to control epoch length.
            seed: Random seed for sampling.  If None, an unpredictable seed
                is used each epoch.
        """
        self.processed_dir = Path(processed_dir)
        self.lookback = lookback
        self.n_samples = n_samples
        self.seed = seed

        # --- Build valid-sample index ---
        # Pre-compute for each (ts_code, target_date) the row offset of
        # target_date in the parquet file.  Only keep samples with enough
        # preceding rows (≥ lookback).
        self._rng = random.Random(seed)
        self.valid_samples: list[tuple[str, int]] = []  # (ts_code, target_row_offset)
        self._build_index(split_manifest)
        logger.info("Dataset ready: %d valid samples, n_samples=%d", len(self.valid_samples), n_samples)

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self, split_manifest: list[list[str]]) -> None:
        """Scan parquet files and record valid ``(ts_code, row_offset)`` pairs."""
        # Group by ts_code for efficient single-file scanning
        by_code: dict[str, list[str]] = {}
        for ts_code, trade_date in split_manifest:
            by_code.setdefault(ts_code, []).append(trade_date)

        for ts_code, target_dates in by_code.items():
            fpath = self.processed_dir / f"{ts_code}.parquet"
            if not fpath.exists():
                logger.debug("File not found: %s", fpath)
                continue

            df = pd.read_parquet(fpath, columns=["trade_date"])
            df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "").str[:8]
            date_to_offset: dict[str, int] = {
                td: i for i, td in enumerate(df["trade_date"])
            }

            for target_date in target_dates:
                offset = date_to_offset.get(target_date)
                if offset is None:
                    continue
                if offset >= self.lookback:  # need ≥ lookback rows before target
                    self.valid_samples.append((ts_code, offset))

    # ------------------------------------------------------------------
    # Core sampling
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a single sample.  *idx* is ignored (uniform random draw)."""
        if not self.valid_samples:
            raise RuntimeError("No valid samples in dataset.")

        ts_code, target_offset = self._rng.choice(self.valid_samples)

        # Read exactly the rows we need: [context_start, target]
        start_offset = target_offset - self.lookback
        fpath = self.processed_dir / f"{ts_code}.parquet"
        df = pd.read_parquet(
            fpath,
            columns=FEATURE_COLS,
        ).iloc[start_offset : target_offset + 1]
        data = df[FEATURE_COLS].values.astype(np.float32)  # [257, 6]

        # --- Normalize with context-only statistics ---
        context = data[: self.lookback]   # [256, 6]
        mean = context.mean(axis=0)       # [6]
        std = context.std(axis=0, ddof=0) # [6]  (D-006: ddof=0)
        std = np.maximum(std, 1e-5)       # eps

        data = (data - mean) / std
        data = np.clip(data, -5.0, 5.0)

        context_x = torch.from_numpy(data[: self.lookback]).float()  # [256, 6]
        target_x = torch.from_numpy(data[self.lookback :]).float()   # [1, 6]

        return {"x": context_x, "target_x": target_x}

    # ------------------------------------------------------------------
    # Epoch management
    # ------------------------------------------------------------------

    def set_epoch_seed(self, epoch: int) -> None:
        """Reseed the internal RNG for reproducibility across epochs.

        When *seed* is not None, the per-epoch seed is ``seed + epoch``.
        Validation datasets should share a fixed seed (e.g. 100) so the
        sample sequence is deterministic.
        """
        if self.seed is not None:
            self._rng = random.Random(self.seed + epoch)


def build_stamp(context_length: int, batch_size: int) -> np.ndarray:
    """Build dummy time-stamp features for daily data.

    For daily bars the original Kronos temporal embedding expects
    ``[minute, hour, weekday, day, month]``.  Since minute/hour are
    constant for daily data, we provide zeros and rely on the learned
    daily-level periodicity in the other three dimensions.

    Args:
        context_length: Number of time steps (256).
        batch_size: Batch dimension.

    Returns:
        ndarray of shape ``[batch_size, context_length, 5]``.
    """
    # Use a simple monotonic index; the model's TemporalEmbedding will
    # learn any remaining structure from the positional RoPE encoding.
    stamps = np.zeros((batch_size, context_length, 5), dtype=np.float32)
    stamps[:, :, 2] = 1  # placeholder weekday (non-zero to activate embedding)
    stamps[:, :, 3] = np.arange(context_length) % 31 + 1  # day
    stamps[:, :, 4] = (np.arange(context_length) // 20) % 12 + 1  # month proxy
    return stamps
