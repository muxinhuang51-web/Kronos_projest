"""Corporate-action adjustment for daily OHLC data.

Accepted contract (D-009):
- adjusted OHLC = raw OHLC × same-day adjustment factor;
- volume and amount retain their raw provider values;
- no future-anchored adjustment factor is used;
- adjustment factor gaps are forward-filled (adj factors only change on
  ex-dividend dates);
- adjustment inputs and formulas are recorded in the dataset manifest.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PRICE_COLS = ["open", "high", "low", "close"]
VOL_AMT_COLS = ["vol", "amount"]


def apply_backward_adjustment(
    daily_df: pd.DataFrame,
    adj_df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply backward (post-) adjustment to OHLC prices.

    Args:
        daily_df: Daily bars with columns ``[trade_date, open, high, low, close, vol, amount]``.
        adj_df: Adjustment factors with columns ``[trade_date, adj_factor]``.

    Returns:
        DataFrame with the same columns as *daily_df*, with OHLC adjusted
        and volume/amount unchanged.
    """
    merged = daily_df.merge(adj_df, on="trade_date", how="left")
    merged["adj_factor"] = merged["adj_factor"].ffill()

    for col in PRICE_COLS:
        if col in merged.columns:
            merged[col] = merged[col] * merged["adj_factor"]

    # Restore original NaN in columns where adj_factor was never available
    still_missing = merged["adj_factor"].isna()
    for col in PRICE_COLS:
        merged.loc[still_missing, col] = daily_df.loc[still_missing, col]

    return merged.drop(columns=["adj_factor"])


def process_all_stocks(raw_root: str, output_dir: str) -> None:
    """Apply backward adjustment to every stock and write results.

    Args:
        raw_root: Path containing ``daily/`` and ``adj_factors/`` subdirectories.
        output_dir: Directory to write adjusted parquet files.
    """
    raw_root = Path(raw_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_dir = raw_root / "daily"
    adj_dir = raw_root / "adj_factors"

    daily_files = sorted(daily_dir.glob("*.parquet"))
    logger.info("Processing %d stocks for adjustment...", len(daily_files))

    for fpath in daily_files:
        ts_code = fpath.stem
        output_path = output_dir / f"{ts_code}.parquet"

        if output_path.exists():
            continue

        daily_df = pd.read_parquet(fpath)
        adj_path = adj_dir / f"{ts_code}.parquet"

        if not adj_path.exists():
            logger.debug("No adj factor file for %s, copying raw data.", ts_code)
            daily_df.to_parquet(output_path, index=False)
            continue

        adj_df = pd.read_parquet(adj_path)
        adjusted = apply_backward_adjustment(daily_df, adj_df)
        adjusted.to_parquet(output_path, index=False)

    logger.info("Adjustment complete. Output: %s", output_dir)
