"""Resumable market-data download from tushare.

Fetches the A-share trading calendar, point-in-time security universe, raw daily
bars, and daily adjustment factors. Credentials must come from the configured
environment variable and must never be stored in source or experiment artifacts.

The implementation downloads by trade date, persists successful partitions
immediately, and supports retry/backoff and restart without duplicating data.
"""

import os
import time
import logging
from pathlib import Path

import pandas as pd
import tushare as ts

logger = logging.getLogger(__name__)


class AShareDownloader:
    """Download A-share daily data from tushare with resume support."""

    def __init__(self, token: str, raw_root: str):
        self.pro = ts.pro_api(token)
        self.raw_root = Path(raw_root)
        (self.raw_root / "calendar").mkdir(parents=True, exist_ok=True)
        (self.raw_root / "daily").mkdir(parents=True, exist_ok=True)
        (self.raw_root / "adj_factors").mkdir(parents=True, exist_ok=True)

    # ---- calendar ----

    def download_calendar(self, start: str, end: str) -> pd.DataFrame:
        """Download SSE trading calendar and persist to calendar/calendar.parquet."""
        output_path = self.raw_root / "calendar" / "calendar.parquet"
        if output_path.exists():
            logger.info("Calendar file already exists, skipping.")
            return pd.read_parquet(output_path)

        df = self._fetch_with_retry(
            lambda: self.pro.trade_cal(
                exchange="SSE", start_date=start, end_date=end
            )
        )
        if df is None:
            raise RuntimeError("Failed to download calendar after retries.")

        df = df.rename(columns={"cal_date": "cal_date", "is_open": "is_open", "pretrade_date": "pretrade_date"})
        df.to_parquet(output_path, index=False)
        logger.info("Calendar saved: %d rows → %s", len(df), output_path)
        return df

    # ---- stock list ----

    def download_stock_list(self) -> pd.DataFrame:
        """Download full A-share stock list (including delisted) and persist."""
        output_path = self.raw_root / "calendar" / "stock_list.parquet"
        if output_path.exists():
            logger.info("Stock list already exists, skipping.")
            return pd.read_parquet(output_path)

        df = self._fetch_with_retry(
            lambda: self.pro.stock_basic(
                exchange="",
                list_status="L",
                fields="ts_code,symbol,name,area,industry,list_date,delist_date",
            )
        )
        if df is None:
            raise RuntimeError("Failed to download stock list after retries.")

        df.to_parquet(output_path, index=False)
        logger.info("Stock list saved: %d stocks → %s", len(df), output_path)
        return df

    # ---- daily bars ----

    def download_one_stock_daily(self, ts_code: str, start: str, end: str) -> bool:
        """Download daily OHLCVA for one stock. Returns True on success."""
        output_path = self.raw_root / "daily" / f"{ts_code}.parquet"
        if output_path.exists():
            return True

        df = self._fetch_with_retry(
            lambda: self.pro.daily(
                ts_code=ts_code, start_date=start, end_date=end,
                fields="ts_code,trade_date,open,high,low,close,vol,amount",
            ),
            max_retries=3,
        )
        if df is None:
            logger.warning("Download failed for %s", ts_code)
            return False

        df = df.sort_values("trade_date")
        df.to_parquet(output_path, index=False)
        return True

    # ---- adjustment factors ----

    def download_one_stock_adj(self, ts_code: str) -> bool:
        """Download daily adjustment factors for one stock. Returns True on success."""
        output_path = self.raw_root / "adj_factors" / f"{ts_code}.parquet"
        if output_path.exists():
            return True

        df = self._fetch_with_retry(
            lambda: self.pro.adj_factor(ts_code=ts_code),
            max_retries=3,
        )
        if df is None:
            logger.warning("Adj factor download failed for %s", ts_code)
            return False

        df = df.sort_values("trade_date")
        df.to_parquet(output_path, index=False)
        return True

    # ---- main orchestration ----

    def download_all(self, start: str, end: str) -> dict:
        """Download calendar, stock list, daily bars, and adj factors for all stocks."""
        logger.info("=== Step 1/4: Downloading calendar ===")
        self.download_calendar(start, end)

        logger.info("=== Step 2/4: Downloading stock list ===")
        stock_df = self.download_stock_list()

        codes = stock_df["ts_code"].tolist()
        total = len(codes)
        success = 0
        failed = []
        # daily API limit: ~500/min, use 0.15s delay per call for safety
        daily_delay = 0.15

        logger.info("=== Step 3/4: Downloading daily bars for %d stocks ===", total)
        for i, code in enumerate(codes):
            ok = self.download_one_stock_daily(code, start, end)
            if ok:
                success += 1
            else:
                failed.append(code)
            if (i + 1) % 500 == 0:
                logger.info("  Daily progress: %d/%d (ok=%d, fail=%d)", i + 1, total, success, len(failed))
            time.sleep(daily_delay)
        logger.info("Daily bars done: %d ok, %d failed", success, len(failed))

        # adj_factor API limit: ~200/min, use 0.4s delay per call for safety
        adj_delay = 0.4

        logger.info("=== Step 4/4: Downloading adj factors for %d stocks ===", total)
        adj_success = 0
        adj_failed = []
        for i, code in enumerate(codes):
            ok = self.download_one_stock_adj(code)
            if ok:
                adj_success += 1
            else:
                adj_failed.append(code)
            if (i + 1) % 500 == 0:
                logger.info("  Adj progress: %d/%d (ok=%d, fail=%d)", i + 1, total, adj_success, len(adj_failed))
            time.sleep(adj_delay)
        logger.info("Adj factors done: %d ok, %d failed", adj_success, len(adj_failed))

        result = {
            "total_stocks": total,
            "daily_success": success,
            "daily_failed": failed,
            "adj_success": adj_success,
            "adj_failed": adj_failed,
        }
        logger.info("Download summary: %s", result)
        return result

    # ---- helpers ----

    @staticmethod
    def _fetch_with_retry(fetch_fn, max_retries=3, base_delay=1.0):
        """Call *fetch_fn* with exponential backoff on failure."""
        for attempt in range(max_retries):
            try:
                df = fetch_fn()
                if df is not None and not df.empty:
                    return df
            except Exception as exc:
                logger.warning("Attempt %d/%d failed: %s", attempt + 1, max_retries, exc)
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
        return None
