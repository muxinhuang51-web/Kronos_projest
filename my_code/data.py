from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


FEATURE_COLS = ["open", "high", "low", "close", "volume", "amount"]
TIME_COLS = ["minute", "hour", "weekday", "day", "month"]
REQUIRED_COLS = ["timestamps", *FEATURE_COLS]


@dataclass
class PredictionWindow:
    df: pd.DataFrame
    x_df: pd.DataFrame
    x_timestamp: pd.Series
    y_timestamp: pd.Series
    x_time_df: pd.DataFrame
    y_time_df: pd.DataFrame
    x: np.ndarray
    x_norm: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray


def load_kline_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    csv_path = Path(csv_path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)

    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df[REQUIRED_COLS].copy()


def calc_time_stamps(timestamp_series: pd.Series) -> pd.DataFrame:
    time_df = pd.DataFrame()
    time_df["minute"] = timestamp_series.dt.minute
    time_df["hour"] = timestamp_series.dt.hour
    time_df["weekday"] = timestamp_series.dt.weekday
    time_df["day"] = timestamp_series.dt.day
    time_df["month"] = timestamp_series.dt.month
    return time_df


def split_prediction_window(
    df: pd.DataFrame,
    lookback: int,
    pred_len: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    required_len = lookback + pred_len
    if len(df) < required_len:
        raise ValueError(
            f"Data length is insufficient: need at least {required_len}, got {len(df)}"
        )

    x_df = df.loc[: lookback - 1, FEATURE_COLS].copy()
    x_timestamp = df.loc[: lookback - 1, "timestamps"].copy()
    y_timestamp = df.loc[lookback : lookback + pred_len - 1, "timestamps"].copy()
    return x_df, x_timestamp, y_timestamp


def normalize_window(
    x_df: pd.DataFrame,
    clip: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = x_df[FEATURE_COLS].values.astype(np.float32)
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_norm = (x - x_mean) / (x_std + 1e-5)
    x_norm = np.clip(x_norm, -clip, clip)
    return x, x_norm, x_mean, x_std


def prepare_prediction_window(
    csv_path: Union[str, Path],
    lookback: int = 400,
    pred_len: int = 120,
    clip: float = 5.0,
) -> PredictionWindow:
    df = load_kline_csv(csv_path)
    x_df, x_timestamp, y_timestamp = split_prediction_window(df, lookback, pred_len)
    x_time_df = calc_time_stamps(x_timestamp)
    y_time_df = calc_time_stamps(y_timestamp)
    x, x_norm, x_mean, x_std = normalize_window(x_df, clip=clip)

    return PredictionWindow(
        df=df,
        x_df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        x_time_df=x_time_df,
        y_time_df=y_time_df,
        x=x,
        x_norm=x_norm,
        x_mean=x_mean,
        x_std=x_std,
    )


def save_processed_window(window: PredictionWindow, output_path: Union[str, Path]) -> Path:
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        x_norm=window.x_norm,
        x_mean=window.x_mean,
        x_std=window.x_std,
        x_time=window.x_time_df.values.astype(np.float32),
        y_time=window.y_time_df.values.astype(np.float32),
    )
    return output_path


def default_csv_path(project_root: Optional[Union[str, Path]] = None) -> Path:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    else:
        project_root = Path(project_root).expanduser()

    return project_root / "finetune_csv" / "data" / "HK_ali_09988_kline_5min_all.csv"


if __name__ == "__main__":
    csv_path = default_csv_path()
    window = prepare_prediction_window(csv_path)
    output_path = save_processed_window(
        window,
        Path(__file__).resolve().parent / "processed_window.npz",
    )

    print(f"CSV: {csv_path}")
    print(f"x_norm shape: {window.x_norm.shape}")
    print(f"x_time shape: {window.x_time_df.shape}")
    print(f"y_time shape: {window.y_time_df.shape}")
    print(f"saved: {output_path}")
