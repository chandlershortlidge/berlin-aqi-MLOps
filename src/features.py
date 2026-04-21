"""Feature engineering + target variable for Berlin AQI prediction.

Reads a cleaned raw CSV (output of `src.ingest`), builds lag / rolling / time
features and the t+1 AQI category target, time-splits into train / test, and
writes the results to `data/processed/`.

The module is multi-station aware: if the input has a `location_id` column,
all shifts and rolling ops are groupby'd on it so one station's timeline never
leaks into another's. Single-station data (no `location_id`) works the same.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# 5-class target — see memory/custom_aqi_thresholds.md (High + Very High merged)
AQI_BINS = [0, 12, 35.4, 55.4, 150.4, float("inf")]
AQI_LABELS = ["All Clear", "Low Risk", "Elevated", "Significant", "High+"]

PM25_LAGS = list(range(1, 25))
ROLLING_WINDOWS = [6, 12, 24]


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add aqi_category_next — the AQI category for the NEXT hour (t+1 target)."""
    df = df.copy()
    df["_current_cat"] = pd.cut(
        df["pm25"], bins=AQI_BINS, labels=AQI_LABELS, include_lowest=True
    )
    if "location_id" in df.columns:
        df["aqi_category_next"] = df.groupby("location_id")["_current_cat"].shift(-1)
    else:
        df["aqi_category_next"] = df["_current_cat"].shift(-1)
    return df.drop(columns="_current_cat")


def add_lag_features(df: pd.DataFrame, col: str = "pm25") -> pd.DataFrame:
    """Add {col}_lag_1 … {col}_lag_24 — historical values at each past hour."""
    df = df.copy()
    has_loc = "location_id" in df.columns
    for lag in PM25_LAGS:
        name = f"{col}_lag_{lag}"
        if has_loc:
            df[name] = df.groupby("location_id")[col].shift(lag)
        else:
            df[name] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, col: str = "pm25") -> pd.DataFrame:
    """Add {col}_roll{w}_mean and _std for w in 6/12/24 (min_periods = w)."""
    df = df.copy()
    has_loc = "location_id" in df.columns
    for w in ROLLING_WINDOWS:
        mean_name = f"{col}_roll{w}_mean"
        std_name = f"{col}_roll{w}_std"
        if has_loc:
            grp = df.groupby("location_id")[col]
            df[mean_name] = grp.transform(
                lambda s, w=w: s.rolling(w, min_periods=w).mean()
            )
            df[std_name] = grp.transform(
                lambda s, w=w: s.rolling(w, min_periods=w).std()
            )
        else:
            df[mean_name] = df[col].rolling(w, min_periods=w).mean()
            df[std_name] = df[col].rolling(w, min_periods=w).std()
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos hour-of-day, sin/cos day-of-week, and is_weekend flag."""
    df = df.copy()
    dt = pd.to_datetime(df["datetime"], utc=True)
    hour = dt.dt.hour
    dow = dt.dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = dow.isin([5, 6]).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply target + lag + rolling + time features, drop rows with NaN target/features."""
    sort_cols = [c for c in ["location_id", "datetime"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    df = add_target(df)
    df = add_lag_features(df, "pm25")
    df = add_rolling_features(df, "pm25")
    df = add_time_features(df)

    required = (
        ["aqi_category_next"]
        + [f"pm25_lag_{lag}" for lag in PM25_LAGS]
        + [f"pm25_roll{w}_mean" for w in ROLLING_WINDOWS]
        + [f"pm25_roll{w}_std" for w in ROLLING_WINDOWS]
    )
    before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    logger.info("Dropped %d rows with NaN target/lag/rolling; %d remain", before - len(df), len(df))
    return df


def build_inference_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag + rolling + time features with NO target shift.

    Used at serving time: the most recent hour has no t+1 ground truth,
    so we can't drop it via the target-NaN filter the way build_features
    does. Only drops rows still missing lag / rolling features (warmup).
    """
    sort_cols = [c for c in ["location_id", "datetime"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    df = add_lag_features(df, "pm25")
    df = add_rolling_features(df, "pm25")
    df = add_time_features(df)

    required = (
        [f"pm25_lag_{lag}" for lag in PM25_LAGS]
        + [f"pm25_roll{w}_mean" for w in ROLLING_WINDOWS]
        + [f"pm25_roll{w}_std" for w in ROLLING_WINDOWS]
    )
    before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    logger.info("Dropped %d inference rows with NaN lag/rolling; %d remain", before - len(df), len(df))
    return df


def time_train_test_split(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    time_col: str = "datetime",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-station time-based split — the last test_frac of each timeline → test."""
    sort_cols = [c for c in ["location_id", time_col] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    if "location_id" in df.columns:
        rank = df.groupby("location_id").cumcount()
        total = df.groupby("location_id")["location_id"].transform("size")
        is_train = rank < (total * (1 - test_frac)).astype(int)
        train = df[is_train]
        test = df[~is_train]
    else:
        cutoff = int(len(df) * (1 - test_frac))
        train = df.iloc[:cutoff]
        test = df.iloc[cutoff:]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _latest_raw_csv(raw_dir: Path = RAW_DATA_DIR) -> Path:
    files = sorted(raw_dir.glob("berlin_aqi_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No raw CSV found in {raw_dir}")
    return files[-1]


def save(
    features_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, Path]:
    """Write features/train/test CSVs to data/processed/ with a shared UTC timestamp."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outputs = {
        "features": (out_dir / f"features_{stamp}.csv", features_df),
        "train": (out_dir / f"train_{stamp}.csv", train_df),
        "test": (out_dir / f"test_{stamp}.csv", test_df),
    }
    for name, (path, frame) in outputs.items():
        frame.to_csv(path, index=False)
        logger.info("Saved %s — %d rows to %s", name, len(frame), path)
    return {name: path for name, (path, _) in outputs.items()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    raw_path = _latest_raw_csv()
    logger.info("Reading %s", raw_path)
    raw = pd.read_csv(raw_path)

    features = build_features(raw)
    train, test = time_train_test_split(features)

    def _counts(df: pd.DataFrame) -> pd.Series:
        return (
            df["aqi_category_next"]
            .value_counts()
            .reindex(AQI_LABELS)
            .fillna(0)
            .astype(int)
        )

    print("\nClass balance — train:")
    print(_counts(train).to_string())
    print("\nClass balance — test:")
    print(_counts(test).to_string())

    save(features, train, test)
