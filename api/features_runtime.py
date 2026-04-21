"""Build the runtime feature vector for a station from the latest features CSV.

Phase 4 reads from `data/processed/features_*.csv` — the output of
`src.features`. Phase 5 will swap this for an hourly-refreshed cache
(local JSON first, then S3).

For a given station, we return the features of the most recent row
available; the prediction is for that row's `datetime + 1 hour`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def _latest_features_csv() -> Path:
    files = sorted(PROCESSED_DIR.glob("features_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise RuntimeError(f"No features_*.csv found in {PROCESSED_DIR}")
    return files[-1]


def get_feature_row(
    location_id: int, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.Timestamp, float]:
    """Return (X, target_datetime, pm25_current) for the station's latest hour.

    X is a 1-row DataFrame matching `feature_cols` in order (numeric dtypes
    preserved — critical so XGBoost doesn't reject object-typed input).
    target_datetime is `latest_row.datetime + 1h` (what we're predicting for).
    pm25_current is PM2.5 at the latest row (for the threshold rule).
    """
    path = _latest_features_csv()
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    if "location_id" in df.columns:
        station_df = df[df["location_id"] == location_id]
    else:
        station_df = df

    if station_df.empty:
        raise ValueError(f"Location {location_id} not found in {path.name}")

    # iloc[[-1]] (not [-1]) preserves the DataFrame + column dtypes
    latest_row = station_df.sort_values("datetime").iloc[[-1]].reset_index(drop=True)
    target_datetime = latest_row["datetime"].iloc[0] + pd.Timedelta(hours=1)
    pm25_current = float(latest_row["pm25"].iloc[0])
    X = latest_row[feature_cols]
    return X, target_datetime, pm25_current
