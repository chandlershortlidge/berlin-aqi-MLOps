"""Reader for per-station PM2.5 history.

Source is the actuals log that `src.refresh` appends to on every hourly
run. Each row there is {refreshed_at, location_id, observed_datetime,
actual_category, pm25_actual} — so after N refreshes we have up to N
hours of history per station.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ACTUALS_PATH = Path(__file__).resolve().parents[1] / "data" / "monitoring" / "actuals.csv"


def get_recent(location_id: int, hours: int = 24) -> list[dict]:
    """Return a chronologically-sorted list of {datetime, pm25} for the last N hours."""
    if not ACTUALS_PATH.exists():
        return []
    try:
        df = pd.read_csv(ACTUALS_PATH, parse_dates=["observed_datetime"])
    except pd.errors.EmptyDataError:
        return []

    df = df[df["location_id"] == location_id]
    if df.empty:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    df = df[df["observed_datetime"] >= cutoff]
    # Same observed_datetime can appear multiple times (re-runs) — keep latest refresh's value.
    df = (
        df.sort_values(["observed_datetime", "refreshed_at"])
          .drop_duplicates(subset="observed_datetime", keep="last")
          .sort_values("observed_datetime")
    )
    return [
        {"datetime": ts.isoformat(), "pm25": float(v)}
        for ts, v in zip(df["observed_datetime"], df["pm25_actual"])
    ]
