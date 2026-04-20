"""Fetch, validate, clean, and save Berlin AQI data from the OpenAQ v3 API."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

OPENAQ_BASE_URL = "https://api.openaq.org/v3"
POLLUTANTS = ["pm25", "pm10", "o3", "no2", "so2", "co"]
METADATA_COLS = [
    "datetimeLocal",
    "timezone",
    "country_iso",
    "isMobile",
    "isMonitor",
    "owner_name",
    "provider",
    "unit",
]
DROP_COLS = [
    "country_iso",
    "isMobile",
    "isMonitor",
    "owner_name",
    "provider",
    "unit",
    "datetimeLocal",
    "timezone",
]
RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


class IngestError(Exception):
    """Raised when ingestion fails a hard validation check."""


def _client(api_key: str) -> httpx.Client:
    return httpx.Client(
        base_url=OPENAQ_BASE_URL,
        headers={"X-API-Key": api_key},
        timeout=30.0,
    )


def _paginate_hours(
    client: httpx.Client,
    sensor_id: int,
    datetime_from: str,
    datetime_to: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    page = 1
    while True:
        resp = client.get(
            f"/sensors/{sensor_id}/hours",
            params={
                "datetime_from": datetime_from,
                "datetime_to": datetime_to,
                "limit": 1000,
                "page": page,
            },
        )
        resp.raise_for_status()
        body = resp.json()
        batch = body.get("results", [])
        if not batch:
            break
        results.extend(batch)
        if len(batch) < 1000:
            break
        page += 1
    return results


def fetch(
    location_id: int,
    datetime_from: str,
    datetime_to: str,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Pull hourly measurements for all pollutants at a single OpenAQ location."""
    key = api_key or os.getenv("OPENAQ_API_KEY")
    if not key:
        raise IngestError("OPENAQ_API_KEY not set in environment")

    logger.info(
        "Fetching location=%s from=%s to=%s", location_id, datetime_from, datetime_to
    )

    with _client(key) as client:
        loc_resp = client.get(f"/locations/{location_id}")
        loc_resp.raise_for_status()
        location = loc_resp.json()["results"][0]

        metadata = {
            "country_iso": (location.get("country") or {}).get("code"),
            "isMobile": location.get("isMobile"),
            "isMonitor": location.get("isMonitor"),
            "owner_name": (location.get("owner") or {}).get("name"),
            "provider": (location.get("provider") or {}).get("name"),
            "timezone": (location.get("timezone") or {}).get("name")
            if isinstance(location.get("timezone"), dict)
            else location.get("timezone"),
        }

        frames: list[pd.DataFrame] = []
        for sensor in location.get("sensors", []):
            param = (sensor.get("parameter") or {}).get("name")
            if param not in POLLUTANTS:
                continue
            unit = (sensor.get("parameter") or {}).get("units")
            rows = _paginate_hours(
                client, sensor["id"], datetime_from, datetime_to
            )
            if not rows:
                logger.info("No data for parameter=%s sensor=%s", param, sensor["id"])
                continue
            frames.append(
                pd.DataFrame(
                    [
                        {
                            "datetime": (r.get("period") or {})
                            .get("datetimeFrom", {})
                            .get("utc"),
                            "datetimeLocal": (r.get("period") or {})
                            .get("datetimeFrom", {})
                            .get("local"),
                            "parameter": param,
                            "value": r.get("value"),
                            "unit": unit,
                        }
                        for r in rows
                    ]
                )
            )

    if not frames:
        raise IngestError(f"No pollutant data returned for location {location_id}")

    long_df = pd.concat(frames, ignore_index=True)
    long_df["datetime"] = pd.to_datetime(long_df["datetime"], utc=True)

    wide = (
        long_df.pivot_table(
            index="datetime", columns="parameter", values="value", aggfunc="mean"
        )
        .reset_index()
        .rename_axis(columns=None)
    )

    local_map = long_df.groupby("datetime")["datetimeLocal"].first().reset_index()
    wide = wide.merge(local_map, on="datetime", how="left")

    pm25_units = long_df.loc[long_df["parameter"] == "pm25", "unit"].dropna()
    wide["unit"] = pm25_units.iloc[0] if len(pm25_units) else None

    for k, v in metadata.items():
        wide[k] = v

    for p in POLLUTANTS:
        if p not in wide.columns:
            wide[p] = pd.NA

    return wide.sort_values("datetime").reset_index(drop=True)


def validate(df: pd.DataFrame) -> None:
    """Schema, completeness, and range checks. Raises IngestError on hard failure."""
    expected = {"datetime", *POLLUTANTS, *METADATA_COLS}
    missing = expected - set(df.columns)
    if missing:
        raise IngestError(f"Schema check failed — missing columns: {sorted(missing)}")

    if df.empty:
        raise IngestError("Completeness check failed — dataframe is empty")

    pm25_null_frac = df["pm25"].isna().mean()
    if pm25_null_frac > 0.20:
        raise IngestError(
            f"Completeness check failed — PM2.5 is {pm25_null_frac:.1%} null (>20%)"
        )

    pm25 = df["pm25"].dropna()
    if (pm25 < 0).any():
        raise IngestError("Range check failed — negative PM2.5 values present")

    extreme = int((pm25 > 500).sum())
    if extreme:
        logger.warning("Range check: %d PM2.5 values exceed 500 µg/m³", extreme)

    logger.info(
        "Validation passed — rows=%d, PM2.5 null=%.1f%%, extreme=%d",
        len(df),
        pm25_null_frac * 100,
        extreme,
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop metadata columns, drop null-PM2.5 rows, interpolate short gaps."""
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.dropna(subset=["pm25"]).reset_index(drop=True)

    gap_fill = [c for c in POLLUTANTS if c != "pm25" and c in df.columns]
    df[gap_fill] = df[gap_fill].interpolate(
        method="linear", limit=3, limit_direction="both"
    )
    return df


def save(df: pd.DataFrame, out_dir: Path = RAW_DATA_DIR) -> Path:
    """Write the cleaned dataframe to data/raw/ with a UTC-timestamped filename."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"berlin_aqi_{stamp}.csv"
    df.to_csv(path, index=False)
    logger.info("Saved %d rows to %s", len(df), path)
    return path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    load_dotenv()

    location_id = int(os.getenv("BERLIN_LOCATION_ID", "7178"))
    now = datetime.now(timezone.utc)
    iso = "%Y-%m-%dT%H:%M:%SZ"
    datetime_from = (now - timedelta(days=7)).strftime(iso)
    datetime_to = now.strftime(iso)

    raw_df = fetch(location_id, datetime_from, datetime_to)
    validate(raw_df)
    cleaned_df = clean(raw_df)
    save(cleaned_df)
