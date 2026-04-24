"""Fetch, validate, clean, and save Berlin AQI data.

Pollutants come from the OpenAQ v3 API. Weather covariates (temperature,
relative humidity) come from the Open-Meteo historical archive API because
Berlin government PM2.5 stations don't report weather.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from dotenv import load_dotenv

RETRIABLE_STATUSES = {408, 429, 500, 502, 503, 504}

logger = logging.getLogger(__name__)

OPENAQ_BASE_URL = "https://api.openaq.org/v3"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# The 17 OpenAQ stations in the Berlin bbox with PM2.5 and >=2 years of
# history as of Phase 1.1 EDA (2026-04-21). Includes decommissioned
# stations (DEBE051/DEBE063) — fetch will skip them if their window is
# entirely outside the requested range.
ELIGIBLE_BERLIN_STATIONS = [
    2993,     # Berlin Neukölln
    3019,     # Berlin Mitte (v1 reference station)
    3025,     # DEBE063 (decommissioned 2023-12-31)
    3050,     # Potsdam, Großbeerenstr (decommissioned 2025-06-03)
    3096,     # Potsdam, Groß Glienicke
    4582,     # Berlin Grunewald
    4724,     # Blankenfelde-Mahlow
    4761,     # Berlin Wedding
    4762,     # Berlin Schildhornstraße
    4764,     # Berlin Mariendorfer Damm
    4767,     # Berlin Frankfurter Allee
    4768,     # Berlin Friedrichshagen
    4769,     # DEBE051 (decommissioned 2024-01-29)
    2162178,  # Berlin Leipziger Straße
    2162179,  # Berlin Buch
    2162180,  # Berlin Karl-Marx-Straße II
    2162181,  # Berlin Silbersteinstraße 5
]
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
        timeout=60.0,
    )


# Pacing between stations in multi-station ingest — keeps us under
# OpenAQ's per-minute ceiling when paginating lots of history.
INTER_STATION_SLEEP_SECONDS = 3


def _get_with_retry(
    client: httpx.Client,
    url: str,
    params: dict | None = None,
    max_retries: int = 7,
    max_backoff_seconds: int = 60,
) -> httpx.Response:
    """GET with exponential backoff on transient errors (408, 429, 5xx, connection).

    Backoff schedule: 1, 2, 4, 8, 16, 32, 60 seconds — caps at max_backoff_seconds.
    7 attempts with the cap is ~2 minutes of total patience per call, which is
    enough to ride out OpenAQ's short bursts of 408/429 during multi-station pulls.
    """
    for attempt in range(max_retries):
        try:
            resp = client.get(url, params=params)
            if resp.status_code not in RETRIABLE_STATUSES:
                resp.raise_for_status()
                return resp
            last_error: Exception = httpx.HTTPStatusError(
                f"{resp.status_code} {resp.reason_phrase}", request=resp.request, response=resp
            )
        except httpx.TransportError as exc:
            last_error = exc
        if attempt == max_retries - 1:
            raise last_error
        backoff = min(2 ** attempt, max_backoff_seconds)
        logger.warning(
            "Transient error on %s (attempt %d/%d): %s — retrying in %ds",
            url, attempt + 1, max_retries, last_error, backoff,
        )
        time.sleep(backoff)
    raise RuntimeError("unreachable")


PAGE_LIMIT = 500
INTER_PAGE_SLEEP_SECONDS = 0.5


def _paginate_hours(
    client: httpx.Client,
    sensor_id: int,
    datetime_from: str,
    datetime_to: str,
) -> list[dict[str, Any]]:
    # Page size is 500 (not the API max of 1000): at 5yr windows, OpenAQ's
    # hourly endpoint 408s on 1000-row pages for the high-traffic
    # Senatsverwaltung sensors. Halving the page size trades 2x request
    # count for reliability. A small inter-page sleep prevents our
    # back-to-back pagination from looking like a burst.
    results: list[dict[str, Any]] = []
    page = 1
    while True:
        if page > 1:
            time.sleep(INTER_PAGE_SLEEP_SECONDS)
        resp = _get_with_retry(
            client,
            f"/sensors/{sensor_id}/hours",
            params={
                "datetime_from": datetime_from,
                "datetime_to": datetime_to,
                "limit": PAGE_LIMIT,
                "page": page,
            },
        )
        body = resp.json()
        batch = body.get("results", [])
        if not batch:
            break
        results.extend(batch)
        if len(batch) < PAGE_LIMIT:
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
        loc_resp = _get_with_retry(client, f"/locations/{location_id}")
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
    negatives = int((pm25 < 0).sum())
    if negatives:
        logger.warning(
            "Range check: %d negative PM2.5 values (sensor artifact — clamped to 0 in clean)",
            negatives,
        )

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
    """Drop metadata columns, drop null-PM2.5 rows, clamp negatives, interpolate short gaps."""
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.dropna(subset=["pm25"]).reset_index(drop=True)

    # Negative PM2.5 is physically impossible — sensor baseline drift artifact. Clamp to 0.
    df.loc[df["pm25"] < 0, "pm25"] = 0

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


def fetch_coordinates(
    location_id: int, api_key: str | None = None
) -> tuple[float, float]:
    """Return (latitude, longitude) for an OpenAQ location."""
    key = api_key or os.getenv("OPENAQ_API_KEY")
    if not key:
        raise IngestError("OPENAQ_API_KEY not set in environment")
    with _client(key) as client:
        resp = _get_with_retry(client, f"/locations/{location_id}")
        coords = resp.json()["results"][0].get("coordinates") or {}
    if "latitude" not in coords or "longitude" not in coords:
        raise IngestError(f"Location {location_id} has no coordinates")
    return float(coords["latitude"]), float(coords["longitude"])


def fetch_weather(
    latitude: float,
    longitude: float,
    datetime_from: str,
    datetime_to: str,
) -> pd.DataFrame:
    """Fetch hourly temperature + relative humidity from Open-Meteo archive.

    Returns a DataFrame with columns: datetime (UTC), temperature, relative_humidity.
    """
    start_date = pd.Timestamp(datetime_from).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(datetime_to).strftime("%Y-%m-%d")

    logger.info(
        "Fetching weather lat=%s lon=%s %s..%s",
        latitude,
        longitude,
        start_date,
        end_date,
    )

    with httpx.Client(timeout=60.0) as client:
        resp = client.get(
            OPEN_METEO_ARCHIVE_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": "temperature_2m,relative_humidity_2m",
                "timezone": "UTC",
            },
        )
        resp.raise_for_status()
        body = resp.json()

    hourly = body.get("hourly") or {}
    if not hourly.get("time"):
        raise IngestError("Open-Meteo returned no hourly data")

    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(hourly["time"], utc=True),
            "temperature": hourly["temperature_2m"],
            "relative_humidity": hourly["relative_humidity_2m"],
        }
    )
    return df.sort_values("datetime").reset_index(drop=True)


def fetch_all_stations(
    location_ids: list[int],
    datetime_from: str,
    datetime_to: str,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Ingest pollutants + weather for multiple stations, stack into one DataFrame.

    Each station's rows get a `location_id` column. Stations that fail
    (no data in the window, sensor drift, API error after retries) are
    logged and skipped so one bad station never aborts the whole run.
    """
    frames: list[pd.DataFrame] = []
    skipped: list[tuple[int, str]] = []

    for i, lid in enumerate(location_ids, start=1):
        if i > 1:
            time.sleep(INTER_STATION_SLEEP_SECONDS)
        logger.info("=== Station %d (%d/%d) ===", lid, i, len(location_ids))
        try:
            raw = fetch(lid, datetime_from, datetime_to, api_key=api_key)
            validate(raw)
            cleaned = clean(raw)
            lat, lon = fetch_coordinates(lid, api_key=api_key)
            weather = fetch_weather(lat, lon, datetime_from, datetime_to)
            combined = cleaned.merge(weather, on="datetime", how="left")
            combined["location_id"] = lid
            frames.append(combined)
            logger.info("Station %d: %d rows", lid, len(combined))
        except (IngestError, httpx.HTTPError, ValueError) as exc:
            logger.error("Station %d failed — skipping: %s", lid, exc)
            skipped.append((lid, str(exc)))

    if not frames:
        raise IngestError("No stations successfully ingested")

    result = pd.concat(frames, ignore_index=True)
    logger.info(
        "Multi-station ingest done: %d rows across %d stations (%d skipped)",
        len(result), result["location_id"].nunique(), len(skipped),
    )
    if skipped:
        for lid, err in skipped:
            logger.info("  skipped %d: %s", lid, err)
    return result


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    load_dotenv()

    parser = argparse.ArgumentParser(description="Ingest Berlin AQI data from OpenAQ + Open-Meteo.")
    parser.add_argument("--multi", action="store_true", help="Ingest all 17 eligible Berlin stations")
    parser.add_argument("--location-id", type=int, default=None, help="Single station override")
    parser.add_argument("--days", type=int, default=7, help="Days of history to pull")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    iso = "%Y-%m-%dT%H:%M:%SZ"
    datetime_from = (now - timedelta(days=args.days)).strftime(iso)
    datetime_to = now.strftime(iso)

    if args.multi:
        combined_df = fetch_all_stations(
            ELIGIBLE_BERLIN_STATIONS, datetime_from, datetime_to
        )
    else:
        location_id = args.location_id or int(os.getenv("BERLIN_LOCATION_ID", "3019"))
        raw_df = fetch(location_id, datetime_from, datetime_to)
        validate(raw_df)
        cleaned_df = clean(raw_df)

        lat, lon = fetch_coordinates(location_id)
        weather_df = fetch_weather(lat, lon, datetime_from, datetime_to)
        combined_df = cleaned_df.merge(weather_df, on="datetime", how="left")

    save(combined_df)
