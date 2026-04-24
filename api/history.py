"""Reader for per-station pollutant history.

For PM2.5 we still prefer the local actuals log (`src.refresh` appends to
it hourly, so it's cheap and always available). For PM10 and NO₂ — and
as a fallback for PM2.5 when the actuals log is thin — we hit OpenAQ v3
directly: `/locations/{id}` to find the sensor for the requested
parameter, then `/sensors/{sensor_id}/hours` paginated over the window.

Responses from OpenAQ are cached in-process for 5 minutes so repeated
Streamlit reruns don't hammer the upstream API.
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

logger = logging.getLogger(__name__)

ACTUALS_PATH = Path(__file__).resolve().parents[1] / "data" / "monitoring" / "actuals.csv"

OPENAQ_BASE_URL = "https://api.openaq.org/v3"
OPENAQ_PAGE_LIMIT = 500
OPENAQ_TIMEOUT = 30.0

# Maps the user-facing label to OpenAQ's parameter name.
PARAMETER_ALIASES = {
    "pm25": "pm25",
    "pm2.5": "pm25",
    "pm10": "pm10",
    "no2": "no2",
    "no₂": "no2",
}

# (location_id, parameter, hours) -> (fetched_at_epoch, points)
_history_cache: dict[tuple[int, str, int], tuple[float, list[dict]]] = {}
_HISTORY_CACHE_TTL_SECONDS = 300

# location_id -> (fetched_at_epoch, {parameter -> sensor_id})
_sensor_cache: dict[int, tuple[float, dict[str, int]]] = {}
_SENSOR_CACHE_TTL_SECONDS = 3600


def _normalize_parameter(parameter: str) -> str:
    key = parameter.strip().lower()
    if key not in PARAMETER_ALIASES:
        raise ValueError(
            f"Unsupported parameter '{parameter}'. Expected one of: pm25, pm10, no2."
        )
    return PARAMETER_ALIASES[key]


def _from_actuals_log(location_id: int, hours: int) -> list[dict]:
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
    df = (
        df.sort_values(["observed_datetime", "refreshed_at"])
          .drop_duplicates(subset="observed_datetime", keep="last")
          .sort_values("observed_datetime")
    )
    return [
        {"datetime": ts.isoformat(), "value": float(v)}
        for ts, v in zip(df["observed_datetime"], df["pm25_actual"])
    ]


def _lookup_sensors(client: httpx.Client, location_id: int) -> dict[str, int]:
    now = time.time()
    cached = _sensor_cache.get(location_id)
    if cached and now - cached[0] < _SENSOR_CACHE_TTL_SECONDS:
        return cached[1]

    resp = client.get(f"/locations/{location_id}")
    resp.raise_for_status()
    body = resp.json()
    results = body.get("results") or []
    if not results:
        raise LookupError(f"OpenAQ returned no location for id={location_id}")

    sensors: dict[str, int] = {}
    for sensor in results[0].get("sensors", []):
        param = ((sensor.get("parameter") or {}).get("name") or "").lower()
        if param and "id" in sensor:
            sensors[param] = int(sensor["id"])

    _sensor_cache[location_id] = (now, sensors)
    return sensors


def _paginate_sensor_hours(
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
                "limit": OPENAQ_PAGE_LIMIT,
                "page": page,
            },
        )
        resp.raise_for_status()
        batch = resp.json().get("results", [])
        if not batch:
            break
        results.extend(batch)
        if len(batch) < OPENAQ_PAGE_LIMIT:
            break
        page += 1
        if page > 10:
            # 10 * 500 = 5000 hours > 200 days. The UI asks for 30 days
            # max; if we're past this, something is wrong upstream.
            logger.warning("OpenAQ pagination exceeded 10 pages for sensor=%s", sensor_id)
            break
    return results


def _from_openaq(location_id: int, parameter: str, hours: int) -> list[dict]:
    api_key = os.getenv("OPENAQ_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAQ_API_KEY not set — cannot fetch history from OpenAQ.")

    now = datetime.now(timezone.utc).replace(microsecond=0, second=0, minute=0)
    start = now - timedelta(hours=hours)
    datetime_from = start.isoformat().replace("+00:00", "Z")
    datetime_to = now.isoformat().replace("+00:00", "Z")

    with httpx.Client(
        base_url=OPENAQ_BASE_URL,
        headers={"X-API-Key": api_key},
        timeout=OPENAQ_TIMEOUT,
    ) as client:
        sensors = _lookup_sensors(client, location_id)
        sensor_id = sensors.get(parameter)
        if sensor_id is None:
            return []

        rows = _paginate_sensor_hours(client, sensor_id, datetime_from, datetime_to)

    points: list[dict] = []
    for r in rows:
        ts = ((r.get("period") or {}).get("datetimeFrom") or {}).get("utc")
        value = r.get("value")
        if ts is None or value is None:
            continue
        points.append({"datetime": ts, "value": float(value)})
    points.sort(key=lambda p: p["datetime"])
    return points


def get_recent(
    location_id: int,
    hours: int = 24,
    parameter: str = "pm25",
) -> list[dict]:
    """Return chronologically-sorted {datetime, value} points for the last N hours."""
    param = _normalize_parameter(parameter)

    cache_key = (location_id, param, hours)
    cached = _history_cache.get(cache_key)
    if cached and time.time() - cached[0] < _HISTORY_CACHE_TTL_SECONDS:
        return cached[1]

    points: list[dict] = []
    # OpenAQ is the canonical source for all three pollutants. For PM2.5
    # we fall back to the local actuals log when OpenAQ is unavailable
    # (missing key, upstream error) so the dashboard still has *something*.
    try:
        points = _from_openaq(location_id, param, hours)
    except Exception as exc:  # noqa: BLE001 — we fall back on any upstream failure
        if param == "pm25":
            logger.warning("OpenAQ fetch failed for loc=%s (%s); using actuals log.", location_id, exc)
            points = _from_actuals_log(location_id, hours)
        else:
            logger.warning("OpenAQ fetch failed for loc=%s param=%s: %s", location_id, param, exc)
            points = []

    if not points and param == "pm25":
        points = _from_actuals_log(location_id, hours)

    _history_cache[cache_key] = (time.time(), points)
    return points
