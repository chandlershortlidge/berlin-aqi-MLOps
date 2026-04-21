"""Reader for the refresh-job prediction cache.

`src.refresh` writes `data/cache/predictions.json` atomically every hour.
The API reads it on every /predict request — file reads are a few ms,
cheaper than recomputing features + model inference per request.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "cache" / "predictions.json"


def read_all() -> dict[str, dict]:
    """Return the full cache dict, or empty dict if the cache doesn't exist yet."""
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text())
    except json.JSONDecodeError as exc:
        logger.warning("Cache at %s is corrupt: %s", CACHE_PATH, exc)
        return {}


def get(location_id: int) -> dict | None:
    """Return the cached prediction for a station, or None if missing / stale."""
    cache = read_all()
    return cache.get(str(location_id))


def age_seconds(refreshed_at_iso: str) -> int:
    return int((datetime.now(timezone.utc) - datetime.fromisoformat(refreshed_at_iso)).total_seconds())
