"""Hourly refresh: ingest latest, engineer features, predict, cache.

This is the cron-driven path of the serving flow. Every hour:

1. Pull the last ~48h of pollutants + weather for every eligible station
2. Build inference features (lag + rolling + time — NO t+1 target shift)
3. Run the production model on each station's most recent hour
4. Apply the PM2.5 > 150.5 threshold override for High+
5. Atomically write predictions to `data/cache/predictions.json`
6. Append to `data/monitoring/predictions.csv` and
   `data/monitoring/actuals.csv` (Phase 7 consumes these)

Local cron:
    0 * * * * cd /path/to/repo && uv run python -m src.refresh >> logs/refresh.log 2>&1

AWS: EventBridge triggers a scheduled ECS task / Lambda that runs the
same `python -m src.refresh` entry point.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from api import model_loader, threshold_rule
from src.features import AQI_BINS, AQI_LABELS, build_inference_features
from src.ingest import (
    ELIGIBLE_BERLIN_STATIONS,
    clean,
    fetch,
    fetch_coordinates,
    fetch_weather,
    validate,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_DIR = DATA_DIR / "cache"
MONITORING_DIR = DATA_DIR / "monitoring"
PREDICTIONS_CACHE = CACHE_DIR / "predictions.json"
PREDICTIONS_LOG = MONITORING_DIR / "predictions.csv"
ACTUALS_LOG = MONITORING_DIR / "actuals.csv"

# 48h gives full warmup for 24h lag + 24h rolling windows
LOOKBACK_HOURS = 48


def _ingest_one(lid: int, dt_from: str, dt_to: str) -> pd.DataFrame | None:
    try:
        raw = fetch(lid, dt_from, dt_to)
        validate(raw)
        cleaned = clean(raw)
        lat, lon = fetch_coordinates(lid)
        weather = fetch_weather(lat, lon, dt_from, dt_to)
        combined = cleaned.merge(weather, on="datetime", how="left")
        combined["location_id"] = lid
        combined["latitude"] = lat
        combined["longitude"] = lon
        return combined
    except Exception as exc:
        logger.error("Station %d ingest failed — skipping: %s", lid, exc)
        return None


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".refresh_", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        Path(tmp_name).replace(path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _pm25_to_category(pm25: float) -> str:
    """Same 5-class mapping as features.add_target, without pandas overhead."""
    return pd.cut([pm25], bins=AQI_BINS, labels=AQI_LABELS, include_lowest=True).astype(str)[0]


def run(stations: list[int] | None = None) -> dict:
    stations = stations or ELIGIBLE_BERLIN_STATIONS
    now = datetime.now(timezone.utc)
    dt_from = (now - timedelta(hours=LOOKBACK_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    dt_to = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.info("Refresh start — %d stations, %dh lookback", len(stations), LOOKBACK_HOURS)

    frames = [f for f in (_ingest_one(lid, dt_from, dt_to) for lid in stations) if f is not None]
    if not frames:
        raise RuntimeError("Refresh failed: no stations successfully ingested")

    raw = pd.concat(frames, ignore_index=True)
    logger.info("Ingested %d rows across %d stations", len(raw), raw["location_id"].nunique())

    features = build_inference_features(raw)

    state = model_loader.get_state()
    model = state["model"]
    feature_cols = state["feature_cols"]
    label_mapping = state["label_mapping"]

    predictions: dict[str, dict] = {}
    pred_rows, actual_rows = [], []

    for lid, group in features.groupby("location_id"):
        latest = group.sort_values("datetime").iloc[[-1]]
        X = latest[feature_cols]
        proba = model.predict_proba(X)[0]
        pred_idx = int(proba.argmax())
        predicted = label_mapping[pred_idx]
        confidence = float(proba.max())
        pm25_current = float(latest["pm25"].iloc[0])
        target_dt = latest["datetime"].iloc[0] + pd.Timedelta(hours=1)

        final, rule_fired = threshold_rule.apply(predicted, pm25_current)

        predictions[str(int(lid))] = {
            "location_id": int(lid),
            "predicted_category": final,
            "target_datetime": target_dt.isoformat(),
            "pm25_current": pm25_current,
            "confidence": confidence,
            "rule_override": rule_fired,
            "refreshed_at": now.isoformat(),
            "latitude": float(latest["latitude"].iloc[0]) if "latitude" in latest else None,
            "longitude": float(latest["longitude"].iloc[0]) if "longitude" in latest else None,
        }

        # Include the primary feature values so monitor.py can run PSI drift
        # without a separate snapshot log.
        pred_rows.append({
            "refreshed_at": now.isoformat(),
            "location_id": int(lid),
            "target_datetime": target_dt.isoformat(),
            "predicted_category": final,
            "confidence": confidence,
            "rule_override": rule_fired,
            "pm25_current": pm25_current,
            "pm10_current": float(latest["pm10"].iloc[0]) if "pm10" in latest else None,
            "no2_current": float(latest["no2"].iloc[0]) if "no2" in latest else None,
            "temperature_current": float(latest["temperature"].iloc[0]) if "temperature" in latest else None,
            "relative_humidity_current": float(latest["relative_humidity"].iloc[0]) if "relative_humidity" in latest else None,
        })

        # Record the just-observed hour as an "actual" — the previous
        # refresh's prediction was trying to call this one.
        actual_rows.append({
            "refreshed_at": now.isoformat(),
            "location_id": int(lid),
            "observed_datetime": latest["datetime"].iloc[0].isoformat(),
            "actual_category": _pm25_to_category(pm25_current),
            "pm25_actual": pm25_current,
        })

    _atomic_write_json(PREDICTIONS_CACHE, predictions)
    logger.info("Wrote %d predictions to %s", len(predictions), PREDICTIONS_CACHE)

    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pred_rows).to_csv(
        PREDICTIONS_LOG, mode="a", header=not PREDICTIONS_LOG.exists(), index=False
    )
    pd.DataFrame(actual_rows).to_csv(
        ACTUALS_LOG, mode="a", header=not ACTUALS_LOG.exists(), index=False
    )
    logger.info("Appended %d pred / %d actual rows to monitoring logs",
                len(pred_rows), len(actual_rows))

    return predictions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    run()
