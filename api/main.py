"""Berlin AQI FastAPI service — /predict, /health, /metrics, /cache.

The /predict path is cache-only: `src.refresh` writes
`data/cache/predictions.json` on a schedule and the API just reads it.
If the cache is missing or a station isn't in it, the endpoint returns
503 / 404 — not 500, and not an on-the-fly ingest.

This means a fresh container exposes /predict as 503 until a refresh
has populated the cache for the first time. Run `src.refresh` inside
the container (or on a schedule) to bootstrap.
"""
from __future__ import annotations

import logging
from collections import Counter
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query

from api import cache, history, model_loader
from api.schemas import HealthResponse, MetricsResponse, PredictResponse

logger = logging.getLogger(__name__)

# Simple in-memory metrics for /metrics. OK for a single-process dev service;
# a production deployment across multiple workers would need a shared store.
_metrics: dict = {
    "total": 0,
    "by_category": Counter(),
    "rule_overrides": 0,
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # /history calls OpenAQ and needs OPENAQ_API_KEY. Pulling .env at
    # startup keeps `uvicorn api.main:app` self-sufficient in local runs;
    # in prod the key comes from the task/instance role instead.
    load_dotenv()
    model_loader.load()
    if not cache.CACHE_PATH.exists():
        logger.warning(
            "Prediction cache not found at %s. /predict will return 503 "
            "until a refresh populates it. Run `python -m src.refresh` "
            "inside the container (needs OPENAQ_API_KEY) or mount a "
            "pre-populated data/ volume.",
            cache.CACHE_PATH,
        )
    yield


app = FastAPI(
    title="Berlin AQI",
    description="Next-hour air quality predictions for Berlin athletes.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    state = model_loader.get_state()
    return HealthResponse(
        status="ok",
        model_name=model_loader.MODEL_NAME,
        model_version=state.get("version"),
        mlflow_run_id=state.get("run_id"),
    )


@app.get("/predict", response_model=PredictResponse)
def predict(location_id: int = Query(..., description="OpenAQ location_id")) -> PredictResponse:
    if not cache.CACHE_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction cache not populated yet. Run "
                "`python -m src.refresh` (with OPENAQ_API_KEY set) "
                "before querying /predict."
            ),
        )

    cached = cache.get(location_id)
    if cached is None:
        known = sorted(cache.read_all().keys())
        raise HTTPException(
            status_code=404,
            detail=f"Location {location_id} is not in the cache. Known: {known[:20]}"
            + ("..." if len(known) > 20 else ""),
        )

    response = PredictResponse(
        location_id=cached["location_id"],
        predicted_category=cached["predicted_category"],
        target_datetime=cached["target_datetime"],
        pm25_current=cached["pm25_current"],
        confidence=cached["confidence"],
        rule_override=cached["rule_override"],
        refreshed_at=cached["refreshed_at"],
        age_seconds=cache.age_seconds(cached["refreshed_at"]),
        source="cache",
    )
    _metrics["total"] += 1
    _metrics["by_category"][response.predicted_category] += 1
    if response.rule_override:
        _metrics["rule_overrides"] += 1
    return response


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    return MetricsResponse(
        predictions_total=_metrics["total"],
        predictions_by_category=dict(_metrics["by_category"]),
        predictions_with_rule_override=_metrics["rule_overrides"],
    )


@app.get("/history")
def history_for_station(
    location_id: int = Query(..., description="OpenAQ location_id"),
    hours: int = Query(24, ge=1, le=744, description="How many hours back to include"),
    parameter: str = Query(
        "pm25",
        description="Pollutant: pm25, pm10, or no2",
    ),
) -> dict:
    """Per-station pollutant history. PM2.5/PM10/NO₂ served from OpenAQ v3."""
    logger.info(
        "/history location_id=%s parameter=%s hours=%s", location_id, parameter, hours
    )
    try:
        points = history.get_recent(location_id, hours=hours, parameter=parameter)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    logger.info(
        "/history -> %d points for location_id=%s parameter=%s",
        len(points), location_id, parameter,
    )
    return {
        "location_id": location_id,
        "parameter": parameter,
        "hours": hours,
        "points": points,
    }


@app.get("/cache")
def cache_status() -> dict:
    """Full cache contents + freshness. Feeds the frontend map."""
    if not cache.CACHE_PATH.exists():
        return {"exists": False, "path": str(cache.CACHE_PATH), "stations": 0, "predictions": {}}
    entries = cache.read_all()
    newest = max((e["refreshed_at"] for e in entries.values()), default=None)
    return {
        "exists": True,
        "path": str(cache.CACHE_PATH),
        "stations": len(entries),
        "station_ids": sorted(int(k) for k in entries.keys()),
        "newest_refreshed_at": newest,
        "newest_age_seconds": cache.age_seconds(newest) if newest else None,
        "predictions": entries,
    }
