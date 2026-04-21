"""Berlin AQI FastAPI service — /predict, /health, /metrics."""
from __future__ import annotations

import logging
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query

from api import cache, features_runtime, model_loader, threshold_rule
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
    model_loader.load()
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
    # Cache-first path (the plan's architecture — cron pre-computes)
    cached = cache.get(location_id)
    if cached is not None:
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
        _record(response)
        return response

    # Fallback: compute on the fly from the latest training features CSV.
    # Only hit when the refresh cron hasn't run yet for this station.
    logger.warning("Cache miss for location_id=%d — falling back to live compute", location_id)
    state = model_loader.get_state()
    try:
        X, target_dt, pm25_current = features_runtime.get_feature_row(
            location_id, state["feature_cols"]
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    proba = state["model"].predict_proba(X)[0]
    pred_idx = int(proba.argmax())
    predicted = state["label_mapping"][pred_idx]
    confidence = float(proba.max())
    final, rule_fired = threshold_rule.apply(predicted, pm25_current)
    now_iso = datetime.now(timezone.utc).isoformat()

    response = PredictResponse(
        location_id=location_id,
        predicted_category=final,
        target_datetime=target_dt.isoformat(),
        pm25_current=pm25_current,
        confidence=confidence,
        rule_override=rule_fired,
        refreshed_at=now_iso,
        age_seconds=0,
        source="live",
    )
    _record(response)
    return response


def _record(resp: PredictResponse) -> None:
    _metrics["total"] += 1
    _metrics["by_category"][resp.predicted_category] += 1
    if resp.rule_override:
        _metrics["rule_overrides"] += 1


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    return MetricsResponse(
        predictions_total=_metrics["total"],
        predictions_by_category=dict(_metrics["by_category"]),
        predictions_with_rule_override=_metrics["rule_overrides"],
    )
