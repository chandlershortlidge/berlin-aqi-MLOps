"""Pydantic request/response models for the Berlin AQI API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    location_id: int
    predicted_category: str = Field(
        ..., description="One of: All Clear, Low Risk, Elevated, Significant, High+"
    )
    target_datetime: str = Field(..., description="UTC ISO timestamp the prediction is for (t+1)")
    pm25_current: float = Field(..., description="PM2.5 µg/m³ at the current hour (t)")
    confidence: float = Field(..., description="Max class probability from the model (0-1)")
    rule_override: bool = Field(
        ..., description="True when the High+ PM2.5 threshold rule overrode the model"
    )
    refreshed_at: str = Field(..., description="UTC ISO timestamp when the refresh cron computed this")
    age_seconds: int = Field(..., description="Seconds since refreshed_at — clients can reject stale entries")
    source: str = Field(..., description="'cache' (cron) or 'live' (fallback when cache is missing)")


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: int | None = None
    mlflow_run_id: str | None = None


class MetricsResponse(BaseModel):
    predictions_total: int
    predictions_by_category: dict[str, int]
    predictions_with_rule_override: int
