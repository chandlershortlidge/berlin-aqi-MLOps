"""Load the production XGBoost model + label mapping + feature columns from MLflow.

Uses the alias URI `models:/berlin-aqi-xgboost@production` (set by
`src.register`). Downloads the companion `label_mapping.json` and
`feature_columns.json` artifacts from the same run.

The state is cached in a module-level dict after first `load()` call;
subsequent calls are no-ops.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import mlflow.artifacts
import mlflow.xgboost
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

MODEL_NAME = "berlin-aqi-xgboost"
ALIAS = "production"
MODEL_URI = f"models:/{MODEL_NAME}@{ALIAS}"

_state: dict[str, Any] = {}


def load() -> dict[str, Any]:
    """Load model + artifacts. Idempotent."""
    if "model" in _state:
        return _state

    logger.info("Loading model from %s", MODEL_URI)
    client = MlflowClient()
    version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)

    model = mlflow.xgboost.load_model(MODEL_URI)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        label_path = mlflow.artifacts.download_artifacts(
            run_id=version.run_id, artifact_path="label_mapping.json", dst_path=str(tmp_path)
        )
        features_path = mlflow.artifacts.download_artifacts(
            run_id=version.run_id, artifact_path="feature_columns.json", dst_path=str(tmp_path)
        )
        label_mapping = {int(k): v for k, v in json.loads(Path(label_path).read_text()).items()}
        feature_cols = json.loads(Path(features_path).read_text())

    _state.update(
        {
            "model": model,
            "label_mapping": label_mapping,
            "feature_cols": feature_cols,
            "version": int(version.version),
            "run_id": version.run_id,
        }
    )
    logger.info(
        "Loaded %s v%d (run %s) — %d classes, %d features",
        MODEL_NAME, _state["version"], _state["run_id"],
        len(label_mapping), len(feature_cols),
    )
    return _state


def get_state() -> dict[str, Any]:
    if "model" not in _state:
        load()
    return _state
