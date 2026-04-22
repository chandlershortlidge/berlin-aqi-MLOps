"""Load the model baked into the container image at build time.

The image's ./artifacts/ directory is populated pre-build by
`src.bundle`, which extracts the @production MLflow model and its
sidecar JSONs. At runtime there is **no** MLflow tracking-server
contact — the container serves predictions fully self-contained.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow.xgboost

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "model"
LABEL_MAPPING_PATH = ARTIFACTS_DIR / "label_mapping.json"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.json"
METADATA_PATH = ARTIFACTS_DIR / "METADATA.json"

MODEL_NAME = "berlin-aqi-xgboost"
ALIAS = "production"

_state: dict[str, Any] = {}


def load() -> dict[str, Any]:
    """Load model + sidecars from the bundled artifacts dir. Idempotent."""
    if "model" in _state:
        return _state

    if not MODEL_DIR.exists():
        raise RuntimeError(
            f"Model bundle not found at {MODEL_DIR}. "
            f"Run `uv run python -m src.bundle` before building the image."
        )

    logger.info("Loading bundled model from %s", MODEL_DIR)
    model = mlflow.xgboost.load_model(str(MODEL_DIR))
    label_mapping = {int(k): v for k, v in json.loads(LABEL_MAPPING_PATH.read_text()).items()}
    feature_cols = json.loads(FEATURE_COLUMNS_PATH.read_text())
    metadata = json.loads(METADATA_PATH.read_text())

    _state.update({
        "model": model,
        "label_mapping": label_mapping,
        "feature_cols": feature_cols,
        "version": int(metadata["version"]),
        "run_id": metadata["run_id"],
    })
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
