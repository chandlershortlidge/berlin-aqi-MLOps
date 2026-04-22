"""Export the @production MLflow model + sidecars to ./artifacts/.

Run once before building the container image. The Dockerfile copies the
whole ./artifacts/ directory in, so the running container never touches
the MLflow registry at serve time. Re-run after promoting a new model
version, then rebuild the image.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import mlflow.artifacts
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

MODEL_NAME = "berlin-aqi-xgboost"
ALIAS = "production"
OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# Features to baseline for PSI drift. Lags are derived from pm25 (redundant),
# and time features don't drift. These 5 are the independent inputs.
BASELINE_FEATURES = ["pm25", "pm10", "no2", "temperature", "relative_humidity"]
BASELINE_BINS = 10


def _compute_baseline() -> dict:
    """Per-feature histogram of the training features CSV for PSI drift checks."""
    files = sorted(PROCESSED_DIR.glob("features_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        logger.warning("No training features_*.csv — skipping baseline")
        return {}

    df = pd.read_csv(files[-1])
    baseline: dict = {}
    for col in BASELINE_FEATURES:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(values) < 2:
            continue
        edges = np.histogram_bin_edges(values, bins=BASELINE_BINS)
        counts, _ = np.histogram(values, bins=edges)
        baseline[col] = {
            "bin_edges": edges.tolist(),
            "counts": counts.tolist(),
            "n": int(len(values)),
        }
    logger.info("Baseline computed for %d features from %s", len(baseline), files[-1].name)
    return baseline


def main() -> None:
    client = MlflowClient()
    version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    logger.info("Bundling %s v%s (run %s)", MODEL_NAME, version.version, version.run_id)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    # The MLflow model directory (contains MLmodel, model.xgb, conda.yaml, etc.)
    model_dir = mlflow.artifacts.download_artifacts(
        run_id=version.run_id, artifact_path="model", dst_path=str(OUT_DIR)
    )
    logger.info("Model artifact dir: %s", model_dir)

    # Sidecar JSONs written alongside by train.py
    mlflow.artifacts.download_artifacts(
        run_id=version.run_id, artifact_path="label_mapping.json", dst_path=str(OUT_DIR)
    )
    mlflow.artifacts.download_artifacts(
        run_id=version.run_id, artifact_path="feature_columns.json", dst_path=str(OUT_DIR)
    )

    (OUT_DIR / "METADATA.json").write_text(json.dumps({
        "model_name": MODEL_NAME,
        "version": int(version.version),
        "run_id": version.run_id,
        "alias": ALIAS,
    }, indent=2))

    baseline = _compute_baseline()
    if baseline:
        (OUT_DIR / "feature_baseline.json").write_text(json.dumps(baseline, indent=2))
        logger.info("Wrote feature_baseline.json (%d features)", len(baseline))

    logger.info("Bundled into %s", OUT_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
