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
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

MODEL_NAME = "berlin-aqi-xgboost"
ALIAS = "production"
OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


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

    logger.info("Bundled into %s", OUT_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
