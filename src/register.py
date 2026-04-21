"""Register the latest tuned MLflow run and set a production alias.

Usage:
    uv run python -m src.register                 # auto-pick latest tuned run
    uv run python -m src.register --run-id <id>   # explicit run
"""
from __future__ import annotations

import argparse
import logging

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "berlin-aqi-xgboost"
MODEL_NAME = "berlin-aqi-xgboost"


def latest_tuned_run_id(client: MlflowClient) -> str:
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"MLflow experiment '{EXPERIMENT_NAME}' does not exist")
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.run_type = 'tuned'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No runs tagged run_type='tuned' found")
    return runs[0].info.run_id


def register_and_promote(run_id: str, alias: str = "production") -> int:
    """Register the run's model and set `alias` on the new version. Returns the version."""
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    logger.info("Registering %s as '%s'", model_uri, MODEL_NAME)
    registered = mlflow.register_model(model_uri, MODEL_NAME)
    version = int(registered.version)
    client.set_registered_model_alias(MODEL_NAME, alias, version)
    logger.info(
        "Model '%s' version %d now has alias '%s'", MODEL_NAME, version, alias
    )
    return version


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Register the latest tuned MLflow model under an alias."
    )
    parser.add_argument("--run-id", default=None, help="Explicit MLflow run ID")
    parser.add_argument("--alias", default="production", help="Alias to set on the version")
    args = parser.parse_args()

    client = MlflowClient()
    run_id = args.run_id or latest_tuned_run_id(client)
    logger.info("Source run: %s", run_id)
    version = register_and_promote(run_id, args.alias)
    print(f"\nRegistered '{MODEL_NAME}' v{version} with alias '{args.alias}'")
