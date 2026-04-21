"""XGBoost training + MLflow tracking for Berlin AQI prediction.

Loads the latest train/test CSVs from `data/processed/`, trains an
XGBClassifier with inverse-frequency sample weights, evaluates on F2
(primary) plus accuracy / precision / recall / confusion matrix, and
logs the whole run to MLflow.

This is the MVP pass: default hyperparameters, no RandomizedSearchCV
tuning yet, no SMOTE, no registry promotion. Those follow once we
have a baseline F2 to beat.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.features import AQI_LABELS

logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[1] / "data" / "models"

TARGET = "aqi_category_next"
# Columns in train/test that are not features
NON_FEATURE_COLS = ["datetime", TARGET]

DEFAULT_XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "objective": "multi:softprob",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}


def _latest_pair(processed_dir: Path = PROCESSED_DATA_DIR) -> tuple[Path, Path]:
    """Return (train_path, test_path) for the most recent timestamped pair."""
    trains = sorted(processed_dir.glob("train_*.csv"), key=lambda p: p.stat().st_mtime)
    tests = sorted(processed_dir.glob("test_*.csv"), key=lambda p: p.stat().st_mtime)
    if not trains or not tests:
        raise FileNotFoundError(f"No train/test CSVs in {processed_dir}")
    return trains[-1], tests[-1]


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """All columns except datetime/target, dropping columns that are entirely NaN."""
    candidates = [c for c in df.columns if c not in NON_FEATURE_COLS]
    return [c for c in candidates if df[c].notna().any()]


def load_data(
    train_path: Path, test_path: Path
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, LabelEncoder, list[str]]:
    """Load train/test CSVs, label-encode the target, return feature matrices + encoder."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    feature_cols = _feature_columns(train)
    logger.info("Using %d feature columns", len(feature_cols))

    encoder = LabelEncoder()
    encoder.fit(AQI_LABELS)

    X_train = train[feature_cols].copy()
    y_train = encoder.transform(train[TARGET].astype(str))
    X_test = test[feature_cols].copy()
    y_test = encoder.transform(test[TARGET].astype(str))

    return X_train, y_train, X_test, y_test, encoder, feature_cols


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights — each class contributes equal total weight."""
    classes, counts = np.unique(y, return_counts=True)
    weight_per_class = len(y) / (len(classes) * counts)
    weight_map = dict(zip(classes, weight_per_class))
    return np.array([weight_map[v] for v in y])


def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weights: np.ndarray,
    n_classes: int,
    params: dict | None = None,
) -> XGBClassifier:
    """Fit XGBClassifier with per-row sample weights."""
    config = {**DEFAULT_XGB_PARAMS, "num_class": n_classes, **(params or {})}
    model = XGBClassifier(**config)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def evaluate(
    model: XGBClassifier,
    X: pd.DataFrame,
    y: np.ndarray,
    encoder: LabelEncoder,
    split_name: str,
) -> dict:
    """Compute F2 (macro + per-class), accuracy, precision, recall."""
    y_pred = model.predict(X)
    labels_present = sorted(set(y) | set(y_pred))
    metrics = {
        f"{split_name}_f2_macro": fbeta_score(y, y_pred, beta=2, average="macro", zero_division=0),
        f"{split_name}_f2_weighted": fbeta_score(y, y_pred, beta=2, average="weighted", zero_division=0),
        f"{split_name}_accuracy": accuracy_score(y, y_pred),
        f"{split_name}_precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
        f"{split_name}_recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
    }
    # Per-class F2
    per_class_f2 = fbeta_score(y, y_pred, beta=2, average=None, labels=labels_present, zero_division=0)
    for label_idx, score in zip(labels_present, per_class_f2):
        label_name = encoder.inverse_transform([label_idx])[0].replace(" ", "_").replace("+", "plus")
        metrics[f"{split_name}_f2_{label_name}"] = float(score)
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, encoder: LabelEncoder, out_path: Path
) -> Path:
    """Save a confusion matrix PNG using the label names."""
    label_idx = list(range(len(encoder.classes_)))
    cm = confusion_matrix(y_true, y_pred, labels=label_idx)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(label_idx, labels=encoder.classes_, rotation=45, ha="right")
    ax.set_yticks(label_idx, labels=encoder.classes_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def run(
    train_path: Path | None = None,
    test_path: Path | None = None,
    experiment_name: str = "berlin-aqi-xgboost",
) -> dict:
    """End-to-end training run. Returns the metrics dict."""
    if train_path is None or test_path is None:
        train_path, test_path = _latest_pair()
    logger.info("Train: %s", train_path)
    logger.info("Test:  %s", test_path)

    X_train, y_train, X_test, y_test, encoder, feature_cols = load_data(train_path, test_path)

    train_counts = pd.Series(y_train).map(lambda v: encoder.inverse_transform([v])[0]).value_counts()
    logger.info("Train class counts:\n%s", train_counts.to_string())

    sample_weights = compute_sample_weights(y_train)
    n_classes = len(encoder.classes_)
    params = {**DEFAULT_XGB_PARAMS, "num_class": n_classes}

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run_ctx:
        mlflow.log_params(params)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("class_weighting", "inverse_frequency")

        model = train_model(X_train, y_train, sample_weights, n_classes)

        metrics = {
            **evaluate(model, X_train, y_train, encoder, "train"),
            **evaluate(model, X_test, y_test, encoder, "test"),
        }
        mlflow.log_metrics(metrics)

        print("\n=== Classification report (test) ===")
        print(classification_report(
            y_test, model.predict(X_test),
            target_names=encoder.classes_, zero_division=0,
        ))

        # Artifacts: confusion matrix, label mapping, feature list
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cm_path = MODEL_DIR / "confusion_matrix.png"
        plot_confusion_matrix(y_test, model.predict(X_test), encoder, cm_path)
        mlflow.log_artifact(str(cm_path))

        mapping_path = MODEL_DIR / "label_mapping.json"
        mapping_path.write_text(json.dumps(
            {int(i): str(lbl) for i, lbl in enumerate(encoder.classes_)}, indent=2
        ))
        mlflow.log_artifact(str(mapping_path))

        features_path = MODEL_DIR / "feature_columns.json"
        features_path.write_text(json.dumps(feature_cols, indent=2))
        mlflow.log_artifact(str(features_path))

        # Log the model
        mlflow.xgboost.log_model(model, name="model")

        # Local copy for Phase 5 serving
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        local_model_path = MODEL_DIR / f"xgb_{stamp}.json"
        model.save_model(local_model_path)

        logger.info("MLflow run: %s", run_ctx.info.run_id)
        logger.info("Model saved to %s", local_model_path)

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    metrics = run()

    print("\n=== Metrics ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k:40s} {v:.4f}")
