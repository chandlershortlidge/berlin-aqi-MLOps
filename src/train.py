"""XGBoost training + MLflow tracking for Berlin AQI prediction.

Loads the latest train/test CSVs from `data/processed/`, trains an
XGBClassifier with inverse-frequency sample weights, evaluates on F2
(primary) plus accuracy / precision / recall / confusion matrix, and
logs the whole run to MLflow.

Two entry points:
- `python -m src.train`         → baseline with default hyperparams
- `python -m src.train --tune`  → RandomizedSearchCV over F2 macro,
                                   then refit best params on full
                                   train with sample weights

SMOTE fallback and registry promotion come in follow-up commits.
"""
from __future__ import annotations

import argparse
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
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
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

# Search space for RandomizedSearchCV (see planning doc Step 3.4)
PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 300, 500, 800],
    "max_depth": [3, 4, 5, 6, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "min_child_weight": [1, 3, 5, 10],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
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


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_iter: int = 20,
    cv_splits: int = 5,
    random_state: int = 42,
) -> tuple[dict, float, pd.DataFrame]:
    """RandomizedSearchCV with StratifiedKFold, scored on F2 macro.

    Two adaptations for rare classes:
    - Classes with fewer rows than cv_splits are dropped from the search
      set (StratifiedKFold can't put them in every fold's train half).
      At Mitte this drops the single High+ row.
    - Remaining labels are re-encoded to contiguous 0..k-1 because
      XGBoost rejects non-contiguous label sets.

    Stratification is used instead of TimeSeriesSplit: rare classes would
    otherwise be absent from early time-ordered folds. The held-out test
    set is still strictly time-based (from features.py) so final
    evaluation remains honest.

    Sample weights are NOT applied during the search — that would require
    sklearn metadata routing. The caller refits the winning params on the
    full train set (all 5 classes) with sample_weight applied.

    Returns (best_params, best_cv_f2, cv_results_df).
    """
    unique, counts = np.unique(y_train, return_counts=True)
    viable = unique[counts >= cv_splits]
    mask = np.isin(y_train, viable)
    dropped = int((~mask).sum())
    if dropped:
        excluded = sorted(set(unique) - set(viable))
        logger.warning(
            "Dropping %d row(s) with rare classes %s from CV search (< %d per class)",
            dropped, excluded, cv_splits,
        )

    X_cv = X_train[mask].reset_index(drop=True)
    remap = {old: new for new, old in enumerate(sorted(viable))}
    y_cv = np.array([remap[v] for v in y_train[mask]])
    n_cv_classes = len(viable)

    f2_scorer = make_scorer(fbeta_score, beta=2, average="macro", zero_division=0)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    base = XGBClassifier(
        num_class=n_cv_classes,
        objective="multi:softprob",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        base,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring=f2_scorer,
        cv=cv,
        random_state=random_state,
        n_jobs=1,  # avoid oversubscription — XGBoost already uses all cores
        verbose=1,
        refit=False,
    )
    logger.info("Starting RandomizedSearchCV: %d trials × %d folds = %d fits",
                n_iter, cv_splits, n_iter * cv_splits)
    search.fit(X_cv, y_cv)
    logger.info("Best CV F2 macro: %.4f", search.best_score_)
    logger.info("Best params: %s", search.best_params_)

    return search.best_params_, float(search.best_score_), pd.DataFrame(search.cv_results_)


def run(
    train_path: Path | None = None,
    test_path: Path | None = None,
    experiment_name: str = "berlin-aqi-xgboost",
    override_params: dict | None = None,
    cv_results: pd.DataFrame | None = None,
    run_tag: str = "baseline",
) -> dict:
    """End-to-end training run. Returns the metrics dict.

    If `override_params` is passed (e.g. from RandomizedSearchCV), they
    merge on top of DEFAULT_XGB_PARAMS. If `cv_results` is passed, the
    full search results are logged as a CSV artifact.
    """
    if train_path is None or test_path is None:
        train_path, test_path = _latest_pair()
    logger.info("Train: %s", train_path)
    logger.info("Test:  %s", test_path)

    X_train, y_train, X_test, y_test, encoder, feature_cols = load_data(train_path, test_path)

    train_counts = pd.Series(y_train).map(lambda v: encoder.inverse_transform([v])[0]).value_counts()
    logger.info("Train class counts:\n%s", train_counts.to_string())

    sample_weights = compute_sample_weights(y_train)
    n_classes = len(encoder.classes_)
    params = {**DEFAULT_XGB_PARAMS, **(override_params or {}), "num_class": n_classes}

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(tags={"run_type": run_tag}) as run_ctx:
        mlflow.log_params(params)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("class_weighting", "inverse_frequency")

        model = train_model(X_train, y_train, sample_weights, n_classes, params=params)

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

        if cv_results is not None:
            cv_path = MODEL_DIR / "cv_results.csv"
            cv_results.to_csv(cv_path, index=False)
            mlflow.log_artifact(str(cv_path))

        # Log the model
        mlflow.xgboost.log_model(model, name="model")

        # Local copy for Phase 5 serving
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        local_model_path = MODEL_DIR / f"xgb_{run_tag}_{stamp}.json"
        model.save_model(local_model_path)

        logger.info("MLflow run: %s", run_ctx.info.run_id)
        logger.info("Model saved to %s", local_model_path)

    return metrics


def run_tuning(
    train_path: Path | None = None,
    test_path: Path | None = None,
    n_iter: int = 20,
    cv_splits: int = 5,
    experiment_name: str = "berlin-aqi-xgboost",
) -> dict:
    """Tune via RandomizedSearchCV, then refit best params on full train with sample weights."""
    if train_path is None or test_path is None:
        train_path, test_path = _latest_pair()

    X_train, y_train, _, _, _, _ = load_data(train_path, test_path)
    best_params, best_cv_f2, cv_results = tune_hyperparameters(
        X_train, y_train, n_iter=n_iter, cv_splits=cv_splits
    )

    metrics = run(
        train_path=train_path,
        test_path=test_path,
        experiment_name=experiment_name,
        override_params=best_params,
        cv_results=cv_results,
        run_tag="tuned",
    )
    metrics["best_cv_f2_macro"] = best_cv_f2
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train Berlin AQI XGBoost model.")
    parser.add_argument("--tune", action="store_true", help="RandomizedSearchCV over F2 macro")
    parser.add_argument("--n-iter", type=int, default=20, help="Number of tuning trials")
    parser.add_argument("--cv-splits", type=int, default=5, help="TimeSeriesSplit folds")
    args = parser.parse_args()

    if args.tune:
        metrics = run_tuning(n_iter=args.n_iter, cv_splits=args.cv_splits)
    else:
        metrics = run()

    print("\n=== Metrics ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k:40s} {v:.4f}")
