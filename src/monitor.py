"""Phase 7a — model performance monitoring.

Consumes the monitoring logs that `src.refresh` writes every hour:
- data/monitoring/predictions.csv   (one row per station per refresh)
- data/monitoring/actuals.csv       (the PM2.5-derived truth for the same hour)

Joins them on (location_id, target_datetime = observed_datetime) — the
prediction at 12:00 targets 13:00, and the 13:00 refresh records the
actual for 13:00 — and reports:
- rolling 24h F2 macro + per-class
- rolling 24h accuracy
- per-station 24h F2
- all-time F2
- a threshold-triggered alert on the rolling window

Writes data/monitoring/metrics.json for downstream dashboards and
logs a WARNING when rolling F2 drops below the alert threshold.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, fbeta_score

from src.features import AQI_LABELS

logger = logging.getLogger(__name__)

MONITORING_DIR = Path(__file__).resolve().parents[1] / "data" / "monitoring"
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
PREDICTIONS_LOG = MONITORING_DIR / "predictions.csv"
ACTUALS_LOG = MONITORING_DIR / "actuals.csv"
METRICS_OUT = MONITORING_DIR / "metrics.json"
BASELINE_PATH = ARTIFACTS_DIR / "feature_baseline.json"

# Tier-2 rubric signal — if rolling 24h F2 drops below this, alert.
# Baseline test F2 macro on Mitte was 0.66; 0.50 is a meaningful regression.
F2_ALERT_THRESHOLD = 0.50
# PSI thresholds per the planning doc: <0.1 ok, 0.1-0.2 watch, >0.2 retrain.
PSI_ALERT_THRESHOLD = 0.20

# Which columns in predictions.csv correspond to which baseline features
PSI_COLUMN_MAP = {
    "pm25": "pm25_current",
    "pm10": "pm10_current",
    "no2": "no2_current",
    "temperature": "temperature_current",
    "relative_humidity": "relative_humidity_current",
}


def load_joined() -> pd.DataFrame:
    """Return a DataFrame of matched (prediction, actual) rows."""
    if not PREDICTIONS_LOG.exists() or not ACTUALS_LOG.exists():
        raise FileNotFoundError(
            f"Monitoring logs not found under {MONITORING_DIR}. "
            "Run `python -m src.refresh` at least twice (predict + observe)."
        )

    preds = pd.read_csv(
        PREDICTIONS_LOG, parse_dates=["refreshed_at", "target_datetime"]
    )
    actuals = pd.read_csv(
        ACTUALS_LOG, parse_dates=["refreshed_at", "observed_datetime"]
    )

    joined = preds.merge(
        actuals[["location_id", "observed_datetime", "actual_category", "pm25_actual"]],
        left_on=["location_id", "target_datetime"],
        right_on=["location_id", "observed_datetime"],
        how="inner",
    )
    return joined


def _metrics_for(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"f2_macro": None, "accuracy": None, "f2_per_class": {}, "n": 0}
    y_true = df["actual_category"]
    y_pred = df["predicted_category"]
    labels = sorted(set(y_true) | set(y_pred))
    per_class = fbeta_score(y_true, y_pred, beta=2, average=None, labels=labels, zero_division=0)
    return {
        "f2_macro": float(fbeta_score(y_true, y_pred, beta=2, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f2_per_class": {lbl: float(s) for lbl, s in zip(labels, per_class)},
        "n": int(len(df)),
    }


def compute_psi(
    baseline_counts: list[int],
    baseline_edges: list[float],
    new_values: np.ndarray,
    smoothing: float = 1e-4,
) -> float:
    """Population Stability Index between a baseline histogram and new data.

    PSI = Σ (p_new - p_base) * ln(p_new / p_base), per bin.
    Small smoothing keeps zeros from blowing up the log.

    Returns: PSI value. Common thresholds (planning doc):
        < 0.10  — no drift
        0.10 – 0.20 — moderate drift (watch)
        > 0.20 — significant drift (retrain)
    """
    if len(new_values) == 0:
        return float("nan")
    new_counts, _ = np.histogram(new_values, bins=baseline_edges)
    base_total = max(sum(baseline_counts), 1)
    new_total = max(len(new_values), 1)
    base_props = np.array(baseline_counts, dtype=float) / base_total + smoothing
    new_props = new_counts.astype(float) / new_total + smoothing
    return float(np.sum((new_props - base_props) * np.log(new_props / base_props)))


def check_drift(hours: int = 24) -> dict:
    """Compute PSI per feature using the baked baseline + recent predictions.csv."""
    if not BASELINE_PATH.exists():
        logger.warning("No baseline at %s — skipping PSI drift check", BASELINE_PATH)
        return {"available": False, "reason": f"no baseline at {BASELINE_PATH}"}
    if not PREDICTIONS_LOG.exists():
        return {"available": False, "reason": "no predictions log"}

    baseline = json.loads(BASELINE_PATH.read_text())
    preds = pd.read_csv(PREDICTIONS_LOG, parse_dates=["refreshed_at"])
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    recent = preds[preds["refreshed_at"] >= cutoff]

    per_feature: dict[str, dict] = {}
    for feature, col in PSI_COLUMN_MAP.items():
        if feature not in baseline:
            continue
        if col not in recent.columns:
            per_feature[feature] = {"psi": None, "n_recent": 0, "note": f"{col} missing in predictions log"}
            continue
        values = pd.to_numeric(recent[col], errors="coerce").dropna().values
        if len(values) == 0:
            per_feature[feature] = {"psi": None, "n_recent": 0}
            continue
        psi = compute_psi(
            baseline[feature]["counts"],
            baseline[feature]["bin_edges"],
            values,
        )
        per_feature[feature] = {
            "psi": psi,
            "n_recent": int(len(values)),
            "drift_flag": psi > PSI_ALERT_THRESHOLD,
        }

    any_drift = any(v.get("drift_flag") for v in per_feature.values())
    return {
        "available": True,
        "window_hours": hours,
        "threshold": PSI_ALERT_THRESHOLD,
        "per_feature": per_feature,
        "any_drift": any_drift,
    }


def run(f2_threshold: float = F2_ALERT_THRESHOLD) -> dict:
    joined = load_joined()
    now = datetime.now(timezone.utc)
    cutoff_24h = now - timedelta(hours=24)

    last_24h = joined[joined["target_datetime"] >= cutoff_24h]

    per_station: dict[int, dict] = {}
    for lid, group in last_24h.groupby("location_id"):
        per_station[int(lid)] = _metrics_for(group)

    drift = check_drift(hours=24)

    report = {
        "generated_at": now.isoformat(),
        "total_matched_records": int(len(joined)),
        "last_24h": _metrics_for(last_24h),
        "all_time": _metrics_for(joined),
        "per_station_24h": per_station,
        "drift_24h": drift,
        "alert": False,
    }

    f2 = report["last_24h"]["f2_macro"]
    if report["last_24h"]["n"] > 0 and f2 is not None and f2 < f2_threshold:
        report["alert"] = True
        logger.warning(
            "ALERT: rolling 24h F2 macro (%.3f) below threshold (%.2f) — retraining recommended",
            f2, f2_threshold,
        )
    elif f2 is not None:
        logger.info("Rolling 24h F2 macro: %.3f (threshold %.2f)", f2, f2_threshold)
    else:
        logger.info("Not enough matched records yet for a 24h rolling F2")

    if drift.get("any_drift"):
        report["alert"] = True
        drifted = [k for k, v in drift["per_feature"].items() if v.get("drift_flag")]
        logger.warning(
            "ALERT: PSI > %.2f on features %s — retraining recommended",
            PSI_ALERT_THRESHOLD, drifted,
        )

    # Best-effort email — no-op if SMTP env vars aren't set, never raises
    from src import alerting
    alerting.send_if_alert(report)

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Wrote %s", METRICS_OUT)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    report = run()

    print("\n=== Summary ===")
    print(f"Total matched records: {report['total_matched_records']}")
    print(f"\nLast 24h (n={report['last_24h']['n']}):")
    if report["last_24h"]["n"] > 0:
        print(f"  F2 macro:  {report['last_24h']['f2_macro']:.3f}")
        print(f"  Accuracy:  {report['last_24h']['accuracy']:.3f}")
        for lbl in AQI_LABELS:
            if lbl in report["last_24h"]["f2_per_class"]:
                print(f"  F2 {lbl:12s} {report['last_24h']['f2_per_class'][lbl]:.3f}")

    if report["drift_24h"].get("available"):
        print(f"\nDrift (PSI, 24h window, threshold {report['drift_24h']['threshold']}):")
        for feat, v in report["drift_24h"]["per_feature"].items():
            if v["psi"] is None:
                print(f"  {feat:24s} psi=--     (n={v['n_recent']})")
            else:
                mark = "  DRIFT" if v.get("drift_flag") else ""
                print(f"  {feat:24s} psi={v['psi']:.3f}  (n={v['n_recent']}){mark}")

    if report["alert"]:
        print("\nALERT: threshold crossed — see logs above")
