"""Email alerting for monitor threshold crossings.

Called from `monitor.run()` at the end of a monitoring cycle. Sends
exactly one email when the alert flag is set AND the throttle window
has elapsed since the last send. If SMTP config is missing, no-ops
with an INFO log so local dev and CI aren't blocked.

Required env vars:
    SMTP_HOST             smtp.gmail.com / email-smtp.<region>.amazonaws.com / etc.
    SMTP_USER
    SMTP_PASSWORD         Use an app password for Gmail.
    ALERT_EMAIL_TO        Comma-separated recipients.
    ALERT_EMAIL_FROM      From: header.

Optional:
    SMTP_PORT             Default 587 (STARTTLS).
    ALERT_MIN_INTERVAL_SECONDS  Minimum seconds between sends, default 3600.
"""
from __future__ import annotations

import json
import logging
import os
import smtplib
import ssl
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_PATH = Path(__file__).resolve().parents[1] / "data" / "monitoring" / "alert_state.json"
DEFAULT_MIN_INTERVAL = 3600
SMTP_TIMEOUT_SECONDS = 30


def _smtp_config() -> dict | None:
    cfg = {
        "host": os.getenv("SMTP_HOST"),
        "port": int(os.getenv("SMTP_PORT", "587")),
        "user": os.getenv("SMTP_USER"),
        "password": os.getenv("SMTP_PASSWORD"),
        "to": os.getenv("ALERT_EMAIL_TO"),
        "from": os.getenv("ALERT_EMAIL_FROM"),
    }
    missing = [k for k in ("host", "user", "password", "to", "from") if not cfg[k]]
    if missing:
        logger.debug("SMTP config missing fields: %s — skipping email", missing)
        return None
    return cfg


def _read_last_sent() -> datetime | None:
    if not STATE_PATH.exists():
        return None
    try:
        return datetime.fromisoformat(json.loads(STATE_PATH.read_text())["last_sent_at"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def _write_last_sent(when: datetime) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps({"last_sent_at": when.isoformat()}))


def format_subject(report: dict) -> str:
    parts = []
    last = report.get("last_24h", {})
    drift = report.get("drift_24h", {})
    if last.get("f2_macro") is not None and report.get("alert"):
        # Distinguish which threshold fired
        from src.monitor import F2_ALERT_THRESHOLD
        if last["f2_macro"] < F2_ALERT_THRESHOLD:
            parts.append(f"F2 {last['f2_macro']:.2f}")
    if drift.get("any_drift"):
        drifted = [f for f, v in drift["per_feature"].items() if v.get("drift_flag")]
        parts.append(f"drift: {', '.join(drifted)}")
    suffix = " — ".join(parts) if parts else "alert"
    return f"[Berlin AQI] {suffix}"


def format_body(report: dict) -> str:
    lines = [f"Berlin AQI monitor — generated_at {report['generated_at']}", ""]
    last = report.get("last_24h", {})
    if last.get("n", 0) > 0:
        lines.append(f"Rolling 24h  n={last['n']}  F2 macro={last['f2_macro']:.3f}  accuracy={last['accuracy']:.3f}")
        for lbl, score in sorted(last.get("f2_per_class", {}).items()):
            lines.append(f"             F2 {lbl:12s} {score:.3f}")
    else:
        lines.append("Rolling 24h: no matched (prediction, actual) pairs yet.")

    drift = report.get("drift_24h", {})
    if drift.get("available"):
        lines += ["", f"PSI drift (24h, threshold {drift['threshold']}):"]
        for feat, v in drift["per_feature"].items():
            if v["psi"] is None:
                lines.append(f"  {feat:24s} --          n_recent={v['n_recent']}")
            else:
                tag = "  DRIFT" if v.get("drift_flag") else ""
                lines.append(f"  {feat:24s} psi={v['psi']:6.3f}  n_recent={v['n_recent']}{tag}")

    lines += ["", "Full report: /app/data/monitoring/metrics.json",
                  "Runtime log: /var/log/berlin-aqi-monitor.log on EC2"]
    return "\n".join(lines)


def send_if_alert(report: dict, min_interval_seconds: int | None = None) -> bool:
    """Return True iff an email was actually sent (not suppressed/skipped)."""
    if not report.get("alert"):
        return False

    cfg = _smtp_config()
    if cfg is None:
        logger.info("Alert raised but SMTP not configured — no email sent")
        return False

    interval = min_interval_seconds or int(
        os.getenv("ALERT_MIN_INTERVAL_SECONDS", str(DEFAULT_MIN_INTERVAL))
    )
    last_sent = _read_last_sent()
    now = datetime.now(timezone.utc)
    if last_sent and (now - last_sent).total_seconds() < interval:
        logger.info(
            "Alert suppressed (throttle): last email %ds ago, min %ds",
            int((now - last_sent).total_seconds()), interval,
        )
        return False

    msg = EmailMessage()
    msg["Subject"] = format_subject(report)
    msg["From"] = cfg["from"]
    msg["To"] = cfg["to"]
    msg.set_content(format_body(report))

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=SMTP_TIMEOUT_SECONDS) as server:
            server.starttls(context=ctx)
            server.login(cfg["user"], cfg["password"])
            server.send_message(msg)
    except Exception:
        # Don't let a mail server hiccup take the monitor down.
        logger.exception("Failed to send alert email")
        return False

    _write_last_sent(now)
    logger.info("Sent alert email to %s", cfg["to"])
    return True
