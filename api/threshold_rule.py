"""Serving-layer override for the High+ AQI category.

The model cannot learn the High+ class reliably (5 training examples across
2 years / 15 stations — see CLAUDE.md's "Known limitations"). Instead, the
API applies a deterministic PM2.5 threshold rule as a safety net so extreme
events never slip past.

The rule uses current PM2.5 (at time t) as a proxy for t+1: if current PM2.5
is already above the High+ bin boundary, conditions almost certainly persist
into the next hour.
"""
from __future__ import annotations

HIGH_PLUS_THRESHOLD_UGM3 = 150.5
HIGH_PLUS_LABEL = "High+"


def apply(predicted_category: str, pm25_current: float) -> tuple[str, bool]:
    """Return (category, rule_fired). Overrides to High+ when PM2.5 >= 150.5."""
    if pm25_current >= HIGH_PLUS_THRESHOLD_UGM3:
        return HIGH_PLUS_LABEL, True
    return predicted_category, False
