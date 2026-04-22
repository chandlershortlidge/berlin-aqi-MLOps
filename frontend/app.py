"""Berlin AQI — athlete dashboard.

Run locally:
    uv run streamlit run frontend/app.py

Talks to the FastAPI service via two endpoints:
- GET /cache    -> which stations have fresh predictions
- GET /predict  -> per-station next-hour forecast

Set API_BASE=http://3.71.44.98:8000 to point at the deployed backend.
"""
from __future__ import annotations

import os
from datetime import datetime

import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# Friendly names for the 17 eligible stations (two are decommissioned —
# they won't appear in the cache). Unknown IDs fall back to "Station <id>".
STATION_NAMES = {
    2993: "Berlin Neukölln",
    3019: "Berlin Mitte",
    3050: "Potsdam, Großbeerenstr.",
    3096: "Potsdam, Groß Glienicke",
    4582: "Berlin Grunewald",
    4724: "Blankenfelde-Mahlow",
    4761: "Berlin Wedding",
    4762: "Berlin Schildhornstraße",
    4764: "Berlin Mariendorfer Damm",
    4767: "Berlin Frankfurter Allee",
    4768: "Berlin Friedrichshagen",
    2162178: "Berlin Leipziger Straße",
    2162179: "Berlin Buch",
    2162180: "Berlin Karl-Marx-Straße II",
    2162181: "Berlin Silbersteinstraße 5",
}

# Category-level UI styling + plain-language guidance for athletes.
CATEGORY_BG = {
    "All Clear":   "#2ecc71",
    "Low Risk":    "#f1c40f",
    "Elevated":    "#e67e22",
    "Significant": "#e74c3c",
    "High+":       "#8e44ad",
}
CATEGORY_FG = {
    "All Clear":   "#ffffff",
    "Low Risk":    "#1a1a1a",
    "Elevated":    "#ffffff",
    "Significant": "#ffffff",
    "High+":       "#ffffff",
}
RECOMMENDATIONS = {
    "All Clear":   "Ideal conditions for outdoor training.",
    "Low Risk":    "Fine for most. Sensitive athletes may notice irritation on long sessions.",
    "Elevated":    "Reduce prolonged outdoor exertion. Keep sessions shorter.",
    "Significant": "Move training indoors — air quality is meaningfully degraded.",
    "High+":       "Avoid outdoor exercise. Stay indoors and close windows.",
}


@st.cache_data(ttl=60)
def get_cache() -> dict:
    r = httpx.get(f"{API_BASE}/cache", timeout=10)
    r.raise_for_status()
    return r.json()


def get_predict(location_id: int) -> dict:
    r = httpx.get(f"{API_BASE}/predict", params={"location_id": location_id}, timeout=10)
    r.raise_for_status()
    return r.json()


st.set_page_config(page_title="Berlin AQI — Athletes", layout="centered")
st.title("Berlin AQI")
st.caption("Next-hour air quality forecast for Berlin athletes · " + API_BASE)

try:
    cache_info = get_cache()
except httpx.HTTPError as exc:
    st.error(f"Cannot reach API at {API_BASE}\n\n{exc}")
    st.stop()

if not cache_info.get("exists") or cache_info.get("stations", 0) == 0:
    st.warning(
        "No predictions cached yet. Run `python -m src.refresh` inside the "
        "container (needs OPENAQ_API_KEY) to populate the cache."
    )
    st.stop()

cached_ids = sorted(cache_info["station_ids"])
labels = [
    f"{STATION_NAMES.get(lid, f'Station {lid}')} ({lid})" for lid in cached_ids
]
id_by_label = dict(zip(labels, cached_ids))

selected_label = st.selectbox("Neighbourhood", options=labels)
selected_id = id_by_label[selected_label]

try:
    pred = get_predict(selected_id)
except httpx.HTTPError as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

cat = pred["predicted_category"]
bg = CATEGORY_BG.get(cat, "#7f8c8d")
fg = CATEGORY_FG.get(cat, "#ffffff")

st.markdown(
    f"""
    <div style="background-color:{bg}; color:{fg}; padding: 2.5em 1em;
                border-radius: 12px; text-align: center; margin: 1em 0;">
      <div style="font-size: 0.95em; opacity: 0.85;">Next-hour forecast</div>
      <div style="font-size: 3em; font-weight: 700; margin-top: 0.15em;">{cat}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"**{RECOMMENDATIONS.get(cat, '')}**")

col1, col2, col3 = st.columns(3)
col1.metric("Current PM2.5", f"{pred['pm25_current']:.1f} µg/m³")
col2.metric("Confidence", f"{pred['confidence'] * 100:.1f}%")
col3.metric("Data age", f"{pred['age_seconds'] // 60} min")

target = pred["target_datetime"]
try:
    target_display = datetime.fromisoformat(target).strftime("%Y-%m-%d %H:%M UTC")
except ValueError:
    target_display = target

footer = [f"Target hour: {target_display}", f"Source: {pred['source']}"]
if pred.get("rule_override"):
    footer.append("High+ threshold rule applied")
st.caption(" · ".join(footer))
