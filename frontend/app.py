"""Berlin AQI — athlete dashboard.

Interactive Berlin map with a stacked detail panel below. Clicking a
marker reveals a large forecast banner, recommendation text, data
freshness, key metrics, and a 24-hour PM2.5 sparkline.

Data sources: GET /cache (coords + current prediction) and
GET /history?location_id=X (hourly PM2.5 from the refresh cron's
actuals log).
"""
from __future__ import annotations

import os
from datetime import datetime

import folium
import httpx
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

BERLIN_CENTER = (52.52, 13.41)
DEFAULT_ZOOM = 11

STATION_NAMES = {
    2993: "Berlin Neukölln",
    3019: "Berlin Mitte",
    3025: "DEBE063",
    3050: "Potsdam, Großbeerenstr.",
    3096: "Potsdam, Groß Glienicke",
    4582: "Berlin Grunewald",
    4724: "Blankenfelde-Mahlow",
    4761: "Berlin Wedding",
    4762: "Berlin Schildhornstraße",
    4764: "Berlin Mariendorfer Damm",
    4767: "Berlin Frankfurter Allee",
    4768: "Berlin Friedrichshagen",
    4769: "DEBE051",
    2162178: "Berlin Leipziger Straße",
    2162179: "Berlin Buch",
    2162180: "Berlin Karl-Marx-Straße II",
    2162181: "Berlin Silbersteinstraße 5",
}

CATEGORY_COLOR = {
    "All Clear":   "#2ecc71",
    "Low Risk":    "#f1c40f",
    "Elevated":    "#e67e22",
    "Significant": "#e74c3c",
    "High+":       "#8b0000",
}
CATEGORY_TEXT = {
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


@st.cache_data(ttl=60)
def get_history(location_id: int, hours: int = 24) -> list[dict]:
    r = httpx.get(
        f"{API_BASE}/history",
        params={"location_id": location_id, "hours": hours},
        timeout=10,
    )
    r.raise_for_status()
    return r.json().get("points", [])


def format_target(iso_ts: str) -> str:
    try:
        return datetime.fromisoformat(iso_ts).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return iso_ts


st.set_page_config(page_title="Berlin AQI", layout="wide")

# Prominent header, no API URL in the subtitle.
st.markdown(
    """
    <div style="margin-bottom: 1.25em;">
      <h1 style="font-size: 2.6em; margin-bottom: 0.1em;">Berlin AQI</h1>
      <div style="color: #666; font-size: 1.15em;">
        Next-hour air quality forecast for Berlin athletes
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    cache_info = get_cache()
except httpx.HTTPError as exc:
    st.error(f"Cannot reach API at {API_BASE}\n\n{exc}")
    st.stop()

predictions = cache_info.get("predictions", {})
if not predictions:
    st.warning(
        "No predictions cached yet. Run `python -m src.refresh` inside the "
        "container (needs OPENAQ_API_KEY) to populate the cache."
    )
    st.stop()

plottable = {
    lid: p for lid, p in predictions.items()
    if p.get("latitude") is not None and p.get("longitude") is not None
}
missing_coords = set(predictions) - set(plottable)

# ---- Map ------------------------------------------------------------
m = folium.Map(location=BERLIN_CENTER, zoom_start=DEFAULT_ZOOM, tiles="OpenStreetMap")
for lid, p in plottable.items():
    cat = p["predicted_category"]
    name = STATION_NAMES.get(int(lid), f"Station {lid}")
    radius = 6 + p["confidence"] * 14

    popup_html = (
        f"<div style='font-family: sans-serif; min-width: 180px;'>"
        f"<div style='font-weight: 700; font-size: 1.05em;'>{name}</div>"
        f"<div style='margin-top: 4px;'><b>{cat}</b></div>"
        f"<div style='color: #555; font-size: 0.9em;'>"
        f"PM2.5 {p['pm25_current']:.1f} µg/m³ · conf {p['confidence']*100:.0f}%"
        f"</div></div>"
    )
    folium.CircleMarker(
        location=[p["latitude"], p["longitude"]],
        radius=radius,
        color="#111",
        weight=1,
        fill=True,
        fill_color=CATEGORY_COLOR.get(cat, "#7f8c8d"),
        fill_opacity=0.85,
        tooltip=f"{name} — {cat}",
        popup=folium.Popup(popup_html, max_width=260),
    ).add_to(m)

map_data = st_folium(
    m,
    width=None,
    height=520,
    returned_objects=["last_object_clicked"],
    key="map",
)

# Legend under the map
legend_items = [
    f'<span style="display:inline-block; width:12px; height:12px; background:{c}; '
    f'border-radius:50%; margin-right:4px; vertical-align: middle;"></span>{name}'
    for name, c in CATEGORY_COLOR.items()
]
st.markdown(
    "<div style='margin: 0.8em 0 1.5em 0; color: #555;'>"
    + " &nbsp;&nbsp; ".join(legend_items)
    + " &nbsp;&nbsp;·&nbsp;&nbsp; <em>marker size = model confidence</em>"
    + "</div>",
    unsafe_allow_html=True,
)

# ---- Station detail (below the map) --------------------------------
st.divider()

coord_to_lid = {
    (round(p["latitude"], 5), round(p["longitude"], 5)): lid
    for lid, p in plottable.items()
}
clicked = (map_data or {}).get("last_object_clicked") if map_data else None
selected_lid = None
if clicked:
    key = (round(clicked["lat"], 5), round(clicked["lng"], 5))
    selected_lid = coord_to_lid.get(key)

if selected_lid is None:
    st.info("Click a station on the map to see its next-hour forecast.")
else:
    p = plottable[selected_lid]
    cat = p["predicted_category"]
    name = STATION_NAMES.get(int(selected_lid), f"Station {selected_lid}")
    bg = CATEGORY_COLOR.get(cat, "#7f8c8d")
    fg = CATEGORY_TEXT.get(cat, "#ffffff")

    st.markdown(f"### {name}  <span style='color:#999; font-size:0.7em;'>location_id {selected_lid}</span>",
                unsafe_allow_html=True)

    # Prominent forecast banner
    st.markdown(
        f"""
        <div style="background-color:{bg}; color:{fg}; padding: 2em 1em;
                    border-radius: 12px; text-align: center; margin: 0.6em 0 1em 0;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
          <div style="font-size: 0.95em; letter-spacing: 0.05em; opacity: 0.85;
                      text-transform: uppercase;">Next hour forecast</div>
          <div style="font-size: 3.2em; font-weight: 700; margin-top: 0.1em; line-height: 1.1;">
            {cat}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Freshness + recommendation
    age_s = cache_info.get("newest_age_seconds") or 0
    age_mins = max(0, age_s // 60)
    fresh_line = f"Updated {age_mins} min ago"
    if age_mins == 0:
        fresh_line = "Updated just now"

    st.markdown(
        f"""
        <div style="font-size: 1.1em; margin-bottom: 0.4em;">
          <strong>{RECOMMENDATIONS.get(cat, '')}</strong>
        </div>
        <div style="color: #777; font-size: 0.9em; margin-bottom: 1em;">
          {fresh_line} · Target: {format_target(p['target_datetime'])}
          {' · High+ threshold rule applied' if p.get('rule_override') else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Current PM2.5", f"{p['pm25_current']:.1f} µg/m³")
    col2.metric("Model confidence", f"{p['confidence'] * 100:.1f}%")
    col3.metric("Category", cat)

    # 24h sparkline
    st.markdown("#### PM2.5 — last 24 hours")
    try:
        history_points = get_history(int(selected_lid), hours=24)
    except httpx.HTTPError as exc:
        st.error(f"Couldn't load history: {exc}")
        history_points = []

    if not history_points:
        st.caption(
            "No historical readings yet — the hourly refresh cron builds "
            "up this series one point per run."
        )
    else:
        hist_df = pd.DataFrame(history_points)
        hist_df["datetime"] = pd.to_datetime(hist_df["datetime"])
        hist_df = hist_df.set_index("datetime").sort_index()
        st.line_chart(hist_df["pm25"], height=220)
        st.caption(
            f"{len(hist_df)} hourly point(s) · range "
            f"{hist_df['pm25'].min():.1f}–{hist_df['pm25'].max():.1f} µg/m³"
        )

if missing_coords:
    st.caption(
        f"{len(missing_coords)} station(s) hidden from the map — "
        "their cache entry predates lat/lon baking. Run `python -m src.refresh` to repopulate."
    )
