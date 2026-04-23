"""Berlin AQI — athlete dashboard (Folium map edition).

Interactive Berlin map. Each station is a circle marker coloured by the
predicted next-hour category and sized by model confidence. Click a
marker to see full prediction detail in the sidebar.

Data source: GET /cache on the FastAPI backend. That endpoint now ships
the full prediction dict (including latitude + longitude per station),
so this view is one HTTP round trip per rerun.

Run locally:
    uv run streamlit run frontend/app.py
Set API_BASE to point at the deployed backend:
    API_BASE=http://3.71.44.98:8000 uv run streamlit run frontend/app.py
"""
from __future__ import annotations

import os
from datetime import datetime

import folium
import httpx
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
    "All Clear":   "#2ecc71",  # green
    "Low Risk":    "#f1c40f",  # yellow
    "Elevated":    "#e67e22",  # orange
    "Significant": "#e74c3c",  # red
    "High+":       "#8b0000",  # dark red
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


def format_target(iso_ts: str) -> str:
    try:
        return datetime.fromisoformat(iso_ts).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return iso_ts


st.set_page_config(page_title="Berlin AQI — Athletes", layout="wide")
st.title("Berlin AQI")
st.caption(f"Next-hour forecast for Berlin athletes · {API_BASE}")

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

# Only stations with coordinates can be plotted. Older cache entries
# (pre-lat/lon-baking) show up in the sidebar list as fallback.
plottable = {
    lid: p for lid, p in predictions.items()
    if p.get("latitude") is not None and p.get("longitude") is not None
}
missing_coords = set(predictions) - set(plottable)

# Build the map
m = folium.Map(location=BERLIN_CENTER, zoom_start=DEFAULT_ZOOM, tiles="OpenStreetMap")

for lid, p in plottable.items():
    cat = p["predicted_category"]
    name = STATION_NAMES.get(int(lid), f"Station {lid}")
    radius = 6 + p["confidence"] * 14  # 6px at conf=0 → 20px at conf=1

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

# Two-column layout: map on the left, detail panel on the right.
map_col, detail_col = st.columns([3, 2], gap="large")

with map_col:
    map_data = st_folium(
        m,
        width=None,
        height=560,
        returned_objects=["last_object_clicked"],
        key="map",
    )

# Reverse-lookup the clicked marker by its (lat, lng) rounded to 5 decimals.
coord_to_lid = {
    (round(p["latitude"], 5), round(p["longitude"], 5)): lid
    for lid, p in plottable.items()
}

clicked = (map_data or {}).get("last_object_clicked") if map_data else None
selected_lid = None
if clicked:
    key = (round(clicked["lat"], 5), round(clicked["lng"], 5))
    selected_lid = coord_to_lid.get(key)

with detail_col:
    st.subheader("Station detail")
    if selected_lid is None:
        st.info("Click a station on the map to see its next-hour forecast.")
    else:
        p = plottable[selected_lid]
        cat = p["predicted_category"]
        name = STATION_NAMES.get(int(selected_lid), f"Station {selected_lid}")
        bg = CATEGORY_COLOR.get(cat, "#7f8c8d")
        fg = CATEGORY_TEXT.get(cat, "#ffffff")

        st.markdown(f"**{name}** (location_id {selected_lid})")
        st.markdown(
            f"""
            <div style="background-color:{bg}; color:{fg}; padding: 1.2em 1em;
                        border-radius: 10px; text-align: center; margin: 0.4em 0 0.8em 0;">
              <div style="font-size: 0.85em; opacity: 0.85;">Next-hour forecast</div>
              <div style="font-size: 2em; font-weight: 700; margin-top: 0.1em;">{cat}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"**{RECOMMENDATIONS.get(cat, '')}**")

        col1, col2, col3 = st.columns(3)
        col1.metric("PM2.5", f"{p['pm25_current']:.1f} µg/m³")
        col2.metric("Confidence", f"{p['confidence'] * 100:.1f}%")
        col3.metric(
            "Data age",
            f"{(cache_info.get('newest_age_seconds') or 0) // 60} min",
        )

        footer = [f"Target: {format_target(p['target_datetime'])}"]
        if p.get("rule_override"):
            footer.append("High+ threshold rule applied")
        st.caption(" · ".join(footer))

# Legend
legend_items = [
    f'<span style="display:inline-block; width:12px; height:12px; background:{c}; '
    f'border-radius:50%; margin-right:4px;"></span>{name}'
    for name, c in CATEGORY_COLOR.items()
]
st.markdown(
    "**Legend** &nbsp;&nbsp;" + " &nbsp;&nbsp; ".join(legend_items)
    + " &nbsp;&nbsp; • &nbsp;&nbsp; *marker size = model confidence*",
    unsafe_allow_html=True,
)

if missing_coords:
    st.caption(
        f"{len(missing_coords)} station(s) hidden from the map — "
        "their cache entry predates lat/lon baking. Run `python -m src.refresh` to repopulate."
    )
