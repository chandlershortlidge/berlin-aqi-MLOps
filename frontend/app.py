"""Berlin Air Quality Forecast — dashboard.

Streamlit dashboard showing next-hour PM2.5 risk for Berlin monitoring
stations. Pulls /cache for the map + predictions and /history for the
per-station time series. The FastAPI service that backs these endpoints
runs on `API_BASE` (default http://localhost:8000).

Layout sections, all rendered as cards on a light background:

- Hero: what the site is and how to use it.
- Risk levels card: color legend for the 5 AQI categories.
- Map: folium Leaflet map, station markers colored by predicted category,
  sized by model confidence.
- Selected station panel: details for the clicked station.
- Historical trends: recent PM2.5 for the selected station, with a
  pollutant + time-range selector.
"""
from __future__ import annotations

import os
from datetime import datetime

import altair as alt
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

CATEGORY_ORDER = ["All Clear", "Low Risk", "Elevated", "Significant", "High+"]
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
CATEGORY_DESCRIPTIONS = {
    "All Clear":   "Normal conditions. Good for outdoor activity.",
    "Low Risk":    "Mild pollution signal. Most people should be fine.",
    "Elevated":    "Sensitive groups may want to reduce heavy outdoor exercise.",
    "Significant": "Limit prolonged outdoor activity.",
    "High+":       "Avoid heavy outdoor exercise where possible.",
}

POLLUTANT_OPTIONS = ["PM2.5", "PM10", "NO₂", "O₃"]
# /history currently only returns PM2.5 — the refresh cron's actuals log
# doesn't track other pollutants. Non-PM2.5 selections render a "not
# available" state rather than a broken chart.
AVAILABLE_POLLUTANTS = {"PM2.5"}

TIME_RANGES = {
    "Last 24 hours": 24,
    "Last 7 days": 168,
    "Last 30 days": 720,
}
NOT_AVAILABLE = "Not available"


# ---- Data helpers -----------------------------------------------------

@st.cache_data(ttl=60)
def get_cache() -> dict:
    r = httpx.get(f"{API_BASE}/cache", timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def get_history(location_id: int, hours: int) -> list[dict]:
    r = httpx.get(
        f"{API_BASE}/history",
        params={"location_id": location_id, "hours": hours},
        timeout=10,
    )
    r.raise_for_status()
    return r.json().get("points", [])


def format_target(iso_ts: str | None) -> str:
    if not iso_ts:
        return NOT_AVAILABLE
    try:
        return datetime.fromisoformat(iso_ts).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return iso_ts


def freshness_line(age_seconds: int | None) -> str:
    if age_seconds is None:
        return "Freshness unknown"
    age_mins = max(0, age_seconds // 60)
    if age_mins == 0:
        return "Updated just now"
    if age_mins < 60:
        return f"Updated {age_mins} min ago"
    return f"Updated {age_mins // 60}h ago"


def fmt_pm25(v: float | None) -> str:
    if v is None:
        return NOT_AVAILABLE
    return f"{v:.1f} µg/m³"


# ---- Render components -----------------------------------------------

def render_hero() -> None:
    st.markdown(
        """
        <div class="card card-hero">
          <h1>Berlin Air Quality Forecast</h1>
          <p class="subtitle">
            Real-time pollution risk across Berlin monitoring stations.
            Click a station to see its next-hour forecast and recent
            historical readings.
          </p>
          <div class="howto">
            <span class="step">Check the map</span>
            <span class="arrow">→</span>
            <span class="step">Click a station</span>
            <span class="arrow">→</span>
            <span class="step">Review forecast and history</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_levels_card() -> None:
    rows = "".join(
        f"""
        <li>
          <span class="swatch" style="background:{CATEGORY_COLOR[cat]};"></span>
          <span class="level-name">{cat}</span>
          <span class="level-desc">{CATEGORY_DESCRIPTIONS[cat]}</span>
        </li>
        """
        for cat in CATEGORY_ORDER
    )
    st.markdown(
        f"""
        <div class="card card-risk">
          <h3>Risk levels</h3>
          <ul class="risk-list">{rows}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_map(plottable: dict) -> dict | None:
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
    return st_folium(
        m,
        width=None,
        height=520,
        returned_objects=["last_object_clicked"],
        key="map",
    )


def render_legend() -> None:
    swatches = "".join(
        f'<span class="legend-item">'
        f'<span class="swatch" style="background:{CATEGORY_COLOR[cat]};"></span>'
        f'{cat}</span>'
        for cat in CATEGORY_ORDER
    )
    st.markdown(
        f"""
        <div class="legend">
          <div class="legend-title">Risk Levels</div>
          <div class="legend-row">{swatches}</div>
          <div class="legend-note">marker size = model confidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_selected_station_empty() -> None:
    st.markdown(
        """
        <div class="card card-station card-empty">
          <h3>No station selected</h3>
          <p>Click a station marker on the map to see its next-hour
          forecast, model confidence, and latest readings.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_selected_station(p: dict, name: str, selected_lid: int, cache_info: dict) -> None:
    cat = p.get("predicted_category") or NOT_AVAILABLE
    bg = CATEGORY_COLOR.get(cat, "#7f8c8d")
    fg = CATEGORY_TEXT.get(cat, "#ffffff")
    fresh = freshness_line(cache_info.get("newest_age_seconds"))
    target = format_target(p.get("target_datetime"))
    confidence_raw = p.get("confidence")
    conf = f"{confidence_raw * 100:.0f}%" if confidence_raw is not None else NOT_AVAILABLE
    description = CATEGORY_DESCRIPTIONS.get(cat, "")
    rule_banner = (
        '<div class="rule-banner">High+ threshold rule applied (PM2.5 > 150.5 µg/m³)</div>'
        if p.get("rule_override")
        else ""
    )

    # Only PM2.5 is carried in the refresh cache; the other pollutants
    # would need a backend change to surface.
    measurements = [
        ("PM2.5", fmt_pm25(p.get("pm25_current"))),
        ("PM10", NOT_AVAILABLE),
        ("NO₂", NOT_AVAILABLE),
        ("O₃", NOT_AVAILABLE),
    ]
    measurement_rows = "".join(
        f"<div class='measure'><span class='label'>{k}</span>"
        f"<span class='value'>{v}</span></div>"
        for k, v in measurements
    )

    st.markdown(
        f"""
        <div class="card card-station">
          <div class="station-header">
            <h3>{name}</h3>
            <span class="loc-id">location_id {selected_lid}</span>
          </div>

          <div class="forecast-banner" style="background:{bg}; color:{fg};">
            <div class="banner-label">Next-hour forecast</div>
            <div class="banner-cat">{cat}</div>
            <div class="banner-desc">{description}</div>
          </div>

          <div class="meta-grid">
            <div><span class="meta-k">Model confidence</span><span class="meta-v">{conf}</span></div>
            <div><span class="meta-k">Primary pollutant</span><span class="meta-v">PM2.5</span></div>
            <div><span class="meta-k">Forecast for</span><span class="meta-v">{target}</span></div>
            <div><span class="meta-k">Reporting</span><span class="meta-v">{fresh}</span></div>
          </div>

          <h4>Latest measurements</h4>
          <div class="measurements">{measurement_rows}</div>
          {rule_banner}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_historical_trends(selected_lid: int | None, name: str | None) -> None:
    with st.container(border=True):
        if selected_lid is None:
            st.markdown("### Historical trends")
            st.markdown(
                "<p class='empty'>Select a station on the map to view "
                "historical readings.</p>",
                unsafe_allow_html=True,
            )
            return

        st.markdown(f"### Historical Trends — {name}")

        c1, c2 = st.columns([1, 1])
        with c1:
            pollutant = st.selectbox(
                "Pollutant",
                POLLUTANT_OPTIONS,
                index=0,
                key=f"poll-{selected_lid}",
            )
        with c2:
            range_label = st.selectbox(
                "Time range",
                list(TIME_RANGES.keys()),
                index=0,
                key=f"range-{selected_lid}",
            )
        hours = TIME_RANGES[range_label]

        if pollutant not in AVAILABLE_POLLUTANTS:
            st.info(
                f"{pollutant} history is not available for this station — "
                "the hourly refresh job only records PM2.5. "
                "Select PM2.5 to view historical readings."
            )
            return

        with st.spinner("Loading history…"):
            try:
                points = get_history(int(selected_lid), hours=hours)
            except httpx.HTTPError as exc:
                st.error(f"Couldn't load history: {exc}")
                return

        if not points:
            st.info(
                f"No {pollutant} readings in the {range_label.lower()} "
                "for this station. The history grows one point per hourly refresh."
            )
            return

        df = pd.DataFrame(points)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")

        chart = (
            alt.Chart(df)
            .mark_line(color="#2563eb", strokeWidth=2.2, point=False)
            .encode(
                x=alt.X("datetime:T", title=None),
                y=alt.Y("pm25:Q", title=f"{pollutant} (µg/m³)"),
                tooltip=[
                    alt.Tooltip("datetime:T", title="Time", format="%Y-%m-%d %H:%M"),
                    alt.Tooltip("pm25:Q", title=pollutant, format=".1f"),
                ],
            )
            .properties(height=280)
            .configure_axis(labelColor="#475569", titleColor="#334155")
            .configure_view(strokeOpacity=0)
        )
        st.altair_chart(chart, use_container_width=True)

        st.caption(
            f"{len(df)} hourly point(s) · range "
            f"{df['pm25'].min():.1f}–{df['pm25'].max():.1f} µg/m³"
        )


# ---- Page -------------------------------------------------------------

st.set_page_config(page_title="Berlin Air Quality Forecast", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #f5f7fa; }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }
    .card {
      background: #ffffff;
      border-radius: 14px;
      padding: 1.2em 1.4em;
      box-shadow: 0 1px 2px rgba(15,23,42,0.06);
      margin-bottom: 1em;
    }
    .card h1 { font-size: 2.1em; margin: 0 0 0.15em; color: #0f172a; line-height: 1.15; }
    .card h3 { margin: 0 0 0.6em; color: #0f172a; font-size: 1.1em; }
    .card h4 { margin: 1em 0 0.5em; color: #475569; font-size: 0.95em; }
    .card-hero .subtitle { color: #475569; font-size: 1.02em; margin: 0 0 1em; line-height: 1.5; }
    .howto { display: flex; align-items: center; gap: 0.5em; flex-wrap: wrap;
             color: #1d4ed8; font-weight: 500; }
    .howto .step { background: #eff6ff; padding: 0.3em 0.8em; border-radius: 999px;
                   font-size: 0.88em; }
    .howto .arrow { color: #94a3b8; font-weight: 400; }
    .card-risk ul { list-style: none; padding: 0; margin: 0; }
    .card-risk li {
      display: grid; grid-template-columns: 18px 100px 1fr; gap: 0.55em;
      align-items: center; padding: 0.4em 0; border-bottom: 1px solid #f1f5f9;
    }
    .card-risk li:last-child { border-bottom: none; }
    .card-risk .level-name { font-weight: 600; color: #0f172a; font-size: 0.93em; }
    .card-risk .level-desc { color: #475569; font-size: 0.88em; line-height: 1.35; }
    .swatch {
      display: inline-block; width: 14px; height: 14px; border-radius: 50%;
      border: 1px solid rgba(15,23,42,0.15);
    }
    .legend {
      background: #ffffff; border-radius: 12px; padding: 0.8em 1.1em;
      box-shadow: 0 1px 2px rgba(15,23,42,0.06); margin-top: 0.5em;
      margin-bottom: 1em;
    }
    .legend-title { font-weight: 600; color: #0f172a; margin-bottom: 0.35em; }
    .legend-row { display: flex; gap: 1em; flex-wrap: wrap; color: #334155; font-size: 0.92em; }
    .legend-item { display: inline-flex; align-items: center; gap: 0.35em; }
    .legend-note { color: #64748b; font-size: 0.82em; margin-top: 0.4em; font-style: italic; }
    .card-station .station-header {
      display: flex; align-items: baseline; justify-content: space-between; gap: 0.5em;
    }
    .card-station .loc-id { color: #94a3b8; font-size: 0.78em; }
    .forecast-banner {
      border-radius: 12px; padding: 1.2em 1em; text-align: center;
      margin: 0.8em 0 1em; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .forecast-banner .banner-label {
      font-size: 0.78em; letter-spacing: 0.08em; opacity: 0.85; text-transform: uppercase;
    }
    .forecast-banner .banner-cat {
      font-size: 2.2em; font-weight: 700; margin: 0.15em 0 0.3em; line-height: 1.1;
    }
    .forecast-banner .banner-desc { font-size: 0.93em; opacity: 0.95; line-height: 1.4; }
    .meta-grid {
      display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.6em 1em;
      margin-bottom: 0.3em;
    }
    .meta-grid > div { display: flex; flex-direction: column; gap: 0.15em; }
    .meta-k {
      color: #475569; font-size: 0.72em; text-transform: uppercase;
      letter-spacing: 0.05em; font-weight: 600;
    }
    .meta-v { color: #0f172a; font-size: 0.95em; }
    .measurements { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5em; }
    .measure {
      display: flex; justify-content: space-between; padding: 0.45em 0.7em;
      background: #f8fafc; border-radius: 8px; font-size: 0.92em;
    }
    .measure .label { color: #475569; font-weight: 500; }
    .measure .value { color: #0f172a; }
    .rule-banner {
      background: #fef3c7; color: #92400e; padding: 0.5em 0.8em;
      border-radius: 8px; margin-top: 0.8em; font-size: 0.85em;
    }
    .card-empty p { color: #475569; line-height: 1.5; }
    .empty { color: #64748b; font-style: italic; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Top row: hero + risk card (stack on mobile automatically)
col_hero, col_risk = st.columns([3, 2])
with col_hero:
    render_hero()
with col_risk:
    render_risk_levels_card()

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

# Map + selected-station panel
col_map, col_station = st.columns([3, 2])
with col_map:
    map_data = render_map(plottable)
    render_legend()

coord_to_lid = {
    (round(p["latitude"], 5), round(p["longitude"], 5)): lid
    for lid, p in plottable.items()
}
clicked = (map_data or {}).get("last_object_clicked") if map_data else None
selected_lid: int | None = None
selected_name: str | None = None
if clicked:
    key = (round(clicked["lat"], 5), round(clicked["lng"], 5))
    raw_lid = coord_to_lid.get(key)
    if raw_lid is not None:
        selected_lid = int(raw_lid)
        selected_name = STATION_NAMES.get(selected_lid, f"Station {selected_lid}")

with col_station:
    if selected_lid is None:
        render_selected_station_empty()
    else:
        render_selected_station(
            plottable[str(selected_lid)], selected_name or "", selected_lid, cache_info
        )

# Historical trends section
render_historical_trends(selected_lid, selected_name)

if missing_coords:
    st.caption(
        f"{len(missing_coords)} station(s) hidden from the map — "
        "their cache entry predates lat/lon baking. "
        "Run `python -m src.refresh` to repopulate."
    )
