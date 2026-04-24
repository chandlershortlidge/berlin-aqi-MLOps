"""Berlin Air Quality — dashboard.

Streamlit app for next-hour PM2.5 risk in Berlin. Pulls /cache for the
map + predictions and /history for the per-station time series (served
from OpenAQ v3 via the backend).

Layout:

- Hero: title, subtitle, live "Updated HH:MM" badge, 3 numbered steps.
- Map (full width) with color legend immediately below.
- "What the levels mean" — 5-card row describing each risk level.
- Historical readings: station-driven, pollutant + time range selectors,
  line chart + four summary stat cards.
- Footer disclaimer.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

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
    "All Clear":   "#3B6D11",
    "Low Risk":    "#D4A017",
    "Elevated":    "#D85A30",
    "Significant": "#A32D2D",
    "High+":       "#501313",
}
CATEGORY_TEXT = {cat: "#ffffff" for cat in CATEGORY_ORDER}

# Spec copy for the "What the levels mean" row.
LEVEL_CARDS = [
    {
        "name": "All clear",
        "category": "All Clear",
        "range": "PM2.5 < 12 µg/m³",
        "desc": "Ideal for outdoor exercise and extended time outside.",
    },
    {
        "name": "Low risk",
        "category": "Low Risk",
        "range": "12–35 µg/m³",
        "desc": "Acceptable for most. Sensitive individuals may notice mild irritation.",
    },
    {
        "name": "Elevated",
        "category": "Elevated",
        "range": "35–55 µg/m³",
        "desc": "Reduce prolonged outdoor exertion. Asthmatics should take care.",
    },
    {
        "name": "Significant",
        "category": "Significant",
        "range": "55–150 µg/m³",
        "desc": "Avoid extended outdoor activity. Mask recommended if going out.",
    },
    {
        "name": "High+",
        "category": "High+",
        "range": "> 150 µg/m³",
        "desc": "Hazardous. Stay indoors. Air purifiers recommended.",
    },
]

POLLUTANT_OPTIONS = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "NO₂": "no2",
}
TIME_RANGES = {
    "Last 24 hours": 24,
    "Last 7 days": 168,
    "Last 30 days": 720,
}
STATS_WINDOW_HOURS = 168  # 7 days, for the summary stat cards
WHO_24H_PM25 = 15.0  # µg/m³
NOT_AVAILABLE = "Not available"


# ---- Data helpers -----------------------------------------------------

@st.cache_data(ttl=60)
def get_cache() -> dict:
    r = httpx.get(f"{API_BASE}/cache", timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300)
def get_history(location_id: int, hours: int, parameter: str) -> dict:
    """Fetch history + return the full envelope so we can surface debug info."""
    url = f"{API_BASE}/history"
    params = {"location_id": location_id, "hours": hours, "parameter": parameter}
    print(f"[history] GET {url} params={params}", flush=True)
    r = httpx.get(url, params=params, timeout=30)
    status = r.status_code
    print(f"[history] <- status={status}", flush=True)
    r.raise_for_status()
    body = r.json()
    print(f"[history] <- points={len(body.get('points', []))}", flush=True)
    return {
        "points": body.get("points", []),
        "debug": {
            "url": str(r.request.url),
            "status": status,
            "point_count": len(body.get("points", [])),
        },
    }


def freshness_label(age_seconds: int | None) -> str:
    """Render the Updated HH:MM badge text in local-ish (UTC) clock form."""
    now = datetime.now(timezone.utc)
    if age_seconds is None:
        return now.strftime("Updated %H:%M UTC")
    updated_at = now - timedelta(seconds=age_seconds)
    return updated_at.strftime("Updated %H:%M UTC")


# ---- Render components -----------------------------------------------

def render_hero(cache_info: dict | None) -> None:
    age = (cache_info or {}).get("newest_age_seconds") if cache_info else None
    badge = freshness_label(age)
    st.markdown(
        f"""
        <section class="hero">
          <div class="hero-top">
            <div class="hero-heading">
              <h1>Berlin Air Quality</h1>
              <p class="subtitle">
                Real-time PM2.5 forecasts for Berlin monitoring stations —
                powered by machine learning. Designed for cyclists, runners,
                and anyone planning outdoor time.
              </p>
            </div>
            <div class="update-badge">{badge}</div>
          </div>
          <ol class="howto">
            <li><span class="step-num">1</span>
                Click a station on the map to see its next-hour forecast.</li>
            <li><span class="step-num">2</span>
                Marker size reflects model confidence — larger = more certain.</li>
            <li><span class="step-num">3</span>
                Historical readings for that station appear below automatically.</li>
          </ol>
        </section>
        """,
        unsafe_allow_html=True,
    )


NAME_TO_LID = {name: lid for lid, name in STATION_NAMES.items()}


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
        # Tooltip is exactly the station name. We read it back from
        # `last_object_clicked_tooltip` on click and look up the ID —
        # which, unlike `last_object_clicked`, only populates for actual
        # marker clicks, not generic tile clicks.
        folium.CircleMarker(
            location=[p["latitude"], p["longitude"]],
            radius=radius,
            color="#111",
            weight=1,
            fill=True,
            fill_color=CATEGORY_COLOR.get(cat, "#7f8c8d"),
            fill_opacity=0.85,
            tooltip=name,
            popup=folium.Popup(popup_html, max_width=260),
        ).add_to(m)
    return st_folium(
        m,
        width=None,
        height=520,
        returned_objects=["last_object_clicked", "last_object_clicked_tooltip"],
        key="map",
    )


def _lid_from_tooltip(tooltip: str | None) -> int | None:
    if not tooltip:
        return None
    return NAME_TO_LID.get(tooltip.strip())


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
          <div class="legend-row">{swatches}</div>
          <div class="legend-note">marker size = model confidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_level_cards() -> None:
    cards = "".join(
        f"""
        <div class="level-card">
          <div class="level-bar" style="background:{CATEGORY_COLOR[c['category']]};"></div>
          <div class="level-body">
            <div class="level-name">{c['name']}</div>
            <div class="level-range">{c['range']}</div>
            <div class="level-desc">{c['desc']}</div>
          </div>
        </div>
        """
        for c in LEVEL_CARDS
    )
    st.markdown(
        f"""
        <section class="levels">
          <h2>What the levels mean</h2>
          <div class="level-grid">{cards}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _empty_history_section() -> None:
    st.markdown(
        """
        <section class="history">
          <div class="history-header">
            <h2>Historical readings</h2>
            <span class="station-pill station-pill-empty">No station selected</span>
          </div>
          <div class="history-empty">
            Click a station on the map to load its historical readings.
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _history_stats(points_7d: list[dict]) -> tuple[str, str, str]:
    """Returns (7d avg, 7d peak, days all clear) as formatted strings."""
    if not points_7d:
        return NOT_AVAILABLE, NOT_AVAILABLE, "0 / 7"

    df = pd.DataFrame(points_7d)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    avg = df["value"].mean()
    peak = df["value"].max()

    # "Days all clear" = number of calendar days (UTC) in the window whose
    # daily-max PM2.5 stays below the All Clear threshold.
    df["day"] = df["datetime"].dt.floor("D")
    daily_max = df.groupby("day")["value"].max()
    days_clear = int((daily_max < 12).sum())
    days_total = int(daily_max.shape[0])

    return (
        f"{avg:.1f} µg/m³",
        f"{peak:.1f} µg/m³",
        f"{days_clear} / {days_total}",
    )


def render_history(selected_lid: int | None, selected_name: str | None,
                   plottable: dict) -> None:
    if selected_lid is None:
        _empty_history_section()
        return

    pill_color = CATEGORY_COLOR.get(
        plottable.get(str(selected_lid), {}).get("predicted_category", ""),
        "#475569",
    )

    st.markdown(
        f"""
        <section class="history">
          <div class="history-header">
            <h2>Historical readings</h2>
            <span class="station-pill" style="background:{pill_color};">{selected_name}</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        pollutant_label = st.selectbox(
            "Pollutant",
            list(POLLUTANT_OPTIONS.keys()),
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

    parameter = POLLUTANT_OPTIONS[pollutant_label]
    range_hours = TIME_RANGES[range_label]
    # Fetch enough history for both the chart and the 7-day stat cards.
    fetch_hours = max(range_hours, STATS_WINDOW_HOURS)

    with st.spinner("Loading history from OpenAQ…"):
        try:
            result = get_history(int(selected_lid), hours=fetch_hours, parameter=parameter)
        except httpx.HTTPError as exc:
            st.error(f"Couldn't load history: {exc}")
            with st.expander("Debug — last fetch attempt"):
                st.write({
                    "location_id": int(selected_lid),
                    "parameter": parameter,
                    "hours": fetch_hours,
                    "error": str(exc),
                })
            return

    points = result["points"]
    dbg = result["debug"]

    if not points:
        st.info(
            f"No {pollutant_label} readings available for this station "
            f"in the {range_label.lower()}."
        )
        with st.expander("Debug — last fetch attempt"):
            st.write({
                "location_id": int(selected_lid),
                "parameter": parameter,
                "hours": fetch_hours,
                "request_url": dbg["url"],
                "http_status": dbg["status"],
                "point_count": dbg["point_count"],
            })
        return

    df = pd.DataFrame(points)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime")

    # Chart uses only the user's selected range; stats always use last 7 days.
    cutoff_chart = datetime.now(timezone.utc) - timedelta(hours=range_hours)
    chart_df = df[df["datetime"] >= cutoff_chart]
    cutoff_stats = datetime.now(timezone.utc) - timedelta(hours=STATS_WINDOW_HOURS)
    stats_df = df[df["datetime"] >= cutoff_stats]

    line_color = CATEGORY_COLOR.get(
        plottable.get(str(selected_lid), {}).get("predicted_category", ""),
        "#334155",
    )
    chart = (
        alt.Chart(chart_df)
        .mark_line(color=line_color, strokeWidth=2.4)
        .encode(
            x=alt.X("datetime:T", title=None),
            y=alt.Y("value:Q", title=f"{pollutant_label} (µg/m³)"),
            tooltip=[
                alt.Tooltip("datetime:T", title="Time", format="%Y-%m-%d %H:%M"),
                alt.Tooltip("value:Q", title=pollutant_label, format=".1f"),
            ],
        )
        .properties(height=300)
        .configure_axis(labelColor="#475569", titleColor="#334155", grid=True, gridColor="#eef2f7")
        .configure_view(strokeOpacity=0)
    )
    st.altair_chart(chart, use_container_width=True)

    avg, peak, days_clear = _history_stats(stats_df.to_dict("records"))
    who_label = f"{WHO_24H_PM25:.0f} µg/m³"

    st.markdown(
        f"""
        <div class="stat-grid">
          <div class="stat-card">
            <div class="stat-label">7-day avg</div>
            <div class="stat-value">{avg}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">7-day peak</div>
            <div class="stat-value">{peak}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Days all clear (PM2.5 &lt; 12)</div>
            <div class="stat-value">{days_clear}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">WHO 24h guideline</div>
            <div class="stat-value">{who_label}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <footer class="footnote">
          Forecasts are ML predictions and should not replace official air
          quality advisories. Historical data via OpenAQ. WHO 24-hour PM2.5
          guideline: 15 µg/m³.
        </footer>
        """,
        unsafe_allow_html=True,
    )


# ---- Page -------------------------------------------------------------

st.set_page_config(page_title="Berlin Air Quality", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #f5f7fa; }
    .block-container {
      padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px;
    }
    h1, h2, h3, h4 { color: #0f172a; }

    /* --- Hero -------------------------------------------------------- */
    .hero {
      background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px;
      padding: 1.4em 1.6em; margin-bottom: 1.2em;
    }
    .hero-top {
      display: flex; justify-content: space-between; align-items: flex-start;
      gap: 1em; flex-wrap: wrap;
    }
    .hero h1 { font-size: 2.1em; margin: 0 0 0.3em; line-height: 1.15; }
    .hero .subtitle {
      color: #475569; font-size: 1.02em; margin: 0; line-height: 1.5; max-width: 680px;
    }
    .update-badge {
      background: #f1f5f9; color: #334155; border: 1px solid #e2e8f0;
      padding: 0.35em 0.9em; border-radius: 999px; font-size: 0.85em;
      font-weight: 500; white-space: nowrap;
    }
    .howto {
      list-style: none; padding: 0; margin: 1.2em 0 0; display: grid;
      grid-template-columns: repeat(3, 1fr); gap: 0.8em;
    }
    .howto li {
      display: flex; align-items: flex-start; gap: 0.6em;
      color: #334155; font-size: 0.92em; line-height: 1.4;
    }
    .howto .step-num {
      flex-shrink: 0; width: 24px; height: 24px; border-radius: 50%;
      background: #0f172a; color: #ffffff; display: inline-flex;
      align-items: center; justify-content: center; font-weight: 600;
      font-size: 0.82em;
    }
    @media (max-width: 720px) {
      .howto { grid-template-columns: 1fr; }
    }

    /* --- Legend ------------------------------------------------------ */
    .legend {
      background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px;
      padding: 0.7em 1em; margin-top: 0.6em; margin-bottom: 1.4em;
    }
    .legend-row {
      display: flex; gap: 1.2em; flex-wrap: wrap;
      color: #334155; font-size: 0.92em;
    }
    .legend-item { display: inline-flex; align-items: center; gap: 0.4em; }
    .swatch {
      display: inline-block; width: 14px; height: 14px; border-radius: 50%;
      border: 1px solid rgba(15,23,42,0.15);
    }
    .legend-note {
      color: #64748b; font-size: 0.82em; margin-top: 0.35em; font-style: italic;
    }

    /* --- Level cards ------------------------------------------------- */
    .levels { margin: 0 0 1.6em; }
    .levels h2 { font-size: 1.15em; margin: 0 0 0.7em; }
    .level-grid {
      display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.7em;
    }
    .level-card {
      background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px;
      overflow: hidden; display: flex; flex-direction: column;
    }
    .level-bar { height: 6px; }
    .level-body { padding: 0.8em 0.9em 1em; }
    .level-name {
      font-weight: 600; color: #0f172a; font-size: 0.95em; margin-bottom: 0.15em;
    }
    .level-range {
      color: #475569; font-size: 0.82em; margin-bottom: 0.5em;
      font-variant-numeric: tabular-nums;
    }
    .level-desc { color: #334155; font-size: 0.85em; line-height: 1.4; }
    @media (max-width: 1100px) {
      .level-grid { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 600px) {
      .level-grid { grid-template-columns: 1fr; }
    }

    /* --- History panel ---------------------------------------------- */
    .history {
      background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px;
      padding: 1.2em 1.4em 0.4em; margin-top: 1.4em;
    }
    .history-header {
      display: flex; align-items: center; justify-content: space-between;
      flex-wrap: wrap; gap: 0.7em; margin-bottom: 0.6em;
    }
    .history-header h2 {
      margin: 0; font-size: 1.15em;
    }
    .station-pill {
      display: inline-block; padding: 0.3em 0.9em; border-radius: 999px;
      font-size: 0.82em; font-weight: 600; color: #ffffff;
    }
    .station-pill-empty {
      background: #e2e8f0; color: #64748b; font-weight: 500;
    }
    .history-empty {
      color: #64748b; padding: 1.4em 0.4em; text-align: center;
      border: 1px dashed #e2e8f0; border-radius: 10px; margin-bottom: 1em;
    }
    .stat-grid {
      display: grid; grid-template-columns: repeat(4, 1fr);
      gap: 0.7em; margin: 1em 0 1.2em;
    }
    .stat-card {
      background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
      padding: 0.8em 0.9em;
    }
    .stat-label {
      color: #64748b; font-size: 0.78em; text-transform: uppercase;
      letter-spacing: 0.04em; font-weight: 600; margin-bottom: 0.3em;
    }
    .stat-value {
      color: #0f172a; font-size: 1.25em; font-weight: 700;
      font-variant-numeric: tabular-nums;
    }
    @media (max-width: 820px) {
      .stat-grid { grid-template-columns: repeat(2, 1fr); }
    }

    /* --- Footer ------------------------------------------------------ */
    .footnote {
      margin-top: 1.6em; color: #64748b; font-size: 0.82em;
      line-height: 1.5; text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    cache_info = get_cache()
except httpx.HTTPError as exc:
    st.error(f"Cannot reach API at {API_BASE}\n\n{exc}")
    st.stop()

render_hero(cache_info)

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

map_data = render_map(plottable)
render_legend()
render_level_cards()

coord_to_lid = {
    (round(p["latitude"], 5), round(p["longitude"], 5)): lid
    for lid, p in plottable.items()
}
clicked = (map_data or {}).get("last_object_clicked") if map_data else None
tooltip = (map_data or {}).get("last_object_clicked_tooltip") if map_data else None
selected_lid: int | None = None
selected_name: str | None = None
click_debug: dict = {"clicked": clicked, "tooltip": tooltip, "match": None}

# Primary: decode the station ID from the tooltip we stamped on each
# marker. Tooltips only surface for actual marker/popup clicks, so this
# is immune to tile-click noise in last_object_clicked.
lid_from_tooltip = _lid_from_tooltip(tooltip)
if lid_from_tooltip is not None and str(lid_from_tooltip) in plottable:
    selected_lid = lid_from_tooltip
    click_debug["match"] = "tooltip-id"

# Backup: coord match, only useful for older streamlit-folium versions
# that don't populate last_object_clicked_tooltip. Requires a tight
# match so we don't accept tile clicks.
if selected_lid is None and clicked:
    for precision in (5, 4):
        key = (round(clicked["lat"], precision), round(clicked["lng"], precision))
        raw_lid = coord_to_lid.get(key)
        if raw_lid is not None:
            selected_lid = int(raw_lid)
            click_debug["match"] = f"coord @ {precision}dp"
            break

if selected_lid is None and clicked and tooltip is None:
    click_debug["match"] = "no tooltip and coord miss"
elif selected_lid is None and clicked:
    click_debug["match"] = "tooltip present but unparseable"

if selected_lid is not None:
    selected_name = STATION_NAMES.get(selected_lid, f"Station {selected_lid}")

print(
    f"[click] tooltip={tooltip!r} coords={clicked} -> lid={selected_lid} "
    f"name={selected_name} match={click_debug['match']}",
    flush=True,
)

render_history(selected_lid, selected_name, plottable)

with st.expander("Debug — click & lookup"):
    st.write({
        "last_object_clicked": clicked,
        "last_object_clicked_tooltip": tooltip,
        "parsed_from_tooltip": _lid_from_tooltip(tooltip),
        "match_result": click_debug["match"],
        "selected_location_id": selected_lid,
        "selected_name": selected_name,
        "known_station_coords": {
            int(lid): (round(p["latitude"], 5), round(p["longitude"], 5))
            for lid, p in plottable.items()
        },
    })

render_footer()

if missing_coords:
    st.caption(
        f"{len(missing_coords)} station(s) hidden from the map — "
        "their cache entry predates lat/lon baking. "
        "Run `python -m src.refresh` to repopulate."
    )
