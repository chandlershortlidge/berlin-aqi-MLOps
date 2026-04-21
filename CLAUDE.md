# Berlin AQI MLOps

End-to-end MLOps pipeline that predicts air quality risk categories for athletes in Berlin using XGBoost, deployed on AWS.

## Pipeline

1. **Ingest** — pull pollutants from OpenAQ and weather (temp, RH) from the Open-Meteo archive API. Berlin government PM2.5 stations don't report weather, so the two sources are merged on `datetime`.
2. **Feature engineering** — build training features from raw measurements
3. **Training** — XGBoost classifier, experiments tracked with MLflow
4. **Containerize** — package model + serving layer in Docker
5. **Deploy** — push image to ECR, run on EC2
6. **Serve** — FastAPI inference endpoint
7. **Monitor** — PSI drift detection on features + prediction-vs-actual performance tracking

## Project decisions (locked in from EDA, 2026-04-21)

**Data scope**
- Only stations reporting PM2.5 are used. 17 of 28 in the Berlin bbox have ≥2yr history — train on all 17 (multi-station), not just Berlin Mitte.
- Weather (temperature, relative humidity) comes from **Open-Meteo**, not OpenAQ. Most Berlin government PM2.5 stations don't report temp/RH; the AirGradient sensors that do have <1.1yr history.
- SO₂, CO, BC are **excluded** — sparse station coverage (≤4 stations for SO₂/CO, 0 for BC).

**Data cleaning**
- OpenAQ v3 caps at 1000 results per request — `fetch()` must paginate (see `_paginate_hours` in `src/ingest.py`). Skipping this silently truncates to ~41 days of data.
- Negative PM2.5 values are **clamped to 0** in `clean()`, not dropped — sensor baseline drift, physically impossible but recoverable.

**Target + modeling**
- Target is the custom **5-class `aqi_category`** with "High" + "Very High" merged into a single **"High+"** bin (only 2 "High" hours, 0 "Very High" in 2yr at Mitte — neither individually trainable). See memory `custom_aqi_thresholds.md` for exact bins.
- Prediction horizon: **t+1 (next hour)** for v1. Recursive 6-hour forecast is a v2 stretch goal.
- Primary metric: **F2 score** (recall weighted 2× precision). A false negative — predicting "safe" when air is dangerous — harms athletes; a false positive just keeps them indoors unnecessarily.
- Class imbalance strategy: **class weights first, SMOTE as fallback**. Both logged to MLflow for comparison.

## Conventions

- **Python 3.11** — pinned via `.python-version` / `pyproject.toml`
- **uv** for package management — never `pip install` or `requirements.txt`. Use `uv add <pkg>` and `uv sync`
- **pyproject.toml** is the single source of truth for dependencies
- **macOS prerequisite:** `brew install libomp` — XGBoost's `libxgboost.dylib` links to OpenMP at runtime and won't import without it. Linux distros ship `libgomp` with the system.
- **.env** for secrets — never commit. `python-dotenv` loads it at runtime
- **src/** holds the pipeline code, **api/** the FastAPI app, **frontend/** the UI, **notebooks/** for exploration only (not production paths)
- Raw API pulls land in `data/raw/`, engineered features in `data/processed/` — both gitignored

## Common commands

```bash
uv sync                      # install/refresh env from pyproject.toml
uv add <package>             # add a new dependency
uv run <script.py>           # run inside the managed venv
uv run pytest                # tests
uv run uvicorn api.main:app  # serve API locally
```

## AWS / deployment

- **ECR** hosts the Docker image; **EC2** runs the container
- **MLflow** tracking server — local `mlruns/` for dev, gitignored
- Credentials come from `.env` (local) or instance role (prod) — never hardcode
