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
- **"High+" is not predicted by the model — it's a PM2.5 threshold rule in the serving layer.** With only 5 "High+" training examples across 15 stations × 2yr, the model achieves F2 = 0.0 on this class. The API layer hard-codes "PM2.5 > 150.5 → High+" to catch extreme events deterministically.

## Known limitations + future work

- **Significant-class recall caps at ~50%** on the current 2yr multi-station dataset (F2 ≈ 0.56). Primary lever to improve it: extend station history to **3+ years** — more rare-event examples, less seasonal sampling bias. Revisit after the first full year of production data is collected.

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
uv sync                         # install/refresh env from pyproject.toml
uv add <package>                # add a new dependency
uv run <script.py>              # run inside the managed venv
uv run pytest                   # tests
uv run uvicorn api.main:app     # serve API locally
uv run python -m src.ingest --multi --days 730   # full 2yr multi-station ingest
uv run python -m src.features                    # build features + train/test split
uv run python -m src.train --tune                # train with RandomizedSearchCV
uv run python -m src.register                    # promote latest tuned model to @production
uv run python -m src.refresh                     # one-shot hourly refresh (cache)
docker compose up --build                        # local container stack (API + MLflow UI)
```

## Hourly refresh (Phase 5)

`/predict` returns **pre-computed predictions from a cache**, not live model
inference. A scheduled job (`src.refresh`) ingests the last 48h for all 17
eligible stations, runs the production model on each station's current hour,
and writes `data/cache/predictions.json` + appends monitoring rows.

Locally, add to `crontab -e`:

```cron
0 * * * * cd /Users/chandlershortlidge/berlin-aqi-MLOps && /Users/chandlershortlidge/.local/bin/uv run python -m src.refresh >> logs/refresh.log 2>&1
```

On AWS: EventBridge triggers a scheduled ECS task / Lambda that invokes the
same `python -m src.refresh` entry point. Artifact paths in `mlflow.db`
should be S3 URIs by then (not local absolute paths), so the container
doesn't need the bind-mount hack from `docker-compose.yml`.

## AWS / deployment

- **ECR** hosts the Docker image; **EC2** runs the container
- **MLflow** tracking server — local `mlruns/` for dev, gitignored
- Credentials come from `.env` (local) or instance role (prod) — never hardcode
