# PowerPoint build brief — Berlin AQI MLOps capstone

You are building a slide deck for the **Ironhack Data Science & AI Engineering capstone** presentation. This file is a self-contained brief — read it end-to-end before starting. The author is Chandler Shortlidge; the deck is graded against a Tier 1/2/3 + Bonus rubric (see the rubric alignment section at the bottom).

## 1. Goal and audience

- **Audience:** Ironhack graders (technical, but evaluating against a fixed rubric) plus a live-demo cohort. They want to see that the project hits every rubric tier, not a deep ML lecture.
- **Length:** ~12–15 content slides, plus title and thank-you. Aim for a 10–12 minute talk.
- **Tone:** Capstone-professional. Confident, plain-spoken, light on jargon. The narrator is an athlete-turned-MLOps-engineer building a tool he would actually use.
- **One-line pitch the deck must land:** "A real-time, multi-station air-quality predictor for Berlin athletes — full MLOps loop from OpenAQ ingest to Dockerized FastAPI on AWS, with drift monitoring and an hourly retrain hook."

## 2. Output

- Write to `/Users/chandlershortlidge/berlin-aqi-MLOps/berlin_aqi_deck.pptx`. A `.pptx` already exists at that path — **overwrite it**, don't create a sibling file.
- Use **`python-pptx`**. Add it via `uv add python-pptx` if not already a dep. Run the build script with `uv run python <script>.py`. Do not pip-install.
- 16:9 widescreen (`Inches(13.333)` × `Inches(7.5)`).
- Save the build script to `scripts/build_deck.py` so the deck is reproducible. Do not check in the `.pptx` itself (it's already gitignored implicitly via the untracked status).

## 3. Visual assets you can embed

Located at `/Users/chandlershortlidge/berlin-aqi-MLOps/images for MLOps presentation/`:

| File | Use on slide |
|---|---|
| `dashboard.png` | Frontend / "what the athlete sees" slide |
| `mlflow_f2_chart.png` | MLflow / experiment tracking slide |
| `mlflow_params.png` | Hyperparameter tuning slide |

**Folder name has spaces** — quote the path or use `pathlib.Path`. Don't generate new charts; use what's there. If you need an architecture diagram, build it with `python-pptx` shapes (rounded rectangles + arrows), not a separate image lib.

## 4. Slide-by-slide structure

Follow this order. Each bullet under a slide is roughly one bullet point on the slide — keep the wording tight; don't paste this brief verbatim.

### Slide 1 — Title
- "Berlin AQI MLOps Pipeline"
- Subtitle: "Real-time air quality predictions for athletes in Berlin"
- "Chandler Shortlidge | Ironhack DS & AI Engineering | April 2026"

### Slide 2 — The problem
- Standard WHO/EPA AQI categories under-warn on PM2.5 — a "Good" day can still spike high enough to affect respiratory performance.
- Athletes training outdoors need a glance-level go/no-go, not a data table.
- Goal: predict the **next hour's** PM2.5 risk category for any Berlin neighbourhood.

### Slide 3 — Solution at a glance
- Hourly OpenAQ + Open-Meteo ingest → XGBoost → FastAPI → cached predictions → web dashboard.
- Custom 5-class risk schema calibrated to PM2.5 (not WHO).
- Full MLOps loop: experiment tracking, container deploy, drift monitoring, retrain hook.

### Slide 4 — Architecture diagram
Build with `python-pptx` shapes. Left-to-right flow:
`OpenAQ + Open-Meteo` → `Ingest (paginated, validated)` → `Feature engineering` → `XGBoost training (MLflow)` → `Docker image (ECR)` → `EC2 / FastAPI` → `Cache (JSON/S3)` → `Dashboard`. Loop arrow back from `Monitor (PSI + F2)` → `Retrain`.

### Slide 5 — Data sources
- **OpenAQ v3** — 17 Berlin stations reporting PM2.5 with ≥2 yr history (of 28 in the bbox). PM2.5, PM10, NO₂, O₃.
- **Open-Meteo archive** — temperature + relative humidity for Berlin coords (52.52°N, 13.41°E). Berlin gov stations don't report weather, so the two sources are merged on `datetime`.
- Excluded: SO₂, CO, BC (sparse station coverage — ≤4 stations / 0 for BC).
- Pagination gotcha: OpenAQ caps at 1000 results/request; without pagination you silently get ~41 days instead of 2 years.

### Slide 6 — Custom risk categories
Show the table. Emphasize this is **not** WHO/EPA — it's calibrated for athlete decision-making.

| Category | PM2.5 (µg/m³) | Athlete guidance |
|---|---|---|
| All Clear | 0–12 | Train outdoors freely |
| Low Risk | 12.1–35.4 | Fine for most |
| Elevated | 35.5–55.4 | Reduce prolonged exertion |
| Significant | 55.5–150.4 | Move indoors |
| High+ | >150.4 | Avoid all outdoor exercise |

Note: original 5-class schema had "High" and "Very High" — only 2 "High" hours and 0 "Very High" in 2 yr at Berlin Mitte, so they were merged into **High+**. The model doesn't predict High+; the API layer applies a deterministic `PM2.5 > 150.5` threshold rule (see `api/threshold_rule.py`).

### Slide 7 — Feature engineering
- 24 lag features (PM2.5 at t-1 … t-24)
- Rolling mean + std over 6h, 12h, 24h windows
- Cyclical time encoding (hour of day, day of week as sin/cos pairs)
- Weather covariates: temp, RH (from Open-Meteo)
- `location_id` as integer (XGBoost handles tree-friendly categoricals natively — no one-hot)
- Time-based **stratified** 80/20 split, no shuffle (avoids future leakage while preserving rare-class proportions)

### Slide 8 — Model & metric
- **XGBoost** classifier — strong on tabular, fast retrains, native class-weight support, interpretable.
- **F2 score** as primary metric — recall weighted 2× precision. False negatives (predicting "safe" when air is dangerous) hurt athletes; false positives just keep them indoors.
- Class imbalance: class weights first, SMOTE as fallback (both logged to MLflow for comparison).

### Slide 9 — MLflow experiment tracking
- Embed `mlflow_f2_chart.png`.
- Brief: every run logs params, F2/precision/recall, confusion matrix, feature importance, training-data baseline (used later for drift).
- Best model promoted via alias to `@production`; deploy pipeline pulls only from registry. Currently `berlin-aqi-xgboost v2 @production` (`artifacts/METADATA.json`).

### Slide 10 — Hyperparameter tuning
- Embed `mlflow_params.png`.
- `RandomizedSearchCV`, scored on F2, time-aware 5-fold CV.
- Tuned: `max_depth`, `learning_rate`, `n_estimators`, `min_child_weight`, `subsample`, `colsample_bytree`, class-weight scaling.

### Slide 11 — Honest results & limitations
- Significant-class recall caps at ~50% on the current 2-yr multi-station dataset (F2 ≈ 0.56).
- Rare-event ceiling is a **data scarcity** problem, not a model problem — primary lever is extending station history to 3+ years.
- High+ handled deterministically in serving layer because only ~5 training examples across 15 stations × 2 yr.
- Don't oversell. The story is "honest engineering of a known-hard problem," not "we beat the baseline."

### Slide 12 — Containerization & AWS deploy
- **Self-contained image**: `src.bundle` extracts `@production` from `mlruns/` + `mlflow.db` into `./artifacts/`; the Dockerfile `COPY`s it in. Runtime needs zero external dependencies — no MLflow tracking server, no registry lookup. Every image = one model.
- Build flow: `uv run python -m src.bundle` → `docker build` → push to **ECR** → run on **EC2** (t2.micro, free tier).
- `/predict`, `/health`, `/metrics` endpoints; Pydantic schemas validate I/O.

### Slide 13 — Hourly refresh & serving
- `/predict` returns **pre-computed** predictions from a JSON cache, not live inference.
- `src.refresh` runs hourly: ingest last 48h for all 17 stations, run model, write `data/cache/predictions.json`, append monitoring rows.
- Local: `cron`. Cloud: **EventBridge** → ECS task / Lambda invokes the same `python -m src.refresh` entry point.

### Slide 14 — Frontend
- Embed `dashboard.png`.
- Card-based UI: neighbourhood selector, current category (color-coded), next-hour prediction, plain-language recommendation, 24h trend chart from OpenAQ.

### Slide 15 — Monitoring
- **PSI** (Population Stability Index) for feature drift, computed against the training-data baseline saved as an MLflow artifact. Thresholds: <0.1 OK, 0.1–0.2 watch, >0.2 retrain.
- **Prediction-vs-actual** logging — every hour the prior hour's prediction is graded against the new reading; rolling 24h F2 catches gradual drift.
- Alert + retrain hook: if either threshold trips, retrain; the new model only gets promoted if its test F2 beats incumbent.

### Slide 16 — Rubric coverage (closer)
Quick table mapping deliverables to rubric tiers. Use the table from §6 below.

### Slide 17 — Thank you / Q&A
- Repo link, contact, "Questions?"

## 5. Design conventions

- One **accent color** that maps to the AQI palette (green → yellow → orange → red → maroon for All Clear / Low / Elevated / Significant / High+). Use it for category callouts; don't recolor the whole deck.
- Headers: 32–40pt. Body: 18–22pt. No paragraphs — use bullets, max ~6 per slide, max ~10 words each.
- White or near-white background. Dark text. Keep it boring; the content carries the talk.
- No emojis. No clip art. No stock photos.
- Code snippets: use a monospace font (`Consolas` / `Menlo`), shrink to 14pt, only when essential (e.g., the cron line, the `bundle → build` command pair). Prefer prose over code.

## 6. Rubric alignment table (use on the closer slide)

| Tier | Requirement | What to point to |
|---|---|---|
| 1 | Git, venv, version control | GitHub history, `uv` + `pyproject.toml`, `.env.example` |
| 2 | MLflow, AWS deploy, FastAPI, pipeline, monitoring | MLflow registry + alias-based promotion, ECR/EC2, `api/main.py`, `src/ingest.py` validation, `src/monitor.py` PSI + F2 |
| 3 | Docker | Multi-stage Dockerfile, `docker-compose.yml`, ECR push |
| Bonus | Alerting, frontend, CI/CD | Drift alerts in `src/alerting.py`, `frontend/app.py` dashboard, GitHub Actions (if present) |

## 7. Key facts cheat-sheet (for accuracy)

Don't make these up — these are the load-bearing numbers in the talk:

- 17 of 28 Berlin bbox stations report PM2.5 with ≥2 yr history; **all 17** are used for training (multi-station, not just Berlin Mitte / location 3019).
- Target horizon: **t+1** (next hour). Recursive 6-hour forecast is a v2 stretch goal, not delivered.
- 5-class custom AQI schema with High+High very rare → merged into "High+". Threshold `> 150.4 µg/m³`.
- Negative PM2.5 readings are clamped to 0 (sensor baseline drift), not dropped.
- Model: `berlin-aqi-xgboost v2 @production`, F2 ≈ 0.56, Significant-class recall ~50%.
- Build flow: `uv run python -m src.bundle` → `docker build -t berlin-aqi:latest .`. Re-bundle + rebuild on every promotion (no runtime hot-swap, by design).

## 8. What NOT to do

- Don't claim live inference at request time — `/predict` reads the cache.
- Don't claim High+ is model-predicted — it's a deterministic threshold rule in `api/threshold_rule.py`.
- Don't quote WHO AQI thresholds; this project uses its own schema (see §6 above).
- Don't fabricate metrics. If a number isn't in this brief or in `mlflow_f2_chart.png`, omit it.
- Don't add slides on topics not implemented (e.g., the v2 6-hour recursive forecast, traffic-data feature) without flagging them as future work.
- Don't use the standard `python-pptx` default font sizes — they are too small for projection.

## 9. When the deck is built

1. Run the build script.
2. Open the `.pptx` and visually scan every slide — embedded images shouldn't overflow, text shouldn't wrap into the title bar.
3. Report: total slide count, any slide where content didn't fit, and the path to the generated file.
