# Berlin AQI MLOps

End-to-end MLOps pipeline that predicts air quality risk categories for athletes in Berlin using XGBoost, deployed on AWS.

## Pipeline

1. **Ingest** — pull pollutant + weather data from the OpenAQ API
2. **Feature engineering** — build training features from raw measurements
3. **Training** — XGBoost classifier, experiments tracked with MLflow
4. **Containerize** — package model + serving layer in Docker
5. **Deploy** — push image to ECR, run on EC2
6. **Serve** — FastAPI inference endpoint
7. **Monitor** — PSI drift detection on features + prediction-vs-actual performance tracking

## Conventions

- **Python 3.11** — pinned via `.python-version` / `pyproject.toml`
- **uv** for package management — never `pip install` or `requirements.txt`. Use `uv add <pkg>` and `uv sync`
- **pyproject.toml** is the single source of truth for dependencies
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
