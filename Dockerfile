# Multi-stage build for the Berlin AQI FastAPI service.
#
# Stage 1 (builder): resolve deps with uv into /app/.venv.
# Stage 2 (runtime): copy only the venv + source; slim final image.

# -------- Stage 1: build --------
FROM python:3.11-slim AS builder

# libgomp1 is required at build time so xgboost can link; runtime also needs it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv is self-contained; copy its binary from the official image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Lockfile-based install for reproducibility
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Source
COPY src/ ./src/
COPY api/ ./api/


# -------- Stage 2: runtime --------
FROM python:3.11-slim

# XGBoost dylib link to OpenMP — required in the runtime image too.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/api /app/api

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Default MLflow tracking URI — overridden by docker-compose for local dev
    MLFLOW_TRACKING_URI=http://mlflow:5000

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
