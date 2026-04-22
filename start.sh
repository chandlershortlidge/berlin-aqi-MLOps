#!/usr/bin/env bash
# Container entrypoint — run FastAPI (8000) and Streamlit (8501) side-by-side.
#
# Both processes share the same filesystem + env. Streamlit reaches the
# API on localhost because they're in the same network namespace.
# If either child dies, the script kills the other and exits with the
# dead child's code so Docker's restart policy takes over.

set -euo pipefail

cleanup() {
  # Kill every process in our group; ignore errors from already-dead ones.
  kill -TERM 0 2>/dev/null || true
}
trap cleanup INT TERM

uvicorn api.main:app --host 0.0.0.0 --port 8000 &
PID_API=$!

streamlit run frontend/app.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --server.headless=true \
  --browser.gatherUsageStats=false &
PID_UI=$!

# wait -n exits as soon as any child exits.
wait -n "$PID_API" "$PID_UI"
EXIT_CODE=$?

cleanup
wait 2>/dev/null || true
exit "$EXIT_CODE"
