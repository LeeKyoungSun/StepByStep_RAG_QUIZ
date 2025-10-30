#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "[ERR] .venv가 없습니다. 아래 명령으로 먼저 만들고 재실행하세요."
  echo "python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

uvicorn app.api.serve_rag:app --host 0.0.0.0 --port "${APP_PORT:-8000}" --reload
