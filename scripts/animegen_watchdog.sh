#!/usr/bin/env bash
set -euo pipefail

cd /home/ding/anime-short-generator

export PYTHONUNBUFFERED=1
export PATH="/home/ding/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH}"

mkdir -p logs

wait_for_ready=false
if [[ "${1:-}" == "--wait" ]]; then
  wait_for_ready=true
fi

exec 9>logs/animegen_watchdog.lock
if ! flock -n 9; then
  exit 0
fi

if uv run anime-short animegen status >/dev/null 2>&1; then
  exit 0
fi

if pgrep -f "anime-short animegen serve" >/dev/null 2>&1; then
  :
else
  nohup uv run anime-short animegen serve --host 127.0.0.1 --port 28180 \
    >> logs/animegen_service.log 2>&1 &
fi

if [[ "${wait_for_ready}" == true ]]; then
  timeout_seconds="${ANIMEGEN_START_TIMEOUT_SECONDS:-1800}"
  deadline=$((SECONDS + timeout_seconds))
  until uv run anime-short animegen status >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "animegen service did not become ready within ${timeout_seconds}s" >&2
      exit 1
    fi
    sleep 10
  done
fi
