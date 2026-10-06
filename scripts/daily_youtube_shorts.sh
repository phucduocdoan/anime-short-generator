#!/usr/bin/env bash
set -euo pipefail

cd /home/ding/anime-short-generator

export PYTHONUNBUFFERED=1
export PATH="/home/ding/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH}"

mkdir -p logs

today="$(TZ=Asia/Ho_Chi_Minh date +%F)"
queue_file="workspace/queue/daily_${today}.yaml"
theme="winter love, missing someone, letting go, quiet self-worth, gentle life philosophy"

if [[ ! -f "${queue_file}" ]]; then
  uv run anime-short queue plan-day --date "${today}" --count 5 --theme "${theme}"
fi

uv run anime-short queue run --date "${today}" --oauth-port 28080

