#!/usr/bin/env bash
# Restore a verified public shard and resume only its unfinished episodes.
set -Eeuo pipefail

if [[ $# -ne 5 ]]; then
  echo 'Usage: run_recovery.sh SOURCE_RUN_ID RECOVERY_RUN_ID TRACK START_EPISODE LIMIT_EPISODES' >&2
  exit 2
fi
SOURCE_RUN_ID="$1"
RECOVERY_RUN_ID="$2"
TRACK="$3"
START="$4"
LIMIT="$5"
[[ "$RECOVERY_RUN_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$ ]] || {
  echo 'Invalid recovery run ID' >&2; exit 2;
}
[[ "$TRACK" == quality || "$TRACK" == speed ]] || {
  echo 'Track must be quality or speed' >&2; exit 2;
}
REPO_ROOT="$(git rev-parse --show-toplevel)"
BENCHMARK="$REPO_ROOT/painter/benchmarks/openrouter-teachers-20260922"
PY="$REPO_ROOT/.painter-cloud-runtime/renderer-env/bin/python"
[[ -x "$PY" ]] || {
  echo 'Cloud setup missing; run cloud/setup.sh first.' >&2; exit 2;
}
[[ ! -e "$BENCHMARK/episodes" ]] || {
  echo 'Episodes already exist; use a fresh cloud task.' >&2; exit 2;
}
"$PY" "$BENCHMARK/cloud/restore_public_shard.py" --source-run-id "$SOURCE_RUN_ID" --root "$BENCHMARK"
export PAINTER_SOURCE_RUN_ID="$SOURCE_RUN_ID"
export PAINTER_RUN_ID="$RECOVERY_RUN_ID"
bash "$BENCHMARK/cloud/run.sh" --track "$TRACK" --start-episode "$START" \
  --limit-episodes "$LIMIT" --renderer-timeout-override 600
