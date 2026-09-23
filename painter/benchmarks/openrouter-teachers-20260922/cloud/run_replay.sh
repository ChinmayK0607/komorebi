#!/usr/bin/env bash
# Replay a verified public shard's timed-out sketches; make no paid model calls.
set -Eeuo pipefail

if [[ $# -lt 2 ]]; then
  echo 'Usage: run_replay.sh SOURCE_RUN_ID REPLAY_RUN_ID [--limit N] [--timeout N]' >&2
  exit 2
fi
SOURCE_RUN_ID="$1"
REPLAY_RUN_ID="$2"
shift 2
[[ "$REPLAY_RUN_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$ ]] || {
  echo 'Invalid replay run ID' >&2; exit 2;
}
REPO_ROOT="$(git rev-parse --show-toplevel)"
BENCHMARK="$REPO_ROOT/painter/benchmarks/openrouter-teachers-20260922"
RUNTIME="$REPO_ROOT/.painter-cloud-runtime"
PY="$RUNTIME/renderer-env/bin/python"
[[ -x "$PY" && -d "$RUNTIME/browsers" ]] || {
  echo 'Cloud renderer missing; run cloud/setup.sh first.' >&2; exit 2;
}
[[ -n "${HF_TOKEN:-}" ]] || {
  echo 'HF_TOKEN is required to publish the verified replay evidence.' >&2; exit 2;
}
export PLAYWRIGHT_BROWSERS_PATH="$RUNTIME/browsers"
OUTPUT="$BENCHMARK/replays/$REPLAY_RUN_ID"
[[ ! -e "$OUTPUT" ]] || {
  echo 'Replay output already exists; refusing to overwrite evidence.' >&2; exit 2;
}
mkdir -p "$OUTPUT"
export PAINTER_RUN_ID="$REPLAY_RUN_ID"
publish_on_exit() {
  status=$?
  trap - EXIT
  if [[ -d "$OUTPUT/episodes" ]]; then
    HF_HUB_DISABLE_XET=1 "$PY" "$BENCHMARK/cloud/publish_results.py" --root "$OUTPUT" || status=1
  fi
  exit "$status"
}
trap publish_on_exit EXIT
"$PY" "$BENCHMARK/cloud/replay_renderer.py" \
  --source-run-id "$SOURCE_RUN_ID" --output "$OUTPUT" \
  --renderer-python "$PY" --browser-path "$RUNTIME/browsers" "$@"
