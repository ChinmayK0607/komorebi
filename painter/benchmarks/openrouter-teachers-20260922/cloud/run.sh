#!/usr/bin/env bash
# Run a bounded benchmark in a prepared Codex cloud environment.
set -Eeuo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
# Match the Node selected during setup, ahead of Codex universal's Node 20.
export PATH="/usr/bin:/bin:$PATH"
BENCHMARK="$REPO_ROOT/painter/benchmarks/openrouter-teachers-20260922"
RUNTIME_ROOT="$REPO_ROOT/.painter-cloud-runtime"
PY="$RUNTIME_ROOT/renderer-env/bin/python"
RENDERER="$REPO_ROOT/painter/vendor/integrations/watercolour/renderer.py"
BROWSERS="$RUNTIME_ROOT/browsers"

[[ -x "$PY" && -f "$RENDERER" && -d "$BROWSERS" ]] || {
  echo 'Cloud setup is missing; run cloud/setup.sh first.' >&2
  exit 2
}
export PLAYWRIGHT_BROWSERS_PATH="$BROWSERS"

for argument in "$@"; do
  if [[ "$argument" == "--dry-run" ]]; then
    exec "$PY" "$BENCHMARK/run.py" --root "$BENCHMARK" --dry-run "$@"
  fi
done

[[ -n "${AI_GATEWAY_API_KEY:-}" ]] || {
  echo 'AI_GATEWAY_API_KEY is required for paid calls.' >&2
  exit 2
}
[[ -n "${HF_TOKEN:-}" ]] || {
  echo 'HF_TOKEN is required to preserve results.' >&2
  exit 2
}
[[ "${PAINTER_RUN_ID:-}" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$ ]] || {
  echo 'Set a stable PAINTER_RUN_ID before paid calls.' >&2
  exit 2
}

publish_on_exit() {
  status=$?
  trap - EXIT
  if [[ -d "$BENCHMARK/episodes" ]]; then
    "$PY" "$BENCHMARK/cloud/publish_results.py" --root "$BENCHMARK" || status=1
  fi
  exit "$status"
}
trap publish_on_exit EXIT

"$PY" "$BENCHMARK/run.py" \
  --root "$BENCHMARK" --run \
  --renderer "$RENDERER" \
  --renderer-python "$PY" \
  --browser-path "$BROWSERS" \
  --run-as-user painter \
  "$@"
