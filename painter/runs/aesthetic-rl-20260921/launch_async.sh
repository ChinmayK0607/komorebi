#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "${1:?usage: launch_async.sh RUN_ROOT}" && pwd -P)"
cd "$ROOT"

exec 9>"$ROOT/aesthetic-launch.lock"
flock -n 9 || exit 0

rm -f "$ROOT/aesthetic-launch.exit"
trap 'rc=$?; printf "%s\n" "$rc" > "$ROOT/aesthetic-launch.exit"' EXIT

if [[ -f "$ROOT/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/env.sh"
fi
: "${PRIME_ROOT:?env.sh must define PRIME_ROOT}"
export PATH="$ROOT/bootstrap/bin:$PRIME_ROOT/.venv/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export BRUSH_ROOT="$ROOT/source/brush-rl"
export BRUSH_MODEL_PROFILE="qwen38_27b"
export HF_TOKEN_PATH="$ROOT/private/hf-token"

RUN_ID="${PAINTER_RUN_ID:-${HF_CHECKPOINT_RUN:-aesthetic-rl-20260921}}"
[[ "$RUN_ID" == "aesthetic-rl-20260921" ]] || {
  echo "HF_CHECKPOINT_RUN must be aesthetic-rl-20260921, got $RUN_ID" >&2
  exit 1
}
export HF_CHECKPOINT_RUN="$RUN_ID"
export PAINTER_RUN_ID="$RUN_ID"
export ADAPTER_ONLY_CHECKPOINTS="1"
export CAPACITY_STATE_COMPLETE="1"
export HF_CHECKPOINT_REPO="CK0607/qwen3.8-27b-brush-painting"
export PAINTING_JUDGE_MODEL="${PAINTING_JUDGE_MODEL:-gpt-5.6-luna}"

for required in training-ready.json rl.toml model-path.txt initial-adapter-path.txt run_from_sft.py; do
  [[ -e "$ROOT/$required" ]] || { echo "missing $ROOT/$required" >&2; exit 1; }
done
export MODEL_DIR="$(cat "$ROOT/model-path.txt")"
export WATERCOLOUR_INIT_ADAPTER="$(cat "$ROOT/initial-adapter-path.txt")"

cd "$PRIME_ROOT"
printf '%s\n' "$$" > "$ROOT/training-launch.pid"
# Native trainer entrypoint: no legacy quality signal or supervisor wrapper.
uv run --no-sync python "$ROOT/run_from_sft.py" @ "$ROOT/rl.toml"
