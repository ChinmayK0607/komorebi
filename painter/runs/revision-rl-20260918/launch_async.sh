#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(realpath "${1:?workspace}");cd "$ROOT"
exec 8>async-launch.lock;flock -n 8
rm -f async-launch.exit
rm -f training-launch.pid
trap 'status=$?; echo "$status" > "$ROOT/async-launch.exit"; python3 "$ROOT/training_handoff.py" state "$ROOT" >/dev/null 2>&1 || true' EXIT
source env.sh
export PATH="$ROOT/bootstrap/bin:$PATH"
export BRUSH_ROOT="$ROOT/source/brush-rl"
export PRIME_ROOT="$ROOT/prime-rl"
export BRUSH_MODEL_PROFILE=qwen38_27b
export PAINTING_JUDGE_MODEL=gpt-5.6-luna
export HF_TOKEN_PATH="$ROOT/private/hf-token"
RUN_ID="${PAINTER_RUN_ID:-$(bootstrap/bin/python -c 'import json; e=json.load(open("EXPERIMENT.json")); print(e.get("restart_run_name") or e["run_name"])')}"
[[ "$RUN_ID" != */* && "$RUN_ID" != "" ]] || { echo 'Invalid PAINTER_RUN_ID' >&2; exit 2; }
export PAINTER_RUN_ID="$RUN_ID"
export MODEL_DIR="$(cat "$ROOT/model-path.txt")"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WATERCOLOUR_INIT_ADAPTER=$(cat initial-adapter-path.txt)
export HF_CHECKPOINT_REPO=CK0607/qwen3.8-27b-brush-painting
export HF_CHECKPOINT_RUN="$RUN_ID"
export ADAPTER_ONLY_CHECKPOINTS=1
export CAPACITY_STATE_COMPLETE=1
cd "$PRIME_ROOT"
uv pip install --python "$PRIME_ROOT/.venv/bin/python" --no-deps -e "$ROOT/environments/painter_revision"
# Keep task scoring self-contained on fresh, hardware-agnostic nodes.  Prime's
# dependency sync can start workers with a clean import path, so stage the
# renderer, contract, and reward protocol beside rl-states.json explicitly.
for f in quality_signal.py render_actions.py contract.py; do
  [[ -f "$ROOT/$f" ]] || { echo "missing scoring dependency: $ROOT/$f" >&2; exit 2; }
done
uv run --no-sync python "$ROOT/prepare_async.py"
echo "$$" > "$ROOT/training-launch.pid"
python3 "$ROOT/training_handoff.py" preflight "$ROOT"
python3 "$ROOT/training_handoff.py" state "$ROOT"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,power.draw --format=csv -l 10 > "$ROOT/logs/gpu-telemetry.csv" &
TELEMETRY_PID=$!
trap 'status=$?; kill "$TELEMETRY_PID" || true; echo "$status" > "$ROOT/async-launch.exit"; python3 "$ROOT/training_handoff.py" state "$ROOT" >/dev/null 2>&1 || true' EXIT
uv run --no-sync python "$ROOT/run_from_sft.py" @ "$ROOT/rl.toml"
