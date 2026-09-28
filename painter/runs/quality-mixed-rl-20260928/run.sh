#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
OLD="$ROOT/../quality-oss-rl-20260927"
export PAINTER_REPO_ROOT="${PAINTER_REPO_ROOT:-$(cd "$ROOT/../../.." && pwd -P)}"
export BRUSH_ROOT="${BRUSH_ROOT:-$PAINTER_REPO_ROOT/painter/rl_helpers}"
export PRIME_ROOT="${PRIME_ROOT:-/root/painter/prime}"
export MODEL_DIR="${MODEL_DIR:-/root/painter/model-cache/qwen/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-/root/painter/private/hf-token}"
GATEWAY_KEY_FILE="${GATEWAY_KEY_FILE:-/root/painter/private/gateway.env}"
export WATERCOLOUR_INIT_ADAPTER="$(tr -d '\r\n' < "$OLD/../photo-curriculum-rl-20260922/initial-adapter-path.txt")"
export PAINTER_RL_RUN_NAME=quality-mixed-anchors-20260928
export PAINTER_RL_CONFIG="$OLD/native-rl-mixed-anchors.toml"
export PAINTER_RL_MANIFEST="${PAINTER_NODE_ROOT:-/root/painter}/mixed-fixed-anchors-v1/train-manifest.json"
export PAINTER_RL_MAX_STEPS=12
export PAINTER_GPUS_PER_NODE=2 PAINTER_NUM_TRAIN_GPUS=1 PAINTER_NUM_INFER_GPUS=1
export PAINTER_RL_ADAPTER_REVISION=adf687e5b321514eda7be6eab050434fdc8e44ff
export PAINTER_RL_ADAPTER_PREFIX=photo-curriculum-sft-20260922-balanced/step_512
export PAINTER_RL_ADAPTER_SHA256=bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740
export PAINTER_JUDGE_PROTOCOL=aesthetic-pairwise-gateway-mimo-v1
export PAINTER_JUDGE_MODEL=xiaomi/mimo-v2.6-flash
export PAINTER_JUDGE_EFFORT=gateway-default

if [[ "${1:-}" == '--preflight' ]]; then
  PAINTER_RL_PREFLIGHT_ONLY=1 bash "$OLD/launch_rl.sh" "$OLD"
  exit
fi
[[ $# == 0 ]] || { echo 'usage: run.sh [--preflight]' >&2; exit 2; }
RUN_OUTPUT="$OLD/async-output/$PAINTER_RL_RUN_NAME"
[[ ! -e "$RUN_OUTPUT" ]] || {
  echo "Run output already exists; preserve or move it before a fresh launch: $RUN_OUTPUT" >&2
  exit 1
}
[[ -r "$GATEWAY_KEY_FILE" ]] || { echo 'Gateway key file missing' >&2; exit 1; }
mkdir -p "$OLD/logs"
"$PRIME_ROOT/.venv/bin/python" "$OLD/gateway_queue_worker.py" \
  --root "$OLD" --key-file "$GATEWAY_KEY_FILE" \
  --seconds 18000 --workers 8 --batch-limit 16 \
  >"$OLD/logs/gateway-worker-mixed.log" 2>&1 &
worker_pid=$!
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  kill "$worker_pid" 2>/dev/null || true
  wait "$worker_pid" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT INT TERM
bash "$OLD/launch_rl.sh" "$OLD"
