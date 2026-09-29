#!/usr/bin/env bash
set -Eeuo pipefail

RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export PAINTER_REPO_ROOT="$(cd "$RUN/../../.." && pwd -P)"
export BRUSH_ROOT="$PAINTER_REPO_ROOT/painter/rl_helpers"
export PRIME_ROOT="${PRIME_ROOT:-/root/painter/prime}"
export MODEL_DIR="${MODEL_DIR:-/root/painter/model-cache/qwen/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-/root/painter/private/hf-token}"
export PAINTER_GATEWAY_ENV_FILE="${PAINTER_GATEWAY_ENV_FILE:-/root/painter/private/gateway.env}"
export WATERCOLOUR_INIT_ADAPTER="$(tr -d '\r\n' < "$RUN/../photo-curriculum-rl-20260922/initial-adapter-path.txt")"
export PAINTER_RL_RUN_NAME="${PAINTER_RL_RUN_NAME:-text-pairwise-rl-20260928}"
export PAINTER_RL_CONFIG="${PAINTER_RL_CONFIG:-$RUN/native-rl.toml}"
export PAINTER_RL_MANIFEST="${PAINTER_RL_MANIFEST:-$RUN/anchors-frozen/train-manifest.json}"
export PAINTER_RL_MAX_STEPS="${PAINTER_RL_MAX_STEPS:-20}"
export PAINTER_GPUS_PER_NODE=2 PAINTER_NUM_TRAIN_GPUS=1 PAINTER_NUM_INFER_GPUS=1
export PAINTER_RL_ADAPTER_REVISION=adf687e5b321514eda7be6eab050434fdc8e44ff
export PAINTER_RL_ADAPTER_PREFIX=photo-curriculum-sft-20260922-balanced/step_512
export PAINTER_RL_ADAPTER_SHA256=bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740
export PAINTER_JUDGE_PROTOCOL="${PAINTER_JUDGE_PROTOCOL:-text-paint-sft-pairwise-v1}"
export PAINTER_JUDGE_MODEL=xiaomi/mimo-v2.6-flash
export PAINTER_JUDGE_EFFORT=gateway-default
export PAINTER_RL_FIXED_OPPONENT=1
export PYTHONPATH="$PAINTER_REPO_ROOT/painter:$RUN${PYTHONPATH:+:$PYTHONPATH}"
[[ "$(uname -s)" == Linux ]] || { echo 'Linux GPU node required' >&2; exit 1; }
[[ -f "$PAINTER_RL_CONFIG" && -f "$PAINTER_RL_MANIFEST" ]] || {
  echo 'frozen anchors and RL config must exist before launch' >&2; exit 1;
}
python3 - "$RUN/rl-config-receipt.json" "$PAINTER_JUDGE_PROTOCOL" <<'PY'
import json,sys
from pathlib import Path
receipt=json.loads(Path(sys.argv[1]).read_text())
if receipt.get('reward') != sys.argv[2]:
    raise SystemExit('judge protocol does not match prepared config receipt')
PY
[[ -f "$PAINTER_GATEWAY_ENV_FILE" && ! -L "$PAINTER_GATEWAY_ENV_FILE" ]] || {
  echo 'private Gateway key file missing' >&2; exit 1;
}
[[ "$(stat -c %a "$PAINTER_GATEWAY_ENV_FILE")" == 600 ]] || {
  echo 'Gateway key file must have mode 600' >&2; exit 1;
}
if [[ "${1:-}" == '--preflight' ]]; then
  PAINTER_RL_PREFLIGHT_ONLY=1 bash "$RUN/../photo-curriculum-rl-20260922/launch_rl.sh" "$RUN"
  exit
fi
[[ $# == 0 ]] || { echo 'usage: run.sh [--preflight]' >&2; exit 2; }
[[ ! -e "$RUN/async-output/$PAINTER_RL_RUN_NAME" ]] || {
  echo 'existing training output; preserve it before another attempt' >&2; exit 1;
}
# Set from the actual rental quote and start time. Reserve time for eval,
# uploading the last checkpoint, and deleting the pod within the $25 cap.
[[ "${PAINTER_GPU_DEADLINE_UTC:-}" =~ ^[0-9]+$ ]] || {
  echo 'set PAINTER_GPU_DEADLINE_UTC to the Unix timestamp for the rental cap' >&2; exit 2;
}
remaining=$((PAINTER_GPU_DEADLINE_UTC-$(date -u +%s)-5400))
(( remaining > 600 )) || { echo 'insufficient rental budget for RL plus evidence upload' >&2; exit 1; }
timeout --signal=INT --kill-after=90 "${remaining}s" \
  bash "$RUN/../photo-curriculum-rl-20260922/launch_rl.sh" "$RUN"
