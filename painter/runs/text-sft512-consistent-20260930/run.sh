#!/usr/bin/env bash
set -Eeuo pipefail

RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$RUN/../../.." && pwd -P)"
NODE_ROOT="${PAINTER_NODE_ROOT:-/root/painter}"
PRIME_ROOT="${PRIME_ROOT:-$NODE_ROOT/prime}"
MODEL_REVISION="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MODEL_DIR="${MODEL_DIR:-$NODE_ROOT/model-cache/qwen/$MODEL_REVISION}"
RUN_NAME="text-sft512-consistent-20260930"
CONFIG="${PAINTER_RL_CONFIG:-$RUN/native.toml}"
MANIFEST="${PAINTER_RL_MANIFEST:-$RUN/anchors-frozen/train-manifest.json}"
SFT_PATH_FILE="${PAINTER_SFT_ADAPTER_PATH_FILE:-$RUN/../photo-curriculum-rl-20260922/initial-adapter-path.txt}"

[[ "$(uname -s)" == Linux ]] || { echo 'Linux GPU node required' >&2; exit 1; }
for required in "$CONFIG" "$MANIFEST" "$RUN/rl-config-receipt.json" \
  "$RUN/launch_rl.sh" "$RUN/run_from_sft.py" "$RUN/train_from_sft.py" \
  "$RUN/vendor/integrations/watercolour/renderer.py" "$RUN/renderer-env/bin/python" \
  "$RUN/sitecustomize.py" "$REPO/painter/consistent_group_admission.py"; do
  [[ -e "$required" ]] || { echo "missing prepared input: $required" >&2; exit 1; }
done
[[ -r "$SFT_PATH_FILE" ]] || { echo 'verified SFT adapter path file missing' >&2; exit 1; }
WATERCOLOUR_INIT_ADAPTER="$(tr -d '\r\n' < "$SFT_PATH_FILE")"
[[ -f "$WATERCOLOUR_INIT_ADAPTER/adapter_model.safetensors" ]] || {
  echo 'verified SFT initializer is unavailable' >&2; exit 1;
}

export PAINTER_REPO_ROOT="$REPO" BRUSH_ROOT="$REPO/painter/rl_helpers"
export PRIME_ROOT MODEL_DIR WATERCOLOUR_INIT_ADAPTER
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$NODE_ROOT/private/hf-token}"
export PAINTER_GATEWAY_ENV_FILE="${PAINTER_GATEWAY_ENV_FILE:-$NODE_ROOT/private/gateway.env}"
export PAINTER_RL_MODE=fresh PAINTER_RL_FIXED_OPPONENT=1
export PAINTER_RL_CONFIG="$CONFIG" PAINTER_RL_MANIFEST="$MANIFEST"
export PAINTER_RL_RUN_NAME="$RUN_NAME" PAINTER_RL_MAX_STEPS=50
export PAINTER_RL_ADAPTER_REVISION=adf687e5b321514eda7be6eab050434fdc8e44ff
export PAINTER_RL_ADAPTER_PREFIX=photo-curriculum-sft-20260922-balanced/step_512
export PAINTER_RL_ADAPTER_SHA256=bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740
export PAINTER_FROZEN_SFT_ALIAS=sft-step512
export PAINTER_JUDGE_PROTOCOL=text-paint-sft-pairwise-consistent-v1
export PAINTER_JUDGE_MODEL=xiaomi/mimo-v2.6-flash PAINTER_JUDGE_EFFORT=gateway-default
export PAINTER_GPUS_PER_NODE="${PAINTER_GPUS_PER_NODE:-2}"
export PAINTER_NUM_TRAIN_GPUS="${PAINTER_NUM_TRAIN_GPUS:-1}"
export PAINTER_NUM_INFER_GPUS="${PAINTER_NUM_INFER_GPUS:-1}"
export PAINTER_GROUP_RANK_MODULE="${PAINTER_GROUP_RANK_MODULE:-painter.consistent_group_admission}"
export PAINTER_WHOLE_GROUP_ADMISSION=1 PAINTER_WHOLE_GROUP_SIZE=8
export PAINTER_CONSISTENT_GUARD_RECEIPT="${PAINTER_CONSISTENT_GUARD_RECEIPT:-$RUN/consistent-group-admission-activation.json}"
export PYTHONPATH="$REPO:$REPO/painter:$RUN${PYTHONPATH:+:$PYTHONPATH}"

[[ -f "$PAINTER_GATEWAY_ENV_FILE" && ! -L "$PAINTER_GATEWAY_ENV_FILE" ]] || {
  echo 'private Gateway key file missing' >&2; exit 1;
}
[[ "$(stat -c %a "$PAINTER_GATEWAY_ENV_FILE")" == 600 ]] || {
  echo 'Gateway key file must be mode 600' >&2; exit 1;
}

if [[ "${1:-}" == --preflight ]]; then
  PAINTER_RL_PREFLIGHT_ONLY=1 PAINTER_RL_SKIP_EDITABLE_INSTALL=1 \
    bash "$RUN/launch_rl.sh" "$RUN"
  exit 0
fi
[[ $# == 0 ]] || { echo 'usage: run.sh [--preflight]' >&2; exit 2; }
[[ "${PAINTER_RL_SUPERVISED:-0}" == 1 ]] || {
  echo 'finite supervisor required: set PAINTER_RL_SUPERVISED=1 only for its owned process group' >&2
  exit 2
}
[[ ! -e "$RUN/async-output/$RUN_NAME" ]] || {
  echo 'existing training output; preserve it before another attempt' >&2; exit 1;
}

# launch_rl.sh is the reviewed Prime entrypoint copied during preparation.  It
# inherits this process group, so finite_supervisor.py can terminate the whole
# training/inference tree by its private PGID at a trend or cost boundary.
exec bash "$RUN/launch_rl.sh" "$RUN"
