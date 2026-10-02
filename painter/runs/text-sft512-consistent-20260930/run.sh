#!/usr/bin/env bash
set -Eeuo pipefail

RUN="${PAINTER_RUN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)}"
REPO="$(cd "$RUN/../../.." && pwd -P)"
NODE_ROOT="${PAINTER_NODE_ROOT:-/root/painter}"
PRIME_ROOT="${PRIME_ROOT:-$NODE_ROOT/prime}"
MODEL_REVISION="${PAINTER_BASE_MODEL_REVISION:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
MODEL_DIR="${MODEL_DIR:-$NODE_ROOT/model-cache/qwen/$MODEL_REVISION}"
RUN_NAME="${PAINTER_RUN_NAME:-text-sft512-consistent-20260930}"
CONFIG="${PAINTER_RL_CONFIG:-$RUN/native.toml}"
MANIFEST="${PAINTER_RL_MANIFEST:-$RUN/anchors-frozen/train-manifest.json}"
SFT_PATH_FILE="${PAINTER_SFT_ADAPTER_PATH_FILE:-$RUN/../photo-curriculum-rl-20260922/initial-adapter-path.txt}"
FROZEN_SFT_PATH_FILE="${PAINTER_FROZEN_SFT_ADAPTER_PATH_FILE:-$RUN/../photo-curriculum-rl-20260922/initial-adapter-path.txt}"
POLICY_MODE="${PAINTER_POLICY_MODE:-fresh}"
POLICY_REVISION="${PAINTER_POLICY_REVISION:-adf687e5b321514eda7be6eab050434fdc8e44ff}"
POLICY_PREFIX="${PAINTER_POLICY_PREFIX:-photo-curriculum-sft-20260922-balanced/step_512}"
POLICY_SHA256="${PAINTER_POLICY_SHA256:-bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740}"

[[ "$(uname -s)" == Linux ]] || { echo 'Linux GPU node required' >&2; exit 1; }
for required in "$CONFIG" "$MANIFEST" "$RUN/rl-config-receipt.json" \
  "$RUN/launch_rl.sh" "$RUN/run_from_sft.py" "$RUN/train_from_sft.py" \
  "$RUN/vendor/integrations/watercolour/renderer.py" "$RUN/renderer-env/bin/python" \
  "$RUN/sitecustomize.py" "$REPO/painter/consistent_group_admission.py"; do
  [[ -e "$required" ]] || { echo "missing prepared input: $required" >&2; exit 1; }
done
[[ -f "$REPO/painter/final_action_credit.py" ]] || {
  echo 'final action credit module is missing from the runtime pack' >&2; exit 1;
}
[[ -r "$SFT_PATH_FILE" ]] || { echo 'verified SFT adapter path file missing' >&2; exit 1; }
WATERCOLOUR_INIT_ADAPTER="$(tr -d '\r\n' < "$SFT_PATH_FILE")"
[[ -f "$WATERCOLOUR_INIT_ADAPTER/adapter_model.safetensors" ]] || {
  echo 'verified SFT initializer is unavailable' >&2; exit 1;
}

export PAINTER_REPO_ROOT="$REPO" BRUSH_ROOT="$REPO/painter/rl_helpers"
export PRIME_ROOT MODEL_DIR WATERCOLOUR_INIT_ADAPTER
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$NODE_ROOT/private/hf-token}"
export PAINTER_GATEWAY_ENV_FILE="${PAINTER_GATEWAY_ENV_FILE:-$NODE_ROOT/private/gateway.env}"
export PAINTER_RL_MODE="$POLICY_MODE" PAINTER_RL_FIXED_OPPONENT=1
export PAINTER_RL_CONFIG="$CONFIG" PAINTER_RL_MANIFEST="$MANIFEST"
export PAINTER_RL_RUN_NAME="$RUN_NAME" PAINTER_RL_MAX_STEPS=50
export PAINTER_RL_ADAPTER_REVISION="$POLICY_REVISION"
export PAINTER_RL_ADAPTER_PREFIX="$POLICY_PREFIX"
export PAINTER_RL_ADAPTER_SHA256="$POLICY_SHA256"
export PAINTER_FROZEN_SFT_ALIAS=sft-step512
export PAINTER_JUDGE_PROTOCOL="${PAINTER_JUDGE_PROTOCOL:-text-paint-sft-pairwise-consistent-v1}"
export PAINTER_JUDGE_FORMAT="${PAINTER_JUDGE_FORMAT:-identity-joint-side-schema-v1}"
export PAINTER_JUDGE_MODEL="${PAINTER_JUDGE_MODEL:-xiaomi/mimo-v2.6-flash}" \
  PAINTER_JUDGE_EFFORT="${PAINTER_JUDGE_EFFORT:-gateway-default}"
export PAINTER_RENDER_WORKERS="${PAINTER_RENDER_WORKERS:-8}"
export PAINTER_GPUS_PER_NODE="${PAINTER_GPUS_PER_NODE:-2}"
export PAINTER_NUM_TRAIN_GPUS="${PAINTER_NUM_TRAIN_GPUS:-1}"
export PAINTER_NUM_INFER_GPUS="${PAINTER_NUM_INFER_GPUS:-1}"
export PAINTER_GROUP_RANK_MODULE="${PAINTER_GROUP_RANK_MODULE:-painter.consistent_group_admission}"
export PAINTER_GROUP_SIZE="${PAINTER_GROUP_SIZE:-8}"
export PAINTER_WHOLE_GROUP_ADMISSION=1 PAINTER_WHOLE_GROUP_SIZE="$PAINTER_GROUP_SIZE"
export PAINTER_CONSISTENT_GUARD_RECEIPT="${PAINTER_CONSISTENT_GUARD_RECEIPT:-$RUN/consistent-group-admission-activation.json}"
export PAINTER_CONSISTENT_GROUP_TELEMETRY="${PAINTER_CONSISTENT_GROUP_TELEMETRY:-$RUN/consistent-group-admission.jsonl}"
export PAINTER_REQUIRE_VALID_FINAL_ACTION=1 PAINTER_JS_SYNTAX_PREFLIGHT=1
export PAINTER_RENDERER_PATH="${PAINTER_RENDERER_PATH:-$RUN/vendor/integrations/watercolour/renderer.py}"
export PAINTER_RENDERER_PYTHON_PATH="${PAINTER_RENDERER_PYTHON_PATH:-$RUN/renderer-env/bin/python}"
[[ -f "$PAINTER_RENDERER_PATH" && -x "$PAINTER_RENDERER_PYTHON_PATH" ]] || {
  echo 'explicit renderer paths are unavailable' >&2; exit 1;
}
if [[ "$POLICY_MODE" == recovered_policy ]]; then
  [[ -r "$FROZEN_SFT_PATH_FILE" ]] || { echo 'frozen SFT adapter path file missing' >&2; exit 1; }
  FROZEN_SFT_ADAPTER="$(tr -d '\r\n' < "$FROZEN_SFT_PATH_FILE")"
  [[ -f "$FROZEN_SFT_ADAPTER/adapter_model.safetensors" && -f "$FROZEN_SFT_ADAPTER/adapter_config.json" ]] || {
    echo 'verified frozen SFT adapter is unavailable' >&2; exit 1;
  }
  [[ "$(cd "$WATERCOLOUR_INIT_ADAPTER" && pwd -P)" != "$(cd "$FROZEN_SFT_ADAPTER" && pwd -P)" ]] || {
    echo 'policy initializer and frozen SFT adapter must be distinct' >&2; exit 1;
  }
  export PAINTER_FROZEN_SFT_ADAPTER="$FROZEN_SFT_ADAPTER"
fi
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
