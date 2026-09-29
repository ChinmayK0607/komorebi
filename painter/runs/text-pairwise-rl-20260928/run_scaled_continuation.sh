#!/usr/bin/env bash
set -Eeuo pipefail

# Finite continuation of public RL step 16, preserving the frozen SFT opponent.
# The caller supplies the actual rental deadline; no recurring job is created.
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$RUN/../../.." && pwd -P)"
PRIME="${PRIME_ROOT:-/root/painter/prime}"
CONFIG="$RUN/native-continued.toml"
MANIFEST="$RUN/continued-anchors/train-manifest.json"
RECEIPT="$RUN/continued-step16-download.json"
SFT_PATH_FILE="$RUN/../photo-curriculum-rl-20260922/initial-adapter-path.txt"
[[ "$(uname -s)" == Linux ]] || { echo 'Linux node required' >&2; exit 1; }
for required in "$CONFIG" "$MANIFEST" "$RECEIPT" "$SFT_PATH_FILE"; do
  [[ -f "$required" ]] || { echo "missing prepared input: $required" >&2; exit 1; }
done
[[ "${PAINTER_GPU_DEADLINE_UTC:-}" =~ ^[0-9]+$ ]] || {
  echo 'set PAINTER_GPU_DEADLINE_UTC from the actual finite rental TTL' >&2; exit 2;
}
export PAINTER_REPO_ROOT="$REPO" BRUSH_ROOT="$REPO/painter/rl_helpers"
export PRIME_ROOT="$PRIME" MODEL_DIR="/root/painter/model-cache/qwen/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-/root/painter/private/hf-token}"
export PAINTER_GATEWAY_ENV_FILE="${PAINTER_GATEWAY_ENV_FILE:-/root/painter/private/gateway.env}"
export WATERCOLOUR_INIT_ADAPTER="$(python3 - "$RECEIPT" <<'PY'
import json,sys
from pathlib import Path
receipt=json.loads(Path(sys.argv[1]).read_text())
assert receipt['verified'] is True
assert receipt['revision']=='ac4d3eda8e5925e2d23992cfdc1a63620ebde573'
assert receipt['adapter_sha256']=='e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c'
print(receipt['adapter_path'])
PY
)"
export PAINTER_FROZEN_SFT_ADAPTER="$(tr -d '\r\n' < "$SFT_PATH_FILE")"
export PAINTER_RL_MODE=recovered_policy PAINTER_RL_FIXED_OPPONENT=1
export PAINTER_RL_LIVE_POLICY_ALIAS="$MODEL_DIR"
export PAINTER_RL_ADAPTER_REVISION=ac4d3eda8e5925e2d23992cfdc1a63620ebde573
export PAINTER_RL_ADAPTER_PREFIX=text-pairwise-scaled-20260929/step_16
export PAINTER_RL_ADAPTER_SHA256=e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c
export PAINTER_RL_CONFIG="$CONFIG" PAINTER_RL_MANIFEST="$MANIFEST"
export PAINTER_RL_RUN_NAME=text-pairwise-continue-20260929 PAINTER_RL_MAX_STEPS=50
export PAINTER_JUDGE_PROTOCOL=text-paint-sft-pairwise-margin-v1
export PAINTER_JUDGE_MODEL=xiaomi/mimo-v2.6-flash PAINTER_JUDGE_EFFORT=gateway-default
export PAINTER_GPUS_PER_NODE=2 PAINTER_NUM_TRAIN_GPUS=1 PAINTER_NUM_INFER_GPUS=1
export PYTHONPATH="$REPO/painter:$RUN${PYTHONPATH:+:$PYTHONPATH}"
[[ -f "$PAINTER_GATEWAY_ENV_FILE" && ! -L "$PAINTER_GATEWAY_ENV_FILE" ]] || {
  echo 'private Gateway key file missing' >&2; exit 1;
}
[[ "$(stat -c %a "$PAINTER_GATEWAY_ENV_FILE")" == 600 ]] || {
  echo 'Gateway key file must be mode 600' >&2; exit 1;
}
if [[ "${1:-}" == --preflight ]]; then
  PAINTER_RL_PREFLIGHT_ONLY=1 bash "$RUN/../photo-curriculum-rl-20260922/launch_rl.sh" "$RUN"
  exit
fi
[[ $# == 0 ]] || { echo 'usage: run_scaled_continuation.sh [--preflight]' >&2; exit 2; }
[[ ! -e "$RUN/async-output/$PAINTER_RL_RUN_NAME" ]] || {
  echo 'existing continuation output; refusing to overwrite' >&2; exit 1;
}
# Reserve 75 minutes for evaluation, public checkpoint/evidence verification,
# and explicit pod release. The provider TTL is the hard outer cost boundary.
remaining=$((PAINTER_GPU_DEADLINE_UTC-$(date -u +%s)-4500))
(( remaining > 600 )) || { echo 'insufficient rental time after evidence reserve' >&2; exit 1; }
timeout --signal=INT --kill-after=90 "${remaining}s" \
  bash "$RUN/../photo-curriculum-rl-20260922/launch_rl.sh" "$RUN"
