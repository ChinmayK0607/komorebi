#!/usr/bin/env bash
set -Eeuo pipefail

RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$RUN/../../.." && pwd -P)"
PRIME="${PRIME_ROOT:-/root/painter/prime}"
CONFIG="$RUN/native.toml"
MANIFEST="$RUN/anchors-all/grounded-pilot-train-manifest.json"
RECEIPT="$RUN/init-adapter.json"
SFT_PATH_FILE="$RUN/../photo-curriculum-rl-20260922/initial-adapter-path.txt"
[[ "$(uname -s)" == Linux ]] || { echo 'Linux node required' >&2; exit 1; }
for required in "$CONFIG" "$MANIFEST" "$RECEIPT" "$SFT_PATH_FILE"; do
  [[ -f "$required" ]] || { echo "missing prepared input: $required" >&2; exit 1; }
done
[[ "${PAINTER_GPU_DEADLINE_UTC:-}" =~ ^[0-9]+$ ]] || {
  echo 'set PAINTER_GPU_DEADLINE_UTC from finite pod rental deadline' >&2; exit 2;
}
eval "$(python3 - "$RECEIPT" <<'PY'
import json,shlex,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
assert r['verified'] is True
for key,name in [('adapter_path','WATERCOLOUR_INIT_ADAPTER'),
                 ('revision','PAINTER_RL_ADAPTER_REVISION'),
                 ('prefix','PAINTER_RL_ADAPTER_PREFIX'),
                 ('adapter_sha256','PAINTER_RL_ADAPTER_SHA256')]:
    print('export '+name+'='+shlex.quote(r[key]))
PY
)"
export PAINTER_REPO_ROOT="$REPO" BRUSH_ROOT="$REPO/painter/rl_helpers"
export PRIME_ROOT="$PRIME" MODEL_DIR="/root/painter/model-cache/qwen/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-/root/painter/private/hf-token}"
export PAINTER_GATEWAY_ENV_FILE="${PAINTER_GATEWAY_ENV_FILE:-/root/painter/private/gateway.env}"
export PAINTER_FROZEN_SFT_ADAPTER="$(tr -d '\r\n' < "$SFT_PATH_FILE")"
export PAINTER_RL_MODE=recovered_policy PAINTER_RL_FIXED_OPPONENT=1
export PAINTER_RL_LIVE_POLICY_ALIAS="$MODEL_DIR"
export PAINTER_RL_CONFIG="$CONFIG" PAINTER_RL_MANIFEST="$MANIFEST"
export PAINTER_RL_RUN_NAME=text-grounded-pilot-20260930 PAINTER_RL_MAX_STEPS=24
export PAINTER_JUDGE_PROTOCOL=text-paint-sft-pairwise-grounded-v1
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
[[ $# == 0 ]] || { echo 'usage: run.sh [--preflight]' >&2; exit 2; }
[[ ! -e "$RUN/async-output/$PAINTER_RL_RUN_NAME" ]] || {
  echo 'existing pilot output; refusing to overwrite' >&2; exit 1;
}
# Reserve three hours for intermediate/final matched evaluation, public
# verification, and explicit removal before the provider rental deadline.
remaining=$((PAINTER_GPU_DEADLINE_UTC-$(date -u +%s)-10800))
(( remaining > 600 )) || { echo 'insufficient rental time after evidence reserve' >&2; exit 1; }
timeout --signal=INT --kill-after=90 "${remaining}s" \
  bash "$RUN/../photo-curriculum-rl-20260922/launch_rl.sh" "$RUN"
