#!/usr/bin/env bash
set -Eeuo pipefail

# Idempotent Linux setup. It does not start training or evaluation.
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$RUN/../../.." && pwd -P)"
NODE_ROOT="${PAINTER_NODE_ROOT:-/root/painter}"
PRIME_ROOT="${PRIME_ROOT:-$NODE_ROOT/prime}"
# Preserve an explicit local-scratch model location supplied by the structured
# arm; the default remains the historical node-root cache.
MODEL_DIR="${MODEL_DIR:-$NODE_ROOT/model-cache/qwen/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
PRIME_COMMIT="26b3131d2716a4f8210b165df584b83b4bc54f61"
PROBE_ONLY=0
if [[ "${1:-}" == --gpu-probe-only ]]; then
  PROBE_ONLY=1
  shift
fi
[[ $# == 0 ]] || { echo 'usage: setup_node.sh [--gpu-probe-only]' >&2; exit 2; }
[[ "$(uname -s)" == Linux ]] || { echo 'Linux node required' >&2; exit 1; }
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
[[ "$GPU_COUNT" =~ ^[0-9]+$ ]] || { echo 'unable to count visible GPUs' >&2; exit 1; }
if [[ "${PAINTER_EVALUATION_ONLY:-0}" == 1 ]]; then
  [[ "$GPU_COUNT" -ge 1 ]] || {
    echo 'one visible GPU required for evaluation-only mode' >&2; exit 1;
  }
else
  [[ "$GPU_COUNT" -ge 2 ]] || {
    echo 'two visible GPUs required' >&2; exit 1;
  }
fi
if (( PROBE_ONLY )); then
  if [[ "${PAINTER_EVALUATION_ONLY:-0}" == 1 ]]; then
    printf '%s\n' "SETUP_NODE_GPU_PROBE_OK: mode=evaluation_only visible_gpus=$GPU_COUNT required_min=1"
  else
    printf '%s\n' "SETUP_NODE_GPU_PROBE_OK: mode=training visible_gpus=$GPU_COUNT required_min=2"
  fi
  exit 0
fi
[[ -f "$REPO/painter/foundation-lock.json" ]] || { echo 'painter source missing' >&2; exit 1; }
mkdir -p "$RUN/logs" "$NODE_ROOT/private" "$MODEL_DIR"
chmod 700 "$NODE_ROOT/private"
for item in launch_rl.sh run_from_sft.py train_from_sft.py; do
  cp "$RUN/../photo-curriculum-rl-20260922/$item" "$RUN/$item"
done
if [[ ! -x "$NODE_ROOT/bootstrap/bin/uv" ]]; then
  python3 -m venv "$NODE_ROOT/bootstrap"
  "$NODE_ROOT/bootstrap/bin/python" -m pip install -q 'uv==0.11.1' huggingface_hub hf_xet
fi
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$NODE_ROOT/private/hf-token}"
MODEL_DIR="$MODEL_DIR" "$NODE_ROOT/bootstrap/bin/python" - <<'PY' >"$RUN/logs/model-download.log" 2>&1 &
import os
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.8-27B',
  revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
  local_dir=os.environ['MODEL_DIR'])
print('PINNED_MODEL_READY',flush=True)
PY
model_pid=$!
trap 'kill "$model_pid" 2>/dev/null || true' EXIT
"$NODE_ROOT/bootstrap/bin/python" "$RUN/../photo-curriculum-rl-20260922/download_initial_adapter.py"
if [[ ! -d "$PRIME_ROOT/.git" && "${PAINTER_PRIME_SEED_ENABLED:-0}" == 1 ]]; then
  SEED_HELPER="${PAINTER_PRIME_SEED_HELPER:-}"
  SEED_SOURCE="${PAINTER_PRIME_SEED_ROOT:-$NODE_ROOT/prime}"
  [[ -f "$SEED_HELPER" ]] || { echo "Prime seed helper is missing: $SEED_HELPER" >&2; exit 1; }
  [[ -d "$SEED_SOURCE/.git" ]] || { echo "Prime seed source is missing: $SEED_SOURCE" >&2; exit 1; }
  if [[ "$SEED_SOURCE" != "$PRIME_ROOT" ]]; then
    python3 "$SEED_HELPER" --source "$SEED_SOURCE" --destination "$PRIME_ROOT" --expected-commit "$PRIME_COMMIT" \
      >"${PAINTER_PRIME_SEED_RECEIPT:-$RUN/logs/prime-seed-receipt.json}"
  fi
fi
if [[ ! -d "$PRIME_ROOT/.git" ]]; then
  git -c http.version=HTTP/1.1 clone -q https://github.com/PrimeIntellect-ai/prime-rl.git "$PRIME_ROOT"
fi
if git -C "$PRIME_ROOT" cat-file -e "$PRIME_COMMIT^{commit}" 2>/dev/null; then
  git -C "$PRIME_ROOT" checkout -q "$PRIME_COMMIT"
else
  git -C "$PRIME_ROOT" -c http.version=HTTP/1.1 fetch -q origin "$PRIME_COMMIT"
  git -C "$PRIME_ROOT" checkout -q "$PRIME_COMMIT"
fi
PRIME_ROOT="$PRIME_ROOT" PAINTER_RENDER_ROOT="$REPO/painter" UV_BIN="$NODE_ROOT/bootstrap/bin/uv" \
  bash "$RUN/../photo-curriculum-rl-20260922/setup_rl_runtime.sh" "$RUN"
wait "$model_pid"
trap - EXIT
[[ -f "$MODEL_DIR/config.json" ]] || { tail -30 "$RUN/logs/model-download.log" >&2; exit 1; }
"$NODE_ROOT/bootstrap/bin/uv" pip install --python "$PRIME_ROOT/.venv/bin/python" \
  --no-deps --no-build-isolation -e "$REPO/painter"
echo 'TEXT_RL_NODE_READY: run render_baseline.sh; then prepare_config.py and run.sh'
