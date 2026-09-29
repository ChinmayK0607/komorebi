#!/usr/bin/env bash
set -Eeuo pipefail

# One-time, idempotent Linux setup for a 50-step step-16 text-paint continuation.
# Ends after Prime's real task/adapter preflight; launch is a separate command.
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="${PAINTER_NODE_ROOT:-/root/painter}"
PRIME="${PRIME_ROOT:-$ROOT/prime}"
ARCHIVE="$ROOT/cache/scaled-final-20260929.tar.gz"
ANCHORS="$RUN/continued-anchors"
[[ "$(uname -s)" == Linux ]] || { echo 'Linux node required' >&2; exit 1; }
bash "$RUN/setup_node.sh"
PY="$PRIME/.venv/bin/python"
[[ -x "$PY" ]] || { echo 'pinned Prime runtime unavailable' >&2; exit 1; }
mkdir -p "$ROOT/cache"
if [[ ! -f "$ARCHIVE" ]]; then
  curl -fLsS --retry 3 --retry-delay 2 \
    'https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/resolve/e39e6e58fc830eb7be0a77a8bdde7d906943d1d3/runs/text-pairwise-rl-20260928/scaled-final-20260929.tar.gz' \
    -o "$ARCHIVE.tmp"
  mv "$ARCHIVE.tmp" "$ARCHIVE"
fi
if [[ ! -f "$ANCHORS/reuse-receipt.json" ]]; then
  "$PY" "$RUN/rehydrate_scaled_anchors.py" --archive "$ARCHIVE" --output "$ANCHORS"
fi
if [[ ! -f "$RUN/continued-step16-download.json" ]]; then
  "$PY" "$RUN/download_step16.py" --receipt "$RUN/continued-step16-download.json"
fi
if [[ ! -f "$RUN/native-continued.toml" ]]; then
  "$PY" "$RUN/prepare_config.py" \
    --model "$ROOT/model-cache/qwen/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" \
    --run-root "$RUN" --manifest "$ANCHORS/train-manifest.json" \
    --output "$RUN/native-continued.toml" --prime-root "$PRIME" \
    --expected-count 119 --max-steps 50 --checkpoint-interval 5 \
    --run-name text-pairwise-continue-20260929 \
    --output-root rl-rollouts-continued \
    --judge-protocol text-paint-sft-pairwise-margin-v1 \
    --mode recovered_policy \
    --adapter-revision ac4d3eda8e5925e2d23992cfdc1a63620ebde573 \
    --adapter-prefix text-pairwise-scaled-20260929/step_16 \
    --adapter-sha256 e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c \
    --learning-rate 1e-6
fi
# The launcher requires a finite provider deadline even in preflight mode.
PAINTER_GPU_DEADLINE_UTC="${PAINTER_GPU_DEADLINE_UTC:-$(($(date -u +%s)+7200))}" \
  bash "$RUN/run_scaled_continuation.sh" --preflight
echo 'SCALE2_PREPARED: 119 anchors, step-16 initializer, 50-step finite config; launch pending'
