#!/usr/bin/env bash
set -Eeuo pipefail

# Finite post-baseline preparation. It freezes SFT opponents, samples one
# independent SFT control for validation and train probe, and preflights Prime.
# It does not launch optimizer updates.
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PRIME_ROOT="${PRIME_ROOT:-/root/painter/prime}"
PY="$PRIME_ROOT/.venv/bin/python"
POOL="$RUN/prepared/scaled-run-pool/pool.json"
BASELINE="$RUN/baseline-scaled-184"
ANCHORS="$RUN/anchors-scaled"
[[ "$(uname -s)" == Linux ]] || { echo 'Linux node required' >&2; exit 1; }
[[ -x "$PY" && -f "$POOL" && -d "$BASELINE" ]] || {
  echo 'pinned runtime, run pool, or baseline is missing' >&2; exit 1;
}
[[ -f /root/painter/baseline-scaled-184.exit && "$(cat /root/painter/baseline-scaled-184.exit)" == 0 ]] || {
  echo 'baseline must finish successfully before freezing anchors' >&2; exit 1;
}
[[ ! -e "$RUN/anchors-scaled" && ! -e "$RUN/anchors-scaled-first" ]] || {
  echo 'scaled anchors already exist; preserve the prior preparation' >&2; exit 1;
}

args=(--pool "$POOL" --eval-root "$BASELINE" --expected-count 184
      --validation-expected 48 --train-simple 12 --train-intermediate 8
      --train-complex 48)
if "$PY" "$RUN/build_anchors.py" "${args[@]}" --output "$RUN/anchors-scaled-first"; then
  ANCHORS="$RUN/anchors-scaled-first"
else
  "$PY" "$RUN/prepare_retry_pool.py" --pool "$POOL" --source "$BASELINE" \
    --output "$RUN/prepared/scaled-retry-1.json"
  PAINTER_POOL_PATH="$RUN/prepared/scaled-retry-1.json" \
    PAINTER_EVAL_OUT="$RUN/baseline-scaled-retry-1" \
    PAINTER_EVAL_SEED=918201 PAINTER_BUILD_ANCHORS=0 \
    bash "$RUN/render_baseline.sh"
  "$PY" "$RUN/build_anchors.py" "${args[@]}" \
    --retry-root "$RUN/baseline-scaled-retry-1" --output "$ANCHORS"
fi
"$PY" "$RUN/freeze_scaled_probe.py" --train "$ANCHORS/train-manifest.json" \
  --output "$ANCHORS/train-probe-manifest.json"
"$PY" "$RUN/control_sets.py" combine \
  --validation "$ANCHORS/validation-manifest.json" \
  --probe "$ANCHORS/train-probe-manifest.json" \
  --output "$RUN/prepared/scaled-control-cases.json"
PAINTER_VALIDATION_MANIFEST="$RUN/prepared/scaled-control-cases.json" \
  PAINTER_CONTROL_POOL="$RUN/prepared/scaled-control-eval-pool.json" \
  PAINTER_CONTROL_OUT="$RUN/validation/scaled-sft-control" \
  PAINTER_JUDGE_PROTOCOL=text-paint-sft-pairwise-margin-v1 \
  bash "$RUN/evaluate_sft_control.sh"
"$PY" "$RUN/control_sets.py" subset \
  --summary "$RUN/validation/scaled-sft-control/comparison/summary.json" \
  --manifest "$ANCHORS/validation-manifest.json" \
  --output "$RUN/validation/scaled-sft-control/validation-summary.json"
"$PY" "$RUN/control_sets.py" subset \
  --summary "$RUN/validation/scaled-sft-control/comparison/summary.json" \
  --manifest "$ANCHORS/train-probe-manifest.json" \
  --output "$RUN/validation/scaled-sft-control/probe-summary.json"

printf '%s\n' "$ANCHORS" > "$RUN/prepared/scaled-anchors-path.txt"
"$PY" "$RUN/prepare_config.py" \
  --model /root/painter/model-cache/qwen/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --run-root "$RUN" --manifest "$ANCHORS/train-manifest.json" \
  --output "$RUN/native-scaled.toml" --prime-root "$PRIME_ROOT" \
  --expected-count 68 --max-steps 24 --checkpoint-interval 8 \
  --run-name text-pairwise-scaled-20260929 \
  --output-root rl-rollouts-scaled \
  --judge-protocol text-paint-sft-pairwise-margin-v1
PAINTER_RL_CONFIG="$RUN/native-scaled.toml" \
  PAINTER_RL_MANIFEST="$ANCHORS/train-manifest.json" \
  PAINTER_RL_RUN_NAME=text-pairwise-scaled-20260929 \
  PAINTER_RL_MAX_STEPS=24 \
  PAINTER_JUDGE_PROTOCOL=text-paint-sft-pairwise-margin-v1 \
  bash "$RUN/run.sh" --preflight
echo "SCALED_RUN_PREPARED anchors=$ANCHORS config=$RUN/native-scaled.toml"
