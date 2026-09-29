#!/usr/bin/env bash
set -Eeuo pipefail

# One-off finite Linux preparation; no training starts until run.sh is called.
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
OLD="$RUN/../text-pairwise-rl-20260928"
ROOT="${PAINTER_NODE_ROOT:-/root/painter}"
PRIME="${PRIME_ROOT:-$ROOT/prime}"
ARCHIVE="$ROOT/cache/scaled-final-20260929.tar.gz"
ALL="$RUN/anchors-all"
[[ "$(uname -s)" == Linux ]] || { echo 'Linux node required' >&2; exit 1; }
[[ $# == 3 ]] || {
  echo 'usage: prepare_node.sh ADAPTER_COMMIT ADAPTER_PREFIX ADAPTER_SHA256' >&2; exit 2;
}
REVISION="$1" PREFIX="$2" ADAPTER_SHA="$3"
[[ "$REVISION" =~ ^[0-9a-f]{40}$ && "$ADAPTER_SHA" =~ ^[0-9a-f]{64}$ ]] || {
  echo 'pinned commit and SHA-256 required' >&2; exit 2;
}
bash "$OLD/setup_node.sh"
# The shared bootstrap stages the renderer under the old run root. Prime's
# launcher resolves these three paths relative to this pilot root, so expose
# the already prepared assets here without reinstalling them.
for item in vendor renderer-env browsers; do
  [[ -e "$OLD/$item" ]] || { echo "missing prepared render asset: $OLD/$item" >&2; exit 1; }
  if [[ ! -e "$RUN/$item" ]]; then
    ln -s "$OLD/$item" "$RUN/$item"
  fi
  [[ -e "$RUN/$item" ]] || { echo "pilot render asset is unavailable: $RUN/$item" >&2; exit 1; }
done
PY="$PRIME/.venv/bin/python"
mkdir -p "$ROOT/cache"
if [[ ! -f "$ARCHIVE" ]]; then
  curl -fLsS --retry 3 --retry-delay 2 \
    'https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/resolve/e39e6e58fc830eb7be0a77a8bdde7d906943d1d3/runs/text-pairwise-rl-20260928/scaled-final-20260929.tar.gz' \
    -o "$ARCHIVE.tmp"
  mv "$ARCHIVE.tmp" "$ARCHIVE"
fi
if [[ ! -f "$ALL/reuse-receipt.json" ]]; then
  "$PY" "$OLD/rehydrate_scaled_anchors.py" --archive "$ARCHIVE" --output "$ALL"
fi
cp "$OLD/evaluate_checkpoint.sh" "$RUN/evaluate_checkpoint.sh"
cp "$OLD/score_validation.py" "$RUN/score_validation.py"
"$PY" "$OLD/curate_grounded_pool.py" \
  --source "$ALL/train-manifest.json" \
  --output "$ALL/grounded-pilot-train-manifest.json"
"$PY" "$RUN/freeze_validation.py" \
  --source "$ALL/validation-manifest.json" \
  --output "$RUN/balanced16-validation-manifest.json"
"$PY" "$OLD/download_public_adapter.py" \
  --revision "$REVISION" --prefix "$PREFIX" --expected-sha256 "$ADAPTER_SHA" \
  --receipt "$RUN/init-adapter.json"
PROPOSED_CONFIG="$RUN/native.toml.proposed.$$"
trap 'rm -f "$PROPOSED_CONFIG"' EXIT
"$PY" "$OLD/prepare_config.py" \
  --model "$ROOT/model-cache/qwen/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" \
  --run-root "$RUN" --manifest "$ALL/grounded-pilot-train-manifest.json" \
  --output "$PROPOSED_CONFIG" --prime-root "$PRIME" \
  --expected-count 40 --max-steps 24 --checkpoint-interval 4 \
  --run-name text-grounded-pilot-20260930 \
  --output-root rl-rollouts-grounded \
  --judge-protocol text-paint-sft-pairwise-grounded-v1 \
  --mode recovered_policy --adapter-revision "$REVISION" \
  --adapter-prefix "$PREFIX" --adapter-sha256 "$ADAPTER_SHA" \
  --learning-rate 1e-6
if [[ -e "$RUN/native.toml" ]]; then
  cmp -s "$PROPOSED_CONFIG" "$RUN/native.toml" || {
    echo 'existing pilot config differs from the pinned proposal' >&2; exit 1;
  }
else
  mv "$PROPOSED_CONFIG" "$RUN/native.toml"
fi
rm -f "$PROPOSED_CONFIG"
trap - EXIT
PAINTER_GPU_DEADLINE_UTC="${PAINTER_GPU_DEADLINE_UTC:-$(($(date -u +%s)+7200))}" \
  bash "$RUN/run.sh" --preflight
echo 'GROUNDED_PILOT_PREPARED: 40 anchors, 24-step finite config; launch pending'
