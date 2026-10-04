#!/usr/bin/env bash
set -Eeuo pipefail

RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$RUN/../../.." && pwd -P)"
BASE="${PAINTER_BASE_RUN:-$REPO/painter/runs/text-sft512-consistent-20260930}"
THROUGHPUT="${PAINTER_RECOVERY_RUN:-$REPO/painter/runs/throughput-recovery-20261003}"
NODE_ROOT="${PAINTER_NODE_ROOT:-/root/painter}"
PRIME_ROOT="${PRIME_ROOT:-$NODE_ROOT/prime}"
FAST_STORAGE_MODE="${PAINTER_FAST_STORAGE_MODE:-profile_default}"
FAST_STORAGE_ROOT="${PAINTER_FAST_STORAGE_ROOT:-/tmp/painter-runtime-fast}"
FAST_STORAGE_HELPER="$THROUGHPUT/prepare_fast_storage.py"
# Keep the shared root flat as well: inherited base scripts derive REPO from
# RUN/../../.. and must resolve it to the checkout's parent repository.
SHARED="$RUN/../rl-judge-matched-20261004-shared"
MODE="${1:-all}"
[[ "$MODE" == all || "$MODE" == --plan-only ]] || { echo 'usage: prepare.sh [all|--plan-only]' >&2; exit 2; }

python3 "$RUN/prepare_campaign.py" --plan >/dev/null
if [[ "$MODE" == --plan-only ]]; then
  echo 'MATCHED_THREE_JUDGE_PLAN_VALIDATED'
  exit 0
fi
[[ "$(uname -s)" == Linux ]] || { echo 'Linux GPU node required for matched preparation' >&2; exit 1; }
[[ -f "$BASE/prepare_node.sh" ]] || { echo "base preparation missing: $BASE/prepare_node.sh" >&2; exit 1; }
[[ -f "$THROUGHPUT/prepare_renderer_runtime.py" ]] || { echo 'reviewed renderer runtime helper is missing' >&2; exit 1; }

# Bind the profile's finite resource and local-scratch setup before the shared
# bootstrap starts.  The matched campaign deliberately does not call the arm
# validator here because no judge identity exists at this shared stage.
source "$RUN/runtime_env.sh"
export PAINTER_RUN_ROOT="$RUN"
apply_matched_resource_limits
case "$FAST_STORAGE_MODE" in
  off)
    unset PAINTER_FAST_STORAGE_HELPER PAINTER_FAST_STORAGE_ROOT UV_CACHE_DIR
    ;;
  profile_default)
    [[ -f "$FAST_STORAGE_HELPER" ]] || { echo "fast-storage helper missing: $FAST_STORAGE_HELPER" >&2; exit 1; }
    python3 "$FAST_STORAGE_HELPER" --phase cache --run-root "$RUN" \
      --fast-root "$FAST_STORAGE_ROOT" --uv-source "${PAINTER_UV_CACHE_SOURCE:-$HOME/.cache/uv}" \
      --nofile-soft "$PAINTER_NOFILE_SOFT" --nofile-hard "$PAINTER_NOFILE_HARD" \
      --nofile-target "$PAINTER_NOFILE_TARGET" --uv-concurrent-installs "$UV_CONCURRENT_INSTALLS"
    export UV_CACHE_DIR="$FAST_STORAGE_ROOT/uv-cache"
    export PAINTER_FAST_STORAGE_HELPER="$FAST_STORAGE_HELPER"
    export PAINTER_FAST_STORAGE_ROOT="$FAST_STORAGE_ROOT"
    ;;
  *)
    echo 'PAINTER_FAST_STORAGE_MODE must be profile_default or off' >&2
    exit 1
    ;;
esac
export PAINTER_RENDERER_RUNTIME_HELPER="$THROUGHPUT/prepare_renderer_runtime.py"

mkdir -p "$RUN/runtime"
# The renderer overlay is source-only preparation.  The base preparation still
# stages the frozen renderer into the shared root; the actual taskset is bound
# to this cancellable derivative after the shared bootstrap completes.
OVERLAY="$RUN/runtime/watercolour-cancellable-centroid-v1"
if [[ ! -f "$OVERLAY/PATCH-MANIFEST.json" ]]; then
  [[ ! -e "$OVERLAY" ]] || { echo 'partial renderer overlay exists; preserve and inspect it' >&2; exit 1; }
  python3 "$REPO/painter/patch_brush_centroid.py" --renderer-overlay --output-root "$OVERLAY" >"$RUN/cancellable-renderer-build.json"
fi
[[ -f "$OVERLAY/cancellable_watercolour_renderer.py" ]] || { echo 'cancellable renderer overlay is incomplete' >&2; exit 1; }

if [[ ! -f "$SHARED/preparation-receipt.json" ]]; then
  [[ ! -e "$SHARED" || -z "$(find "$SHARED" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
    echo 'partial shared preparation exists without a receipt; preserve and inspect it' >&2; exit 1;
  }
  mkdir -p "$SHARED"
  export PAINTER_RUN_ROOT="$SHARED"
  export PAINTER_RUN_NAME=rl-judge-matched-20261004-shared
  export PAINTER_POLICY_MODE=recovered_policy
  export PAINTER_POLICY_REVISION=ac4d3eda8e5925e2d23992cfdc1a63620ebde573
  export PAINTER_POLICY_PREFIX=text-pairwise-scaled-20260929/step_16
  export PAINTER_POLICY_SHA256=e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c
  export PAINTER_POLICY_DOWNLOAD_SCRIPT="$REPO/painter/runs/text-pairwise-rl-20260928/download_public_adapter.py"
  export PAINTER_POLICY_DOWNLOAD_RECEIPT="$SHARED/policy-adapter-receipt.json"
  export PAINTER_SFT_ADAPTER_PATH_FILE="$SHARED/policy-adapter-path.txt"
  export PAINTER_FROZEN_SFT_ADAPTER_PATH_FILE="$SHARED/frozen-sft-adapter-path.txt"
  export PAINTER_GROUP_SIZE=4 PAINTER_INITIAL_INFLIGHT=32 PAINTER_MAX_INFLIGHT=32
  export PAINTER_GPUS_PER_NODE=2 PAINTER_NUM_TRAIN_GPUS=1 PAINTER_NUM_INFER_GPUS=1
  export PAINTER_ACTIVATION_CPU_OFFLOAD=1 PAINTER_JUDGE_PROTOCOL=text-paint-sft-pairwise-consistent-v1
  export PAINTER_CHECKPOINT_INTERVAL=4 PAINTER_RL_CHECKPOINT_INTERVAL=4
  export PAINTER_BOOTSTRAP_RUN="${PAINTER_BOOTSTRAP_RUN:-$REPO/painter/runs/text-pairwise-rl-20260928}"
  FROZEN="$RUN/runtime/watercolour-p5-brush-2.2.1-centroid-fix-v1"
  if [[ ! -f "$FROZEN/PATCH-MANIFEST.json" ]]; then
    [[ ! -e "$FROZEN" ]] || { echo 'partial frozen renderer root exists; preserve and inspect it' >&2; exit 1; }
    python3 "$REPO/painter/patch_brush_centroid.py" --output-root "$FROZEN" >"$RUN/frozen-renderer-build.json"
  fi
  export PAINTER_RENDERER_ROOT="$FROZEN"
  export PAINTER_RENDERER_PATH="$FROZEN/renderer.py"
  export PAINTER_RENDERER_PYTHON_PATH="${PAINTER_RENDERER_PYTHON_PATH:-$SHARED/renderer-env/bin/python}"
  SHARED_PREPARE_TIMEOUT="${PAINTER_SHARED_PREPARE_TIMEOUT_SECONDS:-3600}"
  [[ "$SHARED_PREPARE_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
    echo 'shared preparation timeout must be a positive integer' >&2; exit 1;
  }
  timeout --signal=TERM --kill-after=30 "$SHARED_PREPARE_TIMEOUT" bash "$BASE/prepare_node.sh"
fi

[[ -f "$SHARED/preparation-receipt.json" ]] || { echo 'shared preparation did not produce a receipt' >&2; exit 1; }
[[ -e "$SHARED/runtime" ]] || ln -s "$RUN/runtime" "$SHARED/runtime"
if [[ "$FAST_STORAGE_MODE" == profile_default ]]; then
  python3 "$FAST_STORAGE_HELPER" --phase prime --run-root "$RUN" \
    --fast-root "$FAST_STORAGE_ROOT" --prime-root "$PRIME_ROOT" \
    --uv-source "${PAINTER_UV_CACHE_SOURCE:-$HOME/.cache/uv}" \
    --nofile-soft "$PAINTER_NOFILE_SOFT" --nofile-hard "$PAINTER_NOFILE_HARD" \
    --nofile-target "$PAINTER_NOFILE_TARGET" --uv-concurrent-installs "$UV_CONCURRENT_INSTALLS"
fi
PY="${PAINTER_PREPARE_PYTHON:-$PRIME_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || { echo "pinned Prime Python missing: $PY" >&2; exit 1; }
if [[ "${PAINTER_JS_SYNTAX_PREFLIGHT:-1}" == 1 ]]; then
  RUNTIME_HELPER="$THROUGHPUT/prepare_renderer_runtime.py"
  BROWSER_ROOT="${PAINTER_BROWSERS_PATH:-$REPO/painter/browsers}"
  [[ -f "$RUNTIME_HELPER" ]] || { echo 'reviewed renderer runtime helper is missing' >&2; exit 1; }
  [[ -d "$BROWSER_ROOT" ]] || { echo "browser bundle is missing: $BROWSER_ROOT" >&2; exit 1; }
  python3 "$RUNTIME_HELPER" --run-root "$SHARED" --renderer-root "$REPO/painter" \
    --renderer-python "$SHARED/renderer-env/bin/python" --browsers "$BROWSER_ROOT" \
    --prime-venv "$PRIME_ROOT/.venv"
fi
# The baseline reuses the shared anchor directory before arm materialization;
# create its balanced16 manifest there so the baseline never depends on an
# arm-specific directory being present.
[[ -f "$SHARED/anchors-frozen/validation-manifest.json" ]] || {
  echo 'shared validation manifest is missing' >&2; exit 1;
}
"$PY" - "$RUN/prepare_campaign.py" "$SHARED/anchors-frozen/validation-manifest.json" \
  "$SHARED/anchors-frozen/validation-matched16.json" <<'PY'
import importlib.util
import sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("matched_prepare_manifest", sys.argv[1])
if spec is None or spec.loader is None:
    raise SystemExit("matched preparation module is unavailable")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.matched16_manifest(Path(sys.argv[2]), Path(sys.argv[3]))
PY
PYTHONPATH="$REPO:$REPO/painter:$RUN${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" "$RUN/prepare_campaign.py" --materialize \
  --shared-root "$SHARED" --prime-root "$PRIME_ROOT" \
  --model-dir "${MODEL_DIR:-$NODE_ROOT/model-cache/qwen/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}" \
  --prepare-config "$REPO/painter/runs/text-pairwise-rl-20260928/prepare_config.py"
echo 'MATCHED_THREE_JUDGE_PREPARED: shared source and three isolated twelve-update configs ready'
