#!/usr/bin/env bash
set -Eeuo pipefail

RUN="${PAINTER_RUN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)}"
REPO="$(cd "$RUN/../../.." && pwd -P)"
BASE="${PAINTER_BASE_RUN:-$REPO/painter/runs/text-sft512-consistent-20260930}"
BOOTSTRAP="${PAINTER_BOOTSTRAP_RUN:-$REPO/painter/runs/text-pairwise-rl-20260928}"
NODE_ROOT="${PAINTER_NODE_ROOT:-/root/painter}"
PRIME_ROOT="${PRIME_ROOT:-$NODE_ROOT/prime}"
FROZEN_RENDERER_ROOT="${PAINTER_FROZEN_RENDERER_ROOT:-$RUN/runtime/watercolour-p5-brush-2.2.1-centroid-fix-v1}"
OVERLAY_ROOT="${PAINTER_CANCELLABLE_RENDERER_ROOT:-$RUN/runtime/watercolour-cancellable-centroid-v1}"
CUSTOM_PREPARE_PYTHON="${PAINTER_PREPARE_PYTHON:-}"

export PAINTER_RUN_ROOT="$RUN"
source "$RUN/runtime_env.sh"
validate_recovery_bindings

usage() {
  echo 'usage: prepare_single.sh [all|--plan-only]' >&2
  exit 2
}

MODE="${1:-all}"
[[ "$MODE" == all || "$MODE" == --plan-only ]] || usage

# The plan check is dependency-free and may run through a fake interpreter in
# tests.  It deliberately does not import Prime, pydantic, or any model stack.
PLAN_PY="${CUSTOM_PREPARE_PYTHON:-python3}"
[[ -x "$PLAN_PY" || "$PLAN_PY" == python3 ]] || {
  echo "preparation interpreter is not executable: $PLAN_PY" >&2
  exit 1
}
"$PLAN_PY" - <<'PY'
import json
import os

required = {
    "PAINTER_RUN_NAME": "throughput-recovery-20261003",
    "PAINTER_EXPERIMENT_NAME": "throughput-recovery-flash-20261003",
    "PAINTER_JUDGE_MODEL": "xiaomi/mimo-v2.6-flash",
    "PAINTER_JUDGE_EFFORT": "medium",
    "PAINTER_JUDGE_FORMAT": "identity-joint-side-schema-v2-generic-v1",
    "PAINTER_JUDGE_PROTOCOL": "text-paint-sft-pairwise-consistent-v1",
    "PAINTER_POLICY_REVISION": "ac4d3eda8e5925e2d23992cfdc1a63620ebde573",
    "PAINTER_POLICY_PREFIX": "text-pairwise-scaled-20260929/step_16",
    "PAINTER_POLICY_SHA256": "e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c",
    "PAINTER_GROUP_SIZE": "4",
    "PAINTER_INITIAL_INFLIGHT": "32",
    "PAINTER_MAX_INFLIGHT": "32",
    "PAINTER_RL_MAX_STEPS": "4",
    "PAINTER_RL_CHECKPOINT_INTERVAL": "1",
    "ADAPTER_ONLY_CHECKPOINTS": "1",
    "PAINTER_ACTIVATION_CPU_OFFLOAD": "1",
    "PAINTER_P5_BRUSH_API_REFERENCE_V2": "1",
    "PAINTER_REQUIRE_VALID_FINAL_ACTION": "1",
}
actual = {key: os.environ.get(key) for key in required}
if actual != required:
    raise SystemExit("single-arm plan binding mismatch: " + json.dumps(actual, sort_keys=True))
topology = {key: int(os.environ[key]) for key in (
    "PAINTER_GPUS_PER_NODE", "PAINTER_NUM_TRAIN_GPUS", "PAINTER_NUM_INFER_GPUS"
)}
if topology["PAINTER_NUM_TRAIN_GPUS"] != 1 or topology["PAINTER_NUM_TRAIN_GPUS"] + topology["PAINTER_NUM_INFER_GPUS"] != topology["PAINTER_GPUS_PER_NODE"]:
    raise SystemExit("single-arm plan topology is not one trainer plus remaining inference GPUs")
print(json.dumps({"schema": "throughput-recovery-single-arm-plan-v1", "topology": topology}, sort_keys=True))
PY

if [[ "$MODE" == --plan-only ]]; then
  echo 'THROUGHPUT_RECOVERY_SINGLE_ARM_PLAN_VALIDATED'
  exit 0
fi

[[ "$(uname -s)" == Linux ]] || { echo 'Linux GPU node required for single-arm preparation' >&2; exit 1; }
[[ -f "$BASE/prepare_node.sh" ]] || { echo "base preparation missing: $BASE/prepare_node.sh" >&2; exit 1; }
[[ -f "$BOOTSTRAP/setup_node.sh" ]] || { echo "reviewed bootstrap setup missing: $BOOTSTRAP/setup_node.sh" >&2; exit 1; }
mkdir -p "$RUN/runtime"

if [[ -f "$FROZEN_RENDERER_ROOT/PATCH-MANIFEST.json" ]]; then
  [[ -f "$FROZEN_RENDERER_ROOT/renderer.py" && -f "$FROZEN_RENDERER_ROOT/assets/p5.brush.js" ]] || {
    echo 'existing frozen centroid runtime is incomplete' >&2
    exit 1
  }
else
  [[ ! -e "$FROZEN_RENDERER_ROOT" ]] || { echo 'partial frozen centroid runtime exists without a manifest' >&2; exit 1; }
  python3 "$REPO/painter/patch_brush_centroid.py" --output-root "$FROZEN_RENDERER_ROOT" >"$RUN/frozen-renderer-build.json"
fi

if [[ -f "$OVERLAY_ROOT/PATCH-MANIFEST.json" ]]; then
  [[ -f "$OVERLAY_ROOT/cancellable_watercolour_renderer.py" && -f "$OVERLAY_ROOT/frozen_watercolour_renderer.py" ]] || {
    echo 'existing cancellable renderer overlay is incomplete' >&2
    exit 1
  }
  python3 - "$REPO" "$OVERLAY_ROOT" <<'PY'
import importlib.util
import sys
from pathlib import Path

repo, root = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("patch_brush_centroid", repo / "painter/patch_brush_centroid.py")
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.validate_renderer_overlay(root)
PY
else
  [[ ! -e "$OVERLAY_ROOT" ]] || { echo 'partial cancellable renderer overlay exists without a manifest' >&2; exit 1; }
  python3 "$REPO/painter/patch_brush_centroid.py" --renderer-overlay --output-root "$OVERLAY_ROOT" >"$RUN/cancellable-renderer-build.json"
fi

# The existing bootstrap/staging flow is reused for dependencies, public
# adapters, and the fixed 40/47 manifests.  Its renderer staging check needs a
# frozen ``renderer.py``; the profile writer below switches the actual taskset
# to the cancellable derivative immediately after staging.
export PAINTER_PATCHED_RENDERER_ROOT="$FROZEN_RENDERER_ROOT"
export PAINTER_RENDERER_ROOT="$FROZEN_RENDERER_ROOT"
export PAINTER_RENDERER_PATH="$FROZEN_RENDERER_ROOT/renderer.py"
export PAINTER_RENDERER_PYTHON_PATH="${PAINTER_RENDERER_PYTHON_PATH:-$RUN/renderer-env/bin/python}"
export PAINTER_BOOTSTRAP_RUN="$BOOTSTRAP"
export PAINTER_POLICY_DOWNLOAD_SCRIPT="$REPO/painter/runs/text-pairwise-rl-20260928/download_public_adapter.py"
export PAINTER_CHECKPOINT_INTERVAL=1 PAINTER_RL_CHECKPOINT_INTERVAL=1
export PAINTER_ACTIVATION_CPU_OFFLOAD=1 ADAPTER_ONLY_CHECKPOINTS=1
export PAINTER_JUDGE_MODEL=xiaomi/mimo-v2.6-flash PAINTER_JUDGE_EFFORT=medium
export PAINTER_JUDGE_FORMAT=identity-joint-side-schema-v2-generic-v1

if [[ -f "$RUN/preparation-receipt.json" ]] && grep -q 'throughput-recovery-single-arm-preparation-v1' "$RUN/preparation-receipt.json"; then
  echo 'reusing existing profile-bound single-arm preparation'
else
  bash "$BASE/prepare_node.sh"
fi

PY="$PRIME_ROOT/.venv/bin/python"
[[ -x "$PY" ]] || { echo "pinned Prime Python missing after bootstrap: $PY" >&2; exit 1; }
export PAINTER_RUNTIME_VALIDATE_PYTHON="$PY"
validate_recovery_profile_environment "$PY"

# Make the opt-in renderer explicit in every run-local fixed manifest.
"$PY" - "$RUN" "$OVERLAY_ROOT/cancellable_watercolour_renderer.py" "$PAINTER_RENDERER_PYTHON_PATH" <<'PY'
import json
import sys
from pathlib import Path

run, renderer, renderer_python = map(Path, sys.argv[1:])
for name, key in (("anchors-frozen/train-manifest.json", "rows"), ("anchors-frozen/validation-manifest.json", "cases")):
    path = run / name
    value = json.loads(path.read_text())
    rows = value.get(key)
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{name} has no {key}")
    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit(f"{name} contains a malformed row")
        row["renderer"] = str(renderer)
        row["renderer_python"] = str(renderer_python)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY

"$PY" "$RUN/prepare_profile_single.py" \
  --repo "$REPO" --run-root "$RUN" --model "$MODEL_DIR" --prime-root "$PRIME_ROOT" \
  --manifest "$RUN/anchors-frozen/train-manifest.json" \
  --renderer "$OVERLAY_ROOT/cancellable_watercolour_renderer.py" \
  --renderer-python "$PAINTER_RENDERER_PYTHON_PATH"

echo 'THROUGHPUT_RECOVERY_SINGLE_ARM_PREPARED: profile-bound four-update Flash arm ready'
