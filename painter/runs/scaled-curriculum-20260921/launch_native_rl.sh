#!/usr/bin/env bash
set -Eeuo pipefail

# Preparation/launch recipe for the pinned native Prime run.  This script is
# intentionally inert until an operator invokes it on a prepared Linux node.
ROOT="$(cd "${1:?usage: launch_native_rl.sh RUN_ROOT}" && pwd -P)"
REPO_ROOT="$(cd "${PAINTER_REPO_ROOT:?set PAINTER_REPO_ROOT to the repository checkout}" && pwd -P)"
PRIME_ROOT="$(cd "${PRIME_ROOT:?set PRIME_ROOT to the pinned Prime checkout}" && pwd -P)"
PRIME_COMMIT="26b3131d2716a4f8210b165df584b83b4bc54f61"
BASE_MODEL_ID="Qwen/Qwen3.8-27B"
BASE_MODEL_REVISION="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SFT_ADAPTER_REVISION="b7103dd9923f19449ccf814f9b67bb2dcfed0909"
SFT_ADAPTER_PREFIX="contract-quality-20260918/step_24"
SFT_ADAPTER_SHA256="d10eeb3b7b53fc539aa9d5e71eb98bcfe1c59d6a4878c7a69c93b8b09b59bcfa"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the staged base model directory}"
WATERCOLOUR_INIT_ADAPTER="${WATERCOLOUR_INIT_ADAPTER:?set WATERCOLOUR_INIT_ADAPTER to the step-24 adapter directory}"
GPUS_PER_NODE="${PAINTER_GPUS_PER_NODE:-2}"
NUM_TRAIN_GPUS="${PAINTER_NUM_TRAIN_GPUS:-1}"
NUM_INFER_GPUS="${PAINTER_NUM_INFER_GPUS:-1}"
CONFIG="${PAINTER_RL_CONFIG:-$ROOT/native-rl.toml}"
MODEL_RECEIPT="${PAINTER_MODEL_RECEIPT:-$ROOT/model-receipt.json}"
PYTHON="${PRIME_ROOT}/.venv/bin/python"
export BRUSH_ROOT="$ROOT/source/brush-rl"
export BRUSH_MODEL_PROFILE="qwen38_27b"

[[ -x "$PYTHON" ]] || { echo "missing pinned Prime Python: $PYTHON" >&2; exit 1; }
[[ "$(git -C "$PRIME_ROOT" rev-parse HEAD)" == "$PRIME_COMMIT" ]] || {
  echo "PRIME_ROOT is not pinned to $PRIME_COMMIT" >&2
  exit 1
}
[[ -d "$MODEL_DIR" ]] || { echo "model directory is missing: $MODEL_DIR" >&2; exit 1; }
(( NUM_TRAIN_GPUS > 0 && NUM_INFER_GPUS > 0 && NUM_TRAIN_GPUS + NUM_INFER_GPUS <= GPUS_PER_NODE )) || {
  echo "GPU roles must be positive and fit gpus_per_node ($NUM_TRAIN_GPUS + $NUM_INFER_GPUS > $GPUS_PER_NODE)" >&2
  exit 1
}
[[ -f "$WATERCOLOUR_INIT_ADAPTER/adapter_model.safetensors" ]] || {
  echo "portable SFT adapter is missing adapter_model.safetensors" >&2
  exit 1
}
[[ -f "$MODEL_RECEIPT" ]] || { echo "model receipt is missing: $MODEL_RECEIPT" >&2; exit 1; }
"$PYTHON" - "$MODEL_RECEIPT" "$WATERCOLOUR_INIT_ADAPTER" \
  "$BASE_MODEL_ID" "$BASE_MODEL_REVISION" "$SFT_ADAPTER_REVISION" \
  "$SFT_ADAPTER_PREFIX" "$SFT_ADAPTER_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

receipt_path, adapter_path, model_id, model_revision, adapter_revision, prefix, expected_sha = sys.argv[1:]
receipt = json.loads(Path(receipt_path).read_text())
model = receipt.get("model", {})
adapter = receipt.get("adapters", {}).get("sft", {})
checks = {
    "model.id": (model.get("id"), model_id),
    "model.revision": (model.get("revision"), model_revision),
    "model.resolved_revision": (model.get("resolved_revision"), model_revision),
    "adapters.sft.revision": (adapter.get("revision"), adapter_revision),
    "adapters.sft.resolved_revision": (adapter.get("resolved_revision"), adapter_revision),
    "adapters.sft.prefix": (adapter.get("prefix"), prefix),
    "adapters.sft.sha256": (adapter.get("sha256"), expected_sha),
}
for name, (actual, expected) in checks.items():
    if actual != expected:
        raise SystemExit(f"{name} mismatch: {actual!r} != {expected!r}")
weight = Path(adapter_path) / "adapter_model.safetensors"
digest = hashlib.sha256(weight.read_bytes()).hexdigest()
if digest != expected_sha:
    raise SystemExit(f"adapter bytes mismatch: {digest} != {expected_sha}")
PY
[[ -f "$ROOT/scaled-training-manifest.json" ]] || {
  echo "scaled-training-manifest.json is missing from $ROOT" >&2
  exit 1
}
for required in aesthetic_judge.py judge_queue.py; do
  [[ -f "$ROOT/$required" ]] || { echo "missing $ROOT/$required" >&2; exit 1; }
done

# Install the shared native taskset as top-level Prime/Verifiers plugins.  The
# editable install is dependency-free; the pinned node bootstrap owns the
# verifier and renderer dependencies.
uv pip install --python "$PYTHON" --no-deps --no-build-isolation -e "$REPO_ROOT/painter"

GENERATOR="$REPO_ROOT/painter/prepare_scaled_native_rl.py"
if [[ ! -f "$CONFIG" ]]; then
  "$PYTHON" "$GENERATOR" \
    --output "$CONFIG" \
    --model "$MODEL_DIR" \
    --run-root "$ROOT" \
    --gpus-per-node "$GPUS_PER_NODE" \
    --num-train-gpus "$NUM_TRAIN_GPUS" \
    --num-infer-gpus "$NUM_INFER_GPUS" \
    --prime-root "$PRIME_ROOT" \
    --validate-prime
fi

RUNNER="$REPO_ROOT/painter/runs/scaled-curriculum-20260921/run_from_sft.py"
[[ -f "$RUNNER" ]] || RUNNER="$ROOT/run_from_sft.py"
[[ -f "$RUNNER" ]] || { echo "SFT wrapper is missing: $RUNNER" >&2; exit 1; }
[[ -f "$(dirname "$RUNNER")/train_from_sft.py" ]] || {
  echo "trainer wrapper is missing beside $RUNNER" >&2
  exit 1
}
for required in \
  "$ROOT/source/brush-rl/integrations/prime/watercolour_sft" \
  "$ROOT/source/brush-rl/integrations/speedpainting/mixed_sft"; do
  [[ -d "$required" ]] || { echo "pinned brush helper directory is missing: $required" >&2; exit 1; }
done
export PRIME_ROOT MODEL_DIR WATERCOLOUR_INIT_ADAPTER PAINTER_RUN_ROOT="$ROOT"
export PYTHONPATH="$REPO_ROOT/painter:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "$PRIME_ROOT"
exec "$PYTHON" "$RUNNER" @ "$CONFIG"
