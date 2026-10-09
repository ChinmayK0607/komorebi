#!/usr/bin/env bash
set -Eeuo pipefail

# Linux-node-only four-update, three-rank strict SFT warmup.  The controller must provide
# already-approved flow-agent JSONL; this script does not render, author, or
# repair targets.  It also never copies HF_TOKEN_PATH into the warmup output.
RUN="${PAINTER_RUN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)}"
source "$RUN/runtime_env.sh"
# Warmup consumes already-reviewed rich JSONL and is outside RL group
# admission.  Override a stale orchestrator role inherited from a controller
# so its processor/trainer imports cannot attempt the RL guard.
export PAINTER_PRIME_ROLE=warmup
validate_structured_bindings
[[ "$(uname -s)" == Linux ]] || { echo 'warmup training requires the Linux GPU node' >&2; exit 1; }

: "${PAINTER_WARMUP_TRAIN_JSONL:?set PAINTER_WARMUP_TRAIN_JSONL to the approved train JSONL (nominally 24 or 32 rows)}"
: "${PAINTER_WARMUP_DEV_JSONL:?set PAINTER_WARMUP_DEV_JSONL to the approved dev JSONL (nominally 8 rows)}"
: "${HF_CHECKPOINT_REPO:?set the public checkpoint repository for every warmup checkpoint}"
: "${HF_TOKEN_PATH:?set the node credential file without writing it to artifacts}"
[[ -f "$HF_TOKEN_PATH" ]] || { echo 'HF_TOKEN_PATH is not a readable node credential file' >&2; exit 1; }

PYTHON="${PAINTER_WARMUP_PYTHON:-$PRIME_ROOT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || { echo "pinned Prime Python is required for code-flow warmup: $PYTHON" >&2; exit 1; }
if [[ -d "$RUN/sft_source/mixed_sft" ]]; then
  SOURCE_ROOT="${PAINTER_WARMUP_SOURCE_ROOT:-$RUN/sft_source}"
elif [[ -d "$RUN/../structured-flow-rl-20261009/sft_source/mixed_sft" ]]; then
  # Reuse the tracked, hash-reviewed packaged source from the prior flow run.
  # The Diverse checkout deliberately does not copy or relabel those helpers;
  # warmup.py preserves its original manifest/provenance and adds only the
  # run-local token-refreshing uploader facade at materialization time.
  SOURCE_ROOT="${PAINTER_WARMUP_SOURCE_ROOT:-$RUN/../structured-flow-rl-20261009/sft_source}"
else
  SOURCE_ROOT="${PAINTER_WARMUP_SOURCE_ROOT:-$PAINTER_BOOTSTRAP_RUN/source/brush-rl}"
fi
PRIME="${PAINTER_WARMUP_PRIME_ROOT:-$PRIME_ROOT}"
MODEL="${PAINTER_WARMUP_MODEL_DIR:-$MODEL_DIR}"
OUTPUT="${PAINTER_WARMUP_OUTPUT:-$RUN/warmup-output}"
RUN_NAME="${PAINTER_WARMUP_RUN_NAME:-Diverse-flow-sft-warmup}"
export HF_CHECKPOINT_RUN="${HF_CHECKPOINT_RUN:-diverse-flow-rl-20261009-sft-warmup}"

# Fail before data preparation or optimizer construction when the nested
# Prime interpreter cannot import the image and mixed_sft runtime used by the
# approved teacher JSONL.  A fallback system interpreter would silently
# produce a different processor environment and is therefore disallowed.
"$PYTHON" - "$SOURCE_ROOT" <<'PY'
import importlib.util
import sys
from pathlib import Path

source_root = Path(sys.argv[1]).resolve()
try:
    import PIL  # noqa: F401
except Exception as exc:
    raise SystemExit(f"Prime Python cannot import Pillow: {type(exc).__name__}: {exc}")
mixed = source_root / "integrations" / "speedpainting" / "mixed_sft"
if not mixed.is_dir():
    mixed = source_root / "mixed_sft"
if not mixed.is_dir():
    raise SystemExit(f"mixed_sft runtime is missing under {source_root}")
train = mixed / "train.py"
scheduled = mixed / "scheduled_data.py"
if not train.is_file() or not scheduled.is_file():
    raise SystemExit("mixed_sft train.py and scheduled_data.py are required")
print({"schema": "diverse-flow-prime-pillow-mixed-sft-probe-v1", "python": sys.executable, "pillow": True, "mixed_sft": str(mixed)})
PY

if [[ -r "${PAINTER_WARMUP_PARENT_ADAPTER:-}" ]]; then
  PARENT="$PAINTER_WARMUP_PARENT_ADAPTER"
else
  PARENT="${PAINTER_WARMUP_PARENT_ADAPTER:-$PAINTER_SFT_ADAPTER_PATH_FILE}"
fi
[[ -e "$PARENT" ]] || { echo "warmup parent adapter or pointer is missing: $PARENT" >&2; exit 1; }

PREPARE_ARGS=(
  prepare --train "$PAINTER_WARMUP_TRAIN_JSONL" --dev "$PAINTER_WARMUP_DEV_JSONL"
  --output "$OUTPUT" --model-dir "$MODEL" --parent-adapter "$PARENT" --run-name "$RUN_NAME"
)
if [[ -n "${PAINTER_WARMUP_APPROVAL_MANIFEST:-}" ]]; then
  PREPARE_ARGS+=(--approval-manifest "$PAINTER_WARMUP_APPROVAL_MANIFEST")
fi
"$PYTHON" "$RUN/warmup.py" "${PREPARE_ARGS[@]}"

# ``train`` repeats the processor audit immediately before torchrun and fails
# before the first optimizer construction when any mask or length check fails.
"$PYTHON" "$RUN/warmup.py" train \
  --prepared "$OUTPUT" --source-root "$SOURCE_ROOT" --prime-root "$PRIME" \
  --parent-adapter "$PARENT" --output-dir "$OUTPUT/train-output" \
  --run-name "$RUN_NAME" --python "$PYTHON"

"$PYTHON" "$RUN/warmup.py" receipt \
  --output "$OUTPUT" --checkpoint-root "$OUTPUT/train-output" --parent-adapter "$PARENT"
# Make the verified warmup adapter the policy initializer for the subsequent
# RL process. Keep the original Flash12 pointer and receipt intact as the
# warmup parent identity; run.sh detects this separate pointer explicitly.
WARMUP_POINTER="$OUTPUT/warmup-adapter-path.txt"
[[ -r "$WARMUP_POINTER" ]] || { echo 'warmup adapter pointer was not produced' >&2; exit 1; }
WARMUP_ADAPTER="$(tr -d '\r\n' < "$WARMUP_POINTER")"
[[ -f "$WARMUP_ADAPTER/adapter_model.safetensors" ]] || { echo 'warmup adapter weights are missing' >&2; exit 1; }
cp "$WARMUP_POINTER" "$RUN/active-policy-adapter-path.txt"
"$PYTHON" - "$RUN" "$OUTPUT" "$WARMUP_ADAPTER" <<'PY'
import hashlib, json, sys
from pathlib import Path
run, output, adapter = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
receipt = json.loads((output / "warmup-receipt.json").read_text())
weights = adapter / "adapter_model.safetensors"
sha = hashlib.sha256(weights.read_bytes()).hexdigest()
if sha != receipt.get("selected_adapter_sha256"):
    raise SystemExit("warmup pointer bytes differ from the verified receipt")
value = {
    "schema": "Diverse-flow-active-policy-v1",
    "status": "verified_warmup_adapter",
    "adapter_path_file": str(output / "warmup-adapter-path.txt"),
    "adapter_sha256": sha,
    "warmup_receipt": str(output / "warmup-receipt.json"),
    "warmup_receipt_sha256": hashlib.sha256((output / "warmup-receipt.json").read_bytes()).hexdigest(),
    "parent_adapter_sha256": receipt["parent_adapter"]["sha256"],
    "steps": receipt["training"]["max_steps"],
    "fresh_optimizer": receipt["training"]["fresh_optimizer"],
}
(run / "active-policy.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY
echo "DIVERSE_FLOW_WARMUP_READY: $OUTPUT/warmup-adapter-path.txt"
