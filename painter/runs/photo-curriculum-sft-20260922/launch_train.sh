#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "${1:?usage: launch_train.sh RUN_ROOT}" && pwd -P)"
cd "$ROOT"
exec 8>"$ROOT/train.lock"
flock -n 8 || { echo "training already running" >&2; exit 1; }
TELEMETRY_PID=""
trap 'rc=$?; printf "%s\n" "$rc" > "$ROOT/training.exit"; if [[ -n "${TELEMETRY_PID:-}" ]]; then kill "$TELEMETRY_PID" 2>/dev/null || true; wait "$TELEMETRY_PID" 2>/dev/null || true; fi; python3 "$ROOT/summarize_telemetry.py" --input "$ROOT/gpu-telemetry.csv" --output "$ROOT/gpu-telemetry-summary.json" 2>/dev/null || true' EXIT

source "$ROOT/env.sh"
[[ -f "$ROOT/training-ready.json" && -f "$ROOT/sft.toml" && -f "$ROOT/model-path.txt" ]] || { echo "bootstrap/config incomplete" >&2; exit 1; }
[[ -f "$ROOT/data/train.jsonl" && -f "$ROOT/data/validation.jsonl" ]] || { echo "balanced data is incomplete" >&2; exit 1; }

export HF_CHECKPOINT_REPO="${HF_CHECKPOINT_REPO:?set a public Hugging Face checkpoint repository}"
export HF_CHECKPOINT_RUN="${HF_CHECKPOINT_RUN:-photo-curriculum-sft-20260922}"
export WATERCOLOUR_INIT_ADAPTER=""
export CAPACITY_STATE_COMPLETE=1
export PRL_OUTPUT_DIR="$ROOT/train-output"
export MIXED_SFT_TRACE="$ROOT/batch-shapes.jsonl"
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export ALLOW_DATA_SHUFFLE="${ALLOW_DATA_SHUFFLE:-0}"

RESUME_PRESENT="$(python3 - <<'PY'
import tomllib
with open("sft.toml", "rb") as stream:
    print("1" if tomllib.load(stream).get("resume") else "0")
PY
)"
if [[ "$RESUME_PRESENT" == "1" ]]; then
  export FRESH_BASE=0
  export ADAPTER_ONLY_CHECKPOINTS="${ADAPTER_ONLY_CHECKPOINTS:-0}"
  [[ "$ADAPTER_ONLY_CHECKPOINTS" == "0" ]] || {
    echo "checkpoint resume requires full native checkpoints (set ADAPTER_ONLY_CHECKPOINTS=0)" >&2
    exit 1
  }
else
  export FRESH_BASE=1
  # Keep Prime's native DCP checkpoints (keep_last=2) so optimizer/data
  # state can resume; the export hook still publishes lightweight public LoRA
  # adapters at each checkpoint. Set ADAPTER_ONLY_CHECKPOINTS=1 only for an
  # explicitly non-resumable capacity run.
  export ADAPTER_ONLY_CHECKPOINTS="${ADAPTER_ONLY_CHECKPOINTS:-0}"
fi

NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-$(python3 -c 'import tomllib;print(tomllib.load(open("sft.toml","rb"))["deployment"]["num_train_gpus"])')}"
[[ "$NUM_TRAIN_GPUS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid NUM_TRAIN_GPUS" >&2; exit 1; }
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(python3 - "$NUM_TRAIN_GPUS" <<'PY'
import sys
n = int(sys.argv[1])
print(",".join(str(i) for i in range(n)))
PY
)"
fi
export CUDA_VISIBLE_DEVICES
VISIBLE_GPU_COUNT="$(python3 - "$CUDA_VISIBLE_DEVICES" <<'PY'
import sys
print(len([part for part in sys.argv[1].split(',') if part.strip()]))
PY
)"
[[ "$VISIBLE_GPU_COUNT" -ge "$NUM_TRAIN_GPUS" ]] || {
  echo "CUDA_VISIBLE_DEVICES exposes $VISIBLE_GPU_COUNT GPU(s), need $NUM_TRAIN_GPUS" >&2
  exit 1
}

rm -f "$ROOT/training.exit" "$ROOT/gpu-telemetry.csv" "$ROOT/gpu-telemetry-summary.json"
python3 gpu_telemetry.py --output "$ROOT/gpu-telemetry.csv" --interval "${GPU_TELEMETRY_INTERVAL:-10}" &
TELEMETRY_PID=$!
export PATH="$PRIME_ROOT/.venv/bin:$ROOT/bootstrap/bin:$PATH"
cd "$PRIME_ROOT"
torchrun --standalone --nnodes=1 --nproc-per-node="$NUM_TRAIN_GPUS" \
  "$ROOT/train.py" @ "$ROOT/sft.toml" > "$ROOT/train.log" 2>&1
