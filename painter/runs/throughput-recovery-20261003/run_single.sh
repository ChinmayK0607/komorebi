#!/usr/bin/env bash
set -Eeuo pipefail

RUN="${PAINTER_RUN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)}"
REPO="$(cd "$RUN/../../.." && pwd -P)"
NODE_ROOT="${PAINTER_NODE_ROOT:-/root/painter}"
source "$RUN/runtime_env.sh"
validate_recovery_bindings

CONFIG="${PAINTER_RL_CONFIG:-$RUN/native.toml}"
MANIFEST="${PAINTER_RL_MANIFEST:-$RUN/anchors-frozen/train-manifest.json}"
SFT_PATH_FILE="$PAINTER_SFT_ADAPTER_PATH_FILE"
FROZEN_PATH_FILE="$PAINTER_FROZEN_SFT_ADAPTER_PATH_FILE"
RENDERER="${PAINTER_RENDERER_PATH:-$RUN/runtime/watercolour-cancellable-centroid-v1/cancellable_watercolour_renderer.py}"
RENDERER_PYTHON="${PAINTER_RENDERER_PYTHON_PATH:-$RUN/renderer-env/bin/python}"
PY="$PRIME_ROOT/.venv/bin/python"

[[ "$(uname -s)" == Linux ]] || { echo 'Linux GPU node required for the RL process' >&2; exit 1; }
[[ -x "$PY" ]] || { echo "pinned Prime Python is unavailable: $PY" >&2; exit 1; }
validate_recovery_profile_environment "$PY"
for required in "$CONFIG" "$MANIFEST" "$RUN/rl-config-receipt.json" "$RUN/launch_rl.sh" \
  "$RUN/run_from_sft.py" "$RUN/train_from_sft.py" "$RUN/sitecustomize.py" \
  "$RUN/vendor/integrations/watercolour/renderer.py" "$RENDERER" "$RENDERER_PYTHON"; do
  [[ -e "$required" ]] || { echo "missing prepared input: $required" >&2; exit 1; }
done
[[ -f "$REPO/painter/final_action_credit.py" ]] || { echo 'strict final-action module is missing' >&2; exit 1; }
[[ -r "$SFT_PATH_FILE" && -r "$FROZEN_PATH_FILE" ]] || { echo 'verified adapter path files are missing' >&2; exit 1; }
WATERCOLOUR_INIT_ADAPTER="$(tr -d '\r\n' < "$SFT_PATH_FILE")"
FROZEN_SFT_ADAPTER="$(tr -d '\r\n' < "$FROZEN_PATH_FILE")"
[[ -f "$WATERCOLOUR_INIT_ADAPTER/adapter_model.safetensors" && -f "$WATERCOLOUR_INIT_ADAPTER/adapter_config.json" ]] || {
  echo 'verified RL16 initializer is unavailable' >&2
  exit 1
}
[[ -f "$FROZEN_SFT_ADAPTER/adapter_model.safetensors" && -f "$FROZEN_SFT_ADAPTER/adapter_config.json" ]] || {
  echo 'verified frozen SFT512 adapter is unavailable' >&2
  exit 1
}
[[ "$(cd "$WATERCOLOUR_INIT_ADAPTER" && pwd -P)" != "$(cd "$FROZEN_SFT_ADAPTER" && pwd -P)" ]] || {
  echo 'RL16 initializer and frozen SFT512 adapter must be distinct' >&2
  exit 1
}
[[ -f "$RENDERER" && -x "$RENDERER_PYTHON" ]] || { echo 'cancellable renderer runtime is unavailable' >&2; exit 1; }

export PAINTER_REPO_ROOT="$REPO" BRUSH_ROOT="$REPO/painter/rl_helpers"
export PRIME_ROOT MODEL_DIR WATERCOLOUR_INIT_ADAPTER
export PAINTER_FROZEN_SFT_ADAPTER="$FROZEN_SFT_ADAPTER"
export PAINTER_RL_CONFIG="$CONFIG" PAINTER_RL_MANIFEST="$MANIFEST"
export PAINTER_RL_RUN_NAME="$PAINTER_RUN_NAME" PAINTER_RL_MAX_STEPS=4
export PAINTER_RL_ADAPTER_REVISION="$PAINTER_POLICY_REVISION"
export PAINTER_RL_ADAPTER_PREFIX="$PAINTER_POLICY_PREFIX"
export PAINTER_RL_ADAPTER_SHA256="$PAINTER_POLICY_SHA256"
export PAINTER_FROZEN_SFT_ALIAS=sft-step512 PAINTER_RL_FIXED_OPPONENT=1
export PAINTER_JUDGE_PROTOCOL PAINTER_JUDGE_FORMAT PAINTER_JUDGE_MODEL PAINTER_JUDGE_EFFORT
export PAINTER_GPUS_PER_NODE PAINTER_NUM_TRAIN_GPUS PAINTER_NUM_INFER_GPUS
export PAINTER_GROUP_RANK_MODULE=painter.consistent_group_admission
export PAINTER_WHOLE_GROUP_ADMISSION=1 PAINTER_WHOLE_GROUP_SIZE=4
export PAINTER_CONSISTENT_GUARD_RECEIPT="${PAINTER_CONSISTENT_GUARD_RECEIPT:-$RUN/consistent-group-admission-activation.json}"
export PAINTER_CONSISTENT_GROUP_TELEMETRY="${PAINTER_CONSISTENT_GROUP_TELEMETRY:-$RUN/consistent-group-admission.jsonl}"
export PAINTER_RENDERER_PATH="$RENDERER" PAINTER_RENDERER_PYTHON_PATH="$RENDERER_PYTHON"
export PAINTER_REQUIRE_VALID_FINAL_ACTION=1 PAINTER_JS_SYNTAX_PREFLIGHT=1
export ADAPTER_ONLY_CHECKPOINTS=1 PAINTER_RL_CHECKPOINT_INTERVAL=1 PAINTER_ACTIVATION_CPU_OFFLOAD=1
export PAINTER_P5_BRUSH_API_REFERENCE_V2=1
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$NODE_ROOT/private/hf-token}"
export PAINTER_GATEWAY_ENV_FILE="${PAINTER_GATEWAY_ENV_FILE:-$NODE_ROOT/private/gateway.env}"

[[ -f "$PAINTER_GATEWAY_ENV_FILE" && ! -L "$PAINTER_GATEWAY_ENV_FILE" ]] || {
  echo 'private Gateway environment file is missing' >&2
  exit 1
}
[[ "$(stat -c %a "$PAINTER_GATEWAY_ENV_FILE" 2>/dev/null || stat -f %Lp "$PAINTER_GATEWAY_ENV_FILE")" == 600 ]] || {
  echo 'Gateway environment file must be mode 600' >&2
  exit 1
}

if [[ "${1:-}" == --preflight ]]; then
  PAINTER_RL_PREFLIGHT_ONLY=1 PAINTER_RL_SKIP_EDITABLE_INSTALL=1 \
    bash "$RUN/launch_rl.sh" "$RUN"
  exit 0
fi
[[ $# == 0 ]] || { echo 'usage: run_single.sh [--preflight]' >&2; exit 2; }
[[ "${PAINTER_RL_SUPERVISED:-0}" == 1 ]] || {
  echo 'finite supervisor required: set PAINTER_RL_SUPERVISED=1' >&2
  exit 2
}
[[ ! -e "$RUN/async-output/$PAINTER_RUN_NAME" ]] || {
  echo 'existing training output; preserve it before another attempt' >&2
  exit 1
}

mkdir -p "$RUN/logs"
TELEMETRY_PIDS=()
TELEMETRY_STARTED="$(date +%s)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits --loop-ms=10000 \
    >"$RUN/logs/gpu-telemetry.csv" 2>&1 &
  TELEMETRY_PIDS+=("$!")
else
  echo 'nvidia-smi is required to record GPU utilization evidence' >&2
  exit 1
fi
if command -v vmstat >/dev/null 2>&1; then
  vmstat 10 >"$RUN/logs/cpu-telemetry.txt" 2>&1 &
  TELEMETRY_PIDS+=("$!")
fi

cleanup_telemetry() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "${TELEMETRY_PIDS[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${TELEMETRY_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  local telemetry_python="$PRIME_ROOT/.venv/bin/python"
  [[ -x "$telemetry_python" ]] || telemetry_python=python3
  "$telemetry_python" - "$RUN" "$TELEMETRY_STARTED" "$(date +%s)" <<'PY'
import json
import os
import sys
from pathlib import Path

run, started, finished = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
gpu = run / "logs/gpu-telemetry.csv"
cpu = run / "logs/cpu-telemetry.txt"
gpu_rows = 0
gpu_devices = set()
gpu_busy = []
per_device = {}
inference_gpu_count = int(os.environ.get("PAINTER_NUM_INFER_GPUS", "1"))
if gpu.is_file():
    for raw in gpu.read_text(errors="replace").splitlines():
        columns = [part.strip() for part in raw.split(",")]
        if len(columns) < 4 or not columns[1].isdigit():
            continue
        try:
            index = int(columns[1])
            utilization = float(columns[2].split()[0])
        except (ValueError, IndexError):
            continue
        gpu_rows += 1
        gpu_devices.add(index)
        gpu_busy.append(utilization)
        per_device.setdefault(index, []).append(utilization)
cpu_rows = 0
if cpu.is_file():
    cpu_rows = sum(1 for line in cpu.read_text(errors="replace").splitlines() if line.strip() and line.split()[0].isdigit())
report = {
    "schema": "throughput-recovery-resource-telemetry-v1",
    "window": {"started_epoch": started, "finished_epoch": finished, "interval_seconds": 10},
    "gpu": {
        "path": "logs/gpu-telemetry.csv",
        "sample_rows": gpu_rows,
        "device_count_observed": len(gpu_devices),
        "average_utilization_percent": sum(gpu_busy) / len(gpu_busy) if gpu_busy else None,
        "sampled_busy_fraction": sum(value > 0 for value in gpu_busy) / len(gpu_busy) if gpu_busy else None,
        "denominator": "all per-device nvidia-smi samples emitted during the run window",
        "devices": {
            str(index): {
                "role_index": 0 if index < inference_gpu_count else 1,
                "role": "inference" if index < inference_gpu_count else "trainer",
                "sample_rows": len(values),
                "average_utilization_percent": sum(values) / len(values) if values else None,
                "sampled_busy_fraction": sum(value > 0 for value in values) / len(values) if values else None,
                "denominator": "samples for this GPU index in logs/gpu-telemetry.csv",
            }
            for index, values in sorted(per_device.items())
        },
    },
    "cpu": {
        "path": "logs/cpu-telemetry.txt",
        "sample_rows": cpu_rows,
        "denominator": "host vmstat sample rows during the run window",
    },
}
(run / "telemetry-receipt.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY
  exit "$status"
}
trap cleanup_telemetry EXIT INT TERM

bash "$RUN/launch_rl.sh" "$RUN"
