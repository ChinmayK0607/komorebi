#!/usr/bin/env bash
set -Eeuo pipefail

# Finite foreground launch.  Provider setup, credentials, and GPU assignment
# are deliberately supplied by the operator; Prime maps the declared role
# counts onto whatever CUDA devices are visible on the node.
ROOT="$(cd "${1:?usage: launch_rl.sh RUN_ROOT}" && pwd -P)"
REPO_ROOT="$(cd "${PAINTER_REPO_ROOT:?set PAINTER_REPO_ROOT to the checkout}" && pwd -P)"
PRIME_ROOT="$(cd "${PRIME_ROOT:?set PRIME_ROOT to the pinned Prime checkout}" && pwd -P)"
PRIME_COMMIT="26b3131d2716a4f8210b165df584b83b4bc54f61"
BASE_MODEL_REVISION="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
ADAPTER_REVISION="adf687e5b321514eda7be6eab050434fdc8e44ff"
ADAPTER_PREFIX="photo-curriculum-sft-20260922-balanced/step_512"
ADAPTER_SHA256="bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740"
HF_REPO="CK0607/qwen3.8-27b-brush-painting"
RUN_NAME="${PAINTER_RL_RUN_NAME:-photo-curriculum-rl-20260922}"
FROZEN_SFT_ALIAS="${PAINTER_FROZEN_SFT_ALIAS:-sft-step512}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the staged base snapshot}"
BRUSH_ROOT="${BRUSH_ROOT:-$ROOT/source/brush-rl}"
INITIAL_ADAPTER="${WATERCOLOUR_INIT_ADAPTER:?set WATERCOLOUR_INIT_ADAPTER to the public step-512 adapter directory}"
MANIFEST="${PAINTER_RL_MANIFEST:-$ROOT/source-manifest.json}"
MAX_STEPS="${PAINTER_RL_MAX_STEPS:-32}"
GPUS_PER_NODE="${PAINTER_GPUS_PER_NODE:-2}"
NUM_TRAIN_GPUS="${PAINTER_NUM_TRAIN_GPUS:-1}"
NUM_INFER_GPUS="${PAINTER_NUM_INFER_GPUS:-1}"
CONFIG="${PAINTER_RL_CONFIG:-$ROOT/native-rl.toml}"
PYTHON="$PRIME_ROOT/.venv/bin/python"
UV_BIN="${UV_BIN:-uv}"

[[ -x "$PYTHON" ]] || { echo "missing pinned Prime Python: $PYTHON" >&2; exit 1; }
[[ "$(git -C "$PRIME_ROOT" rev-parse HEAD)" == "$PRIME_COMMIT" ]] || {
  echo "PRIME_ROOT is not pinned to $PRIME_COMMIT" >&2
  exit 1
}
[[ -d "$MODEL_DIR" && "$(basename "$MODEL_DIR")" == "$BASE_MODEL_REVISION" ]] || {
  echo "MODEL_DIR must be the staged base snapshot ending in $BASE_MODEL_REVISION" >&2
  exit 1
}
[[ -f "$MANIFEST" ]] || { echo "reviewed RL manifest is missing: $MANIFEST" >&2; exit 1; }
[[ -d "$BRUSH_ROOT" ]] || { echo "BRUSH_ROOT is missing: $BRUSH_ROOT" >&2; exit 1; }
[[ -f "$ROOT/vendor/integrations/watercolour/renderer.py" \
  && -x "$ROOT/renderer-env/bin/python" \
  && -d "$ROOT/browsers" ]] || {
  echo "renderer paths are missing under $ROOT; run setup_rl_runtime.sh with PAINTER_RENDER_ROOT" >&2
  exit 1
}
for required in \
  "$BRUSH_ROOT/integrations/prime/watercolour_sft/init_adapter.py" \
  "$BRUSH_ROOT/integrations/prime/watercolour_sft/export_adapter.py" \
  "$BRUSH_ROOT/integrations/prime/watercolour_sft/model_profiles.py" \
  "$BRUSH_ROOT/scripts/upload_checkpoints.py"; do
  [[ -f "$required" ]] || { echo "reviewed RL helper is missing: $required" >&2; exit 1; }
done
[[ -f "$INITIAL_ADAPTER/adapter_model.safetensors" && -f "$INITIAL_ADAPTER/adapter_config.json" ]] || {
  echo "initial adapter is not a portable LoRA directory" >&2
  exit 1
}
[[ "$FROZEN_SFT_ALIAS" != "$MODEL_DIR" ]] || {
  echo "frozen SFT anchor alias must differ from the live Prime LoRA alias" >&2
  exit 1
}
[[ -r "${HF_TOKEN_PATH:-$ROOT/private/hf-token}" ]] || {
  echo "HF_TOKEN_PATH must point to the local upload token file" >&2
  exit 1
}

export BRUSH_ROOT BRUSH_MODEL_PROFILE=qwen38_27b PRIME_ROOT MODEL_DIR
export PAINTER_FROZEN_SFT_ALIAS="$FROZEN_SFT_ALIAS"
# The frozen opponent is a separate, immutable inference route.  Keep every
# path explicit so the lazy generator cannot fall back to an unrelated local
# Prime checkout or renderer environment on a rented node.
export PAINTER_FROZEN_OPPONENT_ROOT="${PAINTER_FROZEN_OPPONENT_ROOT:-$REPO_ROOT}"
export PAINTER_FROZEN_OPPONENT_PRIME_ROOT="${PAINTER_FROZEN_OPPONENT_PRIME_ROOT:-$PRIME_ROOT}"
export PAINTER_FROZEN_OPPONENT_EVAL="${PAINTER_FROZEN_OPPONENT_EVAL:-$PRIME_ROOT/.venv/bin/eval}"
export PAINTER_FROZEN_OPPONENT_RENDERER="${PAINTER_FROZEN_OPPONENT_RENDERER:-$ROOT/vendor/integrations/watercolour/renderer.py}"
export PAINTER_FROZEN_OPPONENT_RENDERER_PYTHON="${PAINTER_FROZEN_OPPONENT_RENDERER_PYTHON:-$ROOT/renderer-env/bin/python}"
# Frozen opponent uses the direct vLLM endpoint: the pinned local router rejects
# VF's default reasoning_effort=none before forwarding, while direct vLLM accepts
# it. Operators may override this for a router with the matching schema.
export PAINTER_FROZEN_OPPONENT_ENDPOINT="${PAINTER_FROZEN_OPPONENT_ENDPOINT:-http://127.0.0.1:8100/v1}"
export PAINTER_FROZEN_OPPONENT_CONFIG="${PAINTER_FROZEN_OPPONENT_CONFIG:-$ROOT/frozen-opponent-config.json}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$ROOT/browsers}"
export WATERCOLOUR_INIT_ADAPTER="$(cd "$INITIAL_ADAPTER" && pwd -P)"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$ROOT/private/hf-token}"
if [[ -z "${HF_TOKEN:-}" ]]; then
  HF_TOKEN="$(tr -d '\r\n' < "$HF_TOKEN_PATH")"
  [[ -n "$HF_TOKEN" ]] || { echo "HF token file is empty" >&2; exit 1; }
  export HF_TOKEN
fi
export HF_CHECKPOINT_REPO="$HF_REPO" HF_CHECKPOINT_RUN="$RUN_NAME"
export ADAPTER_ONLY_CHECKPOINTS=1 CAPACITY_STATE_COMPLETE=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
# Match the validated CUDA 13 Prime/vLLM node environment.  Native eval's
# local API is intentionally a non-secret sentinel; the frozen model endpoint
# remains loopback-only and receives no provider credential.
export PAINTER_LOCAL_API_KEY="${PAINTER_LOCAL_API_KEY:-local}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
CUDA_HOME="$($PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/cu13")')"
[[ -d "$CUDA_HOME" ]] || { echo "CUDA 13 runtime directory is missing: $CUDA_HOME" >&2; exit 1; }
export CUDA_HOME
export PATH="$PRIME_ROOT/.venv/bin:$CUDA_HOME/bin:$ROOT/bootstrap/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT/painter:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PAINTER_RL_CACHE_ROOT="${PAINTER_RL_CACHE_ROOT:-$ROOT/cache}"
mkdir -p "$ROOT/logs" "$ROOT/private" "$PAINTER_RL_CACHE_ROOT"
chmod 700 "$ROOT/private"

"$PYTHON" - "$ROOT" "$MODEL_DIR" "$WATERCOLOUR_INIT_ADAPTER" "$MANIFEST" \
  "$BASE_MODEL_REVISION" "$ADAPTER_REVISION" "$ADAPTER_PREFIX" "$ADAPTER_SHA256" "$RUN_NAME" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root, model_dir, adapter_dir, manifest, base_revision, adapter_revision, prefix, expected_sha, run_name = sys.argv[1:]
model = Path(model_dir)
adapter = Path(adapter_dir)
weight = adapter / "adapter_model.safetensors"
digest = hashlib.sha256(weight.read_bytes()).hexdigest()
if digest != expected_sha:
    raise SystemExit(f"initial adapter SHA-256 mismatch: {digest}")
metadata = json.loads((adapter / "adapter_config.json").read_text())
targets = {
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
}
if metadata.get("r") != 16 or metadata.get("lora_alpha") != 32 or set(metadata.get("target_modules", [])) != targets:
    raise SystemExit("initial adapter rank/alpha/target modules do not match the pinned RL profile")
if metadata.get("revision") not in (None, base_revision):
    raise SystemExit("initial adapter base revision does not match the pinned model")
config_hash = hashlib.sha256((model / "config.json").read_bytes()).hexdigest()
receipt = {
    "run": run_name,
    "base_model": "Qwen/Qwen3.8-27B",
    "base_revision": base_revision,
    "base_config_sha256": config_hash,
    "adapter_revision": adapter_revision,
    "adapter_prefix": prefix,
    "adapter_sha256": digest,
    "adapter_path": str(adapter),
    "manifest": str(Path(manifest).resolve()),
    "initialization": "trainer_exact_copy_and_inference_preload",
    "live_policy_alias": "Prime served base-model name (reloaded in place)",
    "frozen_sft_alias": os.environ.get("PAINTER_FROZEN_SFT_ALIAS", "sft-step512"),
    "optimizer_resume": False,
    "reward": "terminal_aesthetic_only; no speed reward",
}
Path(root, "initial-adapter-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
PY

if [[ ! -f "$CONFIG" ]]; then
  "$PYTHON" "$ROOT/prepare_rl_config.py" \
    --output "$CONFIG" --model "$MODEL_DIR" --run-root "$ROOT" \
    --manifest "$(python3 -c 'import os,sys;print(os.path.relpath(sys.argv[1],sys.argv[2]))' "$MANIFEST" "$ROOT")" \
    --max-steps "$MAX_STEPS" --gpus-per-node "$GPUS_PER_NODE" \
    --num-train-gpus "$NUM_TRAIN_GPUS" --num-infer-gpus "$NUM_INFER_GPUS" \
    --checkpoint-interval 8 --run-name "$RUN_NAME" --prime-root "$PRIME_ROOT" --validate-prime
fi
[[ -f "$PAINTER_FROZEN_OPPONENT_CONFIG" ]] || {
  echo "frozen opponent operational config is missing: $PAINTER_FROZEN_OPPONENT_CONFIG" >&2
  exit 1
}

# Use the already-installed pinned package; this is dependency-free and does
# not mutate the runtime lock.  Prime's own child processes inherit the same
# source and private-token paths without ever printing the token.
if [[ "${PAINTER_RL_SKIP_EDITABLE_INSTALL:-0}" == "1" ]]; then
  # Prepare-only validation can use the explicit PYTHONPATH below without
  # mutating a shared Prime environment. Normal runs retain the reviewed
  # editable install behavior.
  printf '%s\n' 'Skipping editable painter install (prepare-only mode).'
else
  "$UV_BIN" pip install --python "$PYTHON" --no-deps --no-build-isolation -e "$REPO_ROOT/painter"
fi

# Exercise the exact installed VF data serializer before starting any owned
# GPU process. The orchestrator ships task_data over RPC and the env server
# reconstructs it from Task.data_type(); direct task construction misses this.
"$PYTHON" - "$ROOT" "$CONFIG" "$MANIFEST" <<'PY'
import json
import sys
import tomllib
from pathlib import Path

root, config_path, manifest_path = map(Path, sys.argv[1:])
raw_config = config_path.read_bytes()
if config_path.suffix.lower() == ".json" or raw_config.lstrip().startswith(b"{"):
    config = json.loads(raw_config.decode("utf-8"))
else:
    config = tomllib.loads(raw_config.decode("utf-8"))
taskset_values = config["orchestrator"]["train"]["source"][0]["env"]["taskset"]
if Path(taskset_values["manifest"]).name != Path(manifest_path).name:
    raise SystemExit("resolved taskset manifest does not match launcher manifest")

try:
    from native_painting_rl import NativePaintingRLConfig, NativePaintingRLTaskset
except ImportError:
    from painter.native_painting_rl import NativePaintingRLConfig, NativePaintingRLTaskset

taskset_config = NativePaintingRLConfig.model_validate(taskset_values)
task = next(iter(NativePaintingRLTaskset(taskset_config).load()))
data_type = task.data_type()
wire = task.data.model_dump(mode="json", exclude_none=True)
roundtrip = data_type.model_validate(wire)
if roundtrip.model_dump(mode="json", exclude_none=True) != wire:
    raise SystemExit("RL task data changed during pinned VF model_dump/model_validate roundtrip")
if task.key != type(task)(roundtrip, task.config).key:
    raise SystemExit("RL task key changed during pinned VF data roundtrip")
print(json.dumps({
    "status": "passed",
    "task_id": task.data.task_id,
    "data_type": f"{data_type.__module__}.{data_type.__name__}",
    "rl_wire_fields": len(wire),
}, sort_keys=True))
PY

# Remote prepare-only validation must exercise the installed VF serializer,
# adapter/token paths, and config imports without starting telemetry, servers,
# or Prime.  Keep this exit before every owned GPU process.
if [[ "${PAINTER_RL_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  printf '%s\n' '{"status":"preflight_passed","training_started":false}'
  exit 0
fi

TELEMETRY_PID=""
RL_PID=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$RL_PID" ]]; then
    # setsid gives the owned Prime tree its own process group.  This cleanup
    # also handles a normal child exit where descendants briefly outlive the
    # wrapper process.
    kill -- -"$RL_PID" 2>/dev/null || true
    wait "$RL_PID" 2>/dev/null || true
  fi
  if [[ -n "$TELEMETRY_PID" ]]; then
    kill "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
  fi
  printf '%s\n' "$status" > "$ROOT/rl.exit"
  exit "$status"
}
forward_signal() {
  local signal="$1"
  if [[ -n "$RL_PID" ]]; then
    kill -"$signal" -- -"$RL_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
trap 'forward_signal TERM; exit 143' TERM
trap 'forward_signal INT; exit 130' INT
ATTEMPT_ID="${PAINTER_RL_ATTEMPT_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
TELEMETRY_FILE="${PAINTER_RL_TELEMETRY_FILE:-$ROOT/logs/gpu-telemetry-$ATTEMPT_ID.csv}"
TELEMETRY_ERR_FILE="$ROOT/logs/gpu-telemetry-$ATTEMPT_ID.err"
rm -f "$ROOT/rl.exit"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,power.draw \
    --format=csv -l "${GPU_TELEMETRY_INTERVAL:-10}" > "$TELEMETRY_FILE" 2>"$TELEMETRY_ERR_FILE" &
  TELEMETRY_PID=$!
fi

export PATH="$PRIME_ROOT/.venv/bin:$PATH"
cd "$PRIME_ROOT"
set +e
setsid "$PYTHON" "$ROOT/run_from_sft.py" @ "$CONFIG" &
RL_PID=$!
wait "$RL_PID"
RL_STATUS=$?
set -e
exit "$RL_STATUS"
