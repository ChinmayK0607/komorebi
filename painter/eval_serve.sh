#!/usr/bin/env bash
# Serve both verified adapters on one inference GPU.
# Usage: bash eval_serve.sh /absolute/node/workspace [vLLM args...]
set -Eeuo pipefail

ROOT=$(realpath "${1:?node workspace required}")
shift
cd "$ROOT"
[[ -f "$ROOT/env.sh" ]] || { echo "run eval_node.sh first" >&2; exit 1; }
# shellcheck disable=SC1091
source "$ROOT/env.sh"

: "${INFERENCE_PY:?eval_node.sh did not write INFERENCE_PY}"
: "${INFERENCE_ROOT:?eval_node.sh did not write INFERENCE_ROOT}"
: "${MODEL_DIR:?eval_node.sh did not write MODEL_DIR}"
: "${SFT_ADAPTER:?eval_node.sh did not write SFT_ADAPTER}"
: "${RL_ADAPTER:?eval_node.sh did not write RL_ADAPTER}"
[[ -f "$ROOT/model-receipt.json" ]] || { echo "model receipt is missing" >&2; exit 1; }
[[ -f "$MODEL_DIR/config.json" ]] || { echo "base model path is missing" >&2; exit 1; }
[[ -f "$SFT_ADAPTER/adapter_model.safetensors" ]] || { echo "SFT adapter path is missing" >&2; exit 1; }
[[ -f "$RL_ADAPTER/adapter_model.safetensors" ]] || { echo "RL adapter path is missing" >&2; exit 1; }

# The pinned Prime wheel carries CUDA 13 libraries inside the inference
# environment.  Keep the launch path identical to the known-working renderer
# scripts instead of relying on the image's global LD_LIBRARY_PATH.
CUDA_HOME="$($INFERENCE_PY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/cu13")')"
if [[ -d "$CUDA_HOME" ]]; then
  export CUDA_HOME
  export PATH="$INFERENCE_ROOT/bin:$CUDA_HOME/bin:$ROOT/bootstrap/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
else
  echo "CUDA 13 runtime directory is missing from $INFERENCE_ROOT: $CUDA_HOME" >&2
  exit 1
fi

CONCURRENCY=${EVAL_CONCURRENCY:-4}
[[ "$CONCURRENCY" =~ ^[0-9]+$ ]] && (( CONCURRENCY >= 4 && CONCURRENCY <= 8 )) || {
  echo "EVAL_CONCURRENCY must be an integer from 4 through 8" >&2
  exit 1
}
PORT=${EVAL_PORT:-8000}
HOST=${EVAL_HOST:-127.0.0.1}
MAX_BATCHED=${EVAL_MAX_BATCHED_TOKENS:-32768}

# Confirm the serving runtime before consuming GPU memory.  The Prime lock
# supports this exact pair; this catches an accidental global/host environment
# from being used by the launch command.
"$INFERENCE_PY" - <<'PY'
import importlib.metadata
for package, expected in (("vllm", "0.28.0"), ("transformers", "5.6.2")):
    actual = importlib.metadata.version(package)
    if actual != expected:
        raise SystemExit(f"{package}={actual}; expected pinned {expected}")
import torch
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false; refusing to start a GPU server")
print({"torch": torch.__version__, "cuda": torch.version.cuda, "devices": torch.cuda.device_count()})
PY

export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}

exec "$INFERENCE_ROOT/bin/vllm" serve "$MODEL_DIR" \
  --served-model-name base \
  --host "$HOST" --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-seqs "$CONCURRENCY" \
  --max-num-batched-tokens "$MAX_BATCHED" \
  --gpu-memory-utilization "${EVAL_GPU_MEMORY_UTILIZATION:-0.90}" \
  --limit-mm-per-prompt '{"image":14,"video":0}' \
  --gdn-prefill-backend triton \
  --kernel-config '{"enable_jit_warmup":false}' \
  --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":1048576}' \
  --enable-lora --max-lora-rank 16 --max-loras 2 --max-cpu-loras 2 \
  --lora-modules "sft=$SFT_ADAPTER" "rl=$RL_ADAPTER" "$@"
