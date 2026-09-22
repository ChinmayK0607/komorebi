#!/usr/bin/env bash
# Run the fixed 28-case, one-turn evaluation for SFT steps 512 and 1024.
# This finite launcher owns two vLLM servers, records telemetry, and cleans
# up only the servers it started. Invalid canvases remain native evidence.
set -Eeuo pipefail

PAINTER_ROOT=${PAINTER_ROOT:-/root/painter}
RUN_ROOT=${RUN_ROOT:-$PAINTER_ROOT/runs/photo-curriculum-sft-20260922}
PRIME_ROOT=${PRIME_ROOT:-$PAINTER_ROOT/prime-rl}
MODEL_DIR=${MODEL_DIR:-/tmp/photo-curriculum-sft-20260922-models/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}
ADAPTER_ROOT=${ADAPTER_ROOT:-$RUN_ROOT/train-output/photo-curriculum-sft-20260922/artifacts/adapters}
MANIFEST=${MANIFEST:-$RUN_ROOT/eval-prep/eval-manifest.json}
EVAL_ROOT=${EVAL_ROOT:-$RUN_ROOT/eval-runs/photo-curriculum-sft-20260922}
PYTHON=${PYTHON:-$PRIME_ROOT/.venv/bin/python}
EVAL=${EVAL:-$PRIME_ROOT/.venv/bin/eval}
VLLM=${VLLM:-$PRIME_ROOT/.venv/bin/vllm}
EVAL_GENERATOR=${EVAL_GENERATOR:-$PAINTER_ROOT/native_painting_eval.py}
MIDPOINT_ADAPTER=${MIDPOINT_ADAPTER:-$ADAPTER_ROOT/step_512}
FINAL_ADAPTER=${FINAL_ADAPTER:-$ADAPTER_ROOT/step_1024}
MIDPOINT_GPU=${MIDPOINT_GPU:-0}
FINAL_GPU=${FINAL_GPU:-1}
MIDPOINT_PORT=${MIDPOINT_PORT:-8100}
FINAL_PORT=${FINAL_PORT:-8101}
MAX_CONCURRENT=${MAX_CONCURRENT:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-8192}
RENDER_TIMEOUT=${PAINTER_RENDER_TIMEOUT_SECONDS:-180}
ROLLOUT_TIMEOUT=${ROLLOUT_TIMEOUT_SECONDS:-1800}
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT_SECONDS:-900}
LOG_ROOT=$EVAL_ROOT/logs
mkdir -p "$LOG_ROOT"

[[ $(uname -s) == Linux ]] || { echo "this launcher must run on the Linux node" >&2; exit 2; }
for required in "$PYTHON" "$EVAL" "$VLLM" "$MODEL_DIR/config.json" "$MANIFEST" \
  "$MIDPOINT_ADAPTER/adapter_model.safetensors" "$FINAL_ADAPTER/adapter_model.safetensors" \
  "$EVAL_GENERATOR"; do
  [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 2; }
done
INFERENCE_ROOT=${INFERENCE_ROOT:-$PRIME_ROOT/.venv}
CUDA_HOME="$($PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/cu13")')"
[[ -d "$CUDA_HOME" ]] || { echo "CUDA 13 runtime directory is missing: $CUDA_HOME" >&2; exit 2; }
export CUDA_HOME
export PATH="$INFERENCE_ROOT/bin:$CUDA_HOME/bin:$PAINTER_ROOT/bootstrap/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
"$PYTHON" - <<'PY'
import importlib.metadata
for package, expected in (("vllm", "0.28.0"), ("transformers", "5.6.2")):
    actual = importlib.metadata.version(package)
    if actual != expected:
        raise SystemExit(f"{package}={actual}; expected pinned {expected}")
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise SystemExit("the matched eval requires two CUDA devices in the pinned runtime")
print({"torch": torch.__version__, "cuda": torch.version.cuda, "devices": torch.cuda.device_count()})
PY
[[ "$RENDER_TIMEOUT" =~ ^[0-9]+$ ]] && (( RENDER_TIMEOUT >= 1 && RENDER_TIMEOUT <= 180 )) || {
  echo "PAINTER_RENDER_TIMEOUT_SECONDS must be an integer from 1 through 180" >&2; exit 2;
}
[[ "$ROLLOUT_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo "ROLLOUT_TIMEOUT_SECONDS must be positive" >&2; exit 2; }
[[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_CONCURRENT must be positive" >&2; exit 2; }
[[ "$MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_MODEL_LEN must be positive" >&2; exit 2; }
[[ "$MAX_BATCHED_TOKENS" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_BATCHED_TOKENS must be positive" >&2; exit 2; }
[[ "$SERVER_START_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo "SERVER_START_TIMEOUT_SECONDS must be positive" >&2; exit 2; }

"$PYTHON" - "$MIDPOINT_ADAPTER" "$FINAL_ADAPTER" <<'PY'
import hashlib, json, sys
from pathlib import Path
for raw in sys.argv[1:]:
    folder = Path(raw)
    receipt = json.loads((folder / "hf-upload.json").read_text())
    if receipt.get("verified") is not True or receipt.get("private") is not False:
        raise SystemExit(f"adapter receipt is not public and verified: {folder}")
    weight = folder / "adapter_model.safetensors"
    actual = hashlib.sha256(weight.read_bytes()).hexdigest()
    expected = ((receipt.get("files") or {}).get("adapter_model.safetensors") or {}).get("sha256")
    if not expected or actual != expected:
        raise SystemExit(f"adapter hash mismatch for {folder}: {actual} != {expected}")
    config = json.loads((folder / "adapter_config.json").read_text())
    if config.get("r") != 16:
        raise SystemExit(f"unexpected LoRA rank for {folder}: {config.get('r')}")
    print(json.dumps({"adapter": str(folder), "sha256": actual, "revision": receipt.get("commit"), "path": receipt.get("path")}))
PY

MIDPOINT_CONFIG=$EVAL_ROOT/step_512.toml
FINAL_CONFIG=$EVAL_ROOT/step_1024.toml
MIDPOINT_OUTPUT=$EVAL_ROOT/step_512
FINAL_OUTPUT=$EVAL_ROOT/step_1024
mkdir -p "$EVAL_ROOT"
"$PYTHON" "$EVAL_GENERATOR" \
  --root "$PAINTER_ROOT" --manifest "$MANIFEST" --policy-label photo-sft-step512 \
  --model midpoint512 --output-dir "$MIDPOINT_OUTPUT" --config "$MIDPOINT_CONFIG" \
  --expected-count 28 --max-tokens 8192 --max-turns 1 --rollout-timeout "$ROLLOUT_TIMEOUT" --context-length 16384
"$PYTHON" "$EVAL_GENERATOR" \
  --root "$PAINTER_ROOT" --manifest "$MANIFEST" --policy-label photo-sft-step1024 \
  --model final1024 --output-dir "$FINAL_OUTPUT" --config "$FINAL_CONFIG" \
  --expected-count 28 --max-tokens 8192 --max-turns 1 --rollout-timeout "$ROLLOUT_TIMEOUT" --context-length 16384

# The shared config generator intentionally uses the ordinary 8000 endpoint;
# this run owns one endpoint per GPU, so bind each worker to its matching port.
"$PYTHON" - "$MIDPOINT_CONFIG" "$MIDPOINT_PORT" "$FINAL_CONFIG" "$FINAL_PORT" "$MAX_CONCURRENT" <<'PY'
import sys
from pathlib import Path
configs = [(Path(sys.argv[1]), int(sys.argv[2])), (Path(sys.argv[3]), int(sys.argv[4]))]
max_concurrent = int(sys.argv[5])
for path, port in configs:
    text = path.read_text()
    old = 'base_url = "http://127.0.0.1:8000/v1"'
    new = f'base_url = "http://127.0.0.1:{port}/v1"'
    if text.count(old) != 1 or text.count("max_concurrent = 4") != 1:
        raise SystemExit(f"unexpected endpoint stanza in {path}")
    text = text.replace(old, new).replace("max_concurrent = 4", f"max_concurrent = {max_concurrent}")
    path.write_text(text)
PY

"$PYTHON" - "$EVAL_ROOT/launch.json" "$MIDPOINT_ADAPTER" "$FINAL_ADAPTER" "$MANIFEST" \
  "$MIDPOINT_GPU" "$FINAL_GPU" "$MIDPOINT_PORT" "$FINAL_PORT" "$MAX_CONCURRENT" \
  "$MAX_MODEL_LEN" "$MAX_BATCHED_TOKENS" "$RENDER_TIMEOUT" "$ROLLOUT_TIMEOUT" <<'PY'
import hashlib, json, sys
from pathlib import Path
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
out, midpoint, final, manifest = map(Path, sys.argv[1:5])
midpoint_gpu, final_gpu, midpoint_port, final_port, max_concurrent, max_model_len, max_batched_tokens, render_timeout, rollout_timeout = map(int, sys.argv[5:])
def adapter_record(path):
    receipt = json.loads((path / "hf-upload.json").read_text())
    return {"path": str(path), "weights_sha256": digest(path / "adapter_model.safetensors"),
            "receipt_sha256": digest(path / "hf-upload.json"), "hf_revision": receipt.get("commit"),
            "hf_path": receipt.get("path")}
out.write_text(json.dumps({
    "schema": "photo-curriculum-sft-matched-eval-launch.v1", "status": "prepared",
    "base_model": "Qwen/Qwen3.8-27B", "base_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "prime_revision": "26b3131d2716a4f8210b165df584b83b4bc54f61", "manifest_sha256": digest(manifest),
    "protocol": {"case_count": 28, "max_turns": 1, "seed": 918200, "temperature": 1.0,
                 "max_tokens": 8192, "context_length": 16384, "render_timeout_seconds": render_timeout,
                 "rollout_timeout_seconds": rollout_timeout, "max_model_len": max_model_len,
                 "max_batched_tokens": max_batched_tokens,
                 "server_max_concurrent": max_concurrent, "servers": {"step_512": {"gpu": midpoint_gpu, "port": midpoint_port}, "step_1024": {"gpu": final_gpu, "port": final_port}}},
    "adapters": {"step_512": adapter_record(midpoint), "step_1024": adapter_record(final)},
    "quality_scoring": "offline_only", "invalid_canvases_preserved": True,
}, indent=2) + "\n")
PY

SERVER_PIDS=()
EVAL_PIDS=()
TELEMETRY_PID=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$TELEMETRY_PID" ]] && kill -0 "$TELEMETRY_PID" 2>/dev/null; then
    kill "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
  fi
  for pid in "${SERVER_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${SERVER_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  for pid in "${EVAL_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${EVAL_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  exit "$status"
}
trap cleanup EXIT INT TERM

start_server() {
  local name=$1 gpu=$2 port=$3 adapter=$4 log=$5
  setsid env CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
    VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}" \
    PLAYWRIGHT_BROWSERS_PATH="$PAINTER_ROOT/browsers" \
    "$VLLM" serve "$MODEL_DIR" --served-model-name base --host 127.0.0.1 --port "$port" \
      --dtype bfloat16 --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_CONCURRENT" \
      --max-num-batched-tokens "$MAX_BATCHED_TOKENS" --gpu-memory-utilization "${EVAL_GPU_MEMORY_UTILIZATION:-0.90}" \
      --limit-mm-per-prompt '{"image":14,"video":0}' --gdn-prefill-backend triton \
      --kernel-config '{"enable_jit_warmup":false}' --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":1048576}' \
      --enable-lora --max-lora-rank 16 --max-loras 1 --max-cpu-loras 1 \
      --lora-modules "$name=$adapter" >"$log" 2>&1 &
  SERVER_PIDS+=("$!")
}
wait_server() {
  local port=$1 name=$2 pid=$3 deadline=$((SECONDS + SERVER_START_TIMEOUT))
  while (( SECONDS < deadline )); do
    local state
    state=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d '[:space:]' || true)
    if [[ -z "$state" || "$state" == Z* ]]; then
      echo "server $name on port $port exited before readiness; inspect logs" >&2
      return 1
    fi
    if "$PYTHON" - "$port" "$name" <<'PY' >/dev/null 2>&1
import json, sys, urllib.request
port, expected = sys.argv[1:]
with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
    models = json.load(response).get("data", [])
if not any(item.get("id") == expected for item in models): raise SystemExit(1)
PY
    then return 0; fi
    sleep 5
  done
  echo "server $name on port $port did not become ready; inspect logs" >&2
  return 1
}

start_server midpoint512 "$MIDPOINT_GPU" "$MIDPOINT_PORT" "$MIDPOINT_ADAPTER" "$LOG_ROOT/server-step512.log"
start_server final1024 "$FINAL_GPU" "$FINAL_PORT" "$FINAL_ADAPTER" "$LOG_ROOT/server-step1024.log"
wait_server "$MIDPOINT_PORT" midpoint512 "${SERVER_PIDS[0]}"
wait_server "$FINAL_PORT" final1024 "${SERVER_PIDS[1]}"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -l 10 >"$LOG_ROOT/gpu-telemetry.csv" 2>&1 &
TELEMETRY_PID=$!

run_eval() {
  local config=$1 log=$2
  local pythonpath="$PAINTER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  setsid env PYTHONPATH="$pythonpath" PAINTER_RENDER_TIMEOUT_SECONDS="$RENDER_TIMEOUT" \
    PLAYWRIGHT_BROWSERS_PATH="$PAINTER_ROOT/browsers" TOKENIZERS_PARALLELISM=false \
    WANDB_MODE=disabled PAINTER_LOCAL_API_KEY="${PAINTER_LOCAL_API_KEY:-local}" \
    bash -c 'cd "$1"; exec "$2" @ "$3" --no-serve --no-push >"$4" 2>&1' \
    _ "$PRIME_ROOT" "$EVAL" "$config" "$log"
}
run_eval "$MIDPOINT_CONFIG" "$LOG_ROOT/eval-step512.log" & MIDPOINT_PID=$!; EVAL_PIDS+=("$MIDPOINT_PID")
run_eval "$FINAL_CONFIG" "$LOG_ROOT/eval-step1024.log" & FINAL_PID=$!; EVAL_PIDS+=("$FINAL_PID")
set +e
wait "$MIDPOINT_PID"; MIDPOINT_STATUS=$?
wait "$FINAL_PID"; FINAL_STATUS=$?
set -e

"$PYTHON" - "$EVAL_ROOT/completion.json" "$MIDPOINT_STATUS" "$FINAL_STATUS" <<'PY'
import json, sys
from pathlib import Path
out, midpoint, final = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
out.write_text(json.dumps({"schema": "photo-curriculum-sft-matched-eval-completion.v1",
    "status": "passed" if midpoint == 0 and final == 0 else "worker_failed",
    "step_512_exit": midpoint, "step_1024_exit": final, "quality_scoring": "offline_only",
    "invalid_canvases_preserved": True,
    "evidence": {"step_512": "step_512", "step_1024": "step_1024", "telemetry": "logs/gpu-telemetry.csv"}}, indent=2) + "\n")
PY
(( MIDPOINT_STATUS == 0 && FINAL_STATUS == 0 ))
