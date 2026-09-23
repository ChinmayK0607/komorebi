#!/usr/bin/env bash
# Finite six-turn matched photo evaluation: frozen SFT step 512 vs public RL step 24.
set -Eeuo pipefail

PAINTER_ROOT=${PAINTER_ROOT:-/root/painter}
RUN_ROOT=${RUN_ROOT:-$PAINTER_ROOT/runs/photo-curriculum-rl-20260922}
PRIME_ROOT=${PRIME_ROOT:-$PAINTER_ROOT/prime-rl}
MODEL_DIR=${MODEL_DIR:-$PAINTER_ROOT/cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}
BASE_REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
SOURCE_MANIFEST=${SOURCE_MANIFEST:-$PAINTER_ROOT/runs/photo-curriculum-rl-20260922/eval-source/eval-manifest.json}
SELECTOR=${SELECTOR:-$RUN_ROOT/select_photo_eval.py}
EVAL_GENERATOR=${EVAL_GENERATOR:-$PAINTER_ROOT/native_painting_eval.py}
PYTHON=${PYTHON:-$PRIME_ROOT/.venv/bin/python}
EVAL=${EVAL:-$PRIME_ROOT/.venv/bin/eval}
VLLM=${VLLM:-$PRIME_ROOT/.venv/bin/vllm}

FROZEN_SFT_ADAPTER=${FROZEN_SFT_ADAPTER:-}
if [[ -z "$FROZEN_SFT_ADAPTER" && -f "$RUN_ROOT/initial-adapter-path.txt" ]]; then
  FROZEN_SFT_ADAPTER=$(tr -d '\r\n' < "$RUN_ROOT/initial-adapter-path.txt")
fi
RL_ADAPTER=${RL_ADAPTER:-$RUN_ROOT/checkpoint-preservation/adapters/step_24}
RL_RECEIPT=${RL_RECEIPT:-$RUN_ROOT/checkpoint-preservation/preservation.json}
SFT_REVISION=${SFT_REVISION:-adf687e5b321514eda7be6eab050434fdc8e44ff}
SFT_PREFIX=${SFT_PREFIX:-photo-curriculum-sft-20260922-balanced/step_512}
SFT_SHA256=${SFT_SHA256:-bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740}
RL_REVISION=${RL_REVISION:-ee897c3f6572c7e11595207d80fc6497fdca5972}
RL_PREFIX=${RL_PREFIX:-photo-curriculum-rl-20260922-broadcast-backup/step_24}

EVAL_ROOT=${EVAL_ROOT:-$RUN_ROOT/eval-runs/photo-curriculum-rl-20260922-step24}
LOG_ROOT=$EVAL_ROOT/logs
CACHE_ROOT=$EVAL_ROOT/cache
SELECTED_MANIFEST=$EVAL_ROOT/photo-12-manifest.json
SFT_CONFIG=$EVAL_ROOT/sft-step512.toml
RL_CONFIG=$EVAL_ROOT/rl-step24.toml
SFT_OUTPUT=$EVAL_ROOT/sft-step512
RL_OUTPUT=$EVAL_ROOT/rl-step24
SFT_GPU=${SFT_GPU:-0}; RL_GPU=${RL_GPU:-1}
SFT_PORT=${SFT_PORT:-8100}; RL_PORT=${RL_PORT:-8101}
MAX_CONCURRENT=${MAX_CONCURRENT:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-8192}
RENDER_TIMEOUT=${PAINTER_RENDER_TIMEOUT_SECONDS:-180}
ROLLOUT_TIMEOUT=${ROLLOUT_TIMEOUT_SECONDS:-1800}
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT_SECONDS:-900}
GPU_PHASE_TIMEOUT=${GPU_PHASE_TIMEOUT_SECONDS:-2700}
INFERENCE_ROOT=${INFERENCE_ROOT:-$PRIME_ROOT/.venv}
mkdir -p "$LOG_ROOT" "$CACHE_ROOT/sft-step512" "$CACHE_ROOT/rl-step24"

[[ $(uname -s) == Linux ]] || { echo "this launcher must run on the Linux node" >&2; exit 2; }
[[ "$(basename "$MODEL_DIR")" == "$BASE_REVISION" ]] || { echo "MODEL_DIR is not the pinned Qwen revision" >&2; exit 2; }
for required in "$PYTHON" "$EVAL" "$VLLM" "$MODEL_DIR/config.json" "$SOURCE_MANIFEST" \
  "$SELECTOR" "$EVAL_GENERATOR" "$FROZEN_SFT_ADAPTER/adapter_model.safetensors" \
  "$RL_ADAPTER/adapter_model.safetensors" "$RL_RECEIPT"; do
  [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 2; }
done

run_gpu_phase() {
"$PYTHON" - "$EVAL_ROOT/launch.json" "$SELECTED_MANIFEST" "$SFT_SHA256" "$RL_REVISION" "$RL_PREFIX" "$SFT_GPU" "$RL_GPU" "$SFT_PORT" "$RL_PORT" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, manifest, sft_sha, rl_revision, rl_prefix, sft_gpu, rl_gpu, sft_port, rl_port = sys.argv[1:]
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
Path(out).write_text(json.dumps({
    "schema": "photo-curriculum-rl-matched-eval-launch.v1", "status": "prepared",
    "source_manifest_sha256": digest(manifest), "sft_step512_sha256": sft_sha,
    "rl_step24_revision": rl_revision, "rl_step24_prefix": rl_prefix,
    "protocol": {"case_count": 12, "max_turns": 6, "seed": 918200, "temperature": 1.0,
                 "max_tokens": 8192, "context_length": 32768, "thinking": False,
                 "render_timeout_seconds": 180, "gpu_phase_timeout_seconds": 2700,
                 "invalid_canvases_preserved": True, "quality_scoring": "offline_only"},
    "policies": {"sft_step512": {"model": "sft-step512", "gpu": int(sft_gpu), "port": int(sft_port)},
                 "rl_step24": {"model": "rl-step24", "gpu": int(rl_gpu), "port": int(rl_port)}},
}, indent=2, sort_keys=True) + "\n")
PY

SERVER_PIDS=(); EVAL_PIDS=(); TELEMETRY_PID=""; DEADLINE=""
kill_group() {
  local pid=$1 pgid
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]' || true)
  [[ "$pgid" == "$pid" ]] || return 0
  kill -TERM -- "-$pid" 2>/dev/null || true
}
alive_pid() {
  local state
  state=$(ps -o stat= -p "$1" 2>/dev/null | tr -d '[:space:]' || true)
  [[ -n "$state" && "$state" != Z* ]]
}
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  [[ -z "$TELEMETRY_PID" ]] || { kill "$TELEMETRY_PID" 2>/dev/null || true; wait "$TELEMETRY_PID" 2>/dev/null || true; }
  for pid in "${EVAL_PIDS[@]}"; do kill_group "$pid"; done
  for pid in "${SERVER_PIDS[@]}"; do kill_group "$pid"; done
  for pid in "${EVAL_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  for pid in "${SERVER_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  exit "$status"
}
trap cleanup EXIT INT TERM

start_server() {
  local name=$1 gpu=$2 port=$3 adapter=$4 log=$5
  setsid env CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0 \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}" VLLM_CACHE_ROOT="$CACHE_ROOT/$name/vllm" \
    TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/$name/torchinductor" TRITON_CACHE_DIR="$CACHE_ROOT/$name/triton" \
    PLAYWRIGHT_BROWSERS_PATH="$PAINTER_ROOT/browsers" "$VLLM" serve "$MODEL_DIR" \
      --served-model-name "$name" --host 127.0.0.1 --port "$port" --dtype bfloat16 \
      --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_CONCURRENT" \
      --max-num-batched-tokens "$MAX_BATCHED_TOKENS" --gpu-memory-utilization "${EVAL_GPU_MEMORY_UTILIZATION:-0.90}" \
      --limit-mm-per-prompt '{"image":14,"video":0}' --gdn-prefill-backend triton \
      --kernel-config '{"enable_jit_warmup":false}' --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":1048576}' \
      --enable-lora --max-lora-rank 16 --max-loras 1 --max-cpu-loras 1 \
      --lora-modules "$name=$adapter" >"$log" 2>&1 &
  SERVER_PIDS+=("$!")
}
wait_server() {
  local port=$1 name=$2 pid=$3 deadline=$((SECONDS + SERVER_START_TIMEOUT))
  (( DEADLINE < deadline )) && deadline=$DEADLINE
  while (( SECONDS < deadline )); do
    local state
    state=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d '[:space:]' || true)
    [[ -n "$state" && "$state" != Z* ]] || { echo "server $name exited; inspect logs" >&2; return 1; }
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
  echo "server $name did not become ready" >&2; return 1
}
run_eval() {
  local config=$1 log=$2
  setsid env PYTHONPATH="$PAINTER_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    PAINTER_RENDER_TIMEOUT_SECONDS="$RENDER_TIMEOUT" PLAYWRIGHT_BROWSERS_PATH="$PAINTER_ROOT/browsers" \
    TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled PAINTER_LOCAL_API_KEY="${PAINTER_LOCAL_API_KEY:-local}" \
    bash -c 'cd "$1"; exec "$2" @ "$3" --no-serve --no-push >"$4" 2>&1' \
    _ "$PRIME_ROOT" "$EVAL" "$config" "$log"
}

DEADLINE=$((SECONDS + GPU_PHASE_TIMEOUT))
start_server sft-step512 "$SFT_GPU" "$SFT_PORT" "$FROZEN_SFT_ADAPTER" "$LOG_ROOT/server-sft-step512.log"
start_server rl-step24 "$RL_GPU" "$RL_PORT" "$RL_ADAPTER" "$LOG_ROOT/server-rl-step24.log"
wait_server "$SFT_PORT" sft-step512 "${SERVER_PIDS[0]}"
wait_server "$RL_PORT" rl-step24 "${SERVER_PIDS[1]}"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -l 10 >"$LOG_ROOT/gpu-telemetry.csv" 2>&1 &
TELEMETRY_PID=$!
run_eval "$SFT_CONFIG" "$LOG_ROOT/eval-sft-step512.log" & SFT_PID=$!; EVAL_PIDS+=("$SFT_PID")
run_eval "$RL_CONFIG" "$LOG_ROOT/eval-rl-step24.log" & RL_PID=$!; EVAL_PIDS+=("$RL_PID")
while (( SECONDS < DEADLINE )); do
  sft_alive=0; rl_alive=0
  alive_pid "$SFT_PID" && sft_alive=1 || true
  alive_pid "$RL_PID" && rl_alive=1 || true
  (( sft_alive == 0 && rl_alive == 0 )) && break
  sleep 5
done
SFT_STATUS=0; RL_STATUS=0
if alive_pid "$SFT_PID"; then SFT_STATUS=124; kill_group "$SFT_PID"; fi
if alive_pid "$RL_PID"; then RL_STATUS=124; kill_group "$RL_PID"; fi
set +e
wait "$SFT_PID"; waited=$?; (( SFT_STATUS == 0 )) && SFT_STATUS=$waited
wait "$RL_PID"; waited=$?; (( RL_STATUS == 0 )) && RL_STATUS=$waited
set -e

"$PYTHON" - "$EVAL_ROOT/completion.json" "$SFT_STATUS" "$RL_STATUS" <<'PY'
import json, sys
from pathlib import Path
out, sft, rl = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
out.write_text(json.dumps({"schema": "photo-curriculum-rl-matched-eval-completion.v1",
    "status": "passed" if sft == 0 and rl == 0 else "worker_failed",
    "sft_step512_exit": sft, "rl_step24_exit": rl, "quality_scoring": "offline_only",
    "invalid_canvases_preserved": True,
    "evidence": {"sft_step512": "sft-step512", "rl_step24": "rl-step24", "telemetry": "logs/gpu-telemetry.csv"}},
    indent=2, sort_keys=True) + "\n")
PY
(( SFT_STATUS == 0 && RL_STATUS == 0 ))
}
[[ "$MAX_MODEL_LEN" == 32768 ]] || { echo "MAX_MODEL_LEN is fixed at 32768" >&2; exit 2; }
[[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_CONCURRENT must be positive" >&2; exit 2; }
[[ "$MAX_BATCHED_TOKENS" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_BATCHED_TOKENS must be positive" >&2; exit 2; }
[[ "$ROLLOUT_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo "ROLLOUT_TIMEOUT_SECONDS must be positive" >&2; exit 2; }
[[ "$GPU_PHASE_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo "GPU_PHASE_TIMEOUT_SECONDS must be positive" >&2; exit 2; }
[[ "$GPU_PHASE_TIMEOUT" -le 2700 ]] || { echo "GPU_PHASE_TIMEOUT_SECONDS must be <= 2700" >&2; exit 2; }
[[ "$RENDER_TIMEOUT" =~ ^[0-9]+$ ]] && (( RENDER_TIMEOUT >= 1 && RENDER_TIMEOUT <= 180 )) || {
  echo "PAINTER_RENDER_TIMEOUT_SECONDS must be 1..180" >&2; exit 2;
}

CUDA_HOME="$($PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/cu13")')"
[[ -d "$CUDA_HOME" ]] || { echo "CUDA 13 runtime directory is missing: $CUDA_HOME" >&2; exit 2; }
export CUDA_HOME
export PATH="$INFERENCE_ROOT/bin:$CUDA_HOME/bin:$PAINTER_ROOT/bootstrap/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
PYTHONPATH="$PAINTER_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - <<'PY'
import importlib.metadata
for package, expected in (("vllm", "0.28.0"), ("transformers", "5.6.2")):
    actual = importlib.metadata.version(package)
    if actual != expected: raise SystemExit(f"{package}={actual}; expected pinned {expected}")
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 2: raise SystemExit("two CUDA devices are required")
print({"torch": torch.__version__, "cuda": torch.version.cuda, "devices": torch.cuda.device_count()})
PY

PYTHONPATH="$PAINTER_ROOT" "$PYTHON" "$SELECTOR" --source "$SOURCE_MANIFEST" --output "$SELECTED_MANIFEST" --count 12

"$PYTHON" - "$FROZEN_SFT_ADAPTER" "$RL_ADAPTER" "$SFT_SHA256" "$RL_RECEIPT" \
  "$SFT_REVISION" "$RL_REVISION" "$SFT_PREFIX" "$RL_PREFIX" <<'PY'
import hashlib, json, sys
from pathlib import Path
sft, rl, expected_sft, receipt_path, sft_rev, rl_rev, sft_prefix, rl_prefix = sys.argv[1:]
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify(path, expected, label):
    folder = Path(path); weight = folder / "adapter_model.safetensors"
    if folder.is_symlink() or not folder.is_dir() or not weight.is_file(): raise SystemExit(f"unsafe {label} adapter")
    actual = digest(weight)
    if actual != expected: raise SystemExit(f"{label} hash mismatch: {actual} != {expected}")
    config = json.loads((folder / "adapter_config.json").read_text())
    if config.get("r") != 16 or config.get("lora_alpha") != 32: raise SystemExit(f"{label} rank/alpha mismatch")
    return actual
def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values(): yield from walk(child)
    elif isinstance(value, list):
        for child in value: yield from walk(child)
sft_sha = verify(sft, expected_sft, "SFT")
payload = json.loads(Path(receipt_path).read_text())
record = next((x for x in walk(payload) if x.get("step") == 24 and isinstance(x.get("huggingface"), dict)), None)
if record is None: raise SystemExit("receipt lacks RL step 24")
public = record["huggingface"]
if public.get("verified") is not True or public.get("private") is not False: raise SystemExit("RL step 24 is not public and verified")
if public.get("commit") != rl_rev or public.get("path") != rl_prefix: raise SystemExit("RL public identity mismatch")
expected_rl = record.get("adapter_sha256") or ((public.get("files") or {}).get("adapter_model.safetensors") or {}).get("sha256")
if not expected_rl: raise SystemExit("receipt lacks RL weight hash")
rl_sha = verify(rl, expected_rl, "RL step 24")
print(json.dumps({"sft": {"sha256": sft_sha, "revision": sft_rev, "prefix": sft_prefix}, "rl": {"sha256": rl_sha, "revision": rl_rev, "prefix": rl_prefix}}, sort_keys=True))
PY

for spec in \
  "${SFT_CONFIG}|photo-sft-step512|sft-step512|${SFT_OUTPUT}|${SFT_PORT}" \
  "${RL_CONFIG}|photo-rl-step24|rl-step24|${RL_OUTPUT}|${RL_PORT}"; do
  IFS='|' read -r config label model output port <<< "$spec"
  PYTHONPATH="$PAINTER_ROOT" "$PYTHON" "$EVAL_GENERATOR" --root "$PAINTER_ROOT" \
    --manifest "$SELECTED_MANIFEST" --policy-label "$label" --model "$model" \
    --output-dir "$output" --config "$config" --expected-count 12 --max-tokens 8192 \
    --max-turns 6 --rollout-timeout "$ROLLOUT_TIMEOUT" --context-length 32768
  "$PYTHON" - "$config" "$port" "$MAX_CONCURRENT" <<'PY'
from pathlib import Path
import sys
path, port, concurrency = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
text = path.read_text()
old = 'base_url = "http://127.0.0.1:8000/v1"'
if text.count(old) != 1 or text.count("max_concurrent = 4") != 1: raise SystemExit(f"unexpected config: {path}")
path.write_text(text.replace(old, f'base_url = "http://127.0.0.1:{port}/v1"').replace("max_concurrent = 4", f"max_concurrent = {concurrency}"))
PY
done
run_gpu_phase
