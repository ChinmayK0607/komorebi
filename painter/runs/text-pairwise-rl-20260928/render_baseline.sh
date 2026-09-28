#!/usr/bin/env bash
set -Eeuo pipefail

# Two independent pinned SFT samplers; no trainer and no Gateway calls.
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$RUN/../../.." && pwd -P)"
PRIME_ROOT="${PRIME_ROOT:-/root/painter/prime}"
MODEL_DIR="${MODEL_DIR:-/root/painter/model-cache/qwen/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
PY="$PRIME_ROOT/.venv/bin/python"
VLLM="$PRIME_ROOT/.venv/bin/vllm"
EVAL="$PRIME_ROOT/.venv/bin/eval"
ADAPTER="$(tr -d '\r\n' < "$RUN/../photo-curriculum-rl-20260922/initial-adapter-path.txt")"
OUT="$RUN/baseline"
[[ "$(uname -s)" == Linux ]] || { echo 'Linux required' >&2; exit 1; }
[[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 2 ]] || {
  echo 'two GPUs required for parallel baseline' >&2; exit 1;
}
[[ -f "$ADAPTER/adapter_model.safetensors" ]] || { echo 'SFT adapter missing' >&2; exit 1; }
mkdir -p "$OUT/logs"
"$PY" - "$RUN/prepared/pool.json" "$OUT" <<'PY'
import hashlib,json,sys
from pathlib import Path
pool,out=map(Path,sys.argv[1:]); raw=json.loads(pool.read_text()); rows=raw['cases']
if len(rows)!=72:raise SystemExit('pool count changed')
for i in range(2):
    cases=[]
    for row in rows[i::2]:
        path=(pool.parent/row['reference']).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['reference_sha256']:
            raise SystemExit(f"brief hash mismatch: {row['id']}")
        cases.append({**row,'reference':str(path)})
    (out/f'shard-{i}.json').write_text(json.dumps({'cases':cases},indent=2)+'\n')
PY
for i in 0 1; do
  port=$((8100+i))
  PYTHONPATH="$REPO/painter" "$PY" "$REPO/painter/native_painting_eval.py" \
    --root "$REPO/painter" --manifest "$OUT/shard-$i.json" \
    --policy-label sft-step512 --model sft-step512 \
    --output-dir "$OUT/shard-$i" --config "$OUT/shard-$i.toml" \
    --expected-count 36 --max-turns 3 --max-tokens 8192 \
    --rollout-timeout 1800 >"$OUT/logs/config-$i.json"
  "$PY" - "$OUT/shard-$i.toml" "$port" <<'PY'
from pathlib import Path
import sys
path=Path(sys.argv[1]); before='base_url = "http://127.0.0.1:8000/v1"'
text=path.read_text()
if text.count(before)!=1:raise SystemExit('unexpected VF endpoint')
path.write_text(text.replace(before,f'base_url = "http://127.0.0.1:{sys.argv[2]}/v1"'))
PY
done
if [[ "${1:-}" == '--preflight' ]]; then
  echo 'BASELINE_PREFLIGHT_READY'; exit 0
fi
[[ $# == 0 ]] || { echo 'usage: render_baseline.sh [--preflight]' >&2; exit 2; }

CUDA_HOME="$($PY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/cu13")')"
export CUDA_HOME PATH="$PRIME_ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
pids=()
cleanup() {
  status=$?
  trap - EXIT INT TERM
  for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  exit "$status"
}
trap cleanup EXIT INT TERM
for i in 0 1; do
  port=$((8100+i))
  setsid env CUDA_VISIBLE_DEVICES="$i" TOKENIZERS_PARALLELISM=false \
    VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=6 \
    VLLM_CACHE_ROOT="$OUT/cache-$i/vllm" \
    TORCHINDUCTOR_CACHE_DIR="$OUT/cache-$i/torchinductor" \
    TRITON_CACHE_DIR="$OUT/cache-$i/triton" \
    "$VLLM" serve "$MODEL_DIR" --served-model-name sft-step512 \
    --host 127.0.0.1 --port "$port" --dtype bfloat16 \
    --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image":14,"video":0}' \
    --gdn-prefill-backend triton --kernel-config '{"enable_jit_warmup":false}' \
    --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":1048576}' \
    --enable-lora --max-lora-rank 16 --max-loras 1 --max-cpu-loras 1 \
    --lora-modules "sft-step512=$ADAPTER" >"$OUT/logs/vllm-$i.log" 2>&1 &
  pids+=("$!")
done
for i in 0 1; do
  port=$((8100+i)); deadline=$((SECONDS+900))
  while (( SECONDS < deadline )); do
    kill -0 "${pids[$i]}" || { echo "vLLM $i exited" >&2; exit 1; }
    if "$PY" - "$port" <<'PY' >/dev/null 2>&1
import json,sys,urllib.request
with urllib.request.urlopen(f'http://127.0.0.1:{sys.argv[1]}/v1/models',timeout=5) as r:
    models=json.load(r)['data']
assert any(m['id']=='sft-step512' for m in models)
PY
    then break; fi
    sleep 5
  done
  (( SECONDS < deadline )) || { echo "vLLM $i startup timed out" >&2; exit 1; }
done
for i in 0 1; do
  setsid env PYTHONPATH="$REPO/painter" PAINTER_LOCAL_API_KEY=local \
    PAINTER_RENDER_TIMEOUT_SECONDS=180 PLAYWRIGHT_BROWSERS_PATH="$REPO/painter/browsers" \
    WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
    bash -c 'cd "$1"; exec "$2" @ "$3" --no-serve --no-push' \
    _ "$PRIME_ROOT" "$EVAL" "$OUT/shard-$i.toml" >"$OUT/logs/eval-$i.log" 2>&1 &
  pids+=("$!")
done
wait "${pids[2]}"
wait "${pids[3]}"
"$PY" "$RUN/build_anchors.py" --pool "$RUN/prepared/pool.json" \
  --eval-root "$OUT" --output "$RUN/anchors-frozen"
