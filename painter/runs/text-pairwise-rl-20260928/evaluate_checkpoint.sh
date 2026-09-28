#!/usr/bin/env bash
set -Eeuo pipefail

STEP="${1:?usage: evaluate_checkpoint.sh STEP [8|16]}"
COUNT="${2:-8}"
[[ "$STEP" =~ ^[1-9][0-9]*$ && ( "$COUNT" == 8 || "$COUNT" == 16 ) ]] || {
  echo 'step must be positive; count must be 8 or 16' >&2; exit 2;
}
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$RUN/../../.." && pwd -P)"
PRIME_ROOT="${PRIME_ROOT:-/root/painter/prime}"
MODEL_DIR="${MODEL_DIR:-/root/painter/model-cache/qwen/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
PY="$PRIME_ROOT/.venv/bin/python"
VLLM="$PRIME_ROOT/.venv/bin/vllm"
EVAL="$PRIME_ROOT/.venv/bin/eval"
ADAPTER="$RUN/async-output/text-pairwise-rl-20260928/artifacts/adapters/step_$STEP"
KEY="${PAINTER_GATEWAY_ENV_FILE:-/root/painter/private/gateway.env}"
OUT="$RUN/validation/step-$STEP-n$COUNT"
[[ -f "$ADAPTER/adapter_model.safetensors" ]] || { echo 'checkpoint adapter missing' >&2; exit 1; }
[[ -f "$ADAPTER/hf-upload.json" ]] || { echo 'checkpoint upload verification missing' >&2; exit 1; }
[[ -r "$KEY" && ! -L "$KEY" && "$(stat -c %a "$KEY")" == 600 ]] || {
  echo 'private Gateway key file missing or insecure' >&2; exit 1;
}
[[ ! -e "$OUT/comparison/summary.json" ]] || { echo 'validation already scored' >&2; exit 1; }
mkdir -p "$OUT/logs"
"$PY" - "$RUN/anchors-frozen/validation-manifest.json" "$OUT/cases.json" "$COUNT" <<'PY'
import json,sys
from pathlib import Path
source,target=map(Path,sys.argv[1:3]);count=int(sys.argv[3]);rows=json.loads(source.read_text())['cases']
if count==8:
    simple=[r for r in rows if r['family']=='simple'][:4]
    intermediate=[r for r in rows if r['family']=='intermediate'][:4]
    rows=simple+intermediate
if len(rows)!=count or len({r['id'] for r in rows})!=count:
    raise SystemExit(f'expected {count} distinct available validation rows')
target.write_text(json.dumps({'cases':rows},indent=2,sort_keys=True)+'\n')
PY
PYTHONPATH="$REPO/painter" "$PY" "$REPO/painter/native_painting_eval.py" \
  --root "$REPO/painter" --manifest "$OUT/cases.json" \
  --policy-label "rl-step-$STEP" --model "rl-step-$STEP" \
  --output-dir "$OUT/eval" --config "$OUT/eval.toml" \
  --expected-count "$COUNT" --max-turns 3 --max-tokens 8192 \
  --rollout-timeout 1800 >"$OUT/logs/config.json"
"$PY" - "$OUT/eval.toml" <<'PY'
from pathlib import Path
import sys
path=Path(sys.argv[1]);text=path.read_text()
old='base_url = "http://127.0.0.1:8000/v1"'
if text.count(old)!=1:raise SystemExit('unexpected eval endpoint')
path.write_text(text.replace(old,'base_url = "http://127.0.0.1:8100/v1"'))
PY
if [[ "${3:-}" == '--preflight' ]]; then echo 'VALIDATION_PREFLIGHT_READY'; exit 0; fi
[[ $# -le 2 ]] || { echo 'usage: evaluate_checkpoint.sh STEP [8|16]' >&2; exit 2; }

CUDA_HOME="$($PY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/cu13")')"
export CUDA_HOME PATH="$PRIME_ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
setsid env CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
  VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=6 \
  VLLM_CACHE_ROOT="$OUT/cache/vllm" TORCHINDUCTOR_CACHE_DIR="$OUT/cache/torchinductor" \
  TRITON_CACHE_DIR="$OUT/cache/triton" \
  "$VLLM" serve "$MODEL_DIR" --served-model-name "rl-step-$STEP" \
  --host 127.0.0.1 --port 8100 --dtype bfloat16 \
  --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image":14,"video":0}' \
  --gdn-prefill-backend triton --kernel-config '{"enable_jit_warmup":false}' \
  --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":1048576}' \
  --enable-lora --max-lora-rank 16 --max-loras 1 --max-cpu-loras 1 \
  --lora-modules "rl-step-$STEP=$ADAPTER" >"$OUT/logs/vllm.log" 2>&1 &
server_pid=$!
cleanup() {
  status=$?
  trap - EXIT INT TERM
  kill -TERM -- "-$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT INT TERM
deadline=$((SECONDS+900))
while (( SECONDS < deadline )); do
  kill -0 "$server_pid" || { echo 'validation vLLM exited' >&2; exit 1; }
  if "$PY" - <<'PY' >/dev/null 2>&1
import json,urllib.request
with urllib.request.urlopen('http://127.0.0.1:8100/v1/models',timeout=5) as r:
    models=json.load(r)['data']
assert any(m['id'].startswith('rl-step-') for m in models)
PY
  then break; fi
  sleep 5
done
(( SECONDS < deadline )) || { echo 'validation vLLM startup timeout' >&2; exit 1; }
env PYTHONPATH="$REPO/painter" PAINTER_LOCAL_API_KEY=local \
  PAINTER_RENDER_TIMEOUT_SECONDS=180 PLAYWRIGHT_BROWSERS_PATH="$REPO/painter/browsers" \
  WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
  bash -c 'cd "$1"; exec "$2" @ "$3" --no-serve --no-push' \
  _ "$PRIME_ROOT" "$EVAL" "$OUT/eval.toml" >"$OUT/logs/eval.log" 2>&1
"$PY" "$RUN/score_validation.py" --manifest "$OUT/cases.json" \
  --eval-root "$OUT/eval" --output "$OUT/comparison" --key-file "$KEY"
