#!/usr/bin/env bash
# Matched, finite eight-text-case eval of SFT, RL, and exact LoRA blends.
set -Eeuo pipefail

RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$RUN/../../.." && pwd -P)"
PRIME_ROOT="${PRIME_ROOT:-/root/painter/prime}"
MODEL_DIR="${MODEL_DIR:-/root/painter/model-cache/qwen/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
MANIFEST="${PAINTER_CANARY_MANIFEST:-/root/painter/merge-canary/merge-canary-20260929/manifest.json}"
OUT="${PAINTER_CANARY_OUT:-/root/painter/merge-canary/eval-20260929}"
PY="$PRIME_ROOT/.venv/bin/python"
VLLM="$PRIME_ROOT/.venv/bin/vllm"
EVAL="$PRIME_ROOT/.venv/bin/eval"
[[ "$(uname -s)" == Linux ]] || { echo 'Linux node required' >&2; exit 1; }
[[ -x "$PY" && -x "$VLLM" && -x "$EVAL" ]] || { echo 'pinned Prime runtime missing' >&2; exit 1; }
[[ -f "$MODEL_DIR/config.json" && -f "$MANIFEST" ]] || { echo 'pinned model or manifest missing' >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo 'refusing to overwrite previous canary' >&2; exit 1; }
SFT="$(tr -d '\r\n' < "$RUN/../photo-curriculum-rl-20260922/initial-adapter-path.txt")"
RL="$($PY -c 'import json;print(json.load(open("/root/painter/step16-public-receipt.json"))["adapter_path"])')"
[[ -f "$SFT/adapter_model.safetensors" && -f "$RL/adapter_model.safetensors" ]] || {
  echo 'pinned adapters missing' >&2; exit 1;
}
if [[ "${1:-}" == '--preflight' ]]; then
  [[ $# == 1 ]] || { echo 'usage: run_merge_canary.sh [--preflight]' >&2; exit 2; }
  PYTHONPATH="$REPO/painter" "$PY" - "$REPO/painter" "$MANIFEST" <<'PYCODE'
import sys
from pathlib import Path
from native_painting_eval import validate_manifest
audit=validate_manifest(Path(sys.argv[1]),Path(sys.argv[2]),expected_count=8)
assert audit['count']==8
PYCODE
  echo CANARY_PREFLIGHT_READY
  exit 0
fi
[[ $# == 0 ]] || { echo 'usage: run_merge_canary.sh [--preflight]' >&2; exit 2; }
mkdir -p "$OUT/logs"
"$PY" "$RUN/merge_adapters.py" --sft "$SFT" --rl "$RL" \
  --output "$OUT/mix025" --rl-fraction 0.25 > "$OUT/logs/mix025.json"
"$PY" "$RUN/merge_adapters.py" --sft "$SFT" --rl "$RL" \
  --output "$OUT/mix050" --rl-fraction 0.5 > "$OUT/logs/mix050.json"

for policy in sft rl mix025 mix050; do
  "$PY" "$REPO/painter/native_painting_eval.py" \
    --root "$REPO/painter" --manifest "$MANIFEST" --expected-count 8 \
    --policy-label "$policy" --model "$policy" \
    --output-dir "$OUT/$policy" --config "$OUT/$policy.toml" \
    --max-turns 2 --max-tokens 8192 --rollout-timeout 1800 \
    > "$OUT/logs/config-$policy.json"
  "$PY" - "$OUT/$policy.toml" <<'PYCODE'
from pathlib import Path
import sys
path=Path(sys.argv[1]); text=path.read_text()
before='base_url = "http://127.0.0.1:8000/v1"'
assert text.count(before)==1
path.write_text(text.replace(before,'base_url = "http://127.0.0.1:8100/v1"'))
PYCODE
done
CUDA_HOME="$($PY -c 'import sysconfig;print(sysconfig.get_paths()["purelib"]+"/nvidia/cu13")')"
export CUDA_HOME PATH="$PRIME_ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
setsid env CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
  VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=6 \
  VLLM_CACHE_ROOT="$OUT/cache/vllm" TORCHINDUCTOR_CACHE_DIR="$OUT/cache/torchinductor" \
  TRITON_CACHE_DIR="$OUT/cache/triton" \
  "$VLLM" serve "$MODEL_DIR" --served-model-name canary-base \
  --host 127.0.0.1 --port 8100 --dtype bfloat16 \
  --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":14,"video":0}' \
  --gdn-prefill-backend triton --kernel-config '{"enable_jit_warmup":false}' \
  --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":1048576}' \
  --enable-lora --max-lora-rank 32 --max-loras 4 --max-cpu-loras 4 \
  --lora-modules "sft=$SFT" "rl=$RL" "mix025=$OUT/mix025" "mix050=$OUT/mix050" \
  > "$OUT/logs/vllm.log" 2>&1 &
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
  kill -0 "$server_pid" || { echo 'vLLM exited before readiness' >&2; tail -30 "$OUT/logs/vllm.log" >&2; exit 1; }
  if "$PY" - <<'PYCODE' >/dev/null 2>&1
import json,urllib.request
with urllib.request.urlopen('http://127.0.0.1:8100/v1/models',timeout=5) as response:
    names={entry['id'] for entry in json.load(response)['data']}
assert {'sft','rl','mix025','mix050'} <= names
PYCODE
  then break; fi
  sleep 5
done
(( SECONDS < deadline )) || { echo 'vLLM readiness timeout' >&2; exit 1; }

for policy in sft rl mix025 mix050; do
  printf 'START %s %s\n' "$policy" "$(date -u +%FT%TZ)" | tee -a "$OUT/logs/progress.log"
  env PYTHONPATH="$REPO/painter" PAINTER_LOCAL_API_KEY=local \
    PAINTER_RENDER_TIMEOUT_SECONDS=180 PLAYWRIGHT_BROWSERS_PATH="$REPO/painter/browsers" \
    WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
    bash -c 'cd "$1"; exec "$2" @ "$3" --no-serve --no-push' \
    _ "$PRIME_ROOT" "$EVAL" "$OUT/$policy.toml" > "$OUT/logs/eval-$policy.log" 2>&1
  printf 'DONE %s %s\n' "$policy" "$(date -u +%FT%TZ)" | tee -a "$OUT/logs/progress.log"
done
echo CANARY_EVAL_COMPLETE
