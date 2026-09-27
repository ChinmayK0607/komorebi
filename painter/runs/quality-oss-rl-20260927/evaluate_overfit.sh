#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate the frozen step-512 policy and a public RL step on identical four
# references and sampling settings. Run only after the finite trainer exits.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$ROOT/../../.." && pwd -P)"
PAINTER_ROOT="$REPO/painter"
PRIME_ROOT="${PRIME_ROOT:-/root/painter/prime}"
MODEL_DIR="${MODEL_DIR:-/root/painter/model-cache/qwen/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
STEP="${RL_STEP:-4}"
[[ "$STEP" =~ ^[1-9][0-9]*$ ]] || { echo 'RL_STEP must be a positive integer' >&2; exit 2; }
OUT="$ROOT/matched-eval-step-$STEP"
SFT_ADAPTER="$(tr -d '\r\n' < "$ROOT/../photo-curriculum-rl-20260922/initial-adapter-path.txt")"
RL_ADAPTER="$ROOT/async-output/gateway-overfit-four-20260928/artifacts/adapters/step_$STEP"
PYTHON="$PRIME_ROOT/.venv/bin/python"
VLLM="$PRIME_ROOT/.venv/bin/vllm"
EVAL="$PRIME_ROOT/.venv/bin/eval"
mkdir -p "$OUT/logs" "$OUT/cache/sft" "$OUT/cache/rl"

"$PYTHON" - "$ROOT/profile-data/overfit-four-manifest.json" "$OUT" "$SFT_ADAPTER" "$RL_ADAPTER" "$STEP" <<'PY'
import hashlib, json, sys
from pathlib import Path
manifest, out, sft, rl, step = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5])
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
rows=json.loads(manifest.read_text())['rows']
if len(rows)!=4: raise SystemExit('Expected exactly four references')
cases=[]
for row in rows:
    reference=(manifest.parent/row['reference']).resolve()
    if digest(reference)!=row['reference_sha256']: raise SystemExit(f"Reference hash mismatch: {row['id']}")
    cases.append({'id':row['id'], 'source_group':row['id'], 'reference':str(reference), 'reference_sha256':row['reference_sha256']})
if digest(sft/'adapter_model.safetensors')!='bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740':
    raise SystemExit('SFT adapter hash mismatch')
upload=json.loads((rl/'hf-upload.json').read_text())
if upload.get('verified') is not True or upload.get('private') is not False or upload.get('path')!=f'gateway-overfit-four-20260928/step_{step}':
    raise SystemExit('RL adapter is not public and verified for requested step')
expected=upload['files']['adapter_model.safetensors']['sha256']
if digest(rl/'adapter_model.safetensors')!=expected: raise SystemExit('RL adapter hash mismatch')
(out/'cases.json').write_text(json.dumps({'cases':cases},indent=2,sort_keys=True)+'\n')
(out/'identity.json').write_text(json.dumps({'manifest_sha256':digest(manifest),'sft_sha256':digest(sft/'adapter_model.safetensors'),
    'rl_sha256':expected,'rl_public_commit':upload['commit'],'rl_public_path':upload['path'],
    'max_turns':4,'temperature':1.0,'seed':918200,'max_tokens_per_turn':8192},indent=2,sort_keys=True)+'\n')
print('MATCHED_INPUTS_VERIFIED', step, upload['commit'], flush=True)
PY

for spec in "sft|sft-step512|8100" "rl|rl-step-$STEP|8101"; do
  IFS='|' read -r policy model port <<< "$spec"
  PYTHONPATH="$PAINTER_ROOT" "$PYTHON" "$PAINTER_ROOT/native_painting_eval.py" \
    --root "$PAINTER_ROOT" --manifest "$OUT/cases.json" \
    --policy-label "$model" --model "$model" \
    --output-dir "$OUT/$policy" --config "$OUT/$policy.toml" \
    --expected-count 4 --max-tokens 8192 --max-turns 4 \
    --rollout-timeout 1800 --context-length 32768 >"$OUT/logs/prepare-$policy.json"
  "$PYTHON" - "$OUT/$policy.toml" "$port" <<'PY'
from pathlib import Path
import sys
path,port=Path(sys.argv[1]),sys.argv[2]
value=path.read_text()
old='base_url = "http://127.0.0.1:8000/v1"'
if value.count(old)!=1: raise SystemExit('Unexpected eval config URL')
path.write_text(value.replace(old,f'base_url = "http://127.0.0.1:{port}/v1"'))
PY
done
if [[ "${1:-}" == "--preflight" ]]; then
  echo 'MATCHED_EVAL_READY'
  exit 0
fi
[[ $# == 0 ]] || { echo 'usage: evaluate_overfit.sh [--preflight]' >&2; exit 2; }

CUDA_HOME="$($PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/cu13")')"
export CUDA_HOME PATH="$PRIME_ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
PIDS=()
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "${PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  exit "$status"
}
trap cleanup EXIT INT TERM
start_server() {
  local policy=$1 gpu=$2 port=$3 adapter=$4 model=$5
  setsid env CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
    VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=6 \
    VLLM_CACHE_ROOT="$OUT/cache/$policy/vllm" \
    TORCHINDUCTOR_CACHE_DIR="$OUT/cache/$policy/torchinductor" \
    TRITON_CACHE_DIR="$OUT/cache/$policy/triton" \
    "$VLLM" serve "$MODEL_DIR" --served-model-name "$model" \
    --host 127.0.0.1 --port "$port" --dtype bfloat16 \
    --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image":14,"video":0}' \
    --gdn-prefill-backend triton --kernel-config '{"enable_jit_warmup":false}' \
    --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":1048576}' \
    --enable-lora --max-lora-rank 16 --max-loras 1 --max-cpu-loras 1 \
    --lora-modules "$model=$adapter" >"$OUT/logs/server-$policy.log" 2>&1 &
  PIDS+=("$!")
}
wait_server() {
  local policy=$1 port=$2 pid=$3 model=$4 deadline=$((SECONDS+900))
  while (( SECONDS < deadline )); do
    kill -0 "$pid" 2>/dev/null || { echo "$policy server exited" >&2; return 1; }
    if "$PYTHON" - "$port" "$model" <<'PY' >/dev/null 2>&1
import json,sys,urllib.request
with urllib.request.urlopen(f'http://127.0.0.1:{sys.argv[1]}/v1/models',timeout=5) as r: models=json.load(r)['data']
if not any(m.get('id')==sys.argv[2] for m in models): raise SystemExit(1)
PY
    then return 0; fi
    sleep 5
  done
  echo "$policy server startup timed out" >&2; return 1
}
start_server sft 0 8100 "$SFT_ADAPTER" sft-step512
start_server rl 1 8101 "$RL_ADAPTER" "rl-step-$STEP"
wait_server sft 8100 "${PIDS[0]}" sft-step512
wait_server rl 8101 "${PIDS[1]}" "rl-step-$STEP"
for policy in sft rl; do
  setsid env PYTHONPATH="$PAINTER_ROOT" PAINTER_LOCAL_API_KEY=local \
    PAINTER_RENDER_TIMEOUT_SECONDS=180 PLAYWRIGHT_BROWSERS_PATH="$PAINTER_ROOT/browsers" \
    TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled \
    bash -c 'cd "$1"; exec "$2" @ "$3" --no-serve --no-push' \
    _ "$PRIME_ROOT" "$EVAL" "$OUT/$policy.toml" >"$OUT/logs/eval-$policy.log" 2>&1 &
  PIDS+=("$!")
done
set +e
wait "${PIDS[2]}"; sft_status=$?
wait "${PIDS[3]}"; rl_status=$?
set -e
"$PYTHON" - "$OUT/completion.json" "$sft_status" "$rl_status" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]); sft=int(sys.argv[2]); rl=int(sys.argv[3])
out.write_text(json.dumps({'status':'passed' if sft==0 and rl==0 else 'failed',
    'sft_exit':sft,'rl_exit':rl,'quality_scoring':'offline'},indent=2)+'\n')
PY
(( sft_status == 0 && rl_status == 0 ))
