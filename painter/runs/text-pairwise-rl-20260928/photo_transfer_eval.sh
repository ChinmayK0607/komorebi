#!/usr/bin/env bash
set -Eeuo pipefail

# Seven held-out photos, with identical fresh sampling from all three policies.
# Run only after text RL training has stopped. GPU 1 and port 8101 permit a
# separate text validation on GPU 0/8100 if the node has CPU headroom.
STEP="${1:?usage: photo_transfer_eval.sh FINAL_STEP [--preflight]}"
[[ "$STEP" =~ ^[1-9][0-9]*$ ]] || { echo 'step must be positive' >&2; exit 2; }
[[ $# == 1 || ( $# == 2 && "$2" == --preflight ) ]] || exit 2
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$RUN/../../.." && pwd -P)"
PRIME="${PRIME_ROOT:-/root/painter/prime}"
PY="$PRIME/.venv/bin/python"
VLLM="$PRIME/.venv/bin/vllm"
EVAL="$PRIME/.venv/bin/eval"
MODEL_DIR="${MODEL_DIR:-/root/painter/model-cache/qwen/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
PACK="${PAINTER_PHOTO_PACK:-/root/painter/cache/photo-transfer/extracted/mixed-fixed-anchors-v1}"
SFT="$(tr -d '\r\n' < "$RUN/../photo-curriculum-rl-20260922/initial-adapter-path.txt")"
RL16="$($PY - "$RUN/continued-step16-download.json" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
assert r['verified'] and r['adapter_sha256']=='e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c'
print(r['adapter_path'])
PY
)"
FINAL="$RUN/async-output/text-pairwise-continue-20260929/artifacts/adapters/step_$STEP"
OUT="$RUN/photo-transfer-step-$STEP"
mkdir -p "$OUT/logs"
"$PY" - "$PACK" "$OUT" "$SFT" "$RL16" "$FINAL" "$STEP" <<'PY'
import hashlib,json,sys
from pathlib import Path
pack,out,sft,rl16,final=map(Path,sys.argv[1:6]);step=int(sys.argv[6])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
heldout=pack/'heldout-cases.json'
assert sha(heldout)=='26833a8161b276d24a9dd07ad04fb88b06701e481b8c5a06b57f7234c14c4f4e'
cases=json.loads(heldout.read_text())['cases']
assert len(cases)==7 and len({r['id'] for r in cases})==7
rows=[]
for row in cases:
 ref=(pack/row['reference']).resolve()
 assert sha(ref)==row['reference_sha256']
 rows.append({'id':row['id'],'source_group':row['id'],'reference':str(ref),
              'reference_sha256':row['reference_sha256']})
assert sha(sft/'adapter_model.safetensors')=='bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740'
assert sha(rl16/'adapter_model.safetensors')=='e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c'
receipt=json.loads((final/'hf-upload.json').read_text())
assert receipt['verified'] is True and receipt['private'] is False
assert receipt['path']==f'text-pairwise-continue-20260929/step_{step}'
assert sha(final/'adapter_model.safetensors')==receipt['files']['adapter_model.safetensors']['sha256']
(out/'cases.json').write_text(json.dumps({'cases':rows},indent=2,sort_keys=True)+'\n')
(out/'identity.json').write_text(json.dumps({'heldout_manifest_sha256':sha(heldout),
 'sft_sha256':sha(sft/'adapter_model.safetensors'),
 'rl16_sha256':sha(rl16/'adapter_model.safetensors'),
 'final_sha256':sha(final/'adapter_model.safetensors'),
 'final_public_commit':receipt['commit'],'max_turns':4,'temperature':1.0,
 'seed':918200,'max_tokens_per_turn':8192},indent=2,sort_keys=True)+'\n')
print('PHOTO_TRANSFER_INPUTS_VERIFIED',step,receipt['commit'],flush=True)
PY
for label in sft rl16 final; do
  model="photo-$label"
  PYTHONPATH="$REPO/painter" "$PY" "$REPO/painter/native_painting_eval.py" \
    --root "$REPO/painter" --manifest "$OUT/cases.json" \
    --policy-label "$model" --model "$model" \
    --output-dir "$OUT/$label" --config "$OUT/$label.toml" \
    --expected-count 7 --max-turns 4 --max-tokens 8192 \
    --rollout-timeout 1800 --context-length 32768 >"$OUT/logs/prepare-$label.json"
  "$PY" - "$OUT/$label.toml" <<'PY'
from pathlib import Path
import sys
path=Path(sys.argv[1]);value=path.read_text()
old='base_url = "http://127.0.0.1:8000/v1"'
if value.count(old)!=1:raise SystemExit('unexpected eval endpoint')
path.write_text(value.replace(old,'base_url = "http://127.0.0.1:8101/v1"'))
PY
done
if [[ "${2:-}" == --preflight ]]; then echo PHOTO_TRANSFER_READY; exit 0; fi
CUDA_HOME="$($PY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/cu13")')"
export CUDA_HOME PATH="$PRIME/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
setsid env CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false \
  VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=6 \
  VLLM_CACHE_ROOT="$OUT/cache/vllm" TORCHINDUCTOR_CACHE_DIR="$OUT/cache/torchinductor" \
  TRITON_CACHE_DIR="$OUT/cache/triton" \
  "$VLLM" serve "$MODEL_DIR" --served-model-name photo-sft photo-rl16 photo-final \
  --host 127.0.0.1 --port 8101 --dtype bfloat16 \
  --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image":14,"video":0}' \
  --gdn-prefill-backend triton --kernel-config '{"enable_jit_warmup":false}' \
  --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":1048576}' \
  --enable-lora --max-lora-rank 16 --max-loras 3 --max-cpu-loras 3 \
  --lora-modules "photo-sft=$SFT" "photo-rl16=$RL16" "photo-final=$FINAL" \
  >"$OUT/logs/vllm.log" 2>&1 &
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
  kill -0 "$server_pid" || { echo 'photo vLLM exited' >&2; exit 1; }
  if "$PY" - <<'PY' >/dev/null 2>&1
import json,urllib.request
with urllib.request.urlopen('http://127.0.0.1:8101/v1/models',timeout=5) as r:
 ids={m['id'] for m in json.load(r)['data']}
assert {'photo-sft','photo-rl16','photo-final'}<=ids
PY
  then break; fi
  sleep 5
done
(( SECONDS < deadline )) || { echo 'photo vLLM startup timeout' >&2; exit 1; }
for label in sft rl16 final; do
  env PYTHONPATH="$REPO/painter" PAINTER_LOCAL_API_KEY=local \
    PAINTER_RENDER_TIMEOUT_SECONDS=180 PLAYWRIGHT_BROWSERS_PATH="$REPO/painter/browsers" \
    WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
    bash -c 'cd "$1"; exec "$2" @ "$3" --no-serve --no-push' \
    _ "$PRIME" "$EVAL" "$OUT/$label.toml" >"$OUT/logs/eval-$label.log" 2>&1
done
"$PY" "$RUN/photo_transfer_score.py" --manifest "$OUT/cases.json" \
  --sft "$OUT/sft" --rl16 "$OUT/rl16" --final "$OUT/final" \
  --key-file "${PAINTER_GATEWAY_ENV_FILE:-/root/painter/private/gateway.env}" \
  --output "$OUT/review"
echo "PHOTO_TRANSFER_COMPLETE step=$STEP"
