#!/usr/bin/env bash
# Frozen 28-photo two-turn evaluation; compare first paints and their revisions
# against the same baseline on GPU 1, then evaluate the public final adapter.
set -Eeuo pipefail
RUN="$(cd "${1:?usage: launch_multiturn_eval.sh RUN_DIRECTORY baseline|trained}" && pwd -P)"
POLICY="${2:?choose baseline or trained}"
[[ "$POLICY" == baseline || "$POLICY" == trained ]] || exit 2
[[ "$(uname -s)" == Linux && -f "$RUN/env.sh" && -x "$RUN/painter/renderer-env/bin/python" ]] || {
  echo 'training bootstrap and eval renderer setup are required' >&2; exit 2;
}
source "$RUN/env.sh"
export PLAYWRIGHT_BROWSERS_PATH="$RUN/painter/browsers"
export PYTHONPATH="$RUN${PYTHONPATH:+:$PYTHONPATH}"
export PAINTER_LOCAL_API_KEY=local PAINTER_RENDER_TIMEOUT_SECONDS=180
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
PY="$PRIME_ROOT/.venv/bin/python"
EVAL="$PRIME_ROOT/.venv/bin/eval"
VLLM="$PRIME_ROOT/.venv/bin/vllm"
MODEL_DIR="$(cat "$RUN/model-path.txt")"
MANIFEST="$RUN/painter/eval-prep/eval-manifest.json"
EVAL_ROOT="$RUN/eval/$POLICY"
PORT=8100
GPU="${EVAL_GPU:-1}"
mkdir -p "$EVAL_ROOT" "$EVAL_ROOT/cache"
exec 8>"$EVAL_ROOT/eval.lock"
flock -n 8 || { echo 'evaluation already running for this policy' >&2; exit 2; }
if [[ "$POLICY" == baseline ]]; then
  ADAPTER="$(cat "$RUN/initial-adapter-path.txt")"
else
  STEP="$($PY - "$RUN/data/mix-manifest.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['optimizer_steps_at_batch_4'])
PY
)"
  ADAPTER="$RUN/train-output/multiturn-sft-20260925-v1/artifacts/adapters/step_$STEP"
fi
"$PY" - "$RUN" "$POLICY" "$ADAPTER" "$MANIFEST" <<'PY'
import hashlib,json,pathlib,sys
root,policy,adapter,manifest=pathlib.Path(sys.argv[1]),sys.argv[2],pathlib.Path(sys.argv[3]),pathlib.Path(sys.argv[4])
expected_manifest='9943878cc43d48f1cf9ff60e5af633089bf2bd8d45141216531363cad8f1c878'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
if sha(manifest)!=expected_manifest:raise ValueError('frozen 28-photo manifest SHA-256 mismatch')
weight=adapter/'adapter_model.safetensors'
if not weight.is_file():raise ValueError('evaluation adapter missing')
if policy=='baseline':
    if sha(weight)!='bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740':
        raise ValueError('baseline adapter hash mismatch')
else:
    receipt=json.loads((adapter/'hf-upload.json').read_text())
    if receipt.get('verified') is not True or receipt.get('private') is not False:
        raise ValueError('trained adapter is not yet public/hash-verified')
    expected=((receipt.get('files') or {}).get('adapter_model.safetensors') or {}).get('sha256')
    if sha(weight)!=expected:raise ValueError('trained adapter hash differs from public receipt')
print(json.dumps({'policy':policy,'adapter_sha256':sha(weight),'manifest_sha256':sha(manifest)}))
PY
CUDA_HOME="$($PY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/cu13")')"
[[ -d "$CUDA_HOME" ]] || { echo 'CUDA 13 runtime missing' >&2; exit 2; }
export CUDA_HOME PATH="$PRIME_ROOT/.venv/bin:$CUDA_HOME/bin:$RUN/bootstrap/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
"$PY" "$RUN/painter/native_painting_eval.py" \
  --root "$RUN/painter" --manifest "$MANIFEST" --policy-label "multiturn-sft-$POLICY" \
  --model "$POLICY" --output-dir "$EVAL_ROOT/results" --config "$EVAL_ROOT/eval.toml" \
  --expected-count 28 --max-tokens 8192 --max-turns 2 \
  --rollout-timeout 1800 --context-length 16384 > "$EVAL_ROOT/protocol.json"
"$PY" - "$EVAL_ROOT/eval.toml" "$PORT" <<'PY'
import pathlib,sys
p=pathlib.Path(sys.argv[1]);s=p.read_text();old='base_url = "http://127.0.0.1:8000/v1"'
if s.count(old)!=1:raise ValueError('unexpected eval endpoint stanza')
p.write_text(s.replace(old,f'base_url = "http://127.0.0.1:{sys.argv[2]}/v1"'))
PY
SERVER_PID=''
cleanup() {
  local status=$?
  printf '%s\n' "$status" > "$EVAL_ROOT/eval.exit"
  if [[ -n "$SERVER_PID" ]]; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
setsid env CUDA_VISIBLE_DEVICES="$GPU" TOKENIZERS_PARALLELISM=false \
  VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=6 \
  VLLM_CACHE_ROOT="$EVAL_ROOT/cache/vllm" \
  TORCHINDUCTOR_CACHE_DIR="$EVAL_ROOT/cache/torchinductor" \
  TRITON_CACHE_DIR="$EVAL_ROOT/cache/triton" \
  "$VLLM" serve "$MODEL_DIR" --served-model-name base --host 127.0.0.1 --port "$PORT" \
    --dtype bfloat16 --max-model-len 16384 --max-num-seqs 4 \
    --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image":14,"video":0}' --gdn-prefill-backend triton \
    --kernel-config '{"enable_jit_warmup":false}' \
    --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":1048576}' \
    --enable-lora --max-lora-rank 16 --max-loras 1 --max-cpu-loras 1 \
    --lora-modules "$POLICY=$ADAPTER" > "$EVAL_ROOT/server.log" 2>&1 &
SERVER_PID=$!
ready=0
for _ in {1..180}; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo 'vLLM exited before readiness' >&2; exit 1; fi
  if "$PY" - "$PORT" "$POLICY" <<'PY' >/dev/null 2>&1
import json,sys,urllib.request
with urllib.request.urlopen(f'http://127.0.0.1:{sys.argv[1]}/v1/models',timeout=5) as r:
    models=json.load(r).get('data',[])
if not any(x.get('id')==sys.argv[2] for x in models):raise SystemExit(1)
PY
  then ready=1; break; fi
  sleep 5
done
(( ready == 1 )) || { echo 'vLLM readiness timed out' >&2; exit 1; }
started=$(date +%s)
(cd "$PRIME_ROOT" && "$EVAL" @ "$EVAL_ROOT/eval.toml" --no-serve --no-push) > "$EVAL_ROOT/eval.log" 2>&1
ended=$(date +%s)
"$PY" - "$RUN" "$EVAL_ROOT" "$POLICY" "$ADAPTER" "$started" "$ended" <<'PY'
import hashlib,json,pathlib,sys
root,out,policy,adapter=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),sys.argv[3],pathlib.Path(sys.argv[4])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'schema':'painter.multiturn-matched-eval.v1','status':'completed',
         'policy':policy,'manifest_sha256':sha(root/'painter/eval-prep/eval-manifest.json'),
         'adapter_sha256':sha(adapter/'adapter_model.safetensors'),
         'config_sha256':sha(out/'eval.toml'),'started_unix':int(sys.argv[5]),
         'ended_unix':int(sys.argv[6]),'elapsed_seconds':int(sys.argv[6])-int(sys.argv[5]),
         'case_count':28,'max_turns':2,'quality_scoring':'offline_pairwise_per_turn_and_final',
         'invalid_canvases_preserved':True}
(out/'completion.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
PY
