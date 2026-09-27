#!/usr/bin/env bash
# Frozen photo evaluation for the untouched base and fresh-base SFT adapters.
set -Eeuo pipefail
RUN="$(cd "${1:?usage: base_control_eval.sh EXTRACTED_PACKAGE base|step40|step160}" && pwd -P)"
POLICY="${2:?choose base, step40 or step160}"
[[ "$POLICY" == base || "$POLICY" == step40 || "$POLICY" == step160 ]] || exit 2
[[ "$(uname -s)" == Linux && -f "$RUN/env.sh" && -f "$RUN/eval-preflight.json" ]] || exit 2
source "$RUN/env.sh"
export PLAYWRIGHT_BROWSERS_PATH="$RUN/painter/browsers"
export PYTHONPATH="$RUN/painter:$RUN${PYTHONPATH:+:$PYTHONPATH}"
export PAINTER_LOCAL_API_KEY=local PAINTER_RENDER_TIMEOUT_SECONDS=180
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
PY="$PRIME_ROOT/.venv/bin/python"
EVAL="$PRIME_ROOT/.venv/bin/eval"
VLLM="$PRIME_ROOT/.venv/bin/vllm"
MODEL_DIR="$(cat "$RUN/model-path.txt")"
OUT="$RUN/eval-base-control/$POLICY"
mkdir -p "$OUT/cache"
exec 8>"$OUT/eval.lock"
flock -n 8 || { echo 'evaluation already running' >&2; exit 2; }
if [[ "$POLICY" == base ]]; then
  MODEL_LABEL=base
  ADAPTER=''
else
  STEP="${POLICY#step}"
  MODEL_LABEL="$POLICY"
  ADAPTER="$RUN/train-output/brush-base-control-20260927-v1/artifacts/adapters/step_$STEP"
fi
"$PY" - "$RUN" "$POLICY" "$ADAPTER" <<'PY'
import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]);policy=sys.argv[2];a=pathlib.Path(sys.argv[3]) if sys.argv[3] else None
manifest=r/'painter/eval-prep/eval-manifest.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='9943878cc43d48f1cf9ff60e5af633089bf2bd8d45141216531363cad8f1c878'
assert (r/'base-control-training.exit').read_text().strip()=='0'
if a:
    w=a/'adapter_model.safetensors';receipt=json.loads((a/'hf-upload.json').read_text())
    assert receipt['verified'] is True and receipt['private'] is False
    assert hashlib.sha256(w.read_bytes()).hexdigest()==receipt['files']['adapter_model.safetensors']['sha256']
print(json.dumps({'policy':policy,'manifest_verified':True,'public_adapter_verified':bool(a)}))
PY
if [[ -f "$OUT/completion.json" ]]; then
  echo "Completed $POLICY evaluation already exists"; exit 0
fi
CUDA_HOME="$($PY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/cu13")')"
[[ -d "$CUDA_HOME" ]] || exit 2
export CUDA_HOME PATH="$PRIME_ROOT/.venv/bin:$CUDA_HOME/bin:$RUN/bootstrap/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
"$PY" "$RUN/painter/native_painting_eval.py" \
  --root "$RUN/painter" --manifest "$RUN/painter/eval-prep/eval-manifest.json" \
  --policy-label "base-control-$POLICY" --model "$MODEL_LABEL" \
  --output-dir "$OUT/results" --config "$OUT/eval.toml" \
  --expected-count 28 --max-tokens 8192 --max-turns 2 \
  --rollout-timeout 1800 --context-length 32768 > "$OUT/protocol.json"
"$PY" - "$OUT/eval.toml" <<'PY'
import pathlib,sys
p=pathlib.Path(sys.argv[1]);s=p.read_text();old='base_url = "http://127.0.0.1:8000/v1"'
assert s.count(old)==1
p.write_text(s.replace(old,'base_url = "http://127.0.0.1:8103/v1"'))
PY
SERVER_PID=''
cleanup() {
  local status=$?
  printf '%s\n' "$status" > "$OUT/eval.exit"
  if [[ -n "$SERVER_PID" ]]; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
EXTRA=()
if [[ -n "$ADAPTER" ]]; then
  EXTRA=(--enable-lora --max-lora-rank 16 --max-loras 1 --max-cpu-loras 1 --lora-modules "$MODEL_LABEL=$ADAPTER")
fi
setsid env CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" TOKENIZERS_PARALLELISM=false \
  VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=6 \
  VLLM_CACHE_ROOT="$OUT/cache/vllm" TORCHINDUCTOR_CACHE_DIR="$OUT/cache/torchinductor" \
  TRITON_CACHE_DIR="$OUT/cache/triton" \
  "$VLLM" serve "$MODEL_DIR" --served-model-name base --host 127.0.0.1 --port 8103 \
    --dtype bfloat16 --max-model-len 32768 --max-num-seqs 4 \
    --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image":14,"video":0}' --gdn-prefill-backend triton \
    --kernel-config '{"enable_jit_warmup":false}' \
    --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":1048576}' \
    "${EXTRA[@]}" > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
ready=0
for _ in {1..180}; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo 'vLLM exited before readiness' >&2; exit 1; fi
  if "$PY" - "$MODEL_LABEL" <<'PY' >/dev/null 2>&1
import json,sys,urllib.request
with urllib.request.urlopen('http://127.0.0.1:8103/v1/models',timeout=5) as response:
    models=json.load(response).get('data',[])
assert any(item.get('id')==sys.argv[1] for item in models)
PY
  then ready=1; break; fi
  sleep 5
done
(( ready == 1 )) || { echo 'vLLM readiness timed out' >&2; exit 1; }
started=$(date +%s)
(cd "$PRIME_ROOT" && "$EVAL" @ "$OUT/eval.toml" --no-serve --no-push) > "$OUT/eval.log" 2>&1
ended=$(date +%s)
"$PY" - "$RUN" "$OUT" "$POLICY" "$ADAPTER" "$started" "$ended" <<'PY'
import hashlib,json,pathlib,sys
r,o,policy=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),sys.argv[3]
adapter=pathlib.Path(sys.argv[4]) if sys.argv[4] else None
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'schema':'painter.base-control-eval.v1','status':'completed','policy':policy,
         'manifest_sha256':sha(r/'painter/eval-prep/eval-manifest.json'),
         'adapter_sha256':sha(adapter/'adapter_model.safetensors') if adapter else None,
         'config_sha256':sha(o/'eval.toml'),'started_unix':int(sys.argv[5]),
         'ended_unix':int(sys.argv[6]),'elapsed_seconds':int(sys.argv[6])-int(sys.argv[5]),
         'case_count':28,'max_turns':2,'context_length':32768}
(o/'completion.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
PY
