#!/usr/bin/env bash
set -Eeuo pipefail

STEP="${1:?usage: evaluate.sh STEP [all|balanced16] [--preflight]}"
SCOPE="${2:-all}"
MODE="${3:-}"
MAX_CONCURRENT="${PAINTER_EVAL_MAX_CONCURRENT:-16}"
EXPECTED_COUNT="${PAINTER_EVAL_EXPECTED_COUNT:-47}"
JUDGE_PROTOCOL="${PAINTER_JUDGE_PROTOCOL:-text-paint-sft-pairwise-consistent-v1}"
JUDGE_FORMAT="${PAINTER_JUDGE_FORMAT:-identity-joint-side-schema-v1}"
JUDGE_MODEL="${PAINTER_JUDGE_MODEL:-xiaomi/mimo-v2.6-flash}"
[[ "$STEP" =~ ^[1-9][0-9]*$ && ( "$SCOPE" == all || "$SCOPE" == balanced16 ) ]] || {
  echo 'STEP must be positive and SCOPE must be all or balanced16' >&2; exit 2;
}
[[ -z "$MODE" || "$MODE" == --preflight ]] || { echo 'usage: evaluate.sh STEP [all|balanced16] [--preflight]' >&2; exit 2; }
[[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]] || { echo 'PAINTER_EVAL_MAX_CONCURRENT must be a positive integer' >&2; exit 2; }
[[ "$EXPECTED_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo 'PAINTER_EVAL_EXPECTED_COUNT must be a positive integer' >&2; exit 2; }
RUN="${PAINTER_RUN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)}"
REPO="$(cd "$RUN/../../.." && pwd -P)"
NODE_ROOT="${PAINTER_NODE_ROOT:-/root/painter}"
PRIME_ROOT="${PRIME_ROOT:-$NODE_ROOT/prime}"
PY="$PRIME_ROOT/.venv/bin/python"
VLLM="$PRIME_ROOT/.venv/bin/vllm"
EVAL="$PRIME_ROOT/.venv/bin/eval"
MODEL_DIR="${MODEL_DIR:-$NODE_ROOT/model-cache/qwen/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
NAME="${PAINTER_RUN_NAME:-text-sft512-consistent-20260930}"
ADAPTER="$RUN/async-output/$NAME/artifacts/adapters/step_$STEP"
ALL_MANIFEST="${PAINTER_EVAL_MANIFEST:-$RUN/anchors-frozen/validation-manifest.json}"
OUT="$RUN/validation/step-$STEP-$SCOPE"
RENDERER="${PAINTER_RENDERER_PATH:-$RUN/vendor/integrations/watercolour/renderer.py}"
RENDERER_PYTHON="${PAINTER_RENDERER_PYTHON_PATH:-$RUN/renderer-env/bin/python}"
EVAL_EXPECTED_COUNT="$EXPECTED_COUNT"
[[ "$SCOPE" == balanced16 ]] && EVAL_EXPECTED_COUNT=16
[[ "$OUT" == /* ]] || { echo 'evaluation output path must be absolute' >&2; exit 1; }

[[ -x "$PY" && -x "$VLLM" && -x "$EVAL" ]] || { echo 'pinned Prime runtime is unavailable' >&2; exit 1; }
CHECKPOINT_RECEIPT="$ADAPTER/hf-upload.json"
if [[ -n "${PAINTER_EVAL_PUBLIC_RECEIPT:-}" ]]; then
  CHECKPOINT_RECEIPT="$PAINTER_EVAL_PUBLIC_RECEIPT"
  [[ -f "$ADAPTER/adapter_model.safetensors" && -f "$CHECKPOINT_RECEIPT" ]] || {
    echo 'requested public checkpoint adapter or supplied public receipt is missing' >&2; exit 1;
  }
  "$PY" - "$CHECKPOINT_RECEIPT" "$ADAPTER" <<'PY'
import hashlib, json, sys
from pathlib import Path
receipt_path, adapter = Path(sys.argv[1]), Path(sys.argv[2])
receipt = json.loads(receipt_path.read_text())
if (receipt.get('provenance_type') != 'public_download' or
        receipt.get('verified') is not True or receipt.get('private') is not False or
        receipt.get('public_metadata_verified') is not True):
    raise SystemExit('supplied public checkpoint receipt is not a verified public download')
weight = adapter / 'adapter_model.safetensors'
expected = receipt.get('files', {}).get('adapter_model.safetensors', {}).get('sha256')
if not isinstance(expected, str):
    raise SystemExit('supplied public checkpoint receipt has no adapter SHA')
digest = hashlib.sha256()
with weight.open('rb') as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
if digest.hexdigest() != expected:
    raise SystemExit('supplied public checkpoint receipt does not match adapter bytes')
PY
else
  [[ -f "$ADAPTER/adapter_model.safetensors" && -f "$CHECKPOINT_RECEIPT" ]] || {
    echo 'requested public checkpoint adapter is missing' >&2; exit 1;
  }
fi
[[ -f "$ALL_MANIFEST" ]] || { echo 'frozen validation manifest is missing' >&2; exit 1; }
[[ -f "$RENDERER" && -x "$RENDERER_PYTHON" ]] || { echo 'explicit renderer runtime is missing' >&2; exit 1; }
# Verify the exact lexical virtualenv interpreter before starting vLLM.  Do not
# resolve this path: renderer-env/bin/python may be a symlink whose enclosing
# virtualenv supplies Playwright and its browser dependencies.
"$RENDERER_PYTHON" - <<'PY'
try:
    import playwright  # noqa: F401
except Exception as exc:
    raise SystemExit(f'renderer interpreter cannot import playwright: {exc}')
PY
[[ ! -e "$OUT" ]] || { echo 'evaluation output already exists; preserve it before retrying' >&2; exit 1; }
mkdir -p "$OUT/logs"

"$PY" - "$ALL_MANIFEST" "$OUT/cases.json" "$SCOPE" "$EXPECTED_COUNT" <<'PY'
import json, sys
from pathlib import Path
source, target = Path(sys.argv[1]), Path(sys.argv[2])
scope = sys.argv[3]
expected_count = int(sys.argv[4])
rows = json.loads(source.read_text())['cases']
if len(rows) != expected_count or len({row['id'] for row in rows}) != expected_count:
    raise SystemExit(f'expected immutable {expected_count}-case validation set')
if scope == 'balanced16':
    simple = [row for row in rows if row['family'] == 'simple']
    intermediate = [row for row in rows if row['family'] == 'intermediate']
    complex_rows = [row for row in rows if row['family'] == 'complex']
    if len(simple) < 4 or len(intermediate) < 4 or len(complex_rows) < 8:
        raise SystemExit('balanced16 requires four simple, four intermediate, and eight complex cases')
    rows = simple[:4] + intermediate[:4] + complex_rows[:8]
if len(rows) != (16 if scope == 'balanced16' else expected_count):
    raise SystemExit('requested validation scope is incomplete')
target.write_text(json.dumps({'cases': rows}, indent=2, sort_keys=True) + '\n')
PY

PYTHONPATH="$REPO/painter" "$PY" "$REPO/painter/native_painting_eval.py" \
  --root "$REPO/painter" --manifest "$OUT/cases.json" \
  --policy-label "consistent-step-$STEP" --model "consistent-step-$STEP" \
  --output-dir "$OUT/eval" --config "$OUT/eval.toml" \
  --expected-count "$EVAL_EXPECTED_COUNT" \
  --max-turns 3 --max-tokens 8192 --rollout-timeout 1800 \
  --max-concurrent "$MAX_CONCURRENT" --renderer "$RENDERER" \
  --renderer-python "$RENDERER_PYTHON" >"$OUT/logs/config.json"
"$PY" - "$OUT/eval.toml" "$MAX_CONCURRENT" <<'PY'
from pathlib import Path
import os, sys
path = Path(sys.argv[1])
max_concurrent = int(sys.argv[2])
text = path.read_text()
if text.count('base_url = "http://127.0.0.1:8000/v1"') != 1:
    raise SystemExit('unexpected evaluation endpoint')
if text.count('seed = 918200') != 1:
    raise SystemExit('unexpected evaluation seed')
if text.count(f'max_concurrent = {max_concurrent}') != 1:
    raise SystemExit('unexpected evaluation concurrency')
seed = os.environ.get('PAINTER_EVAL_SEED', '918210')
if not seed.isdigit():
    raise SystemExit('PAINTER_EVAL_SEED must be numeric')
path.write_text(text.replace('base_url = "http://127.0.0.1:8000/v1"', 'base_url = "http://127.0.0.1:8100/v1"').replace('seed = 918200', f'seed = {seed}'))
PY
if [[ "$MODE" == --preflight ]]; then
  echo 'CONSISTENT_EVALUATION_GENERATION_READY'
  exit 0
fi

CUDA_HOME="$($PY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/cu13")')"
export CUDA_HOME PATH="$PRIME_ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
setsid env CUDA_VISIBLE_DEVICES="${PAINTER_EVAL_GPU:-0}" TOKENIZERS_PARALLELISM=false \
  VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=6 \
  VLLM_CACHE_ROOT="$OUT/cache/vllm" TORCHINDUCTOR_CACHE_DIR="$OUT/cache/torchinductor" \
  TRITON_CACHE_DIR="$OUT/cache/triton" \
  "$VLLM" serve "$MODEL_DIR" --served-model-name "consistent-step-$STEP" \
  --host 127.0.0.1 --port 8100 --dtype bfloat16 --max-model-len 32768 \
  --max-num-seqs 16 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":14,"video":0}' --gdn-prefill-backend triton \
  --kernel-config '{"enable_jit_warmup":false}' \
  --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":1048576}' \
  --enable-lora --max-lora-rank 16 --max-loras 1 --max-cpu-loras 1 \
  --lora-modules "consistent-step-$STEP=$ADAPTER" >"$OUT/logs/vllm.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  status=$?
  trap - EXIT INT TERM
  kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT INT TERM
deadline=$((SECONDS + 900))
while (( SECONDS < deadline )); do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo 'evaluation vLLM exited' >&2; exit 1; }
  if STEP_CHECK="$STEP" "$PY" - <<'PY' >/dev/null 2>&1
import json, os, urllib.request
with urllib.request.urlopen('http://127.0.0.1:8100/v1/models', timeout=5) as response:
    models = json.load(response)['data']
assert any(model.get('id') == 'consistent-step-' + os.environ['STEP_CHECK'] for model in models)
PY
  then break; fi
  sleep 5
done
(( SECONDS < deadline )) || { echo 'evaluation vLLM startup timeout' >&2; exit 1; }
env PYTHONPATH="$REPO/painter" PAINTER_LOCAL_API_KEY=local \
  PAINTER_RENDER_TIMEOUT_SECONDS=180 PLAYWRIGHT_BROWSERS_PATH="$REPO/painter/browsers" \
  PAINTER_REQUIRE_VALID_FINAL_ACTION=1 PAINTER_JS_SYNTAX_PREFLIGHT=1 \
  PAINTER_JUDGE_PROTOCOL="$JUDGE_PROTOCOL" PAINTER_JUDGE_FORMAT="$JUDGE_FORMAT" \
  PAINTER_JUDGE_MODEL="$JUDGE_MODEL" PAINTER_RENDER_WORKERS="${PAINTER_RENDER_WORKERS:-8}" \
  WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
  bash -c 'cd "$1"; exec "$2" @ "$3" --no-serve --no-push' \
  _ "$PRIME_ROOT" "$EVAL" "$OUT/eval.toml" >"$OUT/logs/eval.log" 2>&1

PYTHONPATH="$REPO/painter" "$PY" - "$OUT" "$OUT/eval/evaluation-rollouts" "$ADAPTER" "$STEP" "$SCOPE" "$MAX_CONCURRENT" "$CHECKPOINT_RECEIPT" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
from native_painting_eval import audit_evaluation_artifacts

out = Path(sys.argv[1]).resolve()
artifact_root = Path(sys.argv[2]).resolve()
adapter, step, scope, max_concurrent, checkpoint_receipt = (
    Path(sys.argv[3]), int(sys.argv[4]), sys.argv[5], int(sys.argv[6]), Path(sys.argv[7])
)

try:
    artifact_root_relative = artifact_root.relative_to(out).as_posix()
except ValueError as exc:
    raise SystemExit('evaluation artifacts escaped the archived run root') from exc
artifact_audit = audit_evaluation_artifacts(artifact_root)
if artifact_audit['valid_receipts_with_png'] < 1:
    raise SystemExit('evaluation produced no valid PNG-backed painting receipt')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
upload = json.loads(checkpoint_receipt.read_text())
expected = upload['files']['adapter_model.safetensors']['sha256']
actual = sha(adapter / 'adapter_model.safetensors')
if upload.get('verified') is not True or upload.get('private') is not False or actual != expected:
    raise SystemExit('checkpoint public receipt or local adapter hash failed')
(out / 'evaluation-artifact-receipt.json').write_text(json.dumps({
    'schema': 'text-sft512-evaluation-artifacts-v1',
    'artifact_root': artifact_root_relative,
    **artifact_audit,
}, indent=2, sort_keys=True) + '\n')
(out / 'generation-receipt.json').write_text(json.dumps({
    'schema': 'text-paint-consistent-generation-v2', 'run_name': out.parents[1].name, 'step': step,
    'scope': scope, 'adapter_sha256': actual, 'public_commit': upload.get('commit'),
    'checkpoint_provenance': upload.get('provenance_type', 'uploaded_checkpoint'),
    'judge_deferred': True, 'scoring_protocol': os.environ.get('PAINTER_JUDGE_PROTOCOL', 'text-paint-sft-pairwise-consistent-v1'),
    'judge_format': os.environ.get('PAINTER_JUDGE_FORMAT', 'identity-joint-side-schema-v1'),
    'judge_model': os.environ.get('PAINTER_JUDGE_MODEL', 'xiaomi/mimo-v2.6-flash'),
    'matched_control': 'sft-step512', 'max_turns': 3, 'max_tokens': 8192,
    'max_concurrent': max_concurrent,
    'evaluation_artifacts': {
        'artifact_root': artifact_root_relative,
        'receipt_count': artifact_audit['receipt_count'],
        'valid_receipts': artifact_audit['valid_receipts'],
        'valid_receipts_with_png': artifact_audit['valid_receipts_with_png'],
        'invalid_receipts': artifact_audit['invalid_receipts'],
    },
}, indent=2, sort_keys=True) + '\n')
PY
echo "CONSISTENT_GENERATION_COMPLETE: step=$STEP scope=$SCOPE; score locally after node release"
