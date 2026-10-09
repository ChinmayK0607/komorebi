#!/usr/bin/env bash
set -Eeuo pipefail

# Matched pre-warmup Flash12 evaluation. It uses the same 16 admitted
# structured-flow cases, renderer, seed, three-turn cap, and native
# evaluation path as the post-RL evaluator. The temporary server loads both
# Flash12 and sft-step512 so it can materialize lazy anchors first.
RUN="${PAINTER_RUN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)}"
REPO="$(cd "$RUN/../../.." && pwd -P)"
source "$RUN/runtime_env.sh"
# This script owns an ungrouped baseline/evaluation phase.  Override an
# inherited RL role before the vLLM server or any Prime helper starts; the
# native policy evaluation below switches to the flow-enabled evaluation role
# explicitly.  The RL orchestrator remains role-gated in run.sh/Prime.
export PAINTER_PRIME_ROLE=baseline
validate_structured_bindings
[[ "$(uname -s)" == Linux ]] || { echo 'matched baseline requires the Linux node' >&2; exit 1; }
[[ $# == 0 || ( $# == 1 && "$1" == --preflight ) ]] || { echo 'usage: matched_baseline.sh [--preflight]' >&2; exit 2; }

PY="${PRIME_ROOT:-$PAINTER_NODE_ROOT/prime}/.venv/bin/python"
VLLM="${PRIME_ROOT:-$PAINTER_NODE_ROOT/prime}/.venv/bin/vllm"
EVAL="${PRIME_ROOT:-$PAINTER_NODE_ROOT/prime}/.venv/bin/eval"
CUDA_HOME="$($PY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/cu13")')"
FLASH_POINTER="${PAINTER_SFT_ADAPTER_PATH_FILE:-$RUN/policy-adapter-path.txt}"
FROZEN_POINTER="${PAINTER_FROZEN_SFT_ADAPTER_PATH_FILE:-$RUN/frozen-sft-adapter-path.txt}"
TRAIN_MANIFEST="${PAINTER_RL_MANIFEST:-$RUN/curriculum/pilot-train-manifest.json}"
EVAL_MANIFEST="${PAINTER_EVAL_MANIFEST:-$RUN/curriculum/matched16-manifest.json}"
OUT="${PAINTER_MATCHED_BASELINE_OUTPUT:-$RUN/matched-baseline/pre-step12}"
RENDERER="${PAINTER_RENDERER_PATH:-$RUN/vendor/integrations/watercolour/renderer.py}"
RENDERER_PYTHON="${PAINTER_RENDERER_PYTHON_PATH:-$RUN/renderer-env/bin/python}"
BROWSER_ROOT="${PAINTER_BROWSER_ROOT:-${PLAYWRIGHT_BROWSERS_PATH:-$RUN/browsers}}"
MODEL="${MODEL_DIR:-$PAINTER_NODE_ROOT/model-cache/qwen/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
PORT="${PAINTER_BASELINE_PORT:-8000}"

for required in "$PY" "$VLLM" "$EVAL" "$MODEL/config.json" "$FLASH_POINTER" "$FROZEN_POINTER" "$RUN/anchor_bank.py" "$RENDERER" "$RENDERER_PYTHON"; do
  [[ -e "$required" ]] || { echo "missing baseline input: $required" >&2; exit 1; }
done
[[ -d "$BROWSER_ROOT" ]] || { echo "native evaluation browser root is missing: $BROWSER_ROOT" >&2; exit 1; }
[[ -d "$CUDA_HOME" ]] || { echo "CUDA 13 runtime directory is missing: $CUDA_HOME" >&2; exit 1; }
export CUDA_HOME
export PATH="${PRIME_ROOT:-$PAINTER_NODE_ROOT/prime}/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
FLASH="$(tr -d '\r\n' < "$FLASH_POINTER")"
FROZEN="$(tr -d '\r\n' < "$FROZEN_POINTER")"
"$PY" - "$FLASH" "$FROZEN" "$PAINTER_POLICY_SHA256" "$PAINTER_FROZEN_SFT_SHA256" <<'PY'
import hashlib, sys
from pathlib import Path
flash, frozen, flash_sha, frozen_sha = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
for path, expected, label in ((flash, flash_sha, "Flash12"), (frozen, frozen_sha, "SFT512")):
    weight = path / "adapter_model.safetensors"
    if path.is_symlink() or not weight.is_file():
        raise SystemExit(f"{label} adapter is incomplete")
    actual = hashlib.sha256(weight.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"{label} adapter hash mismatch: {actual}")
if flash.resolve() == frozen.resolve():
    raise SystemExit("baseline policy and frozen opponent adapters must be distinct")
PY

if [[ "${1:-}" == --preflight ]]; then
  echo 'MATCHED_FLASH12_BASELINE_PREFLIGHT_READY'
  exit 0
fi
[[ ! -e "$OUT" ]] || { echo "baseline output already exists: $OUT" >&2; exit 1; }
mkdir -p "$OUT/logs"
# Prove the ungrouped evaluator role and run-local flow hook with the actual
# Prime interpreter before starting the model server.  This imports only the
# taskset/hook contract; it does not contact a model, judge, or renderer.
EVALUATION_ROLE_SMOKE="$OUT/logs/evaluation-role-smoke.json"
PYTHONPATH="$REPO:$REPO/painter:$RUN${PYTHONPATH:+:$PYTHONPATH}" \
  PAINTER_PRIME_ROLE=evaluation PAINTER_STRUCTURED_FLOW_POLICY=1 \
  PAINTER_RL_MANIFEST="$TRAIN_MANIFEST" PAINTER_EVAL_MANIFEST="$EVAL_MANIFEST" \
  PAINTER_FLOW_HOOK_RECEIPT="$EVALUATION_ROLE_SMOKE" \
  "$PY" - "$EVALUATION_ROLE_SMOKE" <<'PY'
import json
import os
import sys
from pathlib import Path

receipt = Path(sys.argv[1])
if os.environ.get("PAINTER_PRIME_ROLE") != "evaluation":
    raise SystemExit("evaluator role smoke did not run as evaluation")
if os.environ.get("PAINTER_STRUCTURED_FLOW_POLICY") != "1":
    raise SystemExit("evaluator role smoke did not enable structured flow")
if not receipt.is_file():
    raise SystemExit(f"flow hook receipt was not written: {receipt}")
value = json.loads(receipt.read_text(encoding="utf-8"))
if value.get("schema") != "structured-flow-prompt-hook-v1" or not value.get("installed"):
    raise SystemExit(f"flow hook did not install: {value}")
if not value.get("evaluation_taskset_patched"):
    raise SystemExit(f"evaluation taskset was not patched: {value}")
# Loading the real taskset is the boundary that exercises frozen
# NativePaintingData/Task models.  Do not replace this with a SimpleNamespace
# contract: a successful import alone would miss immutable-data failures.
from painter.native_painting import NativePaintingConfig, NativePaintingTaskset

manifest = Path(os.environ["PAINTER_EVAL_MANIFEST"]).resolve()
config = NativePaintingConfig(
    root=str(manifest.parent),
    manifest=str(manifest),
    expected_count=16,
    max_turns=3,
    policy_label="flash12-pre-warmup",
)
tasks = list(NativePaintingTaskset(config).load())
if len(tasks) != 16:
    raise SystemExit(f"evaluation taskset loaded {len(tasks)} tasks, expected 16")
if any("[STRUCTURED_FLOW_POLICY]" not in str(task.data.system_prompt) for task in tasks):
    raise SystemExit("structured flow hint missing from a loaded evaluation task")
print(json.dumps({
    "role": os.environ["PAINTER_PRIME_ROLE"],
    "flow_policy": os.environ["PAINTER_STRUCTURED_FLOW_POLICY"],
    "flow_hook_installed": True,
    "evaluation_taskset_patched": True,
    "evaluation_tasks_loaded": len(tasks),
}, sort_keys=True))
PY
SERVER_PID=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$SERVER_PID" ]]; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

setsid env CUDA_VISIBLE_DEVICES="${PAINTER_BASELINE_GPU:-0}" TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=6 \
  VLLM_CACHE_ROOT="$OUT/cache/vllm" TORCHINDUCTOR_CACHE_DIR="$OUT/cache/torchinductor" TRITON_CACHE_DIR="$OUT/cache/triton" \
  "$VLLM" serve "$MODEL" --served-model-name flash12-baseline --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 --max-model-len 32768 --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image":14,"video":0}' \
  --gdn-prefill-backend triton --kernel-config '{"enable_jit_warmup":false}' \
  --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":1048576}' \
  --enable-lora --max-lora-rank 16 --max-loras 2 --max-cpu-loras 2 \
  --lora-modules "flash12-baseline=$FLASH" "sft-step512=$FROZEN" >"$OUT/logs/vllm.log" 2>&1 &
SERVER_PID=$!
deadline=$((SECONDS + 900))
while (( SECONDS < deadline )); do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo 'baseline vLLM exited' >&2; exit 1; }
  if "$PY" - "$PORT" <<'PY' >/dev/null 2>&1
import json, sys, urllib.request
with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/v1/models", timeout=5) as response:
    names = {item.get("id") for item in json.load(response).get("data", [])}
if {"flash12-baseline", "sft-step512"} - names:
    raise SystemExit(1)
PY
  then break
  fi
  sleep 5
done
(( SECONDS < deadline )) || { echo 'baseline vLLM startup timeout' >&2; exit 1; }

PYTHONPATH="$REPO:$REPO/painter:$RUN${PYTHONPATH:+:$PYTHONPATH}" PAINTER_PRIME_ROLE=baseline PAINTER_STRUCTURED_FLOW_POLICY=0 PAINTER_BROWSER_ROOT="$BROWSER_ROOT" PLAYWRIGHT_BROWSERS_PATH="$BROWSER_ROOT" PAINTER_FROZEN_OPPONENT_ENDPOINT="http://127.0.0.1:$PORT/v1" \
  "$PY" "$RUN/anchor_bank.py" materialize \
  --train-manifest "$TRAIN_MANIFEST" --heldout-manifest "$EVAL_MANIFEST" \
  --anchor-root "${PAINTER_ANCHOR_ROOT:-$RUN/curriculum/anchors}" \
  --output-dir "${PAINTER_ANCHOR_STAGING:-$RUN/curriculum/.materialized-pilot}" \
  --run-root "$RUN" >"$OUT/logs/anchor-materialization.json"

PYTHONPATH="$REPO:$REPO/painter:$RUN${PYTHONPATH:+:$PYTHONPATH}" \
  PAINTER_PRIME_ROLE=evaluation PAINTER_STRUCTURED_FLOW_POLICY=1 PAINTER_BROWSER_ROOT="$BROWSER_ROOT" PLAYWRIGHT_BROWSERS_PATH="$BROWSER_ROOT" PAINTER_RL_MANIFEST="$TRAIN_MANIFEST" PAINTER_EVAL_MANIFEST="$EVAL_MANIFEST" \
  PAINTER_FLOW_HOOK_RECEIPT="$OUT/structured-flow-hook-receipt.json" \
  "$PY" "$REPO/painter/native_painting_eval.py" --root "$REPO/painter" --manifest "$EVAL_MANIFEST" \
  --policy-label flash12-pre-warmup --model flash12-baseline --output-dir "$OUT/evaluation-rollouts" \
  --config "$OUT/eval.toml" --expected-count 16 --max-turns 3 --max-tokens 8192 --max-concurrent 8 \
  --renderer "$RENDERER" --renderer-python "$RENDERER_PYTHON" >"$OUT/logs/config.json"
"$PY" - "$OUT/eval.toml" "$PORT" "${PAINTER_EVAL_SEED:-918210}" <<'PY'
from pathlib import Path
import sys
path, port, seed = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
text = path.read_text()
old = 'base_url = "http://127.0.0.1:8000/v1"'
if text.count(old) != 1 or text.count('seed = 918200') != 1:
    raise SystemExit('unexpected baseline evaluation defaults')
path.write_text(text.replace(old, f'base_url = "http://127.0.0.1:{port}/v1"').replace('seed = 918200', f'seed = {seed}'))
PY
env PYTHONPATH="$REPO/painter:$RUN${PYTHONPATH:+:$PYTHONPATH}" PAINTER_LOCAL_API_KEY=local \
  PAINTER_BROWSER_ROOT="$BROWSER_ROOT" PLAYWRIGHT_BROWSERS_PATH="$BROWSER_ROOT" PAINTER_REQUIRE_VALID_FINAL_ACTION=1 PAINTER_JS_SYNTAX_PREFLIGHT=1 \
  PAINTER_JUDGE_PROTOCOL="$PAINTER_JUDGE_PROTOCOL" PAINTER_JUDGE_FORMAT="$PAINTER_JUDGE_FORMAT" PAINTER_JUDGE_MODEL="$PAINTER_JUDGE_MODEL" \
  PAINTER_STRUCTURED_FLOW_POLICY=1 PAINTER_RL_MANIFEST="$TRAIN_MANIFEST" PAINTER_EVAL_MANIFEST="$EVAL_MANIFEST" \
  PAINTER_PRIME_ROLE=evaluation PAINTER_FLOW_HOOK_RECEIPT="$OUT/structured-flow-hook-receipt.json" \
  bash -c 'cd "$1"; exec "$2" @ "$3" --no-serve --no-push' _ "$PRIME_ROOT" "$EVAL" "$OUT/eval.toml" >"$OUT/logs/eval.log" 2>&1

"$PY" - "$OUT" "$FLASH" "$EVAL_MANIFEST" "${PAINTER_EVAL_SEED:-918210}" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, flash, manifest, seed = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
receipt = {
    "schema": "structured-flow-matched-flash12-baseline-v1", "status": "complete",
    "policy": {"label": "flash12-pre-warmup", "revision": "1227a2427a8ae16014d2fdca4d1adb6855d3fa1c", "prefix": "rl-judge-matched-20261004-flash/step_12", "adapter_sha256": sha(flash / "adapter_model.safetensors")},
    "manifest_sha256": sha(manifest), "manifest_cases": 16, "max_turns": 3, "seed": seed,
    "judge_deferred": True, "quality_scoring": "offline_only",
    "limitations": ["Invalid, censored, and missing-final-action receipts remain in evaluation-rollouts for audit."],
}
(out / "baseline-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY
echo "MATCHED_FLASH12_BASELINE_COMPLETE: $OUT"
