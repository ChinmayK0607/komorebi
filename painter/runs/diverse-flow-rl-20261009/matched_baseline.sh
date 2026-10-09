#!/usr/bin/env bash
set -Eeuo pipefail

# Matched pre-warmup Flash12 evaluation, plus the optional post-warmup
# control. Both phases use the same 32 admitted Diverse-flow cases, renderer,
# seed, three-turn cap, and native
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
PHASE="pre-warmup"
if [[ "${1:-}" == --post-warmup ]]; then
  PHASE="post-warmup"
  export PAINTER_PRIME_ROLE=evaluation
  shift
fi
[[ $# == 0 || ( $# == 1 && "$1" == --preflight ) ]] || { echo 'usage: matched_baseline.sh [--post-warmup] [--preflight]' >&2; exit 2; }

PY="${PRIME_ROOT:-$PAINTER_NODE_ROOT/prime}/.venv/bin/python"
VLLM="${PRIME_ROOT:-$PAINTER_NODE_ROOT/prime}/.venv/bin/vllm"
EVAL="${PRIME_ROOT:-$PAINTER_NODE_ROOT/prime}/.venv/bin/eval"
CUDA_HOME="$($PY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/cu13")')"
POLICY_POINTER="${PAINTER_SFT_ADAPTER_PATH_FILE:-$RUN/policy-adapter-path.txt}"
FROZEN_POINTER="${PAINTER_FROZEN_SFT_ADAPTER_PATH_FILE:-$RUN/frozen-sft-adapter-path.txt}"
TRAIN_MANIFEST="${PAINTER_RL_MANIFEST:-$RUN/curriculum/pilot-train-manifest.json}"
EVAL_MANIFEST="${PAINTER_EVAL_MANIFEST:-$RUN/curriculum/matched32-manifest.json}"
if [[ "$PHASE" == "post-warmup" ]]; then
  POLICY_POINTER="${PAINTER_ACTIVE_POLICY_PATH_FILE:-$RUN/active-policy-adapter-path.txt}"
  OUT="${PAINTER_POST_WARMUP_OUTPUT:-$RUN/matched-baseline/post-warmup}"
  POLICY_ALIAS="warmup-step4"
  POLICY_LABEL="warmup-post-sft"
else
  OUT="${PAINTER_MATCHED_BASELINE_OUTPUT:-$RUN/matched-baseline/pre-step12}"
  POLICY_ALIAS="flash12-baseline"
  POLICY_LABEL="flash12-pre-warmup"
fi
export PAINTER_POLICY_ALIAS="$POLICY_ALIAS" PAINTER_EVAL_POLICY_LABEL="$POLICY_LABEL"
RENDERER="${PAINTER_RENDERER_PATH:-$RUN/vendor/integrations/watercolour/renderer.py}"
RENDERER_PYTHON="${PAINTER_RENDERER_PYTHON_PATH:-$RUN/renderer-env/bin/python}"
BROWSER_ROOT="${PAINTER_BROWSER_ROOT:-${PLAYWRIGHT_BROWSERS_PATH:-$RUN/browsers}}"
MODEL="${MODEL_DIR:-$PAINTER_NODE_ROOT/model-cache/qwen/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
PORT="${PAINTER_BASELINE_PORT:-8000}"

# A same-run retry may reuse the one hash-pinned public panel.  The branch is
# entered only for pre-warmup; post-warmup evaluation remains a fresh policy
# comparison against the restored baseline.
SAME_RUN_RESUME="${PAINTER_RESUME_SAME_RUN_ARCHIVE:-0}"
REUSE_MATCHED_BASELINE="${PAINTER_REUSE_MATCHED_BASELINE_EVIDENCE:-0}"
[[ "$SAME_RUN_RESUME" == 0 || "$SAME_RUN_RESUME" == 1 ]] || {
  echo 'PAINTER_RESUME_SAME_RUN_ARCHIVE must be 0 or 1' >&2; exit 1;
}
[[ "$REUSE_MATCHED_BASELINE" == 0 || "$REUSE_MATCHED_BASELINE" == 1 ]] || {
  echo 'PAINTER_REUSE_MATCHED_BASELINE_EVIDENCE must be 0 or 1' >&2; exit 1;
}
[[ "$SAME_RUN_RESUME" == 1 ]] && REUSE_MATCHED_BASELINE=1
[[ "$REUSE_MATCHED_BASELINE" == 0 || "$SAME_RUN_RESUME" == 1 ]] || {
  echo 'matched-baseline reuse requires the canonical same-run archive resume' >&2; exit 1;
}

for required in "$PY" "$VLLM" "$EVAL" "$MODEL/config.json" "$POLICY_POINTER" "$FROZEN_POINTER" "$RUN/anchor_bank.py" "$RUN/renderer-selection-receipt.json" "$RENDERER" "$RENDERER_PYTHON"; do
  [[ -e "$required" ]] || { echo "missing baseline input: $required" >&2; exit 1; }
done
SELECTOR_VALUES="$($PY - "$RUN/renderer-selection-receipt.json" <<'PY'
import json, sys
from pathlib import Path
selected = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("selected")
if not isinstance(selected, dict):
    raise SystemExit("renderer selection receipt has no selected renderer")
print("\t".join((str(selected.get("path", "")), str(selected.get("renderer_sha256", "")), str(selected.get("assets_sha256", "")))))
PY
)"
IFS=$'\t' read -r SELECTED_RENDERER SELECTED_RENDERER_SHA SELECTED_ASSETS_SHA <<< "$SELECTOR_VALUES"
RENDERER_RESOLVED="$($PY - "$RENDERER" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).resolve())
PY
)"
SELECTED_RENDERER_RESOLVED="$($PY - "$SELECTED_RENDERER" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).resolve())
PY
)"
[[ "$SELECTED_RENDERER_RESOLVED" == "$RENDERER_RESOLVED" && "$SELECTED_RENDERER_SHA" =~ ^[0-9a-f]{64}$ && "$SELECTED_ASSETS_SHA" =~ ^[0-9a-f]{64}$ ]] || {
  echo 'renderer path/hash is not bound to renderer-selection-receipt.json' >&2; exit 1;
}
if [[ "$PHASE" == pre-warmup && "$REUSE_MATCHED_BASELINE" == 1 ]]; then
  CACHE_ARCHIVE_SHA="${PAINTER_SAME_RUN_ARCHIVE_SHA256:-${PAINTER_MATCHED_BASELINE_EVIDENCE_SHA256:-${PAINTER_ANCHOR_EVIDENCE_SHA256:-41f527d104e1e5fe52ef83369e3fd089ba153ce2e3112ae25d4da25f682881ef}}}"
  CACHE_URL="${PAINTER_SAME_RUN_ARCHIVE_URL:-${PAINTER_MATCHED_BASELINE_EVIDENCE_URL:-${PAINTER_ANCHOR_EVIDENCE_URL:-https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/resolve/2de12b59875fbdd5cc0f206f79eed12c8ee9f43b/runs/diverse-flow-rl-20261009/evidence.tar.gz}}}"
  for value in "${PAINTER_SAME_RUN_ARCHIVE_SHA256:-}" "${PAINTER_ANCHOR_EVIDENCE_SHA256:-}" "${PAINTER_MATCHED_BASELINE_EVIDENCE_SHA256:-}"; do
    [[ -z "$value" || "$value" == 41f527d104e1e5fe52ef83369e3fd089ba153ce2e3112ae25d4da25f682881ef ]] || {
      echo 'same-run baseline archive SHA-256 is not the pinned public archive' >&2; exit 1;
    }
  done
  for value in "${PAINTER_SAME_RUN_ARCHIVE_URL:-}" "${PAINTER_ANCHOR_EVIDENCE_URL:-}" "${PAINTER_MATCHED_BASELINE_EVIDENCE_URL:-}"; do
    [[ -z "$value" || "$value" == https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/resolve/2de12b59875fbdd5cc0f206f79eed12c8ee9f43b/runs/diverse-flow-rl-20261009/evidence.tar.gz ]] || {
      echo 'same-run baseline archive URL is not the pinned credential-free public revision' >&2; exit 1;
    }
  done
  [[ "$CACHE_ARCHIVE_SHA" == 41f527d104e1e5fe52ef83369e3fd089ba153ce2e3112ae25d4da25f682881ef ]] || {
    echo 'same-run baseline archive SHA-256 is not the pinned public archive' >&2; exit 1;
  }
  [[ "$CACHE_URL" == https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/resolve/2de12b59875fbdd5cc0f206f79eed12c8ee9f43b/runs/diverse-flow-rl-20261009/evidence.tar.gz ]] || {
    echo 'same-run baseline archive URL is not the pinned credential-free public revision' >&2; exit 1;
  }
  [[ -z "${PAINTER_MATCHED_BASELINE_EVIDENCE_ARCHIVE:-}" && -z "${PAINTER_ANCHOR_EVIDENCE_ARCHIVE:-}" ]] || {
    echo 'same-run baseline reuse accepts only the pinned public URL; local archive overrides are disabled' >&2; exit 1;
  }
  [[ -x "$PY" && -f "$RUN/restore_anchor_evidence.py" && -f "$EVAL_MANIFEST" ]] || {
    echo 'same-run baseline reuse requires Prime Python, the restore helper, and the restored 32-case manifest' >&2; exit 1;
  }
  CACHE_RESTORE_ARGS=(
    --run-root "$RUN" --sha256 "$CACHE_ARCHIVE_SHA" --mode baseline
    --renderer-sha256 "${PAINTER_RENDERER_SHA256:-$SELECTED_RENDERER_SHA}"
    --renderer "$RENDERER" --eval-manifest "$EVAL_MANIFEST"
    --flow-source "$RUN/flow_demonstrations.py"
    --policy-sha256 "$PAINTER_POLICY_SHA256"
    --policy-revision "$PAINTER_POLICY_REVISION"
    --policy-prefix "$PAINTER_POLICY_PREFIX"
    --seed "${PAINTER_EVAL_SEED:-918210}"
    --url "$CACHE_URL"
  )
  PYTHONPATH="$REPO:$REPO/painter:$RUN${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" "$RUN/restore_anchor_evidence.py" "${CACHE_RESTORE_ARGS[@]}" >"$RUN/baseline-restore.json"
  mkdir -p "$OUT/logs"
  "$PY" - "$OUT" "$RUN" "$CACHE_ARCHIVE_SHA" <<'PY'
import json
import sys
from pathlib import Path

out, run, archive_sha = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
restore = json.loads((run / "baseline-restore.json").read_text(encoding="utf-8"))
validation = restore.get("validation")
if restore.get("source_archive_sha256") != archive_sha or not isinstance(validation, dict):
    raise SystemExit("same-run baseline archive identity changed during restore")
evaluation = validation.get("eval", {})
rollouts = validation.get("rollouts", {})
if evaluation.get("seed") != 918210 or evaluation.get("max_turns") != 3 or evaluation.get("max_tokens") != 8192:
    raise SystemExit("same-run baseline is not the pinned 32-case Flash12 evaluation")
if validation.get("prompt", {}).get("case_count") != 32 or rollouts.get("receipt_count", 0) < 32:
    raise SystemExit("same-run baseline does not contain the complete 32-case panel")
value = {
    "schema": "Diverse-flow-matched-baseline-reuse-v1",
    "status": "reused_cached_baseline_no_generation",
    "source_archive_sha256": archive_sha,
    "baseline_restore_receipt": str(run / "baseline-restore.json"),
    "generator_calls": 0,
    "model_calls": 0,
    "renderer_calls": 0,
    "judge_calls": 0,
    "train_rows": 96,
    "heldout_cases": 32,
    "invalid_or_censored_retained": bool(validation.get("invalid_or_censored_retained")),
    "limitations": [
        "This retry reused the hash-verified same-run Flash12 panel; no baseline model, renderer, or judge call was made.",
        "Invalid and censored baseline receipts remain in the restored evidence tree.",
    ],
}
out.mkdir(parents=True, exist_ok=True)
(out / "baseline-reuse-receipt.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(value, sort_keys=True))
PY
  echo "MATCHED_${PHASE^^}_BASELINE_REUSED: $OUT"
  exit 0
fi
[[ -d "$BROWSER_ROOT" ]] || { echo "native evaluation browser root is missing: $BROWSER_ROOT" >&2; exit 1; }
[[ -d "$CUDA_HOME" ]] || { echo "CUDA 13 runtime directory is missing: $CUDA_HOME" >&2; exit 1; }
export CUDA_HOME
export PATH="${PRIME_ROOT:-$PAINTER_NODE_ROOT/prime}/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
POLICY="$(tr -d '\r\n' < "$POLICY_POINTER")"
FROZEN="$(tr -d '\r\n' < "$FROZEN_POINTER")"
if [[ "$PHASE" == "post-warmup" ]]; then
  [[ -r "$RUN/active-policy.json" ]] || { echo 'post-warmup evaluation requires the verified active-policy receipt' >&2; exit 1; }
  POLICY_SHA="$($PY - "$RUN/active-policy.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
sha = value.get("adapter_sha256")
if not isinstance(sha, str) or len(sha) != 64:
    raise SystemExit("active-policy receipt has no adapter SHA-256")
print(sha)
PY
)"
else
  POLICY_SHA="$PAINTER_POLICY_SHA256"
fi
"$PY" - "$POLICY" "$FROZEN" "$POLICY_SHA" "$PAINTER_FROZEN_SFT_SHA256" <<'PY'
import hashlib, sys
from pathlib import Path
policy, frozen, policy_sha, frozen_sha = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
for path, expected, label in ((policy, policy_sha, "policy"), (frozen, frozen_sha, "SFT512")):
    weight = path / "adapter_model.safetensors"
    if path.is_symlink() or not weight.is_file():
        raise SystemExit(f"{label} adapter is incomplete")
    actual = hashlib.sha256(weight.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"{label} adapter hash mismatch: {actual}")
if policy.resolve() == frozen.resolve():
    raise SystemExit("policy and frozen opponent adapters must be distinct")
PY

if [[ "${1:-}" == --preflight ]]; then
  echo "MATCHED_${PHASE^^}_EVALUATION_PREFLIGHT_READY"
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
if value.get("schema") != "Diverse-flow-prompt-hook-v1" or not value.get("installed"):
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
    expected_count=32,
    max_turns=3,
    policy_label=os.environ.get("PAINTER_EVAL_POLICY_LABEL", "flash12-pre-warmup"),
)
tasks = list(NativePaintingTaskset(config).load())
if len(tasks) != 32:
    raise SystemExit(f"evaluation taskset loaded {len(tasks)} tasks, expected 32")
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

BASELINE_GPUS="${PAINTER_BASELINE_GPUS:-2,3}"
[[ "$BASELINE_GPUS" =~ ^[0-9]+,[0-9]+$ && "${BASELINE_GPUS%,*}" != "${BASELINE_GPUS#*,}" ]] || {
  echo 'PAINTER_BASELINE_GPUS must name two distinct inference GPU ordinals' >&2; exit 1;
}
setsid env CUDA_VISIBLE_DEVICES="$BASELINE_GPUS" TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=6 \
  VLLM_CACHE_ROOT="$OUT/cache/vllm" TORCHINDUCTOR_CACHE_DIR="$OUT/cache/torchinductor" TRITON_CACHE_DIR="$OUT/cache/triton" \
  "$VLLM" serve "$MODEL" --served-model-name "$POLICY_ALIAS" --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 --max-model-len 32768 --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image":14,"video":0}' \
  --tensor-parallel-size 2 \
  --gdn-prefill-backend triton --kernel-config '{"enable_jit_warmup":false}' \
  --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":1048576}' \
  --enable-lora --max-lora-rank 16 --max-loras 2 --max-cpu-loras 2 \
  --lora-modules "$POLICY_ALIAS=$POLICY" "sft-step512=$FROZEN" >"$OUT/logs/vllm.log" 2>&1 &
SERVER_PID=$!
deadline=$((SECONDS + 900))
while (( SECONDS < deadline )); do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo 'baseline vLLM exited' >&2; exit 1; }
if "$PY" - "$PORT" <<'PY' >/dev/null 2>&1
import json, sys, urllib.request
import os
with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/v1/models", timeout=5) as response:
    names = {item.get("id") for item in json.load(response).get("data", [])}
if {os.environ["PAINTER_POLICY_ALIAS"], "sft-step512"} - names:
    raise SystemExit(1)
PY
  then break
  fi
  sleep 5
done
(( SECONDS < deadline )) || { echo 'baseline vLLM startup timeout' >&2; exit 1; }

if [[ "$PHASE" == "pre-warmup" ]]; then
  PYTHONPATH="$REPO:$REPO/painter:$RUN${PYTHONPATH:+:$PYTHONPATH}" PAINTER_PRIME_ROLE=baseline PAINTER_STRUCTURED_FLOW_POLICY=0 PAINTER_BROWSER_ROOT="$BROWSER_ROOT" PLAYWRIGHT_BROWSERS_PATH="$BROWSER_ROOT" PAINTER_FROZEN_OPPONENT_ENDPOINT="http://127.0.0.1:$PORT/v1" \
    "$PY" "$RUN/anchor_bank.py" materialize \
    --train-manifest "$TRAIN_MANIFEST" --heldout-manifest "$EVAL_MANIFEST" \
    --anchor-root "${PAINTER_ANCHOR_ROOT:-$RUN/curriculum/anchors}" \
    --output-dir "${PAINTER_ANCHOR_STAGING:-$RUN/curriculum/.materialized-pilot}" \
    --run-root "$RUN" >"$OUT/logs/anchor-materialization.json"
fi

PYTHONPATH="$REPO:$REPO/painter:$RUN${PYTHONPATH:+:$PYTHONPATH}" \
  PAINTER_PRIME_ROLE=evaluation PAINTER_STRUCTURED_FLOW_POLICY=1 PAINTER_BROWSER_ROOT="$BROWSER_ROOT" PLAYWRIGHT_BROWSERS_PATH="$BROWSER_ROOT" PAINTER_RL_MANIFEST="$TRAIN_MANIFEST" PAINTER_EVAL_MANIFEST="$EVAL_MANIFEST" \
  PAINTER_FLOW_HOOK_RECEIPT="$OUT/Diverse-flow-hook-receipt.json" \
  "$PY" "$REPO/painter/native_painting_eval.py" --root "$REPO/painter" --manifest "$EVAL_MANIFEST" \
  --policy-label "$POLICY_LABEL" --model "$POLICY_ALIAS" --output-dir "$OUT/evaluation-rollouts" \
  --config "$OUT/eval.toml" --expected-count 32 --max-turns 3 --max-tokens 8192 --max-concurrent 8 \
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
  PAINTER_PRIME_ROLE=evaluation PAINTER_FLOW_HOOK_RECEIPT="$OUT/Diverse-flow-hook-receipt.json" \
  bash -c 'cd "$1"; exec "$2" @ "$3" --no-serve --no-push' _ "$PRIME_ROOT" "$EVAL" "$OUT/eval.toml" >"$OUT/logs/eval.log" 2>&1

"$PY" - "$RUN" "$OUT" "$POLICY" "$POLICY_SHA" "$EVAL_MANIFEST" "${PAINTER_EVAL_SEED:-918210}" "$PHASE" "$RENDERER" "$SELECTED_RENDERER_SHA" "$SELECTED_ASSETS_SHA" "$POLICY_LABEL" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
run, out, policy, policy_sha, manifest, seed, phase, renderer, renderer_sha, assets_sha, policy_label = (
    Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], Path(sys.argv[5]), int(sys.argv[6]),
    sys.argv[7], Path(sys.argv[8]), sys.argv[9], sys.argv[10], sys.argv[11]
)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
if sha(policy / "adapter_model.safetensors") != policy_sha:
    raise SystemExit("evaluation policy bytes differ from the bound adapter identity")
selector_path = run / "renderer-selection-receipt.json"
selector = json.loads(selector_path.read_text(encoding="utf-8"))
selected = selector.get("selected")
if not isinstance(selected, dict):
    raise SystemExit("renderer selection receipt has no selected renderer")
if Path(str(selected.get("path"))).resolve() != renderer.resolve():
    raise SystemExit("evaluation renderer path differs from the selected renderer receipt")
if selected.get("renderer_sha256") != renderer_sha or selected.get("assets_sha256") != assets_sha:
    raise SystemExit("evaluation renderer hashes differ from the selected renderer receipt")
asset_map = selected.get("assets")
if not isinstance(asset_map, dict):
    raise SystemExit("selected renderer receipt has no asset map")
eval_toml = out / "eval.toml"
text = eval_toml.read_text(encoding="utf-8")
for key, expected in (("num_tasks", 32), ("max_turns", 3), ("max_tokens", 8192), ("max_concurrent", 8), ("seed", seed)):
    match = re.search(rf"(?m)^\s*{key}\s*=\s*(\d+)\s*$", text)
    if not match or int(match.group(1)) != expected:
        raise SystemExit(f"evaluation task config is not pinned for {key}: {expected}")
if 'policy_label = "' + policy_label + '"' not in text:
    raise SystemExit("evaluation task config has the wrong policy label")
rollouts = out / "evaluation-rollouts"
painting = sorted(rollouts.rglob("painting.json"))
if len(painting) < 32:
    raise SystemExit(f"evaluation produced only {len(painting)} painting receipts; expected at least 32")
valid = 0
for path in painting:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("valid") is not True:
        continue
    valid += 1
    if value.get("renderer_sha256") != renderer_sha:
        raise SystemExit(f"painting receipt renderer mismatch: {path}")
    rendered_assets = value.get("assets")
    if not isinstance(rendered_assets, dict) or any(rendered_assets.get(key) != asset_map.get(key) for key in ("p5.brush.js", "p5.min.js")):
        raise SystemExit(f"painting receipt asset mismatch: {path}")
if valid < 1:
    raise SystemExit("evaluation produced no valid PNG-backed painting receipts")
receipt = {
    "schema": "Diverse-flow-matched-flash12-baseline-v1" if phase == "pre-warmup" else "Diverse-flow-post-warmup-evaluation-v1",
    "status": "complete", "phase": phase,
    "policy": {"label": policy_label, "revision": "1227a2427a8ae16014d2fdca4d1adb6855d3fa1c", "prefix": "rl-judge-matched-20261004-flash/step_12", "adapter_sha256": policy_sha},
    "manifest_sha256": sha(manifest), "manifest_cases": 32, "max_turns": 3, "max_tokens": 8192, "max_concurrent": 8, "seed": seed,
    "renderer": {"selector_receipt": str(selector_path), "path": str(renderer.resolve()), "renderer_sha256": renderer_sha, "assets_sha256": assets_sha},
    "task_config": {"path": str(eval_toml), "sha256": sha(eval_toml)},
    "evaluation_artifacts": {"root": str(rollouts), "painting_receipts": len(painting), "valid_painting_receipts": valid},
    "judge_deferred": True, "quality_scoring": "offline_only",
    "limitations": ["Invalid, censored, and missing-final-action receipts remain in evaluation-rollouts for audit."],
}
name = "baseline-receipt.json" if phase == "pre-warmup" else "post-warmup-receipt.json"
(out / name).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
echo "MATCHED_${PHASE^^}_EVALUATION_COMPLETE: $OUT"
