#!/usr/bin/env bash
set -Eeuo pipefail
STEP="${1:-}"
SCOPE="${2:-all}"
MODE="${3:-}"
[[ $# -ge 1 && $# -le 3 ]] || { echo 'usage: evaluate.sh STEP [all] [--preflight]' >&2; exit 2; }
[[ "$STEP" =~ ^[1-9][0-9]*$ && "$SCOPE" == all ]] || {
  echo 'STEP must be positive and SCOPE must be all' >&2; exit 2;
}
[[ -z "$MODE" || "$MODE" == --preflight ]] || {
  echo 'usage: evaluate.sh STEP [all] [--preflight]' >&2; exit 2;
}
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BASE="$(cd "$RUN/../text-sft512-consistent-20260930" && pwd -P)"
# The l40s_4 default remains PAINTER_EVAL_GPUS:-2,3; the profile resolver
# below substitutes the h100_2 single inference ordinal when selected.
# Evaluation must select the registered final checkpoint even when invoked
# from a shell that inherited the previous eight-update setting.
export PAINTER_RL_MAX_STEPS=20 PAINTER_MAX_UPDATES=20 PAINTER_CHECKPOINT_INTERVAL=4 PAINTER_EVAL_PREFERRED_STEP=20
source "$RUN/runtime_env.sh"
# Evaluation is intentionally outside the RL whole-group admission.  Set the
# role before the transient evaluator patcher starts so an inherited
# orchestrator value cannot activate the fail-closed RL guard.
export PAINTER_PRIME_ROLE=evaluation
validate_structured_bindings
export PAINTER_RUN_ROOT="$RUN" PAINTER_RUN_NAME=diverse-flow-rl-20261009
export PAINTER_EVAL_MANIFEST="$RUN/curriculum/matched32-manifest.json"
export PAINTER_EVAL_EXPECTED_COUNT=32 PAINTER_EVAL_SCOPE="$SCOPE"
export PAINTER_JUDGE_MODEL=xiaomi/mimo-v2.6-flash PAINTER_JUDGE_EFFORT=medium
export PAINTER_JUDGE_FORMAT=identity-joint-side-schema-v2-generic-v1
PY_AUDIT="${PRIME_ROOT:-$PAINTER_NODE_ROOT/prime}/.venv/bin/python"
[[ -x "$PY_AUDIT" ]] || { echo "pinned Prime Python is required for evaluation receipts: $PY_AUDIT" >&2; exit 1; }
SELECTOR_VALUES="$($PY_AUDIT - "$RUN/renderer-selection-receipt.json" <<'PY'
import json, sys
from pathlib import Path
selected = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("selected")
if not isinstance(selected, dict):
    raise SystemExit("renderer selection receipt has no selected renderer")
print("\t".join((str(selected.get("path", "")), str(selected.get("renderer_sha256", "")), str(selected.get("assets_sha256", "")))))
PY
)"
IFS=$'\t' read -r SELECTED_RENDERER_PATH SELECTED_RENDERER_SHA SELECTED_ASSETS_SHA <<< "$SELECTOR_VALUES"
[[ -f "$SELECTED_RENDERER_PATH" && "$SELECTED_RENDERER_SHA" =~ ^[0-9a-f]{64}$ && "$SELECTED_ASSETS_SHA" =~ ^[0-9a-f]{64}$ ]] || {
  echo 'renderer path/hashes are not bound to renderer-selection-receipt.json' >&2; exit 1;
}
export PAINTER_RENDERER_PATH="$SELECTED_RENDERER_PATH"
# The generic evaluator assigns PYTHONPATH itself, which would hide the
# run-owned sitecustomize/flow hook.  Generate a transient local copy with the
# same proven commands and add the repository plus run root to every Python
# invocation.  The copy is removed on exit and contains no credentials.
PATCHED_EVALUATOR="$RUN/.evaluate-runtime.$$"
trap 'rm -f "$PATCHED_EVALUATOR"' EXIT
python3 - "$BASE/evaluate.sh" "$PATCHED_EVALUATOR" <<'PY'
from pathlib import Path
import os
import sys

source, target = map(Path, sys.argv[1:])
text = source.read_text(encoding="utf-8")
needle = 'PYTHONPATH="$REPO/painter"'
replacement = (
    'PYTHONPATH="$REPO:$REPO/painter:$RUN" '
    'PAINTER_PRIME_ROLE=evaluation '
    'PAINTER_STRUCTURED_FLOW_POLICY=1'
)
if text.count(needle) < 2:
    raise SystemExit("generic evaluator contract changed: expected two Python path bindings")
text = text.replace(needle, replacement)
text = text.replace(
    'PLAYWRIGHT_BROWSERS_PATH="$REPO/painter/browsers"',
    'PAINTER_BROWSER_ROOT="$RUN/browsers" PLAYWRIGHT_BROWSERS_PATH="$RUN/browsers"',
)
text = text.replace(
    'CUDA_VISIBLE_DEVICES="${PAINTER_EVAL_GPU:-0}"',
    'CUDA_VISIBLE_DEVICES="${PAINTER_EVAL_GPUS:-' + os.environ.get("PAINTER_INFER_GPU_IDS", "2,3") + '}"',
)
text = text.replace(
    ' --gpu-memory-utilization 0.90 \\\n',
    ' --gpu-memory-utilization 0.90 --tensor-parallel-size ' + os.environ.get("PAINTER_INFERENCE_TENSOR_PARALLEL", "2") + ' \\\n',
)
target.write_text(text, encoding="utf-8")
target.chmod(0o700)
PY
export PAINTER_PRIME_ROLE=evaluation PAINTER_STRUCTURED_FLOW_POLICY=1
EVALUATOR_ARGS=("$STEP" "$SCOPE")
[[ -z "$MODE" ]] || EVALUATOR_ARGS+=("$MODE")
bash "$PATCHED_EVALUATOR" "${EVALUATOR_ARGS[@]}"
STATUS=$?
if (( STATUS == 0 )); then
  # The base evaluator owns rollout generation.  Bind its completed output to
  # the renderer selected by prepare_node, and audit native painting receipts
  # before the run can report a final evaluation result.  This deliberately
  # reads the selector receipt instead of trusting a parent-shell default.
  OUT="$RUN/validation/step-$STEP-$SCOPE"
  "$PY_AUDIT" - "$RUN" "$OUT" "$PAINTER_EVAL_MANIFEST" "$SELECTED_RENDERER_SHA" "$SELECTED_ASSETS_SHA" <<'PY'
import hashlib, json, sys
from pathlib import Path
run, out, manifest, renderer_sha, assets_sha = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
selector_path = run / "renderer-selection-receipt.json"
selector = json.loads(selector_path.read_text(encoding="utf-8"))
selected = selector.get("selected")
if not isinstance(selected, dict) or selected.get("renderer_sha256") != renderer_sha or selected.get("assets_sha256") != assets_sha:
    raise SystemExit("final evaluation renderer hashes are not bound to renderer-selection-receipt.json")
asset_map = selected.get("assets")
if not isinstance(asset_map, dict):
    raise SystemExit("selected renderer receipt has no asset map")
eval_toml = out / "eval" / "eval.toml"
if not eval_toml.is_file():
    eval_toml = out / "eval.toml"
text = eval_toml.read_text(encoding="utf-8")
for key, expected in (("num_tasks", 32), ("max_turns", 3), ("max_tokens", 8192), ("max_concurrent", 8), ("seed", 918210)):
    import re
    match = re.search(rf"(?m)^\s*{key}\s*=\s*(\d+)\s*$", text)
    if not match or int(match.group(1)) != expected:
        raise SystemExit(f"final evaluation task config is not pinned for {key}: {expected}")
rollouts = out / "eval" / "evaluation-rollouts"
painting = sorted(rollouts.rglob("painting.json"))
if len(painting) < 32:
    raise SystemExit(f"final evaluation produced only {len(painting)} painting receipts")
valid = 0
for receipt_path in painting:
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    if value.get("valid") is not True:
        continue
    valid += 1
    if value.get("renderer_sha256") != renderer_sha:
        raise SystemExit(f"final evaluation painting receipt renderer mismatch: {receipt_path}")
    rendered_assets = value.get("assets")
    if not isinstance(rendered_assets, dict) or any(rendered_assets.get(key) != asset_map.get(key) for key in ("p5.brush.js", "p5.min.js")):
        raise SystemExit(f"final evaluation painting receipt asset mismatch: {receipt_path}")
if valid < 1:
    raise SystemExit("final evaluation produced no valid painting receipts")
(out / "Diverse-flow-final-evaluation-receipt.json").write_text(json.dumps({
    "schema": "Diverse-flow-final-evaluation-v1", "status": "complete",
    "manifest_sha256": sha(manifest), "manifest_cases": 32,
    "renderer": {"selector_receipt": str(selector_path), "path": str(Path(str(selected["path"])).resolve()), "renderer_sha256": renderer_sha, "assets_sha256": assets_sha},
    "task_config": {"path": str(eval_toml), "sha256": sha(eval_toml)},
    "evaluation_artifacts": {"root": str(rollouts), "painting_receipts": len(painting), "valid_painting_receipts": valid},
    "max_turns": 3, "max_tokens": 8192, "max_concurrent": 8, "seed": 918210,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  STATUS=$?
fi
exit "$STATUS"
