#!/usr/bin/env bash
set -Eeuo pipefail
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BASE="$(cd "$RUN/../text-sft512-consistent-20260930" && pwd -P)"
# Evaluation must select the registered final checkpoint even when invoked
# from a shell that inherited the previous eight-update setting.
export PAINTER_RL_MAX_STEPS=20 PAINTER_MAX_UPDATES=20 PAINTER_CHECKPOINT_INTERVAL=4 PAINTER_EVAL_PREFERRED_STEP=20
source "$RUN/runtime_env.sh"
validate_structured_bindings
export PAINTER_RUN_ROOT="$RUN" PAINTER_RUN_NAME=structured-flow-rl-20261009
export PAINTER_EVAL_MANIFEST="$RUN/curriculum/matched16-manifest.json"
export PAINTER_EVAL_EXPECTED_COUNT=16 PAINTER_EVAL_SCOPE=all
export PAINTER_JUDGE_MODEL=xiaomi/mimo-v2.6-flash PAINTER_JUDGE_EFFORT=medium
export PAINTER_JUDGE_FORMAT=identity-joint-side-schema-v2-generic-v1
# The generic evaluator assigns PYTHONPATH itself, which would hide the
# run-owned sitecustomize/flow hook.  Generate a transient local copy with the
# same proven commands and add the repository plus run root to every Python
# invocation.  The copy is removed on exit and contains no credentials.
PATCHED_EVALUATOR="$RUN/.evaluate-runtime.$$"
trap 'rm -f "$PATCHED_EVALUATOR"' EXIT
python3 - "$BASE/evaluate.sh" "$PATCHED_EVALUATOR" <<'PY'
from pathlib import Path
import sys

source, target = map(Path, sys.argv[1:])
text = source.read_text(encoding="utf-8")
needle = 'PYTHONPATH="$REPO/painter"'
replacement = (
    'PYTHONPATH="$REPO:$REPO/painter:$RUN" '
    'PAINTER_PRIME_ROLE=orchestrator '
    'PAINTER_STRUCTURED_FLOW_POLICY=1'
)
if text.count(needle) < 2:
    raise SystemExit("generic evaluator contract changed: expected two Python path bindings")
text = text.replace(needle, replacement)
text = text.replace(
    'PLAYWRIGHT_BROWSERS_PATH="$REPO/painter/browsers"',
    'PAINTER_BROWSER_ROOT="$RUN/browsers" PLAYWRIGHT_BROWSERS_PATH="$RUN/browsers"',
)
target.write_text(text, encoding="utf-8")
target.chmod(0o700)
PY
export PAINTER_PRIME_ROLE=orchestrator PAINTER_STRUCTURED_FLOW_POLICY=1
exec bash "$PATCHED_EVALUATOR" "$1" all "${2:-}"
