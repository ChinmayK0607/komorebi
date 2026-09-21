#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "${1:?usage: activate_async.sh RUN_ROOT}" && pwd -P)"
cd "$ROOT"

exec 9>"$ROOT/aesthetic-activation.lock"
flock -n 9 || exit 0

rm -f "$ROOT/aesthetic-activation.exit"
trap 'rc=$?; printf "%s\n" "$rc" > "$ROOT/aesthetic-activation.exit"' EXIT

READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-2700}"
started_at="$(date +%s)"
while [[ ! -f "$ROOT/training-ready.json" ]]; do
  if [[ -f "$ROOT/training-setup.exit" ]]; then
    echo "training setup failed" >&2
    exit 1
  fi
  now="$(date +%s)"
  if (( now - started_at >= READY_TIMEOUT_SECONDS )); then
    echo "timed out waiting for training-ready.json" >&2
    exit 1
  fi
  sleep 10
done

if [[ -f "$ROOT/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/env.sh"
fi
: "${PRIME_ROOT:?env.sh must define PRIME_ROOT}"
export PATH="$ROOT/bootstrap/bin:$PRIME_ROOT/.venv/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$PRIME_ROOT"
PACKAGE="$ROOT/environments/painter_revision"
if [[ ! -f "$PACKAGE/pyproject.toml" ]]; then
  mkdir -p "$ROOT/environments"
  uv run --no-sync init painter-revision --path "$ROOT/environments"
fi

mkdir -p "$PACKAGE/painter_revision"
cp "$ROOT/taskset_impl.py" "$PACKAGE/painter_revision/taskset.py"
cat > "$PACKAGE/painter_revision/__init__.py" <<'PY'
"""Run-local Prime environment exports for the aesthetic taskset."""

from .taskset import (
    PainterRevisionConfig,
    PainterRevisionEnv,
    PainterRevisionEnvConfig,
    PainterRevisionTask,
    PainterRevisionTaskset,
)

__all__ = [
    "PainterRevisionConfig",
    "PainterRevisionEnv",
    "PainterRevisionEnvConfig",
    "PainterRevisionTask",
    "PainterRevisionTaskset",
]
PY

for required in taskset_impl.py aesthetic_judge.py judge_queue.py render_actions.py rl-states.json aesthetic-states.json aesthetic-data-receipt.json; do
  [[ -f "$ROOT/$required" ]] || { echo "missing $ROOT/$required" >&2; exit 1; }
done

# Install only the local package into the already bootstrapped environment.  The
# package has no dependency install here; setup/download work belongs to bootstrap.
uv pip install --python .venv/bin/python --no-deps -e "$PACKAGE"

uv run --no-sync python - "$ROOT" "$PACKAGE" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
package = pathlib.Path(sys.argv[2])
runtime_path = root / "rl-states.json"
rows = json.loads(runtime_path.read_text())
expected = sum(max(1, len(row.get("anchors", []))) for row in rows)
if len(rows) != 24:
    raise SystemExit(f"expected 24 train states, found {len(rows)}")
if len({row["source_group"] for row in rows}) != 24:
    raise SystemExit("expected 24 source groups")
if len({row["family"] for row in rows}) != 12:
    raise SystemExit("expected 12 task families")
if any(row.get("split") != "train" for row in rows):
    raise SystemExit("runtime data contains a non-train state")

sys.path.insert(0, str(package))
from painter_revision import PainterRevisionConfig, PainterRevisionEnv, PainterRevisionTaskset

if PainterRevisionEnv is None:
    raise SystemExit("painter_revision does not export PainterRevisionEnv")
taskset = PainterRevisionTaskset(PainterRevisionConfig(root=str(root))).load()
tasks = list(taskset)
if len(tasks) != expected:
    raise SystemExit(f"task count mismatch: manifest={expected}, taskset={len(tasks)}")
indices = [getattr(task.data, "idx", None) for task in tasks]
if any(index is None for index in indices) or len(set(indices)) != len(indices):
    raise SystemExit("task indices are not present and unique")

def image_count(task):
    messages = getattr(task.data, "prompt", ())
    if not messages:
        return 0
    content = getattr(messages[0], "content", ())
    return sum(getattr(part, "type", None) == "image_url" for part in content)

counts = [image_count(task) for task in tasks]
if counts != [2] * len(tasks):
    raise SystemExit(f"expected two initial images per task, counts={sorted(set(counts))}")

validation = {
    "schema": "painter.aesthetic-taskset-validation.v1",
    "states": len(rows),
    "source_groups": len({row["source_group"] for row in rows}),
    "families": len({row["family"] for row in rows}),
    "tasks": len(tasks),
    "expected_tasks_from_manifest": expected,
    "unique_indices": len(set(indices)),
    "initial_images_per_task": sorted(set(counts)),
    "package": str(package),
}
(root / "taskset-validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
print(json.dumps(validation, sort_keys=True))
PY

if [[ -f "$ROOT/prepare_async.py" ]]; then
  uv run --no-sync python "$ROOT/prepare_async.py"
else
  echo "missing $ROOT/prepare_async.py (owned by environment overlay)" >&2
  exit 1
fi
