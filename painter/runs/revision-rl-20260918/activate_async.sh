#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(realpath "${1:?workspace}");cd "$ROOT"
exec 6>async-activation.lock;flock -n 6
rm -f async-activation.exit
trap 'echo "$?" > async-activation.exit' EXIT
start=$SECONDS
while [[ ! -f training-ready.json ]];do
 if [[ -f training-setup.exit ]];then echo 'Setup failed; inspect setup logs';exit 1;fi
 if (( SECONDS-start > 2700 ));then echo 'Setup exceeded 45-minute activation wait; no training launched';exit 1;fi
 sleep 10
done
source env.sh
export PATH="$ROOT/bootstrap/bin:$PATH" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PRIME_ROOT"
if [[ ! -d "$ROOT/environments/painter_revision" ]];then
 uv run --no-sync init painter-revision --path "$ROOT/environments"
fi
# Native scaffold chooses underscore directory names.
PACKAGE="$ROOT/environments/painter_revision"
[[ -f "$PACKAGE/pyproject.toml" ]]
cp "$ROOT/taskset_impl.py" "$PACKAGE/painter_revision/taskset.py"
for f in quality_signal.py render_actions.py contract.py; do
 [[ -f "$ROOT/$f" ]] || { echo "missing scoring dependency: $ROOT/$f" >&2; exit 2; }
done
uv pip install --python .venv/bin/python --no-deps -e "$PACKAGE"
uv run --no-sync python "$ROOT/prepare_async.py"
uv run --no-sync python - "$ROOT" <<'PY'
import sys,json
from pathlib import Path
from painter_revision import PainterRevisionTaskset
from painter_revision.taskset import PainterRevisionConfig
p=Path(sys.argv[1]);tasks=PainterRevisionTaskset(PainterRevisionConfig(root=str(p))).load()
assert len(tasks)==24
for t in tasks:
 assert len([c for c in t.data.prompt[0].content if c.type=='image_url'])==2
assert not set(json.loads((p/'EXPERIMENT.json').read_text())['training_state_groups']) & set(json.loads((p/'EXPERIMENT.json').read_text())['heldout_groups'])
(p/'taskset-validation.json').write_text(json.dumps(dict(tasks=24,images_per_task=2,heldout_disjoint=True))+'\n')
print('Native taskset loaded: 24 states with reference and canvas')
PY
python3 "$ROOT/supervise_async.py" "$ROOT"
