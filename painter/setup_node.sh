#!/usr/bin/env bash
# Usage: bash setup_node.sh /absolute/node/workspace [render|learning]
# Idempotent setup only. It does not start training or rent a machine.
set -Eeuo pipefail
ROOT=$(realpath "${1:?node workspace required}")
PROFILE=${2:-render}
[[ $(uname -s) == Linux ]]
[[ "$PROFILE" == render || "$PROFILE" == learning ]]
cd "$ROOT"
mkdir -p logs cache
exec 9>setup.lock; flock -n 9
trap 'echo "$?" > setup.exit' EXIT
# A failed setup must not leave a previous success marker looking current.
rm -f ready.json
# Verify the transferred interface before installing or launching anything.
python3 foundation_lock.py > foundation-check.json
python3 -m unittest discover -s . -p 'test_*.py' > foundation-tests.log 2>&1
export UV_CACHE_DIR="$ROOT/cache/uv" PLAYWRIGHT_BROWSERS_PATH="$ROOT/browsers"
if [[ ! -x bootstrap/bin/uv ]]; then
 python3 -m venv bootstrap
 bootstrap/bin/pip -q install 'uv==0.11.1'
fi
bootstrap/bin/uv venv --allow-existing --python python3 renderer-env
bootstrap/bin/uv pip install --python renderer-env/bin/python 'playwright==1.58.0' 'pillow==12.3.0' > logs/renderer-install.log 2>&1
renderer-env/bin/python -m playwright install --with-deps chromium >> logs/renderer-install.log 2>&1
id painter >/dev/null 2>&1 || useradd --create-home --shell /bin/bash painter
# Permit the unprivileged browser account to traverse this workspace's parents.
cursor="$ROOT"
while [[ "$cursor" != / ]]; do chmod o+x "$cursor"; cursor=$(dirname "$cursor"); done
chmod -R a+rX renderer-env browsers vendor
if [[ -d data ]]; then chown -R painter:painter data; chmod -R a+rX data; fi
python3 hardware.py > hardware-plan.json
if [[ "$PROFILE" == learning ]]; then
 bootstrap/bin/uv venv --allow-existing --python python3 runtime
 bootstrap/bin/uv pip install --python runtime/bin/python 'vllm==0.28.0' 'transformers==5.17.0' 'huggingface_hub==1.30.0' > logs/runtime-install.log 2>&1
 runtime/bin/python -c 'import json,torch; print(json.dumps({"torch":torch.__version__,"cuda":torch.version.cuda,"devices":torch.cuda.device_count()}))' > runtime.json
fi
python3 - "$ROOT" "$PROFILE" <<'PY'
import json,shlex,sys
from pathlib import Path
p=Path(sys.argv[1]);env={'PLAYWRIGHT_BROWSERS_PATH':str(p/'browsers'),'HF_HOME':str(p/'cache/huggingface'),'TOKENIZERS_PARALLELISM':'false'}
(p/'env.sh').write_text('\n'.join('export '+k+'='+shlex.quote(v) for k,v in env.items())+'\n')
(p/'ready.json').write_text(json.dumps(dict(profile=sys.argv[2],setup_complete=True,training_started=False,foundation=json.loads((p/'foundation-check.json').read_text())))+'\n')
PY
echo "Ready: $PROFILE environment. Launch remains a separate command."
