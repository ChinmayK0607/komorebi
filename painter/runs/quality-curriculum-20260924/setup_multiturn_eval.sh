#!/usr/bin/env bash
# Install the CPU renderer and verify the frozen 28-photo eval manifest.
set -Eeuo pipefail
RUN="$(cd "${1:?usage: setup_multiturn_eval.sh RUN_DIRECTORY}" && pwd -P)"
[[ "$(uname -s)" == Linux && -x "$RUN/bootstrap/bin/uv" && -f "$RUN/env.sh" ]] || {
  echo 'complete the Linux training-node bootstrap first' >&2; exit 2;
}
source "$RUN/env.sh"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$RUN/cache/uv}"
export PLAYWRIGHT_BROWSERS_PATH="$RUN/painter/browsers"
mkdir -p "$RUN/logs" "$PLAYWRIGHT_BROWSERS_PATH"
"$RUN/bootstrap/bin/uv" venv --allow-existing --python python3 "$RUN/painter/renderer-env"
"$RUN/bootstrap/bin/uv" pip install --python "$RUN/painter/renderer-env/bin/python" \
  'playwright==1.58.0' 'pillow==12.3.0' > "$RUN/logs/eval-renderer-install.log" 2>&1
"$RUN/painter/renderer-env/bin/python" -m playwright install --with-deps chromium \
  >> "$RUN/logs/eval-renderer-install.log" 2>&1
id painter >/dev/null 2>&1 || useradd --create-home --shell /bin/bash painter
cursor="$RUN"
while [[ "$cursor" != / ]]; do chmod o+x "$cursor"; cursor="$(dirname "$cursor")"; done
chmod -R a+rX "$RUN/painter/renderer-env" "$RUN/painter/browsers" \
  "$RUN/painter/vendor" "$RUN/painter/eval-prep"
"$PRIME_ROOT/.venv/bin/python" "$RUN/painter/native_painting_eval.py" \
  --root "$RUN/painter" --manifest "$RUN/painter/eval-prep/eval-manifest.json" \
  --policy-label eval-preflight --model preflight --expected-count 28 \
  --output-dir "$RUN/eval-preflight" --config "$RUN/eval-preflight.toml" \
  --max-tokens 8192 --max-turns 1 --rollout-timeout 1800 --context-length 16384 \
  > "$RUN/eval-preflight.json"
echo 'Renderer installed and frozen 28-photo eval references hash-verified.'
