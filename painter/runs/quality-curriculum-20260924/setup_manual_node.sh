#!/usr/bin/env bash
set -euo pipefail

# Run after the approved minimal archive has been extracted under /srv/painter-manual.
ROOT=/srv/painter-manual
if [[ ! -f "$ROOT/painter/vendor/integrations/watercolour/renderer.py" ]]; then
  echo 'Pinned renderer source is missing' >&2
  exit 2
fi
if ! id painter >/dev/null 2>&1; then
  useradd --create-home --shell /bin/bash painter
fi
python3 -m venv "$ROOT/venv"
"$ROOT/venv/bin/python" -m pip install --disable-pip-version-check 'playwright==1.58.0' 'Pillow==12.3.0'
mkdir -p "$ROOT/browsers" "$ROOT/renders"
PLAYWRIGHT_BROWSERS_PATH="$ROOT/browsers" "$ROOT/venv/bin/python" -m playwright install --with-deps chromium
chown -R painter:painter "$ROOT"
echo "Renderer ready. Use: PLAYWRIGHT_BROWSERS_PATH=$ROOT/browsers runuser -u painter -- $ROOT/venv/bin/python $ROOT/painter/vendor/integrations/watercolour/renderer.py SOURCE.js --output OUTPUT.png --timeout 600"
