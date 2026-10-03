#!/usr/bin/env bash
set -euo pipefail

# Minimal Linux CPU setup for the finite renderer profile.  The script prefers
# the prepared cloud renderer environment and its browser cache.  If those are
# absent, it creates an ignored temporary uv/pip environment and installs only
# Playwright and Pillow.  No inherited environment is passed to renderers.

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "run_cpu_profile.sh is Linux-only" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PAINTER_ROOT="$REPO_ROOT/painter"

INPUT_ROOT=""
OUTPUT_ROOT=""
RUNTIME_ROOT=""
DERIVATIVE_RUNTIME_ROOT=""
BASELINE_RENDERER=""
DERIVATIVE_RENDERER=""
PYTHON_BIN="${PAINTER_RENDERER_PYTHON:-}"
BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-}"
TIMEOUT="180"
TOTAL_LIMIT="900"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-root) INPUT_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --runtime-root) RUNTIME_ROOT="$2"; shift 2 ;;
    --derivative-runtime-root) DERIVATIVE_RUNTIME_ROOT="$2"; shift 2 ;;
    --baseline-renderer) BASELINE_RENDERER="$2"; shift 2 ;;
    --derivative-renderer) DERIVATIVE_RENDERER="$2"; shift 2 ;;
    --renderer-python) PYTHON_BIN="$2"; shift 2 ;;
    --browsers-path) BROWSERS_PATH="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --total-limit) TOTAL_LIMIT="$2"; shift 2 ;;
    --backend)
      if [[ "$2" != "swiftshader" ]]; then
        echo "This vendored Linux runtime is pinned to ANGLE swiftshader" >&2
        exit 2
      fi
      shift 2
      ;;
    --) shift; break ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$INPUT_ROOT" || -z "$OUTPUT_ROOT" ]]; then
  echo "usage: run_cpu_profile.sh --input-root DIR --output-root DIR [runtime options]" >&2
  exit 2
fi

INPUT_ROOT="$(cd "$INPUT_ROOT" && pwd)"
OUTPUT_WAS_CREATED=0
if [[ ! -e "$OUTPUT_ROOT" ]]; then
  mkdir -p "$OUTPUT_ROOT"
  OUTPUT_WAS_CREATED=1
fi
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"

# A prepared cloud image normally exposes this layout.  It is deliberately
# searched by path only; no token or connection setting is printed or copied.
CLOUD_ROOT="${PAINTER_CLOUD_RUNTIME:-$REPO_ROOT/.painter-cloud-runtime}"
if [[ -z "$PYTHON_BIN" && -x "$CLOUD_ROOT/renderer-env/bin/python" ]]; then
  PYTHON_BIN="$CLOUD_ROOT/renderer-env/bin/python"
fi
if [[ -z "$BROWSERS_PATH" && -d "$CLOUD_ROOT/browsers" ]]; then
  BROWSERS_PATH="$CLOUD_ROOT/browsers"
elif [[ -z "$BROWSERS_PATH" && -d "$CLOUD_ROOT/playwright-browsers" ]]; then
  BROWSERS_PATH="$CLOUD_ROOT/playwright-browsers"
fi

TEMP_ROOT=""
cleanup() {
  if [[ -n "$TEMP_ROOT" && -d "$TEMP_ROOT" ]]; then
    rm -rf "$TEMP_ROOT"
  fi
}
trap cleanup EXIT

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v uv >/dev/null 2>&1; then
    TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/painter-cpu-profile.XXXXXX")"
    VENV="$TEMP_ROOT/venv"
    uv venv "$VENV" >/dev/null
    PYTHON_BIN="$VENV/bin/python"
    uv pip install --python "$PYTHON_BIN" playwright pillow >/dev/null
  else
    TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/painter-cpu-profile.XXXXXX")"
    VENV="$TEMP_ROOT/venv"
    python3 -m venv "$VENV"
    "$VENV/bin/python" -m pip install --disable-pip-version-check playwright pillow >/dev/null
    PYTHON_BIN="$VENV/bin/python"
  fi
fi

if [[ -z "$BROWSERS_PATH" ]]; then
  # A prepared browser is preferred.  This fallback is finite setup work and
  # does not contact a model, judge, or HF upload endpoint.
  if [[ -n "$TEMP_ROOT" ]]; then
    BROWSERS_PATH="$TEMP_ROOT/playwright-browsers"
    PLAYWRIGHT_BROWSERS_PATH="$BROWSERS_PATH" "$PYTHON_BIN" -m playwright install chromium >/dev/null
  else
    BROWSERS_PATH="${TMPDIR:-/tmp}/painter-playwright-browsers"
    mkdir -p "$BROWSERS_PATH"
    PLAYWRIGHT_BROWSERS_PATH="$BROWSERS_PATH" "$PYTHON_BIN" -m playwright install chromium >/dev/null
  fi
fi

if [[ -z "$RUNTIME_ROOT" ]]; then
  if [[ -f "$INPUT_ROOT/runtime/renderer.py" ]]; then
    RUNTIME_ROOT="$INPUT_ROOT/runtime"
  elif [[ -f "$INPUT_ROOT/runtime/watercolour-p5-brush-2.2.1-centroid-fix-v1/renderer.py" ]]; then
    RUNTIME_ROOT="$INPUT_ROOT/runtime/watercolour-p5-brush-2.2.1-centroid-fix-v1"
  else
    RUNTIME_ROOT="$INPUT_ROOT/runtime"
  fi
fi
if [[ -z "$DERIVATIVE_RUNTIME_ROOT" ]]; then
  if [[ -f "$INPUT_ROOT/runtime/cancellable_watercolour_renderer.py" ]]; then
    DERIVATIVE_RUNTIME_ROOT="$INPUT_ROOT/runtime"
  elif [[ -f "$INPUT_ROOT/runtime/watercolour-cancellable-centroid-v1/cancellable_watercolour_renderer.py" ]]; then
    DERIVATIVE_RUNTIME_ROOT="$INPUT_ROOT/runtime/watercolour-cancellable-centroid-v1"
  fi
fi

# If the public bundle omitted materialized runtimes, use the checked-in
# centroid helper to build both into the temporary setup root.  Existing
# materialized runtime identities are never overwritten.
if [[ ! -f "$RUNTIME_ROOT/renderer.py" && ! -f "$RUNTIME_ROOT/frozen_watercolour_renderer.py" ]]; then
  TEMP_ROOT="${TEMP_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/painter-cpu-profile.XXXXXX")}"
  RUNTIME_ROOT="$TEMP_ROOT/watercolour-p5-brush-2.2.1-centroid-fix-v1"
  "$PYTHON_BIN" "$PAINTER_ROOT/patch_brush_centroid.py" --output-root "$RUNTIME_ROOT" >/dev/null
fi
if [[ -z "$DERIVATIVE_RUNTIME_ROOT" || ! -f "$DERIVATIVE_RUNTIME_ROOT/cancellable_watercolour_renderer.py" ]]; then
  TEMP_ROOT="${TEMP_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/painter-cpu-profile.XXXXXX")}"
  DERIVATIVE_RUNTIME_ROOT="$TEMP_ROOT/watercolour-cancellable-centroid-v1"
  "$PYTHON_BIN" "$PAINTER_ROOT/patch_brush_centroid.py" --renderer-overlay --source-root "$PAINTER_ROOT/vendor/integrations/watercolour" --output-root "$DERIVATIVE_RUNTIME_ROOT" >/dev/null
fi

# When the cloud launcher is root, the renderer still runs as the unprivileged
# painter account.  Make only setup-owned temporary paths traversable/readable
# before handing them over; public input files are never rewritten here.
if [[ -n "$TEMP_ROOT" ]]; then
  chmod 755 "$TEMP_ROOT" 2>/dev/null || true
  chmod -R a+rX "$TEMP_ROOT" 2>/dev/null || true
fi
if [[ "$OUTPUT_WAS_CREATED" == "1" && "$(id -u)" == "0" ]] && id painter >/dev/null 2>&1 && command -v runuser >/dev/null 2>&1; then
  if [[ -d "$OUTPUT_ROOT" ]]; then
    chown -R painter:painter "$OUTPUT_ROOT" 2>/dev/null || true
  fi
fi

EXTRA=("--input-root" "$INPUT_ROOT" "--output-root" "$OUTPUT_ROOT" "--runtime-root" "$RUNTIME_ROOT" "--derivative-runtime-root" "$DERIVATIVE_RUNTIME_ROOT" "--renderer-python" "$PYTHON_BIN" "--browsers-path" "$BROWSERS_PATH" "--backend" "swiftshader" "--timeout" "$TIMEOUT" "--total-limit" "$TOTAL_LIMIT" "--execute")
[[ -n "$BASELINE_RENDERER" ]] && EXTRA+=("--baseline-renderer" "$BASELINE_RENDERER")
[[ -n "$DERIVATIVE_RENDERER" ]] && EXTRA+=("--derivative-renderer" "$DERIVATIVE_RENDERER")
EXTRA+=("$@")

run_profile=("$PYTHON_BIN" "$SCRIPT_DIR/profile_cpu.py" "${EXTRA[@]}")
if [[ "$(id -u)" == "0" ]] && id painter >/dev/null 2>&1 && command -v runuser >/dev/null 2>&1; then
  exec runuser -u painter -- env -i PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}" HOME="${TMPDIR:-/tmp}/painter-cpu-home" TMPDIR="${TMPDIR:-/tmp}" LANG=C.UTF-8 PLAYWRIGHT_BROWSERS_PATH="$BROWSERS_PATH" "${run_profile[@]}"
else
  exec env -i PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}" HOME="${TMPDIR:-/tmp}/painter-cpu-home" TMPDIR="${TMPDIR:-/tmp}" LANG=C.UTF-8 PLAYWRIGHT_BROWSERS_PATH="$BROWSERS_PATH" "${run_profile[@]}"
fi
