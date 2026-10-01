#!/usr/bin/env bash
set -Eeuo pipefail

RUN="${PAINTER_RUN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)}"
REPO="$(cd "$RUN/../../.." && pwd -P)"
BASE="$REPO/painter/runs/text-sft512-consistent-20260930"
NODE_ROOT="${PAINTER_NODE_ROOT:-/root/painter}"
NODE_BIN="${PAINTER_NODE_BIN:-node}"
PATCHED_ROOT="${PAINTER_PATCHED_RENDERER_ROOT:-$RUN/runtime/watercolour-p5-brush-2.2.1-centroid-fix-v1}"
RUN_NAME="${PAINTER_RUN_NAME:-text-rl16-retry-20261001}"
export PAINTER_RUN_ROOT="$RUN" PAINTER_RUN_NAME="$RUN_NAME"
export PAINTER_GROUP_SIZE=4 PAINTER_INITIAL_INFLIGHT=32 PAINTER_MAX_INFLIGHT=32
export PAINTER_POLICY_MODE=recovered_policy
export PAINTER_POLICY_REVISION=ac4d3eda8e5925e2d23992cfdc1a63620ebde573
export PAINTER_POLICY_PREFIX=text-pairwise-scaled-20260929/step_16
export PAINTER_POLICY_SHA256=e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c
export PAINTER_SFT_ADAPTER_PATH_FILE="$RUN/policy-adapter-path.txt"
export PAINTER_FROZEN_SFT_ADAPTER_PATH_FILE="${PAINTER_FROZEN_SFT_ADAPTER_PATH_FILE:-$RUN/../photo-curriculum-rl-20260922/initial-adapter-path.txt}"
export PAINTER_POLICY_DOWNLOAD_SCRIPT="$REPO/painter/runs/text-pairwise-rl-20260928/download_public_adapter.py"
export PAINTER_POLICY_DOWNLOAD_RECEIPT="$RUN/policy-adapter-receipt.json"
export PAINTER_RENDERER_ROOT="$PATCHED_ROOT"
export PAINTER_RENDERER_PATH="$PATCHED_ROOT/renderer.py"
export PAINTER_RENDERER_PYTHON_PATH="${PAINTER_RENDERER_PYTHON_PATH:-$RUN/renderer-env/bin/python}"
export PAINTER_BOOTSTRAP_RUN="${PAINTER_BOOTSTRAP_RUN:-$REPO/painter/runs/text-pairwise-rl-20260928}"
export PAINTER_REQUIRE_VALID_FINAL_ACTION=1 PAINTER_JS_SYNTAX_PREFLIGHT=1

[[ "$(uname -s)" == Linux ]] || { echo 'Linux GPU node required' >&2; exit 1; }
NODE_BIN_PATH="$(command -v "$NODE_BIN" || true)"
if [[ -z "$NODE_BIN_PATH" && "$NODE_BIN" == node && "$(id -u)" == 0 && -x "$(command -v apt-get || true)" ]]; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nodejs
  NODE_BIN_PATH="$(command -v node || true)"
fi
[[ -n "$NODE_BIN_PATH" ]] || { echo 'Node.js is required for JavaScript syntax preflight' >&2; exit 1; }
"$NODE_BIN_PATH" --version >"$RUN/node-version.txt"
if [[ -f "$PATCHED_ROOT/PATCH-MANIFEST.json" ]]; then
  [[ -f "$PATCHED_ROOT/renderer.py" && -f "$PATCHED_ROOT/assets/p5.brush.js" ]] || {
    echo 'existing patched renderer is incomplete' >&2; exit 1;
  }
else
  [[ ! -e "$PATCHED_ROOT" ]] || { echo 'partial patched renderer exists without manifest' >&2; exit 1; }
  python3 "$REPO/painter/patch_brush_centroid.py" --output-root "$PATCHED_ROOT" \
    >"$RUN/patched-renderer-build.json"
fi
bash "$BASE/prepare_node.sh"
chmod a+rx "$RUN" "$(dirname "$PATCHED_ROOT")" "$PATCHED_ROOT"
find "$PATCHED_ROOT" -type d -exec chmod a+rx {} +
find "$PATCHED_ROOT" -type f -exec chmod a+r {} +
[[ -x "$PAINTER_RENDERER_PYTHON_PATH" ]] || { echo 'renderer Python executable is missing' >&2; exit 1; }
mkdir -p "$RUN/renderer-home"
/usr/sbin/runuser -u painter -- env -i PATH=/usr/bin:/bin HOME="$RUN/renderer-home" LANG=C \
  "$PAINTER_RENDERER_PYTHON_PATH" "$PAINTER_RENDERER_PATH" --help >/dev/null
"${PRIME_ROOT:-$NODE_ROOT/prime}/.venv/bin/python" - "$RUN" "$PATCHED_ROOT" "$NODE_BIN_PATH" <<'PY'
import hashlib, json, sys
from pathlib import Path
run, renderer, node = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
patch = json.loads((renderer / "PATCH-MANIFEST.json").read_text())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
(run / "renderer-receipt.json").write_text(json.dumps({
    "schema": "text-rl16-patched-renderer-v1",
    "renderer_root": str(renderer),
    "renderer_path": str(renderer / "renderer.py"),
    "renderer_sha256": sha(renderer / "renderer.py"),
    "patch_manifest_sha256": sha(renderer / "PATCH-MANIFEST.json"),
    "runtime_id": patch["runtime_id"],
    "node_path": node,
    "node_version": (run / "node-version.txt").read_text().strip(),
}, indent=2, sort_keys=True) + "\n")
PY
echo 'TEXT_RL16_RETRY_PREPARED: group4 and 32 in-flight with explicit patched renderer'
