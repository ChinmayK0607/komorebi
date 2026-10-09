#!/usr/bin/env bash
set -Eeuo pipefail

# Linux-node preparation. Controller-side preparation never calls this file.
RUN="${PAINTER_RUN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)}"
REPO="$(cd "$RUN/../../.." && pwd -P)"
source "$RUN/runtime_env.sh"
validate_structured_bindings
[[ "$(uname -s)" == Linux ]] || { echo 'Linux GPU node required' >&2; exit 1; }
[[ $# == 0 || ( $# == 1 && "$1" == --interpreter-probe ) ]] || {
  echo 'usage: prepare_node.sh [--interpreter-probe]' >&2; exit 2;
}

# Prime config validation imports pydantic from the pinned Prime environment;
# the bootstrap interpreter intentionally has a smaller dependency set. Keep
# the selection explicit and expose a cheap node-local probe for the operator
# and regression tests before any renderer or model setup begins.
PY_PRIME="${PAINTER_PRIME_PYTHON:-$PRIME_ROOT/.venv/bin/python}"
if [[ -n "$PAINTER_LOCAL_SCRATCH_ROOT" ]]; then
  mkdir -p "$PAINTER_LOCAL_SCRATCH_ROOT"
  [[ -d "$PAINTER_LOCAL_SCRATCH_ROOT" && -w "$PAINTER_LOCAL_SCRATCH_ROOT" ]] || {
    echo "local scratch root is not writable: $PAINTER_LOCAL_SCRATCH_ROOT" >&2; exit 1;
  }
fi
if [[ "${1:-}" == --interpreter-probe ]]; then
  [[ -x "$PY_PRIME" ]] || { echo "pinned Prime Python is missing: $PY_PRIME" >&2; exit 1; }
  "$PY_PRIME" - <<'PY'
import json
import sys
try:
    import pydantic  # noqa: F401
except Exception as exc:
    raise SystemExit(f"Prime interpreter cannot import pydantic: {type(exc).__name__}: {exc}")
print(json.dumps({"schema": "Diverse-flow-prime-interpreter-probe-v1", "executable": sys.executable, "pydantic": True}, sort_keys=True))
PY
  exit 0
fi

# Keep setup bounded on the rented node.  These guards run before any package
# install or model/adapter transfer so a constrained image fails cheaply.
OPEN_LIMIT="$(ulimit -Sn)"
if [[ "$OPEN_LIMIT" != unlimited && "$OPEN_LIMIT" =~ ^[0-9]+$ && "$OPEN_LIMIT" -lt 131072 ]]; then
  ulimit -n 131072 2>/dev/null || { echo 'unable to raise soft nofile limit to 131072' >&2; exit 1; }
fi
export UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-32}"
[[ "$UV_CONCURRENT_INSTALLS" =~ ^[1-9][0-9]*$ && "$UV_CONCURRENT_INSTALLS" -le 32 ]] || {
  echo 'UV_CONCURRENT_INSTALLS must be an integer in [1,32]' >&2; exit 1;
}
for scratch in "${SLURM_TMPDIR:-}" "${TMPDIR:-}" "/scratch" "/tmp"; do
  if [[ -n "$scratch" && -d "$scratch" && -w "$scratch" ]]; then
    export TMPDIR="$scratch/diverse-flow-rl-20261009"
    mkdir -p "$TMPDIR"
    break
  fi
done

# Build and bind the same reviewed renderer derivative used by the matched
# campaign before the generic bootstrap starts.  The bootstrap can therefore
# install its runtime against the frozen centroid-fix source, while every
# actual taskset/evaluator later selects the cancellable overlay.  A raw
# vendor renderer is never silently substituted for this run.
THROUGHPUT_RUN="${PAINTER_RECOVERY_RUN:-$REPO/painter/runs/throughput-recovery-20261003}"
PATCH_RENDERER="$REPO/painter/patch_brush_centroid.py"
PAINTER_RENDERER_RUNTIME_HELPER="${PAINTER_RENDERER_RUNTIME_HELPER:-$RUN/prepare_renderer_runtime.py}"
FROZEN_RENDERER_ROOT="${PAINTER_FROZEN_RENDERER_ROOT:-$RUN/runtime/watercolour-p5-brush-2.2.1-centroid-fix-v1}"
CANCELLABLE_RENDERER_ROOT="${PAINTER_CANCELLABLE_RENDERER_ROOT:-$RUN/runtime/watercolour-cancellable-centroid-v1}"
[[ -f "$PATCH_RENDERER" && -f "$PAINTER_RENDERER_RUNTIME_HELPER" ]] || {
  echo 'reviewed centroid patcher or renderer runtime helper is missing' >&2; exit 1;
}
mkdir -p "$RUN/runtime"
if [[ ! -f "$FROZEN_RENDERER_ROOT/PATCH-MANIFEST.json" ]]; then
  [[ ! -e "$FROZEN_RENDERER_ROOT" ]] || {
    echo 'partial frozen centroid renderer exists; preserve and inspect it' >&2; exit 1;
  }
  python3 "$PATCH_RENDERER" --output-root "$FROZEN_RENDERER_ROOT" \
    >"$RUN/frozen-renderer-build.json"
fi
if [[ ! -f "$CANCELLABLE_RENDERER_ROOT/PATCH-MANIFEST.json" ]]; then
  [[ ! -e "$CANCELLABLE_RENDERER_ROOT" ]] || {
    echo 'partial cancellable renderer overlay exists; preserve and inspect it' >&2; exit 1;
  }
  python3 "$PATCH_RENDERER" --renderer-overlay --output-root "$CANCELLABLE_RENDERER_ROOT" \
    >"$RUN/cancellable-renderer-build.json"
fi
# Verify the files, manifest schemas, and exact hashes before any model or
# Prime setup.  This catches an accidentally copied raw vendor tree early.
python3 - "$PATCH_RENDERER" "$FROZEN_RENDERER_ROOT" "$CANCELLABLE_RENDERER_ROOT" <<'PY'
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

patcher, frozen, overlay = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("structured_patch_brush_centroid", patcher)
if spec is None or spec.loader is None:
    raise SystemExit("centroid patcher is not importable")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

expected = {
    "frozen_renderer": "6181fb904a0c5dcd580f5c3cb01eaf341a14b7a15479e811d99ebb10b70bc841",
    "centroid_asset": "db5b0094cb223529a8c5fc3bbba7bf8f2d5b5f5a5c0a847062194feaf917d16b",
    "cancellable_renderer": "dc6b9d837ed7be382784a6c84de1b64911506b17a19b499ce054c0c572e75794",
}
frozen_manifest = json.loads((frozen / "PATCH-MANIFEST.json").read_text())
if frozen_manifest.get("schema") != "watercolour-centroid-fix-v1":
    raise SystemExit("unexpected frozen centroid renderer manifest schema")
checks = {
    "frozen renderer": (frozen / "renderer.py", expected["frozen_renderer"]),
    "frozen centroid asset": (frozen / "assets/p5.brush.js", expected["centroid_asset"]),
}
for label, (path, wanted) in checks.items():
    if not path.is_file() or sha(path) != wanted:
        raise SystemExit(f"{label} hash mismatch: {path}")
if frozen_manifest.get("patch", {}).get("after_sha256") != expected["centroid_asset"]:
    raise SystemExit("frozen centroid patch manifest does not bind the expected asset")
module.validate_renderer_overlay(overlay)
overlay_checks = {
    "cancellable renderer": (overlay / "cancellable_watercolour_renderer.py", expected["cancellable_renderer"]),
    "overlay frozen renderer": (overlay / "frozen_watercolour_renderer.py", expected["frozen_renderer"]),
    "overlay centroid asset": (overlay / "assets/p5.brush.js", expected["centroid_asset"]),
}
for label, (path, wanted) in overlay_checks.items():
    if not path.is_file() or sha(path) != wanted:
        raise SystemExit(f"{label} hash mismatch: {path}")
print("STRUCTURED_RENDERER_DERIVATIVES_VERIFIED")
PY
export PAINTER_RENDERER_RUNTIME_HELPER
export PAINTER_RENDERER_ROOT="$FROZEN_RENDERER_ROOT"
export PAINTER_RENDERER_PATH="$FROZEN_RENDERER_ROOT/renderer.py"
export PAINTER_RENDERER_PYTHON_PATH="${PAINTER_RENDERER_PYTHON_PATH:-$RUN/renderer-env/bin/python}"

# Probe the node's user-namespace/setuid prerequisites before bootstrap can
# start model downloads or install the large Prime environment.  This does
# not weaken Chromium's sandbox; it fails closed when neither supported path
# is available and records the kernel evidence for a useful retry decision.
export PAINTER_BROWSER_TMPDIR="${PAINTER_BROWSER_TMPDIR:-$RUN/runtime/browser-tmp}"
python3 "$PAINTER_RENDERER_RUNTIME_HELPER" --capability-only \
  --run-root "$RUN" --renderer-root "$RUN" --renderer-python "$PAINTER_RENDERER_PYTHON_PATH" \
  --browsers "${PAINTER_BROWSER_PROBE_ROOT:-$REPO/painter/browsers}" --prime-venv "$PRIME_ROOT/.venv"

# Match the proven campaign's bounded fast-cache placement.  In particular,
# keep uv's cache and later browser storage on the selected scratch device;
# this also lets the shared setup script preserve its browser ownership and
# helper permissions instead of creating a private root-owned tree.
PAINTER_FAST_STORAGE_HELPER="${PAINTER_FAST_STORAGE_HELPER:-$THROUGHPUT_RUN/prepare_fast_storage.py}"
PAINTER_FAST_STORAGE_ROOT="${PAINTER_FAST_STORAGE_ROOT:-${TMPDIR:-/tmp}/Diverse-flow-renderer-fast}"
[[ -f "$PAINTER_FAST_STORAGE_HELPER" ]] || { echo 'reviewed fast-storage helper is missing' >&2; exit 1; }
python3 "$PAINTER_FAST_STORAGE_HELPER" --phase cache --run-root "$RUN" \
  --fast-root "$PAINTER_FAST_STORAGE_ROOT" --uv-source "${PAINTER_UV_CACHE_SOURCE:-${HOME:-/root}/.cache/uv}" \
  --uv-concurrent-installs "$UV_CONCURRENT_INSTALLS"
export PAINTER_FAST_STORAGE_HELPER PAINTER_FAST_STORAGE_ROOT
export UV_CACHE_DIR="$PAINTER_FAST_STORAGE_ROOT/uv-cache"

PY_BOOT="${PAINTER_BOOTSTRAP_PYTHON:-$PAINTER_NODE_ROOT/bootstrap/bin/python}"
[[ -x "$PY_BOOT" ]] || PY_BOOT=python3
if [[ "$PAINTER_PRIME_SEED_ENABLED" == 1 && ! -d "$PRIME_ROOT/.git" ]]; then
  [[ -f "$RUN/seed_prime_checkout.py" ]] || { echo 'Prime seed helper is missing' >&2; exit 1; }
  [[ -d "$PAINTER_PRIME_SEED_ROOT/.git" ]] || { echo "verified Prime seed source is missing: $PAINTER_PRIME_SEED_ROOT" >&2; exit 1; }
  mkdir -p "$(dirname "$PRIME_ROOT")"
  "$PY_BOOT" "$RUN/seed_prime_checkout.py" \
    --source "$PAINTER_PRIME_SEED_ROOT" --destination "$PRIME_ROOT" \
    --expected-commit 26b3131d2716a4f8210b165df584b83b4bc54f61 \
    >"${PAINTER_PRIME_SEED_RECEIPT:-$RUN/prime-seed-receipt.json}"
fi
export PAINTER_PRIME_SEED_HELPER="$RUN/seed_prime_checkout.py"
export PAINTER_PRIME_SEED_RECEIPT="${PAINTER_PRIME_SEED_RECEIPT:-$RUN/prime-seed-receipt.json}"
BOOTSTRAP_RUN="${PAINTER_BOOTSTRAP_RUN:-$REPO/painter/runs/text-pairwise-rl-20260928}"
[[ -f "$BOOTSTRAP_RUN/setup_node.sh" ]] || { echo "reviewed bootstrap missing: $BOOTSTRAP_RUN" >&2; exit 1; }
bash "$BOOTSTRAP_RUN/setup_node.sh"
# setup_node has now materialized the verified Prime environment.  Keep its
# bin directory visible for this preparation shell as well as later shells
# that source runtime_env.sh; native renderer syntax checks rely on its Node.
if [[ -d "$PRIME_ROOT/.venv/bin" && ":${PATH:-}:" != *":$PRIME_ROOT/.venv/bin:"* ]]; then
  export PATH="$PRIME_ROOT/.venv/bin:${PATH:-/usr/local/bin:/usr/bin:/bin}"
fi
[[ -x "$PY_PRIME" ]] || { echo "pinned Prime Python is missing: $PY_PRIME" >&2; exit 1; }

# Exercise the pinned verifier runtime boundary before GPU model serving or
# direct selected-adapter download.  Runtime.run_uv_script performs the same
# uv sync, interpreter discovery, and final PEP 723 launch used by task
# rollouts.  The run-local sitecustomize must rewrite only the child
# environment to ``harness`` while this probe process remains the RL
# ``orchestrator`` role.  The verifier uses its existing home cache; this
# probe does not repurpose HOME.
PRIME_RUNTIME_PROBE_RECEIPT="${PAINTER_PRIME_RUNTIME_PROBE_RECEIPT:-$RUN/prime-runtime-harness-probe.json}"
PRIME_RUNTIME_PROBE_TMP="$RUN/.prime-runtime-harness-probe.json.$$"
if ! PAINTER_PRIME_ROLE=orchestrator \
  PAINTER_WHOLE_GROUP_ADMISSION=1 PAINTER_WHOLE_GROUP_SIZE="$PAINTER_GROUP_SIZE" \
  PAINTER_JUDGE_PROTOCOL="$PAINTER_JUDGE_PROTOCOL" PAINTER_STRUCTURED_FLOW_POLICY=0 \
  PYTHONPATH="$REPO:$REPO/painter:$RUN${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY_PRIME" - >"$PRIME_RUNTIME_PROBE_TMP" <<'PY'
import asyncio
import json
import os
import uuid

from verifiers.v1.runtimes.subprocess import SubprocessConfig, SubprocessRuntime


SCRIPT = '''# /// script
# requires-python = ">=3.11"
# ///
import os
print("Diverse-flow-prime-runtime-child", os.environ.get("PAINTER_PRIME_ROLE", ""))
'''


async def main() -> None:
    if os.environ.get("PAINTER_PRIME_ROLE") != "orchestrator":
        raise SystemExit("probe parent role is not orchestrator")
    # Construct the pinned runtime config instead of emulating Runtime.run.
    config = SubprocessConfig()
    if getattr(config, "type", None) != "subprocess":
        raise SystemExit("unexpected pinned SubprocessConfig type")
    runtime = SubprocessRuntime(
        config,
        name=f"Diverse-flow-prime-runtime-probe-{os.getpid()}-{uuid.uuid4().hex[:8]}",
    )
    await runtime.start()
    try:
        result = await runtime.run_uv_script(
            SCRIPT,
            env={"PAINTER_PRIME_ROLE": "orchestrator"},
        )
    finally:
        await runtime.stop()
    if result.exit_code != 0:
        raise SystemExit(
            "runtime child failed: "
            + (result.stderr.strip() or result.stdout.strip() or str(result.exit_code))
        )
    child_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if child_lines != ["Diverse-flow-prime-runtime-child harness"]:
        raise SystemExit(f"runtime child role mismatch: {child_lines!r}")
    if os.environ.get("PAINTER_PRIME_ROLE") != "orchestrator":
        raise SystemExit("probe parent role changed")
    print(json.dumps({
        "schema": "Diverse-flow-prime-runtime-harness-probe-v1",
        "parent_role": "orchestrator",
        "child_role": "harness",
        "runtime": "verifiers.v1.runtimes.subprocess.SubprocessRuntime",
        "api": "Runtime.run_uv_script",
    }, sort_keys=True))


asyncio.run(main())
PY
then
  rm -f "$PRIME_RUNTIME_PROBE_TMP"
  echo 'pinned verifier runtime harness probe failed' >&2
  exit 1
fi
mkdir -p "$(dirname "$PRIME_RUNTIME_PROBE_RECEIPT")"
mv "$PRIME_RUNTIME_PROBE_TMP" "$PRIME_RUNTIME_PROBE_RECEIPT"
DOWNLOAD="${PAINTER_POLICY_DOWNLOAD_SCRIPT:-$REPO/painter/runs/text-pairwise-rl-20260928/download_public_adapter.py}"
[[ -f "$DOWNLOAD" ]] || { echo 'pinned public adapter downloader is missing' >&2; exit 1; }

download_adapter() {
  local revision="$1" prefix="$2" sha="$3" receipt="$4" path_file="$5"
  if [[ ! -f "$receipt" ]]; then
    "$PY_BOOT" "$DOWNLOAD" --revision "$revision" --prefix "$prefix" --expected-sha256 "$sha" --receipt "$receipt"
  fi
  "$PY_BOOT" "$RUN/verify_adapter.py" "$receipt" "$path_file" "$revision" "$prefix" "$sha"
}
download_adapter "$PAINTER_POLICY_REVISION" "$PAINTER_POLICY_PREFIX" "$PAINTER_POLICY_SHA256" \
  "${PAINTER_POLICY_DOWNLOAD_RECEIPT:-$RUN/policy-adapter-receipt.json}" "$PAINTER_SFT_ADAPTER_PATH_FILE"
download_adapter "$PAINTER_FROZEN_SFT_REVISION" "$PAINTER_FROZEN_SFT_PREFIX" "$PAINTER_FROZEN_SFT_SHA256" \
  "${PAINTER_FROZEN_SFT_DOWNLOAD_RECEIPT:-$RUN/frozen-sft-adapter-receipt.json}" "$PAINTER_FROZEN_SFT_ADAPTER_PATH_FILE"

# These entrypoints are intentionally owned by this run: launch_rl.sh wires
# the four-GPU roles and train_from_sft.py selects the run-local DTensor hooks.
# Bind them to the checked-out source blob instead of comparing them with the
# generic legacy bootstrap's incompatible launchers.
RUN_OWNED_VERIFY="$RUN/verify_run_owned_inputs.py"
[[ -f "$RUN_OWNED_VERIFY" ]] || { echo 'run-owned source verifier is missing' >&2; exit 1; }
RUN_RELATIVE_ROOT="${RUN#"$REPO"/}"
[[ "$RUN_RELATIVE_ROOT" != "$RUN" && -n "$RUN_RELATIVE_ROOT" ]] || {
  echo 'run root is outside the checked-out source repository' >&2; exit 1;
}
"$PY_BOOT" "$RUN_OWNED_VERIFY" --repo "$REPO" --run-root "$RUN" \
  --relative-root "$RUN_RELATIVE_ROOT" \
  --file launch_rl.sh --file run_from_sft.py --file train_from_sft.py
for item in sitecustomize.py safe_archive.py node_validate.py verify_adapter.py matched_baseline.sh; do
  [[ -e "$RUN/$item" ]] || { echo "tracked structured helper missing: $item" >&2; exit 1; }
done
if [[ -e "$BOOTSTRAP_RUN/vendor" && ! -e "$RUN/vendor" ]]; then ln -s "$BOOTSTRAP_RUN/vendor" "$RUN/vendor"; fi
for item in renderer-env browsers; do
  [[ -e "$BOOTSTRAP_RUN/$item" ]] || { echo "bootstrap renderer asset missing: $item" >&2; exit 1; }
  [[ -e "$RUN/$item" ]] || ln -s "$BOOTSTRAP_RUN/$item" "$RUN/$item"
done

# Complete the browser/Node binding only after the bootstrap has staged its
# runtime, then select the cancellable renderer for all generated manifests.
# The helper receipt is kept in the run root so later evaluation can prove the
# exact browser sandbox and Prime Node binding used by the renderer.
"$PY_BOOT" "$PAINTER_RENDERER_RUNTIME_HELPER" \
  --run-root "$RUN" --renderer-root "$BOOTSTRAP_RUN" \
  --renderer-python "$RUN/renderer-env/bin/python" --browsers "$RUN/browsers" \
  --prime-venv "$PRIME_ROOT/.venv"
export PAINTER_RENDERER_PATH="$CANCELLABLE_RENDERER_ROOT/cancellable_watercolour_renderer.py"
[[ -f "$PAINTER_RENDERER_PATH" && -x "$PAINTER_RENDERER_PYTHON_PATH" ]] || {
  echo 'verified cancellable renderer runtime is unavailable' >&2; exit 1;
}
"$PY_BOOT" - "$RUN" "$FROZEN_RENDERER_ROOT" "$CANCELLABLE_RENDERER_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

run, frozen, overlay = map(Path, sys.argv[1:])
expected = {
    "frozen_renderer": "6181fb904a0c5dcd580f5c3cb01eaf341a14b7a15479e811d99ebb10b70bc841",
    "centroid_asset": "db5b0094cb223529a8c5fc3bbba7bf8f2d5b5f5a5c0a847062194feaf917d16b",
    "cancellable_renderer": "dc6b9d837ed7be382784a6c84de1b64911506b17a19b499ce054c0c572e75794",
}
def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
assets = {
    item.relative_to(overlay / "assets").as_posix(): sha(item)
    for item in sorted((overlay / "assets").rglob("*"))
    if item.is_file() and not item.is_symlink()
}
asset_sha = hashlib.sha256(
    (json.dumps(assets, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
).hexdigest()
if sha(overlay / "cancellable_watercolour_renderer.py") != expected["cancellable_renderer"]:
    raise SystemExit("cancellable renderer identity changed after validation")
if sha(overlay / "assets/p5.brush.js") != expected["centroid_asset"]:
    raise SystemExit("cancellable renderer centroid asset changed after validation")
receipt = {
    "schema": "Diverse-flow-renderer-selection-v1",
    "selected": {
        "path": str(overlay / "cancellable_watercolour_renderer.py"),
        "renderer_sha256": expected["cancellable_renderer"],
        "assets_sha256": asset_sha,
        "assets": assets,
    },
    "frozen_centroid_root": {
        "path": str(frozen / "renderer.py"),
        "renderer_sha256": expected["frozen_renderer"],
        "centroid_asset_sha256": expected["centroid_asset"],
    },
    "runtime_helper": "renderer-runtime-receipt.json",
    "credentials_touched": False,
}
(run / "renderer-selection-receipt.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(asset_sha)
PY
export PAINTER_RENDERER_ASSETS_SHA256="$("$PY_BOOT" - "$CANCELLABLE_RENDERER_ROOT/assets" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
assets = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(root.rglob('*')) if p.is_file() and not p.is_symlink()}
print(hashlib.sha256((json.dumps(assets, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode()).hexdigest())
PY
)"
export PAINTER_RENDERER_SHA256=dc6b9d837ed7be382784a6c84de1b64911506b17a19b499ce054c0c572e75794

# Cross the exact renderer boundary once before any Qwen-serving process.
# This tiny p5.brush program uses the selected cancellable renderer, the
# verified browser root, and the same sandboxed runuser path as supervision;
# source/PNG/renderer receipts remain available when the probe is censored.
"$PY_PRIME" "$PAINTER_RENDERER_RUNTIME_HELPER" \
  --render-probe --run-root "$RUN" --renderer-root "$BOOTSTRAP_RUN" \
  --renderer "$PAINTER_RENDERER_PATH" --renderer-python "$PAINTER_RENDERER_PYTHON_PATH" \
  --browsers "$RUN/browsers" --prime-venv "$PRIME_ROOT/.venv" \
  --renderer-sha256 "$PAINTER_RENDERER_SHA256" --assets-sha256 "$PAINTER_RENDERER_ASSETS_SHA256"


TRAIN_MANIFEST="${PAINTER_RL_MANIFEST:-$RUN/curriculum/pilot-train-manifest.json}"
EVAL_MANIFEST="${PAINTER_EVAL_MANIFEST:-$RUN/curriculum/matched32-manifest.json}"
TRAIN_PROPOSAL="${PAINTER_TRAIN_PROPOSAL:-$RUN/curriculum/train-proposal.json}"
HELDOUT_PROPOSAL="${PAINTER_HELDOUT_PROPOSAL:-$RUN/curriculum/heldout-proposal.json}"
if [[ ! -f "$TRAIN_PROPOSAL" || ! -f "$HELDOUT_PROPOSAL" ]]; then
  # The controller may ship only the reviewed curriculum source.  Build the
  # deterministic 96/32 proposal pool on the node before admitting its
  # verified anchors; no model or renderer call is involved here.
  mkdir -p "$RUN/curriculum"
  # Keep the stdout receipt outside the strict output directory.  The shell
  # opens a redirection target before Python starts; placing it below
  # $RUN/curriculum would make the clean directory non-empty and correctly
  # trigger prepare_curriculum.py's refusal to overwrite an existing tree.
  CURRICULUM_BUILD_TMP="$RUN/.curriculum-build.json.$$"
  if PYTHONPATH="$REPO:$REPO/painter:$RUN${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY_BOOT" "$RUN/prepare_curriculum.py" build --output "$RUN/curriculum" \
    >"$CURRICULUM_BUILD_TMP"; then
    mv "$CURRICULUM_BUILD_TMP" "${PAINTER_CURRICULUM_BUILD_RECEIPT:-$RUN/curriculum-build.json}"
  else
    rm -f "$CURRICULUM_BUILD_TMP"
    exit 1
  fi
fi
[[ -f "$TRAIN_PROPOSAL" && -f "$HELDOUT_PROPOSAL" ]] || {
  echo 'Diverse-flow train/heldout proposals are missing' >&2; exit 1;
}
# A same-run retry is the only path that may consume paid evidence. It uses
# one immutable public archive for frozen anchors, admitted manifests,
# casebriefs, and the matched Flash12 baseline. The normal fresh path remains
# independent of this archive.
# anchor-evidence restore is intentionally disabled for every fresh path.
SAME_RUN_RESUME="${PAINTER_RESUME_SAME_RUN_ARCHIVE:-0}"
RESTORE_ANCHOR_EVIDENCE="${PAINTER_RESTORE_ANCHOR_EVIDENCE:-0}"
REUSE_MATCHED_BASELINE="${PAINTER_REUSE_MATCHED_BASELINE_EVIDENCE:-0}"
for value in "$SAME_RUN_RESUME" "$RESTORE_ANCHOR_EVIDENCE" "$REUSE_MATCHED_BASELINE"; do
  [[ "$value" == 0 || "$value" == 1 ]] || {
    echo 'resume, anchor restore, and matched-baseline reuse bits must be 0 or 1' >&2; exit 1;
  }
done
if [[ "$SAME_RUN_RESUME" == 1 ]]; then
  RESTORE_ANCHOR_EVIDENCE=1
  REUSE_MATCHED_BASELINE=1
elif [[ "$RESTORE_ANCHOR_EVIDENCE" == 1 || "$REUSE_MATCHED_BASELINE" == 1 ]]; then
  [[ "$RESTORE_ANCHOR_EVIDENCE" == 1 && "$REUSE_MATCHED_BASELINE" == 1 ]] || {
    echo 'anchor restore and matched-baseline reuse must be enabled together for a same-run retry' >&2; exit 1;
  }
  SAME_RUN_RESUME=1
fi
if [[ "$SAME_RUN_RESUME" == 1 ]]; then
  RESTORE_SHA256="${PAINTER_SAME_RUN_ARCHIVE_SHA256:-${PAINTER_ANCHOR_EVIDENCE_SHA256:-ddd90296e2c6562cb24bc8f8033f439c4925e77002ad8eabca2723c38323eaac}}"
  RESTORE_URL="${PAINTER_SAME_RUN_ARCHIVE_URL:-${PAINTER_ANCHOR_EVIDENCE_URL:-https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/resolve/c35b77c6d3a858ab224f0812f1ed73e0150d1525/runs/diverse-flow-rl-20261009/evidence.tar.gz}}"
  for value in "${PAINTER_SAME_RUN_ARCHIVE_SHA256:-}" "${PAINTER_ANCHOR_EVIDENCE_SHA256:-}" "${PAINTER_MATCHED_BASELINE_EVIDENCE_SHA256:-}"; do
    [[ -z "$value" || "$value" == ddd90296e2c6562cb24bc8f8033f439c4925e77002ad8eabca2723c38323eaac ]] || {
      echo 'same-run archive SHA-256 is not the pinned public archive' >&2; exit 1;
    }
  done
  for value in "${PAINTER_SAME_RUN_ARCHIVE_URL:-}" "${PAINTER_ANCHOR_EVIDENCE_URL:-}" "${PAINTER_MATCHED_BASELINE_EVIDENCE_URL:-}"; do
    [[ -z "$value" || "$value" == https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/resolve/c35b77c6d3a858ab224f0812f1ed73e0150d1525/runs/diverse-flow-rl-20261009/evidence.tar.gz ]] || {
      echo 'same-run archive URL is not the pinned credential-free public revision' >&2; exit 1;
    }
  done
  [[ "$RESTORE_SHA256" == ddd90296e2c6562cb24bc8f8033f439c4925e77002ad8eabca2723c38323eaac ]] || {
    echo 'same-run archive SHA-256 is not the pinned public archive' >&2; exit 1;
  }
  [[ "$RESTORE_URL" == https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/resolve/c35b77c6d3a858ab224f0812f1ed73e0150d1525/runs/diverse-flow-rl-20261009/evidence.tar.gz ]] || {
    echo 'same-run archive URL is not the pinned credential-free public revision' >&2; exit 1;
  }
  [[ -z "${PAINTER_ANCHOR_EVIDENCE_ARCHIVE:-}" && -z "${PAINTER_MATCHED_BASELINE_EVIDENCE_ARCHIVE:-}" ]] || {
    echo 'same-run resume accepts only the pinned public URL; local archive overrides are disabled' >&2; exit 1;
  }
  export PAINTER_RESUME_SAME_RUN_ARCHIVE=1
  export PAINTER_SAME_RUN_ARCHIVE_URL="$RESTORE_URL" PAINTER_SAME_RUN_ARCHIVE_SHA256="$RESTORE_SHA256"
  export PAINTER_ANCHOR_EVIDENCE_URL="$RESTORE_URL" PAINTER_ANCHOR_EVIDENCE_SHA256="$RESTORE_SHA256"
  export PAINTER_MATCHED_BASELINE_EVIDENCE_URL="$RESTORE_URL" PAINTER_MATCHED_BASELINE_EVIDENCE_SHA256="$RESTORE_SHA256"
  RESTORE_SCRIPT="$RUN/restore_anchor_evidence.py"
  [[ -f "$RESTORE_SCRIPT" ]] || { echo 'anchor evidence restore helper is missing' >&2; exit 1; }
  RESTORE_ARGS=(--run-root "$RUN" --sha256 "$RESTORE_SHA256" --url "$RESTORE_URL")
  PYTHONPATH="$REPO:$REPO/painter:$RUN${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY_PRIME" "$RESTORE_SCRIPT" "${RESTORE_ARGS[@]}" \
    >"${PAINTER_ANCHOR_RESTORE_SUMMARY:-$RUN/anchor-restore-summary.json}"
  # Restore the exact admitted manifests and casebriefs after the 128-anchor
  # tree. This is metadata-only: the validator makes no model/renderer/judge
  # request and binds every row to the current proposal and flow source.
  ANCHOR_ROOT="${PAINTER_ANCHOR_ROOT:-$RUN/curriculum/anchors}"
  PYTHONPATH="$REPO:$REPO/painter:$RUN${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY_PRIME" - "$RUN" "$TRAIN_MANIFEST" "$EVAL_MANIFEST" "$TRAIN_PROPOSAL" "$HELDOUT_PROPOSAL" "$ANCHOR_ROOT" "$RESTORE_SHA256" "$RESTORE_URL" <<'PY' \
    >"${PAINTER_CACHED_ANCHOR_RESTORE_SUMMARY:-$RUN/cached-anchor-restore-summary.json}"
import importlib.util
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
train_manifest, heldout_manifest = (Path(value) for value in sys.argv[2:4])
train_proposal, heldout_proposal, anchor_root = (Path(value) for value in sys.argv[4:7])
archive_sha, archive_url = sys.argv[7:9]
spec = importlib.util.spec_from_file_location("diverse_flow_restore_resume", run / "restore_anchor_evidence.py")
if spec is None or spec.loader is None:
    raise SystemExit("cannot load same-run restore helper")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
value = module.restore_cached_pilot_manifests(
    run_root=run, expected_sha256=archive_sha,
    train_proposal=train_proposal, heldout_proposal=heldout_proposal,
    anchor_root=anchor_root, train_manifest=train_manifest,
    heldout_manifest=heldout_manifest, flow_source=run / "flow_demonstrations.py",
    url=archive_url,
)
if value.get("generator_calls") != 0 or value.get("train_rows") != 96 or value.get("heldout_cases") != 32:
    raise SystemExit(f"same-run cached manifest restore is not the exact 96/32 set: {json.dumps(value, sort_keys=True)}")
print(json.dumps(value, sort_keys=True))
PY
  [[ -f "$TRAIN_MANIFEST" && -f "$EVAL_MANIFEST" ]] || {
    echo 'same-run archive did not restore the admitted 96/32 manifests' >&2; exit 1;
  }
fi
# The fixed-anchor files are produced after the owned inference server has
# loaded the frozen SFT alias.  Stage an explicit lazy manifest for preflight;
# run_from_sft.py replaces these bytes with the hash-verified pilot manifests
# before Prime creates the orchestrator.
if [[ "$SAME_RUN_RESUME" == 0 ]]; then
  if [[ ! -f "$TRAIN_MANIFEST" || ! -f "$EVAL_MANIFEST" ]]; then
    "$PY_BOOT" "$RUN/anchor_bank.py" prepare-lazy \
      --train-proposal "$TRAIN_PROPOSAL" --heldout-proposal "$HELDOUT_PROPOSAL" \
      --train-output "$TRAIN_MANIFEST" --heldout-output "$EVAL_MANIFEST"
  fi
  # Older lazy manifests may predate casebrief propagation. Repair only those
  # adjacent inputs in place, preserving any existing verified anchor fields,
  # before the evaluator-role smoke loads all 32 real heldout tasks.
  "$PY_BOOT" "$RUN/anchor_bank.py" ensure-casebriefs \
    --train-manifest "$TRAIN_MANIFEST" --heldout-manifest "$EVAL_MANIFEST" \
    --proposal-root "$RUN/curriculum" >"$RUN/casebrief-repair.json"
fi
RENDERER="${PAINTER_RENDERER_PATH:-$RUN/vendor/integrations/watercolour/renderer.py}"
RENDERER_PYTHON="${PAINTER_RENDERER_PYTHON_PATH:-$RUN/renderer-env/bin/python}"
[[ -f "$RENDERER" && -x "$RENDERER_PYTHON" ]] || { echo 'explicit renderer runtime is unavailable' >&2; exit 1; }
PROPOSED="$RUN/native.toml.proposed.$$"
trap 'rm -f "$PROPOSED"' EXIT
CONFIG_ARGS=(
  --output "$PROPOSED" --model "$MODEL_DIR" --run-root "$RUN" --manifest "$TRAIN_MANIFEST" --prime-root "$PRIME_ROOT"
  --expected-count 96 --max-steps 20 --checkpoint-interval 4 --run-name "$PAINTER_RUN_NAME" --output-root rl-rollouts --max-turns 3
  --judge-protocol "$PAINTER_JUDGE_PROTOCOL" --adapter-revision "$PAINTER_POLICY_REVISION" --adapter-prefix "$PAINTER_POLICY_PREFIX" --adapter-sha256 "$PAINTER_POLICY_SHA256"
  --mode recovered_policy --learning-rate 1e-6 --gpus-per-node 4 --num-train-gpus 2 --num-infer-gpus 2 --batch-size 16 --group-size 4
  --initial-inflight 32 --max-inflight 32 --renderer "$RENDERER" --renderer-python "$RENDERER_PYTHON"
)
[[ "$PAINTER_ACTIVATION_CPU_OFFLOAD" == 1 ]] || CONFIG_ARGS+=(--no-activation-cpu-offload)
[[ "$PAINTER_STRUCTURED_FLOW_POLICY" == 1 ]] && CONFIG_ARGS+=(--enable-flow-policy)
[[ "$PAINTER_CODE_SUPERVISION_ENABLED" == 1 ]] && CONFIG_ARGS+=(--enable-code-supervision)
python3 - "$TRAIN_MANIFEST" <<'PY'
import json, sys
from pathlib import Path
value=json.loads(Path(sys.argv[1]).read_text())
rows=value.get("rows")
if not isinstance(rows,list) or len(rows)!=96:
    raise SystemExit("lazy Diverse-flow train manifest must contain 96 rows")
PY
[[ "$(python3 - "$TRAIN_MANIFEST" <<'PY'
import json,sys
from pathlib import Path
v=json.loads(Path(sys.argv[1]).read_text())
print("1" if v.get("status")=="pending_runtime_frozen_anchors" else "0")
PY
)" == 1 ]] && CONFIG_ARGS+=(--allow-pending-anchors)
"$PY_PRIME" "$RUN/prepare_structured_config.py" "${CONFIG_ARGS[@]}"
if [[ -f "$RUN/native.toml" ]]; then
  cmp -s "$PROPOSED" "$RUN/native.toml" || { echo 'existing native.toml differs from frozen structured proposal' >&2; exit 1; }
else
  mv "$PROPOSED" "$RUN/native.toml"
fi
rm -f "$PROPOSED"
trap - EXIT
if [[ "$PAINTER_RUN_MATCHED_BASELINE" == 1 ]]; then
  bash "$RUN/matched_baseline.sh"
  # Anchor admission may censor only exhausted model-invalid train cases.
  # Rebind Prime's expected count and receipt before warmup or any later
  # node validation can consume the pre-admission 32-row config.
  "$PY_PRIME" "$RUN/run_from_sft.py" --refresh-final-config "$RUN/native.toml" "$TRAIN_MANIFEST"
fi
# Produce the rich multimodal warmup rows on the Linux node when the
# controller did not provide an already reviewed set.  supervision_phase.py
# owns the single render pass for all 32 authored programs, sends the
# initial/final pairs to the pinned generic Flash side judge, retains raw
# decisions, and only then exposes filtered train/dev JSONL to warmup.sh.
SUPERVISION_STATUS="not_requested"
export SUPERVISION_STATUS
if [[ "${PAINTER_RUN_WARMUP:-1}" == 1 && "$PAINTER_CODE_SUPERVISION_ENABLED" == 1 && "$PAINTER_RUN_SUPERVISION" == 1 && \
      ( -z "${PAINTER_WARMUP_TRAIN_JSONL:-}" || -z "${PAINTER_WARMUP_DEV_JSONL:-}" ) ]]; then
  SUPERVISION_ROOT="${PAINTER_SUPERVISION_ROOT:-$RUN/code-supervision}"
  FLOW_SOURCE_MANIFEST="$SUPERVISION_ROOT/manifest.json"
  FLOW_PHASE_RECEIPT="$SUPERVISION_ROOT/supervision-phase.json"
  mkdir -p "$SUPERVISION_ROOT"
  # The approved source tree is reused by hash and renderer identity; this
  # default keeps the launch deterministic without rerendering paid pairs.
  RESTORE_SUPERVISION_EVIDENCE="${PAINTER_RESTORE_SUPERVISION_EVIDENCE:-1}"
  [[ "$RESTORE_SUPERVISION_EVIDENCE" == 0 || "$RESTORE_SUPERVISION_EVIDENCE" == 1 ]] || {
    echo 'PAINTER_RESTORE_SUPERVISION_EVIDENCE must be 0 or 1' >&2; exit 1;
  }
  if [[ "$RESTORE_SUPERVISION_EVIDENCE" == 1 ]]; then
    SUPERVISION_RESTORE_SHA256="${PAINTER_SUPERVISION_EVIDENCE_SHA256:-ab2531f9f12032b9c11e75cbeb681bdf08d084a7b1458864db050493292ccb67}"
  # This is the reviewed source archive's original run path.  Its metadata
  # remains structured-flow provenance; the restore receipt records that
  # provenance rather than rewriting it as a Diverse-flow artifact.
  SUPERVISION_RESTORE_URL="${PAINTER_SUPERVISION_EVIDENCE_URL:-https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/resolve/ca0a838605dfd8cfb75a54dde2d480a94bcbef66/runs/structured-flow-rl-20261009/evidence.tar.gz}"
    SUPERVISION_RESTORE_SCRIPT="$RUN/restore_anchor_evidence.py"
    [[ -f "$SUPERVISION_RESTORE_SCRIPT" ]] || { echo 'supervision evidence restore helper is missing' >&2; exit 1; }
    SUPERVISION_RESTORE_ARGS=(--mode supervision --run-root "$RUN" --sha256 "$SUPERVISION_RESTORE_SHA256" --renderer-sha256 "$PAINTER_RENDERER_SHA256")
    if [[ -n "${PAINTER_SUPERVISION_EVIDENCE_ARCHIVE:-}" ]]; then
      SUPERVISION_RESTORE_ARGS+=(--archive "$PAINTER_SUPERVISION_EVIDENCE_ARCHIVE")
    else
      SUPERVISION_RESTORE_ARGS+=(--url "$SUPERVISION_RESTORE_URL")
    fi
    PYTHONPATH="$REPO:$REPO/painter:$RUN${PYTHONPATH:+:$PYTHONPATH}" \
      "$PY_PRIME" "$SUPERVISION_RESTORE_SCRIPT" "${SUPERVISION_RESTORE_ARGS[@]}" \
      >"${PAINTER_SUPERVISION_RESTORE_SUMMARY:-$RUN/supervision-restore-summary.json}"
  fi
  if [[ ! -f "$FLOW_SOURCE_MANIFEST" ]]; then
    # Apply the same outside-the-output-directory rule as the curriculum
    # builder.  This keeps retries and clean-node preparation deterministic
    # even when flow_demonstrations.py later grows a strict output guard.
    FLOW_BUILD_TMP="$RUN/.flow-supervision-build.json.$$"
    if PYTHONPATH="$REPO:$REPO/painter:$RUN${PYTHONPATH:+:$PYTHONPATH}" \
      "$PY_PRIME" "$RUN/flow_demonstrations.py" build --output "$SUPERVISION_ROOT" \
      >"$FLOW_BUILD_TMP"; then
      mv "$FLOW_BUILD_TMP" "${PAINTER_FLOW_BUILD_RECEIPT:-$RUN/flow-supervision-build.json}"
    else
      rm -f "$FLOW_BUILD_TMP"
      exit 1
    fi
  fi
  if [[ "$RESTORE_SUPERVISION_EVIDENCE" == 1 && -f "$SUPERVISION_ROOT/render-manifest.json" && -f "$SUPERVISION_ROOT/approval-candidates.json" ]]; then
    # Reuse only the restored paid render/judge facts.  This validates the
    # current flow prompt/source/renderer hashes, then regenerates the
    # bounded 15/2 rich export without rendering or calling Flash, even if an
    # older archive also carried filtered JSONL from a previous gate.
    PYTHONPATH="$REPO:$REPO/painter:$RUN${PYTHONPATH:+:$PYTHONPATH}" \
      "$PY_PRIME" "$RUN/supervision_phase.py" --data "$SUPERVISION_ROOT" \
      --renderer-sha256 "$PAINTER_RENDERER_SHA256" --reuse-approved \
      >"$SUPERVISION_ROOT/supervision.json"
  elif [[ ! -f "$FLOW_PHASE_RECEIPT" || ! -s "$SUPERVISION_ROOT/train.jsonl" || ! -s "$SUPERVISION_ROOT/dev.jsonl" ]]; then
    GATEWAY="$PAINTER_GATEWAY_ENV_FILE"
    [[ -f "$GATEWAY" && ! -L "$GATEWAY" ]] || {
      echo 'code supervision requires the owner-only generic Flash gateway env file' >&2; exit 1;
    }
    PYTHONPATH="$REPO:$REPO/painter:$RUN${PYTHONPATH:+:$PYTHONPATH}" \
      "$PY_PRIME" "$RUN/supervision_phase.py" --data "$SUPERVISION_ROOT" \
      --renderer "$PAINTER_RENDERER_PATH" --renderer-python "$PAINTER_RENDERER_PYTHON_PATH" \
      --renderer-sha256 "$PAINTER_RENDERER_SHA256" --gateway-env "$GATEWAY" \
      --workers "${PAINTER_SUPERVISION_WORKERS:-4}" \
      >"$SUPERVISION_ROOT/supervision.json"
  fi
  [[ -s "$SUPERVISION_ROOT/train.jsonl" && -s "$SUPERVISION_ROOT/dev.jsonl" && -f "$FLOW_PHASE_RECEIPT" ]] || {
    echo 'Flash-reviewed supervision did not produce rich train/dev JSONL' >&2; exit 1;
  }
  SUPERVISION_STATUS="reviewed_ready_for_warmup"
  export SUPERVISION_STATUS
  export PAINTER_WARMUP_TRAIN_JSONL="$SUPERVISION_ROOT/train.jsonl"
  export PAINTER_WARMUP_DEV_JSONL="$SUPERVISION_ROOT/dev.jsonl"
  export PAINTER_WARMUP_APPROVAL_MANIFEST="$SUPERVISION_ROOT/approval-candidates.json"
  export PAINTER_WARMUP_SUPERVISION_RECEIPT="$FLOW_PHASE_RECEIPT"
else
  if [[ -n "${PAINTER_WARMUP_TRAIN_JSONL:-}" && -n "${PAINTER_WARMUP_DEV_JSONL:-}" ]]; then
    SUPERVISION_STATUS="external_reviewed_inputs"
    export SUPERVISION_STATUS
  elif [[ "$PAINTER_CODE_SUPERVISION_ENABLED" == 0 || "$PAINTER_RUN_SUPERVISION" == 0 ]]; then
    SUPERVISION_STATUS="disabled_explicitly"
    export SUPERVISION_STATUS
  fi
fi
# Code supervision is a real four-step multimodal SFT warmup, not a prompt
# flag. Require its approved rich train/dev JSONL by default. An operator who
# deliberately accepts an RL-only fallback must set PAINTER_WARMUP_REQUIRED=0
# and receives an explicit pending receipt below.
WARMUP_REQUIRED="${PAINTER_WARMUP_REQUIRED:-1}"
[[ "$WARMUP_REQUIRED" == 0 || "$WARMUP_REQUIRED" == 1 ]] || {
  echo 'PAINTER_WARMUP_REQUIRED must be 0 or 1' >&2; exit 1;
}
if [[ -r "$RUN/active-policy-adapter-path.txt" && -r "$RUN/active-policy.json" ]]; then
  echo 'DIVERSE_FLOW_WARMUP_REUSED: verified active adapter already staged'
  export PAINTER_WARMUP_STATUS=verified
elif [[ "${PAINTER_RUN_WARMUP:-1}" == 1 ]]; then
  if [[ -n "${PAINTER_WARMUP_TRAIN_JSONL:-}" && -n "${PAINTER_WARMUP_DEV_JSONL:-}" ]]; then
    bash "$RUN/warmup.sh"
    export PAINTER_WARMUP_STATUS=verified
  elif [[ "$WARMUP_REQUIRED" == 1 ]]; then
    echo 'approved rich warmup JSONL is required; set PAINTER_WARMUP_TRAIN_JSONL and PAINTER_WARMUP_DEV_JSONL' >&2
    exit 1
  else
    echo 'DIVERSE_FLOW_WARMUP_PENDING: explicit RL-only fallback requested' >&2
    export PAINTER_WARMUP_STATUS=pending_explicit_rl_only
  fi
elif [[ "$WARMUP_REQUIRED" == 1 ]]; then
  echo 'PAINTER_RUN_WARMUP=0 requires PAINTER_WARMUP_REQUIRED=0 to record an explicit pending warmup' >&2
  exit 1
else
  export PAINTER_WARMUP_STATUS=pending_explicit_rl_only
fi
# Keep the intervention attribution explicit: the fresh 32-case panel is
# evaluated once after the verified four-update code-flow warmup and before
# the RL optimizer starts.  Evaluation-only and an explicitly requested
# RL-only fallback record a skipped phase rather than silently calling it a
# post-warmup result.
POST_WARMUP_STATUS="not_requested"
if [[ "${PAINTER_RUN_POST_WARMUP_EVAL:-1}" == 1 ]]; then
  if [[ "${PAINTER_WARMUP_STATUS:-}" != verified || ! -r "$RUN/active-policy-adapter-path.txt" || ! -r "$RUN/active-policy.json" ]]; then
    echo 'post-warmup evaluation requires a verified warmup adapter; no RL update may start' >&2
    exit 1
  fi
  bash "$RUN/matched_baseline.sh" --post-warmup
  POST_WARMUP_STATUS="complete"
elif [[ "${PAINTER_RUN_POST_WARMUP_EVAL:-1}" == 0 && "${PAINTER_WARMUP_STATUS:-}" == pending_explicit_rl_only ]]; then
  POST_WARMUP_STATUS="skipped_explicit_rl_only"
else
  echo 'PAINTER_RUN_POST_WARMUP_EVAL must be 0 or 1' >&2
  exit 1
fi
export POST_WARMUP_STATUS
"$PY_BOOT" - "$RUN" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
run = Path(sys.argv[1])
config = run / "native.toml"
receipt = json.loads((run / "rl-config-receipt.json").read_text())
manifest = json.loads((run / "curriculum/pilot-train-manifest.json").read_text())
train_rows = manifest.get("rows")
pilot_train_rows = len(train_rows) if isinstance(train_rows, list) else 0
anchors_verified = str(manifest.get("status", "")).startswith("admitted_with_verified_frozen_sft512_anchors")
value = {
    "schema": "Diverse-flow-rl-preparation-v1",
    "run_name": run.name,
    "profile": "Diverse-flow-rl-v1",
    "status": "prepared_with_verified_anchors" if anchors_verified else "pending_runtime_anchor_generation",
    "base_model_revision": receipt.get("base_model_revision"),
    "prime_revision": receipt.get("prime_revision"),
    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    "pilot_train_rows": pilot_train_rows or 96,
    "matched_heldout_cases": 32,
    "anchor_bank_status": "verified_pilot" if anchors_verified else "pending_runtime_generation",
    "warmup_status": os.environ.get("PAINTER_WARMUP_STATUS", "unknown"),
    "post_warmup_evaluation_status": os.environ.get("POST_WARMUP_STATUS", "unknown"),
    "supervision_status": os.environ.get("SUPERVISION_STATUS", "unknown"),
    "structured_flow_policy_enabled": os.environ.get("PAINTER_STRUCTURED_FLOW_POLICY") == "1",
    "code_supervision_enabled": os.environ.get("PAINTER_CODE_SUPERVISION_ENABLED") == "1",
}
(run / "preparation-receipt.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
# The generic finite controller records immutable identity immediately after
# lazy preflight. Stage aliases and pinned IDs now; run_from_sft.py replaces
# the aliases and refreshes manifest_sha256 after real anchors are generated,
# before Prime creates the orchestrator.
legacy = run / "anchors-frozen"
legacy.mkdir(parents=True, exist_ok=True)
train = Path(os.environ.get("PAINTER_RL_MANIFEST", str(run / "curriculum/pilot-train-manifest.json")))
heldout = Path(os.environ.get("PAINTER_EVAL_MANIFEST", str(run / "curriculum/matched32-manifest.json")))
(legacy / "train-manifest.json").write_bytes(train.read_bytes())
(legacy / "validation-manifest.json").write_bytes(heldout.read_bytes())
evaluation = run / "evaluation"
evaluation.mkdir(parents=True, exist_ok=True)
(evaluation / "matched32-manifest.json").write_bytes(heldout.read_bytes())
identity = {
    "schema": "Diverse-flow-source-identity-v1",
    "run_name": run.name,
    "model_revision": receipt["base_model_revision"],
    "initializer_revision": receipt["policy_initializer"]["revision"],
    "initializer_prefix": receipt["policy_initializer"]["prefix"],
    "initializer_sha256": receipt["policy_initializer"]["sha256"],
    "policy_model_revision": receipt["policy_initializer"]["revision"],
    "frozen_sft_model_revision": receipt["frozen_sft"]["revision"],
    "frozen_sft_initializer": receipt["frozen_sft"],
    "prime_revision": receipt["prime_revision"],
    "protocol": os.environ["PAINTER_JUDGE_PROTOCOL"],
}
identity_path = run / "source-identity.json"
if identity_path.exists():
    existing = json.loads(identity_path.read_text())
    if any(existing.get(key) != item for key, item in identity.items() if key in existing):
        raise SystemExit("existing source identity conflicts with structured pins")
else:
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
PY
echo 'DIVERSE_FLOW_PREPARED: lazy manifests/config staged; runtime will generate 128 verified anchors before the first RL update'
