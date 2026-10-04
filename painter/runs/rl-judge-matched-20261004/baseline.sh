#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$ROOT/../../.." && pwd -P)"
source "$ROOT/runtime_env.sh"
apply_matched_resource_limits
NODE_ROOT="${PAINTER_NODE_ROOT:-/root/painter}"
BASE="${PAINTER_BASE_RUN:-$REPO/painter/runs/text-sft512-consistent-20260930}"
SHARED="${PAINTER_MATCHED_SHARED_ROOT:-$ROOT/../rl-judge-matched-20261004-shared}"
BASELINE_ROOT="${PAINTER_MATCHED_BASELINE_ROOT:-$ROOT/../rl-judge-matched-20261004-baseline}"
NAME=rl-judge-matched-20261004-baseline
PY="${PRIME_ROOT:-$NODE_ROOT/prime}/.venv/bin/python"

[[ "$(uname -s)" == Linux ]] || { echo 'Linux GPU node required for baseline evaluation' >&2; exit 1; }
[[ -x "$PY" ]] || { echo "pinned Prime Python is unavailable: $PY" >&2; exit 1; }
[[ -r "$SHARED/policy-adapter-path.txt" && -r "$SHARED/policy-adapter-receipt.json" ]] || {
  echo 'RL16 initializer download receipt or path receipt is missing' >&2; exit 1;
}
INIT="$(tr -d '\r\n' < "$SHARED/policy-adapter-path.txt")"
PUBLIC_RECEIPT="$BASELINE_ROOT/public-checkpoint-receipt.json"
PUBLIC_PROOF="$ROOT/initializer-public-proof.json"
[[ -f "$PUBLIC_PROOF" ]] || { echo 'anonymous public initializer proof is missing' >&2; exit 1; }
"$PY" "$ROOT/verify_public_download.py" \
  --download-receipt "$SHARED/policy-adapter-receipt.json" \
  --adapter "$INIT" --output "$PUBLIC_RECEIPT" \
  --public-proof "$PUBLIC_PROOF" \
  --repo CK0607/qwen3.8-27b-brush-painting \
  --revision ac4d3eda8e5925e2d23992cfdc1a63620ebde573 \
  --prefix text-pairwise-scaled-20260929/step_16 \
  --expected-sha256 e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c
[[ -f "$SHARED/preparation-receipt.json" ]] || { echo 'shared preparation receipt is missing' >&2; exit 1; }
[[ ! -e "$BASELINE_ROOT/validation" ]] || { echo 'baseline output already exists; preserve it before retrying' >&2; exit 1; }
mkdir -p "$BASELINE_ROOT"
for item in anchors-frozen anchors-all vendor renderer-env browsers runtime; do
  [[ -e "$SHARED/$item" || -L "$SHARED/$item" ]] || { echo "shared input missing: $item" >&2; exit 1; }
  ln -s "$SHARED/$item" "$BASELINE_ROOT/$item"
done
mkdir -p "$BASELINE_ROOT/async-output/$NAME/artifacts/adapters"
ln -s "$INIT" "$BASELINE_ROOT/async-output/$NAME/artifacts/adapters/step_1"
ln -s "$ROOT/../rl-judge-matched-20261004/prepare_arm.sh" "$BASELINE_ROOT/prepare_arm.sh"
"$PY" - "$BASELINE_ROOT/baseline-identity.json.tmp" "$NAME" "$INIT" <<'PY'
import json, sys
from pathlib import Path
path, name, initializer = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
path.write_text(json.dumps({
    "schema": "matched-fresh-rl16-baseline-v1", "run_name": name,
    "mode": "recovered_policy", "display_checkpoint": "fresh_rl16_initializer",
    "trained": False, "optimizer_resume": False, "initializer_path": initializer,
    "initializer_revision": "ac4d3eda8e5925e2d23992cfdc1a63620ebde573",
    "initializer_prefix": "text-pairwise-scaled-20260929/step_16",
    "initializer_sha256": "e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c",
    "scope": "balanced16", "case_count": 16, "seed": 918210,
}, sort_keys=True, indent=2) + "\n")
PY
mv "$BASELINE_ROOT/baseline-identity.json.tmp" "$BASELINE_ROOT/baseline-identity.json"
export PAINTER_RUN_ROOT="$BASELINE_ROOT" PAINTER_RUN_NAME="$NAME" PAINTER_EVAL_MANIFEST="$BASELINE_ROOT/anchors-frozen/validation-matched16.json"
export PAINTER_EVAL_EXPECTED_COUNT=16 PAINTER_EVAL_SEED=918210 PAINTER_EVAL_SCOPE=balanced16
export PAINTER_EVAL_PUBLIC_RECEIPT="$PUBLIC_RECEIPT"
export PAINTER_RENDERER_PATH="$BASELINE_ROOT/runtime/watercolour-cancellable-centroid-v1/cancellable_watercolour_renderer.py"
export PAINTER_RENDERER_PYTHON_PATH="$BASELINE_ROOT/renderer-env/bin/python"
export PAINTER_JUDGE_PROTOCOL=text-paint-sft-pairwise-consistent-v1 PAINTER_JUDGE_FORMAT=identity-joint-side-schema-v2-generic-v1
EVAL_TIMEOUT="${PAINTER_EVAL_TIMEOUT_SECONDS:-1800}"
[[ "$EVAL_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo 'baseline evaluation timeout must be a positive integer' >&2; exit 1; }
timeout --signal=TERM --kill-after=30 "$EVAL_TIMEOUT" bash "$BASE/evaluate.sh" 1 balanced16

# Baseline evidence is published before the outer controller releases the
# node. The archive tool screens receipts and excludes symlinked inputs.
EVIDENCE="$BASELINE_ROOT/local-evidence"
ARCHIVE="$EVIDENCE/baseline-evidence.tar.gz"
UPLOAD="$ARCHIVE.upload.json"
TOKEN="${HF_TOKEN_PATH:-$NODE_ROOT/private/hf-token}"
REPO_ID="${PAINTER_EVIDENCE_REPO_ID:-CK0607/komorebi-painter-experiments}"
REMOTE_PATH="${PAINTER_BASELINE_EVIDENCE_REMOTE_PATH:-runs/rl-judge-matched-20261004/baseline-evidence.tar.gz}"
TIMEOUT="${PAINTER_ARCHIVE_TIMEOUT_SECONDS:-600}"
[[ "$TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo 'baseline archive timeout must be a positive integer' >&2; exit 1; }
[[ -r "$TOKEN" && ! -L "$TOKEN" && "$(stat -c %a "$TOKEN")" == 600 ]] || {
  echo 'owner-only HF token is required for baseline public archive' >&2; exit 1;
}
mkdir -p "$EVIDENCE"
  timeout --signal=TERM --kill-after=30 "$TIMEOUT" \
  "$PY" "$REPO/painter/runs/text-sft512-consistent-20260930/safe_archive.py" \
  --run-root "$BASELINE_ROOT" --output "$ARCHIVE" --token-file "$TOKEN" \
  --repo-id "$REPO_ID" --remote-path "$REMOTE_PATH" --verify-dir "$EVIDENCE/public-verify" \
  >"$EVIDENCE/archive.log" 2>&1
[[ -f "$UPLOAD" ]] || { echo 'baseline public archive receipt is missing' >&2; exit 1; }
"$PY" - "$UPLOAD" <<'PY'
import json, sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text())
if value.get("private") is not False or value.get("public_hash_verified") is not True:
    raise SystemExit("baseline archive was not anonymously verified as public")
PY
echo 'MATCHED_RL16_BASELINE_COMPLETE: balanced16 evaluation and public archive verified'
