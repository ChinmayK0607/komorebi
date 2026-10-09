#!/usr/bin/env bash
set -Eeuo pipefail

# Finite Linux-only takeover for the saved public checkpoint-4 adapter.  The
# parent controller owns pod allocation/release.  This path never calls the
# RL trainer or the generic post-training pipeline.
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$RUN/../../.." && pwd -P)"
START="${1:-}"; PRICE="${2:-}"; CAP="${3:-}"; MODE="${4:-}"
[[ $# == 3 || ( $# == 4 && "$MODE" == --preflight ) ]] || {
  echo 'usage: recover_evaluation.sh RENTAL_START_EPOCH PRICE_USD_PER_HOUR MAX_GPU_USD [--preflight]' >&2; exit 2;
}
source "$RUN/runtime_env.sh"
# This is a physical one-GPU evaluator.  The archived training receipt keeps
# its historical 2-GPU topology, while this opt-in branch validates the live
# node as zero-trainer/one-inference and never advertises training readiness.
export PAINTER_EVALUATION_ONLY=1 PAINTER_GPUS_PER_NODE=1 PAINTER_NUM_TRAIN_GPUS=0 PAINTER_NUM_INFER_GPUS=1
validate_structured_bindings

if [[ "$MODE" == --preflight ]]; then
  required=(prepare_node.sh evaluate.sh finalize.sh safe_archive.py evaluation_recovery.py build_recovery_review.py score_recovery.py)
  for item in "${required[@]}"; do [[ -f "$RUN/$item" ]] || { echo "missing recovery helper: $item" >&2; exit 1; }; done
  python3 - <<'PY'
import json
print(json.dumps({"schema":"structured-flow-evaluation-recovery-preflight-v1","step":4,"cases":16,"training":False,"trainer_gpus":0,"inference_gpus":1}, sort_keys=True))
PY
  exit 0
fi

[[ "$(uname -s)" == Linux ]] || { echo 'evaluation recovery requires a Linux GPU node' >&2; exit 1; }

# A three-dollar cap is the approved recovery bound.  The reserve is only for
# archive/release work; no training reserve is carried into this evaluator.
RECOVERY_DEADLINE="$(python3 - "$START" "$PRICE" "$CAP" <<'PY'
import math, sys, time
start, price, cap = map(float, sys.argv[1:])
if any(not math.isfinite(v) or v <= 0 for v in (start, price, cap)):
    raise SystemExit("rental start, price, and cap must be positive finite numbers")
if cap > 3.0:
    raise SystemExit("evaluation recovery cap exceeds the approved $3.00 bound")
deadline = start + cap / price * 3600.0
if deadline - time.time() <= 360.0:
    raise SystemExit("remaining recovery window is shorter than the archive/release reserve")
print(f"{deadline:.6f}")
PY
)"

export PAINTER_RUN_ROOT="$RUN" PAINTER_RUN_NAME=structured-flow-rl-20261009
export PAINTER_EVALUATION_ONLY=1 PAINTER_EVAL_PREFERRED_STEP=4
export PAINTER_EVAL_SCOPE=all PAINTER_EVAL_EXPECTED_COUNT=16 PAINTER_EVAL_SEED=918210
export PAINTER_MAX_TURNS=3 PAINTER_EVAL_MAX_CONCURRENT=8
export PAINTER_JUDGE_MODEL=xiaomi/mimo-v2.6-flash PAINTER_JUDGE_EFFORT=medium
export PAINTER_JUDGE_FORMAT=identity-joint-side-schema-v2-generic-v1
export PAINTER_JUDGE_PROTOCOL=text-paint-sft-pairwise-consistent-v1

# Restore native anchors from the latest full public evidence archive and the
# paid baseline from its dedicated archive. Both paths preserve failed and
# censored receipts and make zero generation calls during restore.
export PAINTER_RESTORE_ANCHOR_EVIDENCE=1
export PAINTER_ANCHOR_EVIDENCE_SHA256=9ec7942d8666f09ff794d635d306b6f1cfb6d440f55186fec32f6c1befb5ffdb
export PAINTER_ANCHOR_EVIDENCE_URL="https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/resolve/cec8e7133738b03337ee4cae3f1c3fffdd91bc7f/runs/structured-flow-rl-20261009/evidence.tar.gz"
export PAINTER_REUSE_MATCHED_BASELINE_EVIDENCE=1
export PAINTER_MATCHED_BASELINE_EVIDENCE_SHA256=ff4efa37835b1a7e3c5e90a628d685f1e94f39cf3d13760e47abab834c1a90c3
export PAINTER_MATCHED_BASELINE_EVIDENCE_URL="https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/resolve/f72ee86cc24cb5fa3618c649aa1fe32ea22753ae/runs/structured-flow-rl-20261009/evidence.tar.gz"
export PAINTER_RUN_MATCHED_BASELINE=1

# Paired zero flags skip all optional warmup/supervision work. The existing
# prepare_node.sh still verifies the pinned runtime and renderer boundary.
export PAINTER_RUN_WARMUP=0 PAINTER_WARMUP_REQUIRED=0
export PAINTER_RUN_SUPERVISION=0 PAINTER_RESTORE_SUPERVISION_EVIDENCE=0
export PAINTER_CODE_SUPERVISION_ENABLED=0 PAINTER_STRUCTURED_FLOW_POLICY=1

PY_BOOT="${PAINTER_BOOTSTRAP_PYTHON:-${PRIME_ROOT:-}/.venv/bin/python}"
[[ -x "$PY_BOOT" ]] || PY_BOOT=python3
archive_python() {
  if [[ -x "${PRIME_ROOT:-}/.venv/bin/python" ]]; then
    printf '%s\n' "${PRIME_ROOT}/.venv/bin/python"
  else
    printf '%s\n' "$PY_BOOT"
  fi
}

remaining_seconds() {
  python3 - "$RECOVERY_DEADLINE" <<'PY'
import sys, time
print(max(0, int(float(sys.argv[1]) - time.time())))
PY
}
ARCHIVE_ROOT="${PAINTER_RECOVERY_ARCHIVE_ROOT:-$RUN/local-evidence}"
ARCHIVE="$ARCHIVE_ROOT/${PAINTER_RUN_NAME}-evaluation4-evidence.tar.gz"
ARCHIVE_DONE=0
RECOVERY_STATUS=running
RECOVERY_EXIT_CODE=
archive_partial() {
  status=$?
  trap - EXIT
  if (( status != 0 && ARCHIVE_DONE == 0 )); then
    set +e
    mkdir -p "$ARCHIVE_ROOT"
    archive_budget=$(( $(remaining_seconds) - 120 ))
    archive_args=(--run-root "$RUN" --output "$ARCHIVE" \
      --repo-id CK0607/komorebi-painter-experiments \
      --remote-path "runs/$PAINTER_RUN_NAME/evaluation-step-4-evidence.tar.gz" \
      --verify-dir "$ARCHIVE_ROOT/public-verify")
    if [[ -n "${PAINTER_RECOVERY_TOKEN_FILE:-}" ]]; then archive_args+=(--token-file "$PAINTER_RECOVERY_TOKEN_FILE"); fi
    archive_python_path="$(archive_python)"
    if (( archive_budget > 0 )); then
      timeout --signal TERM --kill-after 30 "$archive_budget" "$archive_python_path" "$RUN/safe_archive.py" "${archive_args[@]}" >"$ARCHIVE_ROOT/partial-archive.json" 2>&1
    else
      echo 'archive window exhausted' >"$ARCHIVE_ROOT/partial-archive.json"
    fi
    set -e
  fi
  RECOVERY_EXIT_CODE="$status"
  if (( status == 0 && ARCHIVE_DONE == 1 )); then RECOVERY_STATUS=complete; else RECOVERY_STATUS=failed; fi
  mkdir -p "$ARCHIVE_ROOT"
  set +e
  archive_python_path="$(archive_python)"
  "$archive_python_path" - "$RUN/terminal-receipt.json" "$RECOVERY_STATUS" "$status" "$ARCHIVE" "$RUN/async-output/$PAINTER_RUN_NAME/artifacts/adapters/step_4/hf-upload.json" "$START" "$PRICE" "$CAP" <<'PY'
import hashlib, json, sys
from pathlib import Path

out = Path(sys.argv[1])
status = sys.argv[2]
exit_code = int(sys.argv[3])
archive = Path(sys.argv[4])
checkpoint = Path(sys.argv[5])
rental_start = float(sys.argv[6])
price = float(sys.argv[7])
cap = float(sys.argv[8])

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

report = {}
for candidate in (archive.with_name("archive.json"), archive.with_name("partial-archive.json")):
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        continue
    if isinstance(value, dict):
        report = value
        break
archive_hash = digest(archive) if archive.is_file() and not archive.is_symlink() else None
archive_verified = bool(archive_hash and report.get("sha256") == archive_hash)
try:
    upload = json.loads(checkpoint.read_text(encoding="utf-8"))
except (OSError, UnicodeDecodeError, json.JSONDecodeError):
    upload = {}
checkpoint_public = (
    isinstance(upload, dict)
    and upload.get("provenance_type") == "public_download"
    and upload.get("verified") is True
    and upload.get("private") is False
    and upload.get("public_metadata_verified") is True
)
payload = {
    "schema": "structured-flow-evaluation-terminal-v1",
    "status": status,
    "exit_code": exit_code,
    "mode": "evaluation_only",
    "rental": {"start_epoch": rental_start, "price_usd_per_hour": price, "cap_usd": cap},
    "training": {"started": False, "new_optimizer_updates": 0},
    "topology": {"gpus_per_node": 1, "trainer_gpus": 0, "inference_gpus": 1, "evaluation_gpu": 0},
    "checkpoint4_public_complete": checkpoint_public,
    "checkpoint4_receipt": str(checkpoint),
    "archive": {
        "path": str(archive),
        "sha256": archive_hash,
        "hash_verified": archive_verified,
        "public_hash_verified": bool(report.get("public_hash_verified", False)),
        "report": report,
    },
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  set -e
  exit "$status"
}
trap archive_partial EXIT

# This is the only preparation call. It restores the paid baseline/anchors,
# downloads the small runtime inputs, and records warmup=0/warmup-required=0.
prepare_budget=$(( $(remaining_seconds) - 300 ))
(( prepare_budget > 0 )) || { echo 'evaluation window exhausted before node preparation' >&2; exit 1; }
timeout --signal TERM --kill-after 60 "$prepare_budget" bash "$RUN/prepare_node.sh"
PY_PRIME="${PRIME_ROOT:-}/.venv/bin/python"
[[ -x "$PY_PRIME" ]] || PY_PRIME="$PY_BOOT"
# Bootstrap may have created the pinned Prime environment; prefer it for the
# Hugging Face downloader, staging receipt, and archive dependency set.
PY_BOOT="$PY_PRIME"

DOWNLOAD="${PAINTER_POLICY_DOWNLOAD_SCRIPT:-$REPO/painter/runs/text-pairwise-rl-20260928/download_public_adapter.py}"
DOWNLOAD_RECEIPT="${PAINTER_EVAL_CHECKPOINT_DOWNLOAD_RECEIPT:-$RUN/evaluation-checkpoint4-download.json}"
ADAPTER_SOURCE="${PAINTER_EVAL_ADAPTER_SOURCE:-}"
if [[ -z "$ADAPTER_SOURCE" ]]; then
  [[ -f "$DOWNLOAD" ]] || { echo 'pinned public adapter downloader is missing' >&2; exit 1; }
  download_budget=$(( $(remaining_seconds) - 300 ))
  (( download_budget > 0 )) || { echo 'evaluation window exhausted before checkpoint download' >&2; exit 1; }
  timeout --signal TERM --kill-after 60 "$download_budget" "$PY_BOOT" "$DOWNLOAD" --revision d4ae68d425b5728ffb69aeabce82dd54231486f2 --prefix structured-flow-rl-20261009/step_4 --expected-sha256 51e35b6fb48c2cf019309f928f292cd9852cfdd2af29abe3f8714d6c429922bf --receipt "$DOWNLOAD_RECEIPT"
  ADAPTER_SOURCE="$("$PY_BOOT" - "$DOWNLOAD_RECEIPT" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["adapter_path"])
PY
)"
fi
"$PY_PRIME" "$RUN/evaluation_recovery.py" stage-checkpoint --run-root "$RUN" --download-receipt "$DOWNLOAD_RECEIPT" --adapter-source "$ADAPTER_SOURCE" --step 4 >"$RUN/evaluation-checkpoint4-stage.json"

RECOVERY_SOURCE_COMMIT="${PAINTER_SOURCE_COMMIT:-a7dd0fa}"
"$PY_PRIME" "$RUN/evaluation_recovery.py" write-receipt --run-root "$RUN" --source-commit "$RECOVERY_SOURCE_COMMIT" --manifest "$RUN/curriculum/matched16-manifest.json" --renderer "${PAINTER_RENDERER_PATH:-$RUN/runtime/watercolour-cancellable-centroid-v1/cancellable_watercolour_renderer.py}" --output "$RUN/evaluation-recovery-receipt.json"

# Pass exactly STEP and SCOPE; evaluate.sh owns the fixed argument contract.
# Keep a hard deadline for generation so archive/release still have the
# reserved tail of the approved rental window.
remaining="$(remaining_seconds)"
eval_budget=$((remaining - 300))
(( eval_budget > 0 )) || { echo 'evaluation window exhausted before generation' >&2; exit 1; }
timeout --signal TERM --kill-after 60 "$eval_budget" bash "$RUN/evaluate.sh" 4 all

# The target wrapper routes this to finalize_evaluation_recovery.py. It does
# not invoke the generic SFT512-only finalizer.
PAINTER_EVALUATION_ONLY=1 bash "$RUN/finalize.sh" 4
"$PY_PRIME" "$RUN/build_recovery_review.py" --manifest "$RUN/curriculum/matched16-manifest.json" --baseline-root "$RUN/matched-baseline/pre-step12/evaluation-rollouts" --candidate-root "$RUN/validation/step-4-all/eval" --output "$RUN/recovery-review"

# Build screened local evidence. Publishing is explicit and requires a caller
# supplied owner-only token path; credentials are never read by default.
mkdir -p "$ARCHIVE_ROOT"
ARCHIVE_ARGS=(--run-root "$RUN" --output "$ARCHIVE" --repo-id CK0607/komorebi-painter-experiments --remote-path "runs/$PAINTER_RUN_NAME/evaluation-step-4-evidence.tar.gz" --verify-dir "$ARCHIVE_ROOT/public-verify")
if [[ -n "${PAINTER_RECOVERY_TOKEN_FILE:-}" ]]; then ARCHIVE_ARGS+=(--token-file "$PAINTER_RECOVERY_TOKEN_FILE"); fi
archive_budget=$(( $(remaining_seconds) - 120 ))
(( archive_budget > 0 )) || { echo 'evaluation window exhausted before archive' >&2; exit 1; }
timeout --signal TERM --kill-after 30 "$archive_budget" "$PY_PRIME" "$RUN/safe_archive.py" "${ARCHIVE_ARGS[@]}" >"$ARCHIVE_ROOT/archive.json"
ARCHIVE_DONE=1
echo "STRUCTURED_FLOW_EVALUATION_RECOVERY_COMPLETE: step=4 archive=$ARCHIVE"
