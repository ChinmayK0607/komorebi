#!/usr/bin/env bash
set -Eeuo pipefail

RUN="${PAINTER_RUN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)}"
REPO="$(cd "$RUN/../../.." && pwd -P)"
NODE_ROOT="${PAINTER_NODE_ROOT:-/root/painter}"
PRIME_ROOT="${PRIME_ROOT:-$NODE_ROOT/prime}"
START="${1:?usage: execute_pipeline.sh RENTAL_START_EPOCH PRICE_USD_PER_HOUR MAX_GPU_USD}"
PRICE="${2:?usage: execute_pipeline.sh RENTAL_START_EPOCH PRICE_USD_PER_HOUR MAX_GPU_USD}"
CAP="${3:?usage: execute_pipeline.sh RENTAL_START_EPOCH PRICE_USD_PER_HOUR MAX_GPU_USD}"
[[ $# == 3 ]] || { echo 'usage: execute_pipeline.sh RENTAL_START_EPOCH PRICE_USD_PER_HOUR MAX_GPU_USD' >&2; exit 2; }
RESERVE="${PAINTER_RESERVE_SECONDS:-2700}"
GROUP_SIZE="${PAINTER_GROUP_SIZE:-8}"
EVAL_TIMEOUT="${PAINTER_EVAL_TIMEOUT_SECONDS:-1800}"
INITIAL_METRIC_TIMEOUT="${PAINTER_INITIAL_METRIC_TIMEOUT_SECONDS:-1800}"
ARCHIVE_TIMEOUT="${PAINTER_ARCHIVE_TIMEOUT_SECONDS:-600}"
PARENT_RELEASE_RESERVE="${PAINTER_PARENT_RELEASE_RESERVE_SECONDS:-180}"
COMPLETION_MARGIN="${PAINTER_COMPLETION_MARGIN_SECONDS:-120}"
HF_TOKEN_PATH="${HF_TOKEN_PATH:-$NODE_ROOT/private/hf-token}"
HF_REPO="${PAINTER_EVIDENCE_REPO_ID:-CK0607/komorebi-painter-experiments}"
HF_PATH="${PAINTER_EVIDENCE_REMOTE_PATH:-runs/${PAINTER_RUN_NAME:-text-sft512-consistent-20260930}/evidence.tar.gz}"
NAME="${PAINTER_RUN_NAME:-text-sft512-consistent-20260930}"
METRICS="$RUN/async-output/$NAME/monitors/file/metrics.jsonl"
STATE="$RUN/supervisor-state.json"
TREND="$RUN/reward-trend.jsonl"
TRAINER_LOG="${PAINTER_TRAINER_LOG:-$RUN/async-output/$NAME/logs/attempt_1/trainer.log}"
EVIDENCE="$RUN/local-evidence"
ARCHIVE="$EVIDENCE/$NAME-evidence.tar.gz"
UPLOAD="$ARCHIVE.upload.json"
TERMINAL="$RUN/terminal-receipt.json"
GUARD_RECEIPT="$RUN/consistent-group-admission-activation.json"
PY="$PRIME_ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY=python3
STATUS=0
TRAIN_STATUS=
EVAL_STATUS=
FINALIZE_STATUS=
STEP=

"$PY" - "$START" "$PRICE" "$CAP" "$RESERVE" "$EVAL_TIMEOUT" "$INITIAL_METRIC_TIMEOUT" \
    "$ARCHIVE_TIMEOUT" "$PARENT_RELEASE_RESERVE" "$COMPLETION_MARGIN" <<'PY'
import math, sys
start, price, cap, reserve, eval_timeout, initial_metric_timeout, archive_timeout, parent_reserve, margin = map(float, sys.argv[1:])
if any(not math.isfinite(x) or x <= 0 for x in (start, price, cap, reserve)):
    raise SystemExit('rental start, price, cap, and reserve must be positive finite numbers')
if cap > 20:
    raise SystemExit('GPU cap exceeds the hard $20 limit')
if reserve >= cap / price * 3600:
    raise SystemExit('reserved completion window exceeds rental budget')
if any(not math.isfinite(x) or x <= 0 for x in (eval_timeout, initial_metric_timeout, archive_timeout, parent_reserve, margin)):
    raise SystemExit('completion timeout and release margin values must be positive finite numbers')
if initial_metric_timeout > 3600:
    raise SystemExit('initial metric timeout must be at most 3600 seconds')
if reserve < eval_timeout + archive_timeout + parent_reserve + margin:
    raise SystemExit('completion reserve is shorter than evaluation, archive, release, and margin bounds')
PY
DEADLINE="$($PY - "$START" "$PRICE" "$CAP" <<'PY'
import sys
print(float(sys.argv[1]) + float(sys.argv[3]) / float(sys.argv[2]) * 3600)
PY
)"

write_identity() {
  local commit="${PAINTER_SOURCE_COMMIT:-}"
  [[ "$commit" =~ ^[0-9a-fA-F]{7,64}$ ]] || {
    echo 'PAINTER_SOURCE_COMMIT must be the reviewed source Git commit' >&2
    return 1
  }
  "$PY" - "$RUN" "$REPO" "$commit" <<'PY'
import hashlib, json, sys
from pathlib import Path
run, repo, commit = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
identity_path = run / 'source-identity.json'
identity = json.loads(identity_path.read_text())
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
files = {
    'config_sha256': run / 'native.toml',
    'manifest_sha256': run / 'anchors-frozen/train-manifest.json',
    'foundation_lock_sha256': repo / 'painter/foundation-lock.json',
}
if any(not path.is_file() for path in files.values()):
    raise SystemExit('immutable config, manifest, or foundation lock is missing')
identity.update({key: sha(path) for key, path in files.items()})
policy_revision = (identity.get('initializer_revision') or identity.get('policy_initializer', {}).get('revision')
                   or identity.get('model_revision'))
frozen_revision = (identity.get('frozen_sft_model_revision') or identity.get('sft_model_revision')
                   or identity.get('frozen_sft_initializer', {}).get('revision') or policy_revision)
if not policy_revision or not frozen_revision:
    raise SystemExit('pinned policy and frozen SFT model revisions are missing')
identity.setdefault('source_git_commit', identity.get('source_commit', commit))
identity['completion_source_git_commit'] = commit
identity['policy_model_revision'] = policy_revision
identity['sft_model_revision'] = frozen_revision
tmp = identity_path.with_suffix('.json.tmp')
tmp.write_text(json.dumps(identity, indent=2, sort_keys=True) + '\n')
tmp.replace(identity_path)
PY
}

refresh_prime_python() {
  local candidate="$PRIME_ROOT/.venv/bin/python"
  [[ -x "$candidate" ]] || {
    echo "pinned Prime Python missing after preparation: $candidate" >&2
    return 1
  }
  PY="$candidate"
}

record_post_identity() {
  "$PY" - "$RUN" "${PAINTER_SOURCE_COMMIT:-}" <<'PY'
import json
import re
import sys
from pathlib import Path

run, supplied = Path(sys.argv[1]), sys.argv[2]
path = run / 'source-identity.json'
identity = json.loads(path.read_text())
completion = supplied or identity.get('completion_source_git_commit') or identity.get('source_commit') or identity.get('source_git_commit')
if not isinstance(completion, str) or not re.fullmatch(r'[0-9a-fA-F]{7,64}', completion):
    raise SystemExit('post-training completion source commit is missing or invalid')
source_commit = identity.get('source_commit') or identity.get('source_git_commit')
model_revision = (identity.get('model_revision') or identity.get('policy_model_revision')
                  or identity.get('initializer_revision') or identity.get('policy_initializer', {}).get('revision'))
frozen_revision = (identity.get('frozen_sft_model_revision') or identity.get('sft_model_revision')
                   or identity.get('frozen_sft_initializer', {}).get('revision') or model_revision)
required = (source_commit, identity.get('config_sha256'), identity.get('manifest_sha256'), identity.get('foundation_lock_sha256'), model_revision, frozen_revision)
if any(not item for item in required):
    raise SystemExit('source identity is incomplete; immutable source IDs must be recorded before post-training finalization')
# Preserve source_commit, config, manifest, and foundation fields exactly as
# recorded before training; add only the completion provenance field.
identity['completion_source_git_commit'] = completion
temporary = path.with_name(path.name + '.tmp')
temporary.write_text(json.dumps(identity, indent=2, sort_keys=True) + '\n')
temporary.replace(path)
PY
}

write_terminal() {
  local original_rc="$1" archive_rc="$2" archive_uploaded="$3"
  "$PY" - "$RUN" "$TERMINAL" "$original_rc" "$archive_rc" "$archive_uploaded" \
    "${TRAIN_STATUS:-}" "${EVAL_STATUS:-}" "${FINALIZE_STATUS:-}" "${STEP:-}" "$DEADLINE" <<'PY'
import hashlib
import json
import sys
import time
from pathlib import Path

run, target = Path(sys.argv[1]), Path(sys.argv[2])
original_rc, archive_rc, uploaded = sys.argv[3:6]
train_rc, eval_rc, finalize_rc, step, deadline = sys.argv[6:11]

def sha(path):
    if not path.is_file():
        return None
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

state = run / 'supervisor-state.json'
try:
    state_value = json.loads(state.read_text()) if state.is_file() else None
except (OSError, json.JSONDecodeError):
    state_value = None
report = {
    'schema': 'text-paint-consistent-terminal-v2',
    'run_name': run.name,
    'status': 'complete' if original_rc == '0' and archive_rc == '0' else 'failed',
    'pipeline_exit_code': int(original_rc),
    'archive_exit_code': int(archive_rc) if archive_rc else None,
    'public_archive_hash_verified': uploaded == '1',
    'train_exit_code': int(train_rc) if train_rc else None,
    'evaluation_exit_code': int(eval_rc) if eval_rc else None,
    'finalizer_exit_code': int(finalize_rc) if finalize_rc else None,
    'checkpoint_step': int(step) if step.isdigit() else None,
    'deadline_epoch': float(deadline),
    'source_identity_sha256': sha(run / 'source-identity.json'),
    'config_sha256': sha(run / 'native.toml'),
    'manifest_sha256': sha(run / 'anchors-frozen/train-manifest.json'),
    'supervisor_status': state_value.get('status') if isinstance(state_value, dict) else None,
    'created_epoch': time.time(),
}
temporary = target.with_name(target.name + '.tmp')
temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
temporary.replace(target)
PY
}

archive_rc=1
archive_uploaded=0
on_exit() {
  local original_rc=$?
  trap - EXIT INT TERM
  set +e
  # Write a pre-archive receipt so the evidence bundle contains a terminal
  # record even when a later upload or finalizer operation fails.
  write_terminal "$original_rc" '' 0
  mkdir -p "$EVIDENCE"
  rm -f "$UPLOAD"
  local archive_cmd=("$PY" "$RUN/safe_archive.py" --run-root "$RUN" --output "$ARCHIVE" \
    --repo-id "$HF_REPO" --remote-path "$HF_PATH" --verify-dir "$EVIDENCE/public-verify")
  local missing_token=0
  if [[ -r "$HF_TOKEN_PATH" ]]; then
    archive_cmd+=(--token-file "$HF_TOKEN_PATH")
  else
    missing_token=1
  fi
  timeout --signal=TERM --kill-after=30 "$ARCHIVE_TIMEOUT" "${archive_cmd[@]}" \
    >"$EVIDENCE/archive.log" 2>&1
  archive_rc=$?
  if (( missing_token )); then
    # A local package preserves failure evidence, but a successful run must
    # also prove the public anonymous upload before the node is released.
    archive_rc=3
  fi
  if [[ -f "$UPLOAD" ]]; then
    archive_uploaded=1
  fi
  if (( original_rc == 0 && archive_rc != 0 )); then
    original_rc=$archive_rc
  fi
  write_terminal "$original_rc" "$archive_rc" "$archive_uploaded"
  exit "$original_rc"
}
trap on_exit EXIT
trap 'exit 143' INT TERM

post_training_only="${PAINTER_PIPELINE_POST_TRAINING_ONLY:-0}"
if [[ "$post_training_only" == 1 ]]; then
  [[ -f "$STATE" ]] || { echo 'post-training mode requires supervisor-state.json' >&2; exit 1; }
  "$PY" - "$STATE" <<'PY'
import json, sys
from pathlib import Path
state = json.loads(Path(sys.argv[1]).read_text())
if state.get('status') not in {'stopped', 'completed'}:
    raise SystemExit('post-training mode requires a terminal finite supervisor state')
PY
  "$PY" - "$RUN" <<'PY'
import json, sys
from pathlib import Path
value = json.loads((Path(sys.argv[1]) / 'source-identity.json').read_text())
source_commit = value.get('source_git_commit') or value.get('source_commit')
model_revision = (value.get('model_revision') or value.get('policy_model_revision')
                  or value.get('initializer_revision') or value.get('policy_initializer', {}).get('revision'))
frozen_revision = (value.get('frozen_sft_model_revision') or value.get('sft_model_revision')
                   or value.get('frozen_sft_initializer', {}).get('revision') or model_revision)
required = (source_commit, value.get('config_sha256'), value.get('manifest_sha256'), value.get('foundation_lock_sha256'), model_revision, frozen_revision)
if any(not item for item in required):
    raise SystemExit('source identity is incomplete; parent must record immutable source IDs before post-training finalization')
PY
  if ! record_post_identity; then
    STATUS=1
    exit 1
  fi
  TRAIN_STATUS="$($PY - "$STATE" <<'PY'
import json, sys
value = json.loads(open(sys.argv[1]).read())
print(value.get('exit_code') if value.get('exit_code') is not None else '')
PY
)"
else
  # Preparation is finite and idempotently hash-verifies the frozen public
  # anchor archive.  It does not start training.
  if ! bash "$RUN/prepare_node.sh"; then
    STATUS=1
    exit 1
  fi
  if ! refresh_prime_python; then
    STATUS=1
    exit 1
  fi
  prove_guard() {
    rm -f "$GUARD_RECEIPT"
    PAINTER_PRIME_ROLE=orchestrator \
    PAINTER_JUDGE_PROTOCOL=text-paint-sft-pairwise-consistent-v1 \
    PAINTER_WHOLE_GROUP_ADMISSION=1 PAINTER_WHOLE_GROUP_SIZE="$GROUP_SIZE" \
    PAINTER_GROUP_RANK_MODULE=painter.consistent_group_admission \
    PAINTER_CONSISTENT_GUARD_RECEIPT="$GUARD_RECEIPT" \
    PYTHONPATH="$REPO:$REPO/painter:$RUN${PYTHONPATH:+:$PYTHONPATH}" \
      "$PY" - <<'PY'
import json
import os
from pathlib import Path
receipt = Path(os.environ['PAINTER_CONSISTENT_GUARD_RECEIPT'])
if not receipt.is_file():
    raise SystemExit('sitecustomize did not write guard activation receipt')
value = json.loads(receipt.read_text())
if value.get('role') != 'orchestrator' or value.get('protocol') != 'text-paint-sft-pairwise-consistent-v1':
    raise SystemExit('guard activation receipt has the wrong role or protocol')
expected = int(os.environ['PAINTER_WHOLE_GROUP_SIZE'])
if value.get('group_size') != expected or not (value.get('installed') or value.get('already_installed')):
    raise SystemExit(f'whole-group guard was not installed for group size {expected}')
print('CONSISTENT_GROUP_GUARD_CHILD_ACTIVATED')
PY
  }
  if ! prove_guard; then
    STATUS=1
    exit 1
  fi
  if ! PAINTER_RL_SUPERVISED=1 bash "$RUN/run.sh" --preflight; then
    STATUS=1
    exit 1
  fi
  if ! write_identity; then
    STATUS=1
    exit 1
  fi
  if PAINTER_RL_SUPERVISED=1 "$PY" "$RUN/finite_supervisor.py" \
      --metrics "$METRICS" --trainer-log "$TRAINER_LOG" --state "$STATE" --trend "$TREND" \
      --price-usd-per-hour "$PRICE" --rental-start-epoch "$START" \
      --max-gpu-usd "$CAP" --max-updates 50 --reserve-seconds "$RESERVE" \
      --metric-timeout-seconds 1800 --initial-metric-timeout-seconds "$INITIAL_METRIC_TIMEOUT" \
      -- bash "$RUN/run.sh"; then
    TRAIN_STATUS=0
  else
    TRAIN_STATUS=$?
    STATUS=1
  fi
fi

remaining_seconds() {
  "$PY" - "$DEADLINE" <<'PY'
import sys, time
print(max(0, int(float(sys.argv[1]) - time.time())))
PY
}

STEP="$($PY - "$RUN" "$EVIDENCE/latest-public-checkpoint.json" "${PAINTER_EVAL_PREFERRED_STEP:-}" <<'PY'
import json
import sys
from pathlib import Path
run, target, preferred_raw = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(run))
from safe_archive import select_latest_public_checkpoint, select_preferred_public_checkpoint
preferred = int(preferred_raw) if preferred_raw.isdigit() else None
if preferred is None:
    # Preserve the historical latest-verified selection for existing runs.
    selected = select_latest_public_checkpoint(run)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(selected or {}, indent=2, sort_keys=True) + '\n')
    print(selected['step'] if selected else '')
    raise SystemExit
selected, selection = select_preferred_public_checkpoint(run, preferred)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps({**(selected or {}), "selection": selection}, indent=2, sort_keys=True) + '\n')
print(selected['step'] if selected else '')
PY
)"
if [[ -z "$STEP" ]]; then
  STATUS=1
else
  remaining="$(remaining_seconds)"
  completion_tail=$((ARCHIVE_TIMEOUT + PARENT_RELEASE_RESERVE + COMPLETION_MARGIN))
  eval_timeout_budget=$((remaining - completion_tail - 60))
  if (( eval_timeout_budget >= 600 )); then
    if (( eval_timeout_budget > EVAL_TIMEOUT )); then
      eval_timeout_budget="$EVAL_TIMEOUT"
    fi
    if timeout --signal=TERM --kill-after=60 "$eval_timeout_budget" bash "$RUN/evaluate.sh" "$STEP" all; then
      EVAL_STATUS=0
      remaining_after_eval="$(remaining_seconds)"
      if (( remaining_after_eval > completion_tail )); then
        finalize_timeout=$((remaining_after_eval - completion_tail))
        if timeout --signal=TERM --kill-after=30 "$finalize_timeout" bash "$RUN/finalize.sh" "$STEP"; then
          FINALIZE_STATUS=0
        else
          FINALIZE_STATUS=$?
          STATUS=1
        fi
      else
        FINALIZE_STATUS=124
        STATUS=1
      fi
    else
      EVAL_STATUS=$?
      STATUS=1
    fi
  else
    echo 'insufficient finite GPU budget for generation-only evaluation' >&2
    STATUS=1
  fi
fi

exit "$STATUS"
