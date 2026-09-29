#!/usr/bin/env bash
set -Eeuo pipefail

# One finite post-run publication. The caller releases the GPU node only after
# this returns with a public, hash-verified receipt.
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PY="${PRIME_ROOT:-/root/painter/prime}/.venv/bin/python"
TOKEN="${HF_TOKEN_PATH:-/root/painter/private/hf-token}"
[[ -x "$PY" && -f "$TOKEN" && ! -L "$TOKEN" ]] || {
  echo 'pinned Python runtime or protected HF token file missing' >&2; exit 1;
}
[[ "$(stat -c %a "$TOKEN")" == 600 ]] || {
  echo 'HF token file must have mode 600' >&2; exit 1;
}
step="$(cat "$RUN/evidence/evaluated-step.txt")"
scope="$(cat "$RUN/evidence/evaluation-scope.txt")"
[[ "$step" =~ ^[1-9][0-9]*$ && ( "$scope" == all || "$scope" == balanced16 ) ]] || {
  echo 'missing or malformed final evaluation receipt' >&2; exit 1;
}
eval_dir="scaled-step-$step-$scope"
[[ -f "$RUN/validation/$eval_dir/direct-sft-versus-rl/summary.json" ]] || {
  echo 'direct matched visual comparison missing' >&2; exit 1;
}
# Measure the predeclared fixed train probe when the rental still has room;
# this is read-only evaluation of the same verified checkpoint. Keep enough
# time for publication and deletion even if an episode stalls.
probe_dir="scaled-step-$step-probe8"
eval_dirs=(--eval-dir "$eval_dir")
remaining=$((1790677701-$(date -u +%s)))
if (( remaining >= 2400 )) && [[ ! -e "$RUN/validation/$probe_dir/comparison/summary.json" ]]; then
  if timeout --signal=INT --kill-after=60 1200s \
      env PAINTER_RL_RUN_NAME=text-pairwise-scaled-20260929 \
          PAINTER_VALIDATION_MANIFEST="$RUN/anchors-scaled/train-probe-manifest.json" \
          PAINTER_CHECKPOINT_OUT="$RUN/validation/$probe_dir" \
          PAINTER_JUDGE_PROTOCOL=text-paint-sft-pairwise-margin-v1 \
      bash "$RUN/evaluate_checkpoint.sh" "$step" all; then
    if ! "$PY" "$RUN/analyze_scaled.py" \
      --baseline "$RUN/validation/scaled-sft-control/probe-summary.json" \
      --candidate "$RUN/validation/$probe_dir/comparison/summary.json" \
      --output "$RUN/validation/$probe_dir/analysis.json"; then
      echo 'fixed train-probe comparison failed; preserving raw probe results' >&2
    fi
  fi
fi
if [[ -f "$RUN/validation/$probe_dir/comparison/summary.json" ]]; then
  eval_dirs+=(--eval-dir "$probe_dir")
fi
# A live diagnostic may have copied a partial stream. Recopy only after the
# finite training/evaluation chain has finished so the published audit is final.
rm -f "$RUN/evidence/stream-final.jsonl"
PYTHONPATH="$RUN" "$PY" - "$RUN" <<'PY'
import sys
from pathlib import Path
from archive_scaled_evidence import ensure_screenable_stream
ensure_screenable_stream(Path(sys.argv[1]))
PY
"$PY" "$RUN/audit_scaled_rollouts.py" \
  --stream "$RUN/evidence/stream-final.jsonl" \
  --judge-root "$RUN/rl-rollouts-scaled" \
  --output "$RUN/evidence/rollout-audit.json"
"$PY" "$RUN/archive_scaled_evidence.py" \
  --run-root "$RUN" "${eval_dirs[@]}" \
  --output "$RUN/evidence/scaled-final-20260929.tar.gz" \
  --remote-path runs/text-pairwise-rl-20260928/scaled-final-20260929.tar.gz \
  --token-file "$TOKEN"
[[ -f "$RUN/evidence/scaled-final-20260929.tar.upload.json" ]] || {
  echo 'public upload verification receipt missing' >&2; exit 1;
}
echo 'SCALED_EVIDENCE_PUBLIC_HASH_VERIFIED'
