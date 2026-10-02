#!/usr/bin/env bash
set -Eeuo pipefail
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ARM="${1:?usage: preflight.sh ARM [--evaluation] }"
MODE="${2:-}"
[[ -z "$MODE" || "$MODE" == --evaluation ]] || { echo 'usage: preflight.sh ARM [--evaluation]' >&2; exit 2; }
source "$RUN/arm_env.sh" "$ARM"
[[ -f "$PAINTER_RUN_ROOT/source-identity.json" && -f "$PAINTER_RUN_ROOT/native.toml" ]] || {
  echo "arm is not prepared: $ARM; run prepare.sh on the Linux node" >&2; exit 1;
}
[[ "$PAINTER_JUDGE_MODEL" == xiaomi/mimo-v2.6-flash || "$PAINTER_JUDGE_MODEL" == xiaomi/mimo-v2.6-pro || "$PAINTER_JUDGE_MODEL" == openai/gpt-6-luna ]] || exit 1
[[ "$PAINTER_JUDGE_EFFORT" == medium && "$PAINTER_JUDGE_FORMAT" == identity-joint-side-schema-v2-generic-v1 ]] || exit 1
if [[ "$MODE" == --evaluation ]]; then
  [[ -f "$PAINTER_RUN_ROOT/evaluation/matched16-manifest.json" ]] || {
    echo 'matched16 evaluation manifest is missing' >&2; exit 1;
  }
  echo "RL_JUDGE_ABLATION_EVALUATION_PREFLIGHT_READY: arm=$ARM"
else
  [[ -f "$PAINTER_RUN_ROOT/anchors-frozen/train-manifest.json" && -f "$PAINTER_RUN_ROOT/anchors-frozen/validation-manifest.json" ]] || {
    echo 'frozen train/heldout manifests are missing' >&2; exit 1;
  }
  echo "RL_JUDGE_ABLATION_PREFLIGHT_READY: arm=$ARM"
fi
