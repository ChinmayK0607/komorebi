#!/usr/bin/env bash
# Finite matched evaluation and evidence bundling after fresh-base training.
set -Eeuo pipefail
RUN="$(cd "${1:?usage: base_control_campaign.sh EXTRACTED_PACKAGE}" && pwd -P)"
[[ "$(uname -s)" == Linux && "$(cat "$RUN/base-control-training.exit")" == 0 ]] || {
  echo 'base-control training must finish successfully first' >&2; exit 2;
}
exec 9>"$RUN/base-control-campaign.lock"
flock -n 9 || { echo 'base-control campaign already running' >&2; exit 2; }
for policy in base step40 step160; do
  echo "Evaluating $policy against the frozen 28-photo manifest"
  bash "$RUN/base_control_eval.sh" "$RUN" "$policy"
done
python3 "$RUN/collect_base_control.py" --run "$RUN" \
  --output "$RUN/base-control-evidence.tar.gz"
echo 'Finite campaign complete; inspect and publish the evidence archive before releasing the node.'
