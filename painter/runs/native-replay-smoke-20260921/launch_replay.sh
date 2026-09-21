#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: WATERCOLOUR_INIT_ADAPTER=/path/to/step_24 $0 /path/to/trainer.toml" >&2
  exit 2
fi
: "${WATERCOLOUR_INIT_ADAPTER:?set the exact contract-quality step_24 adapter path}"
: "${PAINTER_TRAIN_FROM_SFT:?set the active train_from_sft.py wrapper path}"
: "${NATIVE_REPLAY_OUTPUT_DIR:?set the replay output directory used by the trainer config}"
: "${REPLAY_SINK_TIMEOUT_SECONDS:=1200}"

# TrainerConfig has no disabled weight-broadcast variant. The trainer's
# startup and final-update broadcasts otherwise wait for an inference
# consumer, which this diagnostic intentionally does not run. WeightSender
# deletes and recreates each step directory, so the local sink must wait for
# .sender_ready before acknowledging it; precreating .receiver_ready would be
# erased. This sink performs no model work and is bounded to the two expected
# versions.
receiver_sink() {
  local step="$1"
  local step_dir="$NATIVE_REPLAY_OUTPUT_DIR/broadcasts/step_${step}"
  local deadline=$((SECONDS + REPLAY_SINK_TIMEOUT_SECONDS))
  while [[ ! -e "$step_dir/.sender_ready" ]]; do
    if (( SECONDS >= deadline )); then
      return 31
    fi
    sleep 0.1
  done
  touch "$step_dir/.receiver_ready"
  while [[ ! -e "$step_dir/.finished" ]]; do
    if (( SECONDS >= deadline )); then
      return 32
    fi
    sleep 0.1
  done
}

# Remove only stale generated broadcast handshakes from an interrupted replay;
# checkpoints, rank batches, and audit files are outside these step dirs.
for broadcast_step in 0 1; do
  rm -rf "$NATIVE_REPLAY_OUTPUT_DIR/broadcasts/step_${broadcast_step}"
done
receiver_sink 0 & sink0_pid=$!
receiver_sink 1 & sink1_pid=$!

set +e
"${UV:-uv}" run --no-sync torchrun --standalone --nproc-per-node=1 \
  "$PAINTER_TRAIN_FROM_SFT" @ "$1"
trainer_status=$?
set -e
kill "$sink0_pid" "$sink1_pid" 2>/dev/null || true
wait "$sink0_pid" "$sink1_pid" 2>/dev/null || true
exit "$trainer_status"
