# Finite launch handoff

Run the preparation and training supervisor on the Linux node. The parent
side `rental_controller.py` runs separately on the Mac with the exact active
pod ID, rental start epoch, listed whole-node price, and `$20` GPU cap. Keep
that controller in the foreground; it owns provider release. This file does
not transport a credential or start a provider command.

```bash
RUN=/root/painter/repo/painter/runs/text-sft512-consistent-20260930
PRICE_USD_PER_HOUR=2.38
RENTAL_START_EPOCH='REPLACE_WITH_PARENT_RENTAL_START_EPOCH'

bash "$RUN/prepare_node.sh"
PAINTER_RL_SUPERVISED=1 python3 "$RUN/finite_supervisor.py" \
  --metrics "$RUN/async-output/text-sft512-consistent-20260930/monitors/file/metrics.jsonl" \
  --state "$RUN/supervisor-state.json" \
  --trend "$RUN/reward-trend.jsonl" \
  --price-usd-per-hour "$PRICE_USD_PER_HOUR" \
  --rental-start-epoch "$RENTAL_START_EPOCH" \
  --max-gpu-usd 20 --max-updates 50 --reserve-seconds 2700 -- bash "$RUN/run.sh"
```

`RENTAL_START_EPOCH` comes from the parent rental controller. The supervisor
reads only admitted optimizer-batch rewards, reserves 45 minutes of the hard
deadline for checkpoint/evaluation/evidence work, and stops its private
process group at a trend, metric, or finite-cost boundary. After training,
preserve and hash the public checkpoint and generation evidence, run the
generation-only evaluation if it is still within the reserved window, then
release the pod through the parent controller. Create its release marker only
after the final GPU operation and evidence preservation have completed; never
touch it automatically after training.
