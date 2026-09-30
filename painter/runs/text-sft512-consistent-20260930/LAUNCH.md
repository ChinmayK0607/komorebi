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

export PAINTER_SOURCE_COMMIT="${PAINTER_SOURCE_COMMIT:?parent supplies the reviewed source commit}"
bash "$RUN/execute_pipeline.sh" "$RENTAL_START_EPOCH" "$PRICE_USD_PER_HOUR" 20
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

If the parent has already run the finite supervisor and is relaying the node
for post-training completion, set `PAINTER_PIPELINE_POST_TRAINING_ONLY=1` and
invoke the same command. That mode requires the terminal supervisor state and
the pre-recorded source identity, then performs checkpoint selection,
generation-only evaluation, finalization, screened public evidence upload, and
the terminal receipt before the parent releases the node.

The local parent-side completion relay is a foreground process and reads its
pod/start inputs from an ignored private directory. It polls the remote
supervisor every 20 seconds, runs the post-training command once the state is
terminal, retrieves receipts with `scp`, and writes `release.ready` only after
the public archive hash is verified (or failure evidence is preserved):

```bash
python3 "$RUN/completion_relay.py" \
  --private-dir "$PRIVATE_RENTAL_DIR" \
  --lium-bin lium \
  --completion-commit "$PAINTER_SOURCE_COMMIT"
```
