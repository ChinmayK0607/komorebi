# Native replay smoke

This run creates one native Prime rank batch from a completed painter group,
then lets the existing SFT-initialized trainer consume that batch for one CUDA
optimizer update. It is a mechanical transport and parameter-change smoke;
it is not a quality result.

The replay is intentionally bounded. It does not call a model provider, run
painter inference or rendering, wait for a judge, invent a batch member, or
retokenize generated actions. It does invoke the pinned offline tokenizer and
multimodal processor to restore omitted tensors, and refuses the run unless
the restored prompt tokens exactly match the native persisted prefix.
`Trace.to_record` rounds serialized logprobs, so the
manifest records that the serialized values were preserved without claiming
original floating-point precision.

## Inputs

- Evidence archive: `painter/runs/revision-rl-neutral-20260921/live/final-evidence.tar.gz`
- Archive SHA-256: `b00964e4e486b4ec7220738e7ae52fa72f63ba2380810ab2b3d93d9f8a38739a`
- Native stream member: `async-output/revision-rl-neutral-20260921/monitors/file/traces/stream/00000.jsonl`
- Complete smoke group: `ff9c7975-6f4c-42b8-9fe6-a3912c1c45c5`
- Prime revision: `26b3131d2716a4f8210b165df584b83b4bc54f61`
- Verifiers revision: `828488fffe31aa3332b9d1bd4bd9ee320e375cf1`
- Model: `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Initializer: contract-quality `step_24`, revision `b7103dd9923f19449ccf814f9b67bb2dcfed0909`

The checked-in `verified-scores.json` is an explicit whitelist prepared from
the reviewed evidence. For the selected group, one valid trace is keyed by its
native ID; deterministic program failures receive `-2.0` only when the
explicit opt-in flag is passed. Three group members are explicitly excluded:
one scoring-timeout trace, one reviewed tie omitted from this smoke whitelist,
and one trace with an uncertain label:

```json
{
  "scores": {"104865ea947745389c54080ea088e465": 1.0}
}
```

Do not add timeout, uncertain, or otherwise excluded traces to this file.

## Build the native rank batch

Run on the Linux training node with the exact model processor available. Keep
the output directory separate from the source checkout and out of Git:

```bash
python painter/runs/native-replay-smoke-20260921/native_replay.py \
  --archive painter/runs/revision-rl-neutral-20260921/live/final-evidence.tar.gz \
  --scores /path/to/reviewed-scores.json \
  --group-id ff9c7975-6f4c-42b8-9fe6-a3912c1c45c5 \
  --model-path /path/to/Qwen3.8-27B \
  --output-dir /path/to/native-replay-output \
  --seq-len 16384 \
  --exclude-trace-id fc6826b027a94b638aa73a5735f6a8a1 \
  --exclude-trace-id 7bdd3ceb8d7e43cb8f75d22e464c7397 \
  --exclude-trace-id 261b04208dfb4522b124b7350abd0001 \
  --allow-invalid-program-penalty
```

The command writes `batches/step_1/rank_0.bin` (fresh Prime trainer progress
starts at step 1) and
`replay-manifest.json`. Check that the manifest reports `training_ready: true`,
`quality_proof: false`, `dummy_batches: 0`, five selected trace IDs from the
validated eight-member source group, and a nonempty
multimodal entry for the painter prompts. A renderer mismatch or missing
processor data is a hard refusal and must be fixed before training.

## Run exactly one update

Derive a flat trainer config from the node's resolved trainer JSON. This keeps
the pinned optimizer, LoRA, VLM, and model implementation while forcing the
replay output, one step, fresh optimizer state, filesystem batch transport,
and one-rank no-offload topology:

```bash
python painter/runs/native-replay-smoke-20260921/prepare_replay_config.py \
  --baseline /path/to/resolved-trainer.json \
  --model-path /path/to/Qwen3.8-27B \
  --output-dir /path/to/native-replay-output \
  --out /path/to/native-replay-output/trainer.json
```

Use the existing active `train_from_sft.py` wrapper and the exact SFT
initializer. The trainer config must point its filesystem batch input/output
at the replay output, use one rank, and set `max_steps = 1`; the existing
trainer loop consumes step 1 and performs one update before stopping. Preserve
the active model, LoRA, sequence length, optimizer, and scheduler settings.

```bash
export WATERCOLOUR_INIT_ADAPTER=/path/to/contract-quality-20260918/step_24
export PAINTER_TRAIN_FROM_SFT=/path/to/revision-rl-20260918/train_from_sft.py
export NATIVE_REPLAY_OUTPUT_DIR=/path/to/native-replay-output
painter/runs/native-replay-smoke-20260921/launch_replay.sh /path/to/native-replay-output/trainer.json
```

The generated JSON may be passed in place of the TOML path shown above. The
launcher runs a bounded local filesystem sink that acknowledges startup
version 0 and final-update version 1 after their `.sender_ready` markers
appear. `TrainerConfig` has no disabled weight-broadcast variant and the
sender recreates each step directory, so this sink avoids waiting for an
inference consumer while keeping the schema valid. Do not use an
inference or rollout command in this step. Record the trainer
step, parameter-change audit, consumed rank-bin hash, GPU busy interval and
measurement window, wall time, and cost. A successful update proves only that
the native replay transport reaches the CUDA optimizer with multimodal inputs.

## Offline checks

```bash
python painter/runs/native-replay-smoke-20260921/test_native_replay.py
```
