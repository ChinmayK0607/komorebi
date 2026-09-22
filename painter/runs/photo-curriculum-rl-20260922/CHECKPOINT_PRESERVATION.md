# Checkpoint preservation

The active RL launch sets `ADAPTER_ONLY_CHECKPOINTS=1`. Its native
`checkpoints/step_N` directories therefore do not contain trainer model or
optimizer state; the observed files are orchestration progress metadata. A
public adapter checkpoint is a portable policy initializer for a fresh RL
optimizer, not an exact native optimizer resume point.

The regular export hook already validates the finite LoRA tensors and uploads
each configured interval to the immutable Hugging Face prefix
`photo-curriculum-rl-20260922/step_N`. It writes `.finished` only after the
upload verification. `preserve_broadcasts.py` is a finite safety process for
completed broadcast directories: it additionally requires `.receiver_ready`,
copies the two adapter inputs before validation, calls the same reviewed
`validate_files` and `upload_checkpoint` helpers, and records a local
`preservation.json`. Only the newest two verified local copies are retained; unverified copies survive upload failures; source
broadcasts and rollout/judge evidence are never deleted. Re-running it is
safe because completed steps in the manifest are skipped and the uploader's
immutable-prefix checks remain authoritative.

Run it on the Linux node with the existing Prime/source helpers, for example:

```sh
export BRUSH_MODEL_PROFILE=qwen38_27b
PRIME_ROOT=/root/painter/prime-rl
RUN_ROOT=/root/painter/runs/photo-curriculum-rl-20260922
"$PRIME_ROOT/.venv/bin/python" "$RUN_ROOT/checkpoint-preservation/preserve_broadcasts.py" \
  --broadcast-root "$RUN_ROOT/async-output/photo-curriculum-rl-20260922/broadcasts" \
  --backup-root "$RUN_ROOT/checkpoint-preservation/adapters" \
  --manifest "$RUN_ROOT/checkpoint-preservation/preservation.json" \
  --export-helper "$RUN_ROOT/source/brush-rl/integrations/prime/watercolour_sft/export_adapter.py" \
  --upload-helper "$RUN_ROOT/source/brush-rl/scripts/upload_checkpoints.py" \
  --repo CK0607/qwen3.8-27b-brush-painting \
  --run photo-curriculum-rl-20260922-broadcast-backup \
  --first-step 9 --max-step 32 --retention 2 --deadline-seconds 172800
```

Set `BRUSH_MODEL_PROFILE=qwen38_27b` in the process environment so the pinned
27B dimensions are used by `validate_files`. The backup namespace is separate
from the regular interval-export namespace; it only re-verifies completed
broadcasts and does not represent additional optimizer updates.

The process has no scheduler, judge dependency, process-control path, or
credential logging. It exits `0` after steps 9 through 32 are preserved and
`2` when its finite deadline expires with a `missing_steps` list.

The process requires an authenticated HF environment for uploads. The deployed
worker obtains it from the already authorized private node credential; the
credential is never written into this example or source. Downloading the
public checkpoints requires no credentials.

Before limits: step10 was published to the primary namespace at commit
`18980c5e008280e06f27a613a3579a2a9c7ccce3`; see
[the verified receipt](step10-public-checkpoint.json). Anonymous config download
matched its recorded hash. Source SHA256 for the deployed worker is
`bd907dd200940b05ea64e7da9da097fe035339ba7f985911c0b0f337e81d4968`.
Four focused tests passed, including an upload failure followed by deletion
of the source broadcast and recovery from the retained copy.

Codex limits do not stop this Linux process, but they may stop Luna judgments.
The run can then wait for rewards while the GPU rental continues billing.
This helper neither stops the pod nor changes the trainer. No Codex scheduled
task or heartbeat was created.

Deployment verified: worker alive with public, hash-verified backup receipts.
See [deployment evidence](checkpoint-preservation-receipt.json).
