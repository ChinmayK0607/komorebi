# Offline reward audit

`offline_reward_audit.py` joins preserved `rl-rollouts/*/sample.json` and
`result.json` files with content-addressed Luna receipts. It computes the
request ID exactly as the environment does:

```text
sha256(json.dumps({reference, anchor, candidate}, sort_keys=True).encode())
```

The action always comes from `result.json`. A shared state canvas therefore
does not turn a recorded `finish` into `paint` (or the reverse). Receipt image
hashes are checked before either reward version is scored. Invalid programs
retain the v1/v2 `-2.0` infrastructure-independent penalty; cutoff and
infrastructure failures remain masked.

Run against the verified archive and preserved batches with:

```sh
python3 offline_reward_audit.py \
  ../revision-rl-neutral-20260921/live/final-evidence.tar.gz \
  --reward-batches ../revision-rl-neutral-20260921/live/reward-batches \
  --exclusions /path/to/reviewed-grounding-exclusions.json \
  --output /tmp/offline-reward-audit.json
```

Repeat `--reward-batches` to identify a supplemental batch explicitly. The
exclusion manifest is JSON with either
`{"exclusions": [{"request_id": "...", "reason": "..."}]}` or a mapping
from request ID to reason. An exclusion masks the whole request independently
of label normalization, while raw label files remain unchanged.

The report contains per-rollout state/action/request links, v1 and v2 scores,
separate mask-reason counts, duplicate request IDs, receipt sources, native
replay-field presence, observed group completeness, and group signal counts.
`native_provenance.training_ready` is always false for this offline audit:
token IDs/logprobs and group admission must be validated by the training
pipeline before a transition can be used for learning. Missing native fields
are reported explicitly rather than reconstructed.
