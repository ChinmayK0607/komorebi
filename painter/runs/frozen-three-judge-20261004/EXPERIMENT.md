# Frozen three-judge fallback — 2026-10-04

This fallback compares the same saved painting pairs with MiMo Flash, MiMo Pro,
and GPT-6 Luna. It is an offline evidence step for choosing a later reviewed
RL judge; it does not render, train, rent a node, or change the live reward
path.

The preparation command freezes the existing 24-pair diagnostic panel, the
brief and image hashes, the current v2 side-keyed rubric and response schema,
the three model IDs, and explicit `reasoning.effort=medium`. Each judge gets
the same fixed left/right sheet, both orders, and two repeats. The three judges
run in one command so Flash is not paid for twice by separate calibration
runners. There are 288 planned order calls. Provider identity, raw valid
responses, usage, reported cost, reasoning-token counts when returned, and
elapsed call time remain in per-call receipts. HTTP and schema failures are
censored; order disagreement is censored rather than converted to a tie.

The generated gallery keeps the images in fixed Left and Right positions and
labels only the judges as MiMo Flash, MiMo Pro, and GPT-6 Luna. It records each
scored choice, comparison reason, and a weakness of the selected image. It does
not expose checkpoint or split labels in the review UI.

Three pair rows carry explicit historical user preferences from the saved
panel. Their provenance is recorded from
`painter/runs/judge-alignment-20260930/PROTOCOL.md` (2026-09-30), including the
saved user statement. The two fan rows are correlated views of one preference,
so they are not two independent accuracy cases. Human agreement is computed
only for these explicit rows; all other rows have no gold label. Repeat winner
stability and cross-judge disagreement are reported separately from that small
human-labeled subset.

The saved local panel has useful same-brief grouping by `group_id`/case ID, but
does not provide complete four-image on-policy rollout groups for GRPO
advantage validation. The selected images are different checkpoints under the
same brief, rather than four on-policy rollouts against one anchor. One
four-image group and several smaller groups are recorded in the manifest source
evidence; no group-normalized credit assignment is claimed and no images are
generated to fill the gap.

This fallback uses exactly two balanced rounds with no adaptive rechecks. That
differs from the live reward path's adaptive 3-of-4 guard, so the panel measures
fixed-image judgment stability and agreement and cannot identify which judge
would train RL best.

Preparation is explicit and offline:

```bash
painter/runs/frozen-three-judge-20261004/run.sh --prepare-only
```

After review, the parent may run the finite paid fallback with an existing
owner-only local key file. The key path is passed through and never printed or
copied into receipts:

```bash
painter/runs/frozen-three-judge-20261004/run.sh --run --key-file /path/to/local/key
```

The run resumes only receipts whose manifest, image, brief, model, order,
repeat, rubric, and request identities match. A truncated or incomplete
receipt is replaced; a mismatched identity stops the run for review.
