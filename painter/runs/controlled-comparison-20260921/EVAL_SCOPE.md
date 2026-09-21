# Controlled comparison evaluation scope — 2026-09-21

This evaluation is a fixed twelve-case pool: one suffix-007 reference from each v2 family (`box`, `bus`, `cup`, `cylinder`, `flower`, `house`, `ribbons`, `sailboat`, `sphere`, `still_life`, `tree`, `vase`). The selection rule is family coverage. It does not use prior model winners, losses, or visual scores to choose cases. Suffix-008 is excluded because the prior curriculum audit found those assets in the package/data pool.

The reference canvases are existing procedural painted targets. I inspected all twelve `turn-3.png` files directly. They show valid, complete targets with clean object identity and composition: connected cube faces; a bus body with aligned windows and attached wheels; a cup with open rim and connected handle; a cylinder with open rim; a five-petal flower with center, leaf and stem; a connected house; three separated ribbons; a mast, two sails and hull; a shaded sphere; an open vessel and fruit; a branched tree; and a connected vase. The manifest stores the local relative PNG, teacher program, renderer receipt, and SHA-256 for each case. No files were copied or rendered.

## Provenance and prior four-case evaluation

`painter/data-v2/manifest.jsonl` marks all twelve suffix-007 rows `split: dev`, with the same reference paths and `generalisation_scope: held-out geometric parameter bands, not unseen template families`. The immutable source reference manifest is `recovery-quality-20260918/evaluation-references.json` (SHA-256 recorded in `eval-manifest.json`). Its render receipts all report `valid: true`, empty `errors`, `finished: true`, and the turn-3 programs begin their `draw()` body with `translate(-300,-300);`.

The earlier aesthetic-RL comparison evaluated only four of these twelve groups: `v2-box-007`, `v2-flower-007`, `v2-bus-007`, and `v2-vase-007`. `aesthetic-rl-20260921/EXPERIMENT.json` records that matched evaluation as a directional four-case check, not a generalization estimate. The new scope retains those four and adds the other eight families so the comparison cannot be driven by the earlier subset.

The contract-quality runbook records 24 evaluation references and 24 disjoint training references for that SFT experiment, with source-group holdout. Its final report says the training/evaluation source groups had no overlap. This is evidence for the experiment’s declared split, not proof that the references were unseen to every earlier pretraining or SFT artifact. The broader curriculum SFT report separately records 288 train and 96 development references and twelve development references for evaluation. Therefore the honest exposure label for this manifest is `split_dev_and_run_declared_holdout; true_unseen_not_provable_from_local_records`.

## Matched checkpoint contract

The planned comparison uses the existing contract-quality SFT `step24` checkpoint against the existing aesthetic-RL `step4` checkpoint. The parent configuration supplies model IDs and revisions; this manifest intentionally does not select or rewrite them. Both checkpoints must receive the same twelve references, prompt/scaffold, sample policy, and judgment protocol. The proposed shared budget is six turns, up to 8,192 generated tokens per turn, and a 32,768-token total context. Historical records differ: curriculum SFT used an eight-turn/16,384-response-token operational cap; aesthetic RL used three turns, 8,192 tokens per turn, and a recorded 16,384-token total context. Those historical settings are not silently represented as the new comparison.

The result should preserve raw labels, ties, uncertainty, valid-canvas status, turn count, and token counts per case. The pool supports a matched directional comparison across all twelve families; it does not support a claim of unseen template-family or photo generalization.

