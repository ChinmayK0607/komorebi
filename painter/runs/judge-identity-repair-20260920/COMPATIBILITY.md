# Live queue integration

The diagnostic preparer reads existing PNGs and never installs rewards. The live implementation is in `painter/runs/revision-rl-20260918/reward_queue.py`, selected through `luna_queue_ctl.py --identity-mode`.

Export preserves original bytes and private request hashes. Identity-mode staging emits `assets.json` and `judge-NN-assets.json` shards with unique assets and explicit reference/A/B aliases. Import binds every alias back to the original request hash, validates observations and label fields, and retains raw Luna judgments. Content identity metadata is separate from visual provenance; self-reported displayed IDs alone do not prove inspection.

Identical candidates require a relative tie and identical absolute acceptability across their aliases. Absolute quality still comes from the actual visual judgment, never from byte equality alone. Independent reversed views pass through the existing preference/acceptability reconciliation; uncertain or inconsistent pairs are excluded. Training rewards and the node's request/receipt contract remain unchanged.

Use a new local queue root and fresh node queue. The default legacy path remains available for historical compatibility and must not be used for this pilot. No original failed-run judgments or rewards were edited or reused.
