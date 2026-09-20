# Initial diagnostic result

Two fresh Luna-low agents reviewed six real pairs in opposite orientations using content-addressed image assets and explicit reference/A/B mappings. All 12 view labels passed exact role/hash bindings. Parent review found that both agents correctly identified the main objects, severe cropping, and the genuinely blank flower candidate; neither reported a visible cube, cup or house as blank.

Absolute acceptability agreed on 6/6 pairs. Preference agreed on 3/6; box, bus and flower had tie-versus-preference disagreement and remain excluded by the existing strict reconciliation. There is no independent gold artistic preference for these six cases. This small test supports a monitored live-reward pilot; it does not establish general judging accuracy or isolate the causal source of prior failures.

Six identity controls were resolved deterministically and are excluded from visual-judge success counts. They do not validate the judge. Byte-identical candidates imply relative tie only, not acceptable quality.

Raw labels, prepared role mappings, source hashes and per-pair reconciliation are preserved in labels-01.json, labels-02.json, bindings.json, assets.json and parent-review.json. Original failed tests are untouched. Luna xhigh implemented the preparation/validation code; three focused tests passed under parent execution. No rendering or ML was performed locally. A fresh GPU pilot is documented in ../revision-rl-identity-20260920/RUNBOOK.md.
