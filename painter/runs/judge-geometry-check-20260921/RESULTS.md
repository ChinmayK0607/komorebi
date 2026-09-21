# Targeted geometry judge check — 2026-09-21

Six fresh-context `gpt-5.6-luna` low judgments ranked three known failure pairs, once in each order. All three pairs had order-consistent rankings, and the parent agreed with all six relative winners. Five of six descriptions avoided a reviewed major geometry error. The reversed filled-face comparison still incorrectly claimed a filled top plane: only the two side planes are filled.

The geometry-first prompt asks for visible filled surfaces and major parts, approximate object bounds, proportions, placement and ground gap before ranking. Reference/A/B images are displayed individually with immediate captions. The judge must not infer a handle, saucer, or surface from an object name, and ambiguous shadows remain unknown. Relative preference and absolute acceptability are separate.

This is a targeted check of three previously inspected failures, not held-out evidence of general judge accuracy. Correct ranking did not eliminate factual hallucination. These labels are diagnostic only; they do not overwrite original receipts or supply rewards to the native replay smoke.

[Structured first-result transcription](RESULTS.json) preserves content hashes, source evidence paths, judge tasks, order reconciliation, and the remaining error. Full original agent outputs remain in the task conversation. No human gold labels were collected in this check.

Next decision: retain the geometry-first observations and order checks for the next bounded judged rollout, review factual disagreements, and keep those rows out of optimization. Do not hold the independent native optimizer smoke waiting for more visual grades.
