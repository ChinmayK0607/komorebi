# Beauty-prompt v2 development calibration (2026-09-29)

This was an API-only test on ten saved step-16 held-out pairs, not a new
student rollout, renderer test, or training reward. We compared the frozen
v1 prompt with the proposed v2 prompt under the same MiMo-v2.6-Flash model,
image pairs, and two image orders. Nine directional labels are interpretations
of the user's visual comments, **not independent blind gold labels**; the
clock was left unlabeled. The sailing boat had no valid RL canvas in this
gallery and was excluded from visual pairwise calibration.

| Prompt | User direction | v1 RL score | v2 RL score | Observation |
| --- | --- | ---: | ---: | --- |
| Potter unloading kiln | SFT anchor | .167 | .167 | Both catch muddy RL darkness. |
| Yarn ball | SFT | .750 | .750 | Both prefer RL's extra needles/strands despite crossed boundaries. |
| Hand fan | SFT | .750 | .833 | V2 worsens the mechanical-detail preference. |
| Observatory | SFT | .250 | .333 | Both prefer SFT. |
| Seahorse | SFT anchor | .250 | .250 | Both prefer SFT. |
| Table lamp | SFT | .250 | .333 | Both prefer SFT. |
| Spinning top | SFT | .167 | .083 | Both prefer SFT. |
| Hand-crank drill | RL | .667 | .833 | Both recognize RL's structural gain. |
| Octopus | RL | .667 | .500 | V2 loses a clear direction. |
| Twin-bell clock | unlabeled anchor comparison | .833 | .833 | Both prefer RL detail; user notes SFT anchor is more soothing. |

Simple directional agreement is 7/9 for v1 and 6/9 for v2. This selected,
nonblind development set cannot estimate population judge accuracy. The two
orders still disagree meaningfully for some pairs; in several rationales the
judge sees missing objects accurately but discounts protruding lines,
jagged fan trim, dark muddling, and visual warmth. **Do not promote v2 to RL
reward.** The next judge experiment should alter the preference tradeoff or
test a stronger model on these exact saved pairs, then reserve different
human-reviewed pairs for confirmation.

Raw local evidence (ignored by Git):
`local-evidence/beauty-v2-calibration-20260929/` with per-order receipts,
image hashes, prompt hashes, and `summary.json`. No credential is included.
