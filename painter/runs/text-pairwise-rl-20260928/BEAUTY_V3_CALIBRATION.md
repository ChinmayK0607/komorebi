# Beauty-prompt v3 development calibration (2026-09-29)

After the user called out dark muddling in the potter scene and the hand
fan's combination of precision and warmth, the v3 draft explicitly allowed
a controlled line-and-wash painting to beat a more literal, cluttered one.
This was an API-only test of ten saved step-16 gallery pairs with
`xiaomi/mimo-v2.6-flash` in both image orders. The nine directional labels
interpret the user's comments; they are selected development examples, not
an independent blind gold set.

The model followed 6/9 directions, the same as v2 and below the old v1
prompt's 7/9. It preferred SFT on the muddy potter, observatory, seahorse,
lamp, and top, and RL on the hand-crank drill. It still preferred RL for
the yarn ball (.75 RL score) and hand fan (.667) despite the user's notes
about boundary crossings and controlled, warm form. It narrowly preferred
SFT for the octopus (.417) where the user favored RL. The clock was not
given a directional label. Order gaps ranged from 0 to .5.

The v3 wording is useful as a rubric proposal but is **not calibrated for RL
reward**. A separate MiMo-v2.6-Pro spot check on v2 also preferred RL on
the yarn and fan, so simply changing to a larger judge did not remove the
detail bias. Keep the old SFT as the default; inspect the matched merge
canary without treating v2/v3 scores as evidence of generalization.

Raw local receipts, image hashes, prompt hash, order-swapped responses,
and summaries (ignored by Git) live in
`local-evidence/beauty-v3-calibration-20260929/` and
`local-evidence/beauty-v2-pro-spot-20260929/`. No credentials are stored.
