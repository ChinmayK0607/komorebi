# Precision-with-warmth judge spot check (2026-09-29)

The v4 draft clarifies that warmth is a positive quality, not a mandatory
style, and that darkness is a problem when it muddies form. Four selected
saved text-paint pairs were sent to `xiaomi/mimo-v2.6-flash` in both image
orders. This is a development spot check using interpreted user preferences,
not a blind validation set or a training reward.

| Pair | User direction | v4 RL score | Result |
| --- | --- | ---: | --- |
| Potter unloading kiln | SFT | .167 | Catches muddy RL darkness. |
| Yarn ball | SFT | .750 | Still favors crossed RL detail. |
| Hand fan | SFT | .833 | Still favors RL over SFT precision/warmth. |
| Hand-crank drill | RL | .667 | Retains RL structural gain. |

The wording did not fix the critical yarn/fan failure and is **not approved
for RL reward**. In particular, repeating aesthetic instructions in a prompt
has not overcome this judge's tendency to prize literal filled detail. Use
matched rendered comparisons and human review before any further optimizer
run. Raw order-swapped receipts, prompt and image hashes are in ignored
`local-evidence/beauty-v4-spot-20260929/`.
