# Parent visual assessment

The parent displayed all twelve references and every available final candidate individually, as well as selected first-turn images and raw programs. There are 11 complete visual pairs; SFT produced no valid sphere canvas. These are previously used procedural families, not a photo-generalization test. The following preferences are subjective observations rather than RL rewards.

| Reference | Parent assessment | Concrete remaining issue |
|---|---|---|
| Box | Slight RL preference; Luna prefers SFT | RL has closer pale pink/grey handling, but both have incorrect top-face geometry and displaced shadows. SFT is more sharply defined. Both finish explanations overstate fidelity. |
| Bus | SFT | SFT corrects its initial clipping by turn5, though it remains too long and left-shifted. RL retains the same clipped program for all six turns. |
| Cup | RL | RL retains a recognizable connected cup and ring handle; SFT's handle is detached and angular. RL's body is too narrow and the handle overlaps incorrectly. |
| Cylinder | RL | RL is a complete centered cylinder; SFT is almost entirely clipped into the upper-left. RL has overly hard rim/base contours and an oversized shadow. |
| Flower | SFT | SFT retains the spread of elongated petals; RL collapses them into a small circular cluster. SFT's leaf and central layering remain wrong. |
| House | SFT | SFT retains connected walls and roof depth. RL loses much of the side volume and distorts the facade into a tapering shape. |
| Ribbons | Both fail; no meaningful parent preference | Both crop almost all of the intended three wave bands into the bottom-right and introduce large wash artifacts. Luna's SFT preference must not be mistaken for an acceptable painting. |
| Sailboat | SFT | Both triangular sails remain readable in SFT; RL collapses the warm sail. SFT still shifts composition and exaggerates the shadow. |
| Sphere | Not a complete visual pair | SFT repeats a `v is not defined` error on all six attempts. RL renders a distorted polygonal mass rather than the reference sphere; execution success is not visual success. |
| Still life | Close; no strong parent preference | Both displace the fruit downward and leave the vessel floating. SFT has a more legible opening but distracting edge ticks; RL is cleaner. Luna prefers RL. |
| Tree | SFT; Luna prefers RL | SFT better connects the branches to the trunk. RL's branch fork floats within the canopy; both change the canopy proportions and placement. |
| Vase | RL; Luna prefers SFT | RL has a fuller connected vessel and less separation from its shadow, but loses the reference's slender neck. SFT has a more distinct slender neck but a smaller body and a large shadow gap. |

The shared conclusion is mixed performance, with clear RL wins on cylinder/cup and clear regressions on bus/flower/house/sailboat. Close preferences differ: this small visual review does not justify treating a judge count as ground truth or promoting the RL checkpoint.

## Rollout mechanism

RL repeats an unchanged valid canvas on **25/61 paint attempts**; SFT does so on **16/56**. Exact repeated programs, including invalid attempts, occur 26 times for RL and 20 for SFT. In the RL bus trace, turns2–6 return exactly the current program while their prose promises changes. The native trace contains four subsequent explicit no-change feedback messages. This failure persists with the corrected feedback and full history; it is not the earlier missing-feedback defect.

The SFT cylinder applies translations inside shape helpers in addition to the draw-time translation, causing clipping. The SFT sphere repeatedly retains an undefined variable after receiving the exact renderer error. Several final explanations claim reference fidelity despite visible geometric discrepancies. These are failures of action revision and visual self-assessment, not evidence that rendering success means painting quality.

Evidence is preserved under `collected-final/`; the complete gallery links every available turn and raw reply. Original blinded Luna-low judgments and both-side critiques remain under `judge-review/`, with the mapped small record in `receipt-visual-review.json`. No new numerical reward or optimizer update used these opinions.
