# Direct visual review — in progress

These are parent observations from individually displayed reference and generated PNGs. The comparison is still running; this document is not an aggregate result or an RL reward receipt. All examples are previously used procedural families, not a photo-generalization test.

- **Box:** SFT stops on turn 2 after one painting despite an incorrect diamond/top-face arrangement and saturated central magenta. RL revises through turn 4, then finishes on turn 6; its pale pink/grey treatment is closer, but it still has the wrong top-face geometry and a displaced shadow. Both final explanations overstate fidelity. Provisional preference: RL, with substantial remaining structural defects.
- **Cylinder:** the SFT final image is almost entirely clipped into the upper-left corner. Its final program applies both the draw-time top-left translation and additional translations inside shape helpers. The RL final cylinder is complete, centered, and recognizably close to the reference, though its rim is sharply outlined, the body too pale, and the shadow too large. Preference: RL, clear structural advantage.
- **Bus:** both first paintings are far too wide and cropped. SFT revises to a fully visible bus by turn 5, establishing a real correction, but its bus remains too long, left-shifted, low-contrast, and has narrower/less legible windows than the reference. Its final claim that proportions are correct is unsupported. RL final not yet reviewed; no pairwise preference recorded.

Evidence: `collected/eval-assets/` and `collected/evaluation-rollouts/{sft,rl}/`; exact final paths and every turn are retained in native episode traces under `collected/controlled-comparison/`. No missing candidate image may be replaced by the reference in the gallery.
