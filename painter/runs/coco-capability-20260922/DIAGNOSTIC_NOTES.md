# Observed failures and interpretation

These notes accompany an inference-only comparison, not a training result. The initial attempt was interrupted after a confirmed context-budget failure; the primary rerun uses the same frozen twelve photos, model pins, system prompt, seed, response budget, and six-turn limit with a larger operational envelope.

## Initial 32k attempt

- The base lunchbox and cat episodes produced valid paintings after repairing their first invalid programs. The cat's later valid painting darkened the chair and animal, improving the tonal match while leaving anatomy and perspective weak. These were parent visual inspections, not blinded preference labels.
- The fourth API request on both hit HTTP 400 because 8192 reserved output tokens plus a prompt of at least 24,577 tokens exceeded 32,768. The last valid paintings are retained as censored episode outputs. This is not evidence of model incapacity.
- Among retained calls, 38 ended with `stop` and 2 with `ProviderError`; maximum completion length was 6,466 tokens and recorded completion tokens totaled 64,387. Interrupted in-flight generations are not included in those token counts. No retained response had a length finish reason.
- Renderer receipts marked unsupported API calls invalid, including `brush.createGradient`, `brush.lineWeight`, and `brush.ellipse`, as well as global-name collisions. A diagnostic PNG can exist after a runtime error. It must not be confused with a valid painting. The parent initially made that distinction incorrectly in commentary and corrected it immediately after reading the receipts.
- Completed SFT photos 009/127/599 episodes repeated short no-code plans across all six turns. The café reply explicitly said code was omitted. Those failures occurred without response/context truncation. They remain distinct from later operator-cancelled episodes.

## Bounded training-data audit

The Luna-xhigh audit of preserved contract-quality, recovery-quality, and curriculum-SFT artifacts found no exact matches for the three stock no-code replies. Contract-quality paint/repair targets include complete JavaScript; preparation masks system/user tokens and verifies that learned tokens equal the full assistant target plus EOS. The parent reviewed that masking check and the mixture record. This narrows the explanation but does not establish the cause of the new photo-prompt behavior.

The 576-example contract-quality mixture was 50% paint,25% repair, 10.07% finish, and 14.93% retention by example. Its assistant-token mass was 64.3% paint, 31.8% repair, 0.285% finish, and 3.635% retention. The audit points away from literal no-code target contamination in the inspected local records. Inference-time fallback or a distribution-specific collapse remain hypotheses; do not report either as proven.

Evidence sources: `painter/runs/contract-quality-20260918/prepare_sft.py`, `recovery_data.py`, `collected-final/evidence/sft-data/mixture.json`, and `training-completion-verification.json`. The raw new attempts will be retained with the final run evidence.

## Primary-run visual observations

Parent review of completed first/final paintings finds recognizable base layouts for the café, kitchen and loft bed, but weak perspective, object proportions and material depiction. The cat is an angular dark silhouette with poor anatomy. The lunchbox's last valid version regresses from a colorful rough arrangement to neon outlines on black. The sink revision adds speckles without resolving the toy's incorrect pose or missing mirror structure. None of these should be admitted automatically as a high-quality teacher target merely because the program ran.

The SFT adapter retains a canvas for only three of the twelve photos: bed, sink and skier. The other nine fail to produce any valid canvas within six turns. Five entire episodes are repeated short no-code plans; other failures repeat unsupported APIs or syntax errors despite explicit renderer feedback. The skier repeats the same valid program for all six turns. A larger per-response budget would not fix replies that voluntarily stop after 9–15 tokens; the primary-run call records distinguish those stops from truncation.

The separate Luna-low policy-pair review covers the three pairs with paintings on both sides. It prefers base for the bed and skier and SFT for the sink. This is a single-pass preference measurement, not calibrated judge accuracy or evidence that either painting is good. The parent agrees on the bed but differs on the sink and skier: the base sink retains a visibly recognizable, albeit badly posed, orange toy; the SFT skier image has the more cohesive palette and broad snow composition, while base's vertical mountain marks are particularly distracting. The sink judge's statement that the base subject is absent or unrecognizable is too strong. Preserve the original labels and these disagreements rather than silently correcting the judge's votes.

The current protocol feeds back the actual reference, retained canvas, full current program and explicit error/no-change message. It does not supply an external visual critique or choose the best intermediate painting. The distinction matters: code repair sometimes works while visual revision still regresses. These results motivate full real-photo teacher programs and teacher corrections of actual failed student states. They do not prove that a different planning mode, prompt or larger sample search cannot improve the base model.

## Exact-program deadline controls

The two terminal base render-timeout programs were replayed unchanged with the supported 180-second deadline. Baseball rendered in 155.652 seconds and clock in 134.797 seconds; original-program, replay-program and renderer receipt hashes match. These are separate post-hoc controls, not replacements for the matched primary results. The first requested 300-second replay never rendered because the CLI accepts at most 180 seconds; that rejected invocation is preserved too.

The parent inspected both new canvases. Baseball includes positioned players and field lines, but its extremely dark, mottled field is far from the bright green reference. The extended clock is more legible than the retained primary schematic: it now has a clock face and pedestal. It still loses the ornate round shape, owl identity and material detail. The limit hid some real progress in the clock case; neither extension establishes high visual quality. A next experiment must match and record the rendering allowance as well as the model's token/context budgets.
