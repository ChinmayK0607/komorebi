# Diverse actual-canvas corrections: matched visual review

Six full replacement programs were authored after the teacher saw the exact first canvas and original input. This is an observed-canvas correction test, not six new reference/prompt pairs. The source bundles and their input/prior/program hashes are in [the public index](PUBLIC_SOURCE_INDEX.json). Renderer SHA-256 was `ad440b1aa8612e293b52fa5e68c2252926722852155eb8cd145d369a862a8a63`. The Cloud Linux CPU run used two render workers; GPU utilization is inapplicable and CPU busy fraction/billed cost were not measured. Its three finite subruns took 58, 19 and 139 seconds of wall time respectively. All were collected from public result archives with anonymous SHA-256 verification.

The matched baseline for each row is its own exact prior canvas, not a different teacher or a separately sampled image. The independent [Luna-low visual audit](../../collected/quality-curriculum-20260926/luna-six-corrections-audit.json) inspected all six exact input/prior/new triads, recorded hashes and described weaknesses of the winner. Parent reviewed the contact sheet and the four strongest outputs full-size. The judgment is nonblind and provisional.

| Input / correction | Render | Relative visual result | Parent training decision |
| --- | --- | --- | --- |
| Grey songbird on branch, Astra-high | Valid | Better eye, body volume and leafy light; still stylized, with a thick foreground branch and abbreviated feet/tail. | Provisional **simple-stage positive**; not a reference-faithfulness exemplar. |
| Soup and broccoli in white bowl, Astra-high | Valid | Broccoli cluster and bowl modeling improve; soup is still a very flat graphic surface. | Provisional **simple-stage positive**; not a final-quality target. |
| Two cream teddy bears, Astra-high | Valid | Better rounded muzzles/limbs and warm layering; smaller bear is partly cropped, paw anatomy simplified. | Provisional **simple-stage positive**; a stylized toy study. |
| School bus, Astra-high | **Invalid** | Browser failed before drawing: `Cannot redefine property: box`; no canvas. | Excluded. A runtime-only helper rename is separately packaged for rerender; the failed program stays in the record. |
| Sheepdog at open gate, Sol-high | Valid | Dog's face/body are clearer, but a tangled orange gate and reduced path depth make the result mixed. | Withheld. |
| Potter unloading kiln, Sol-high | Valid | Kiln, potter, apron, tongs and vessel are more legible; human and hand/vessel relation remain blocky, with stray strokes. | Relative improvement, withheld pending stronger correction. |

Independent Luna-low marked the first three above its high-quality bar. Parent agrees they are improvements but deliberately narrows the claim to a **simple-stage** category: close inspection still shows simplified geometry and weak texture. The [objects](../../collected/quality-curriculum-20260926/rendered-cloud/teacher500-astra-coco-objects-turn2-20260926/review.html), [scenes](../../collected/quality-curriculum-20260926/rendered-cloud/teacher500-astra-coco-scenes-turn2-20260926/review.html), and [text](../../collected/quality-curriculum-20260926/rendered-cloud/teacher500-sol-text-turn2-20260926/review.html) galleries expose every before/after. None has been put in a training export or trained. Relative gains here do not show a student-model gain.

The three provisional simple-stage choices are bound to their exact input, prior canvas, new canvas and program in [SIMPLE_STAGE_CANDIDATES.json](SIMPLE_STAGE_CANDIDATES.json). That record explicitly marks `training_exported: false` and `high_quality_exemplar: false`.

**Next decision.** Render and inspect the school-bus helper fix, the camera/desk/custard corrections, and the remaining simple-photo seeds. Prioritize examples where the next turn can resolve a specific structural or aesthetic fault. A third turn is worthwhile only when the second canvas gives a promising base; do not add turns to satisfy a quota.
