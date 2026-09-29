# Preliminary visual review — 2026-09-29

This is a manual spot review of the first 13 public, hash-verified candidate
episodes assembled in the [local gallery](../../collected/diverse-teacher-wave-20260929/review-preview-20260929-v3/index.html).
All 13 have valid final canvases and four saved turns. They are **not** admitted
to training. The remaining 107 inputs have not been visually evaluated here.
The public archive identities and source hashes are in the gallery's
`candidates.json` and the [campaign](CAMPAIGN.md).

| Mode / ID | Preliminary visual finding |
| --- | --- |
| Photo `coco-val2017-000000046378` | Weak. The cat consuming a bird loses anatomy and the point of contact; broad overlapping polygons obscure the smaller animal. Needs a teacher repair or exclusion. |
| Photo `coco-val2017-000000139872` | Good subject and pose match for the black-and-white dog and pink disc. Grass and fur texture are present, but large flat fields remain. |
| Photo `coco-val2017-000000157601` | Recognizable woman, sandwich and two foreground cups; crop and hand relationship are broadly preserved. Facial geometry is crude. |
| Photo `coco-val2017-000000181859` | Cat in basin and sink fixtures are recognizable. Simplified anatomy and flat color make this a structural rather than finished painterly exemplar. |
| Photo `coco-val2017-000000246963` | Five riders, road and stop sign are spatially clear. Rendering is strongly diagrammatic/vector-like. |
| Photo `coco-val2017-000000558073` | Seated tabby and window composition are readable. Cat proportions and brush character need inspection before admission. |
| Text `fresh-text-001`–`004` | Kettle, figs, bell/postcards, and bottle/orange all satisfy their main prompt objects and arrangement. Large uniform color regions give a partly vector-like finish. |
| Text `fresh-text-013` | Sleeping fox and log have a coherent silhouette and warm palette; overlaid geometric masses weaken the fur/brush feel. |
| Text `fresh-text-043` | Baker, oven and loaf are readable; the two-hand grip and action on the peel are too weak for a high-quality action demonstration. |
| Text `fresh-text-055` | Lighthouse snow globe has a strong silhouette, translucent dome, and coherent atmosphere; promising higher-quality candidate. |

The generated programs do invoke `brush.*`, yet a valid p5.brush call is not
equivalent to painterly visual quality. The common issue is extensive smooth
fills and rigid geometry under sparse brush texture. Before SFT admission,
review prompt/reference fidelity, anatomy/contact, composition, brush feel,
and whether successive turns materially improve the prior canvas. Retain
rejected raw episodes and explicit reasons rather than treating every valid
render as a positive demonstration. The receipts contain token and latency
data, but provider price fields remain missing, so total Gateway spend is
unknown.
