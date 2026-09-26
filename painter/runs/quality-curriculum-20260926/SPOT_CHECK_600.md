# Direct visual spot check of first paintings

The selection was fixed before viewing: Python `random.Random(600).sample` drew six valid text-to-image and six valid image-to-image rows from the hash-pinned [repair inventory](../../collected/quality-curriculum-20260926/teacher600-repair-inventory-v1/manifest.json). Parent inspected each 600×600 canvas directly, including the frog already used in the correction pilot. This is a deliberately small cross-mode spot check, not a population quality estimate or an automatic SFT filter.

| ID | Mode | Direct visual finding |
|---|---|---|
| `t600-537` | Text | The safety pin is recognizable but faint against the background; head, clasp and metal value are weak. |
| `t600-496` | Text | Street sweeper reads as a flat vehicle diagram with distracting scratches, repetitive building windows and little depth. |
| `t600-503` | Text | Hay gatherer has disconnected pale limbs and a transparent figure; the requested person does not hold together structurally. |
| `t600-470` | Text | Piano and player are readable, but ghostly low-contrast forms and awkward anatomy make the painting unattractive. |
| `t600-326` | Text | Frog and leaf are recognizable, but flat, rigid and dominated by repeated rain/droplet marks. |
| `t600-347` | Text | Printmaking press is schematic; the person, paper and machine have little form or spatial relation. |
| `t600-017` | Photo | Cat scene collapses into saturated pink translucent silhouettes and repeated line marks. |
| `t600-143` | Photo | Kite scene conveys a park but uses stick people, repeated oval clouds and simplified buildings. |
| `t600-095` | Photo | Bakery scene is mostly grayscale ghost shapes; worker and bread read poorly. |
| `t600-187` | Photo | Street meter has some recognizable mass but forms and ground are translucent and badly separated. |
| `t600-217` | Photo | Kite and two people are readable, but figure anatomy is stick-like and the grass texture is mechanical. |
| `t600-180` | Photo | Luggage scene is a grid of pale suitcases with texture stripes; materials, edges and depth are weak. |

None of these twelve meets the intended high-quality direct-imitation SFT bar. This does **not** imply that all 595 valid canvases fail or that text and photo modes fail equally. It does corroborate the parent A-nomination review: recognizable first paintings can still be poor training targets. The practical route is scene-specific redraw/correction with source-conditioned comparison. Do not convert native fills blindly to p5.brush; the matched translation pilot lost structure in all eight inspected cases.
