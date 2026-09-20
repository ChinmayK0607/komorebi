# Luna low visual inspection validation

Hypothesis: displaying one identified reference/A/B triplet and recording its observations before loading the next pair reduces image-role confusion and unsupported judgments.

Use existing node-rendered PNGs only. No GPU, training, local rendering or image modification. Twelve underlying pairs produce 24 blinded orientations distributed among three fresh Luna-low agents; reversed orientations go to different agents. Six controls cover two identical-image pairs, three exact-reference/wrong-subject pairs and one exact-reference/blank pair. Six real pairs cover box, bus, cup, flower, house and sailboat.

Controls have known preference answers; real artistic judgments do not. Require every control view to be correct, complete labels, exact image/hash bindings, and direct image display followed by per-pair observations. Report real preference consistency and acceptability consistency separately. Passing is a small inspection-path smoke test, not proof of reliable artistic scoring or permission to claim model improvement. Self-reported paths alone never establish image inspection. Preserve original results on failure.

The earlier pilot used bulk image displays and self-reported reviewed paths. This is a historical comparator, not a randomized causal comparison: failures included role confusion and two batches with unsupported inspection claims. No labels from this validation will be used to train the stopped pilot.

Parent pre-review of the six real pairs found: both boxes are short/flattened and one loses the shadow; both buses are severely cropped at bottom right; both cups are small with blocky handles; one flower is a right-edge fragment and the other is blank; both houses have distorted roofs and differ in size/placement; both sailboats are mostly off-canvas with only a pale sail fragment visible. These observations are kept outside blinded judge manifests and will be used to check descriptive accuracy, not as automated preference labels.

Source code baseline: `9e5c4ab`. Input selection and prepared image manifests will record immutable hashes. Implementation is by Luna xhigh; parent reviews, runs validation, inspects results and commits the research record. No rental or RL restart is part of this validation.

## Follow-up: separate image calls

The original protocol completed before this follow-up: 24 labels, 4/12 correct known-control views, all files and hashes bound. Several false blank descriptions concerned byte-identical images shown together. Hypothesis: displaying each image in a separate tool response, recording its observation before the next image, reduces this presentation/role failure. This is an exploratory follow-up with fresh agents, not a randomized causal attribution to transport, deduplication or model reasoning.

A fresh Luna-low agent described six images individually: two copies of the flattened cube, three copies of the flower and one sailboat. Parent review found all six subject/nonblank descriptions consistent with the actual inputs. This probe contains no blank image, so it cannot test blank specificity. Next, two further fresh Luna-low agents review the same six known controls in opposite orientations with separate image calls; the original 12 control views are the matched-input baseline. Inputs are the existing prepared PNGs. Original labels remain untouched. Passing controls alone would support this inspection method for a further real-pair check, not establish artistic reward reliability.
