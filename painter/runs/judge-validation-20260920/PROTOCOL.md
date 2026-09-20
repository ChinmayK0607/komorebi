# Blinded Luna inspection protocol

Use only the assigned `judge-*.json` manifest. Its IDs and paths are opaque;
do not infer source identities, controls, or expected answers. For every item,
run one `functions.exec` call that displays exactly the three images in this
order, with a role label immediately before each image:

```javascript
text("reference");
image((await tools.view_image({path: pair.reference})).image_url);
text("A");
image((await tools.view_image({path: pair.A})).image_url);
text("B");
image((await tools.view_image({path: pair.B})).image_url);
```

Annotate the reference, A, and B immediately after this call, then atomically
persist that judgment before displaying the next item. Never batch multiple
pairs in one tool call. `reviewed_images` records the three manifest paths in
the same order; those self-reported paths are bookkeeping and do not prove
that an image was visually inspected.

Each shard's label file is `labels/judge-NN.json` with exactly this schema.
For the current local run, `judge-NN-labels.json` at the run root is also
accepted; keep exactly one of these locations per judge so duplicate files
cannot be mistaken for independent labels.

```json
{
  "judge_model": "gpt-5.6-luna",
  "reasoning_effort": "low",
  "judgments": [
    {
      "id": "opaque-manifest-id",
      "winner": "A|B|tie|uncertain",
      "confidence": "high|medium|low",
      "acceptable_A": "yes|no|uncertain",
      "acceptable_B": "yes|no|uncertain",
      "reason": "short concrete comparison",
      "observations": {
        "reference": "what is visibly present",
        "A": "what is visibly present",
        "B": "what is visibly present"
      },
      "reviewed_images": ["absolute reference path", "absolute A path", "absolute B path"]
    }
  ]
}
```

Use `uncertain` when an image or comparison cannot be judged. Missing or
uncertain labels are masked with a reason; they are never converted into ties.
The validator compares reversed views by canonical candidate identity and
reports orientation agreement separately from absolute-acceptability
agreement. Optional per-item latency may be written in a separate
`labels/*.latency.json` sidecar; it is not part of the exact label schema.

## Independent one-image-per-call follow-up

The `separate-calls-manifest.json` and
`separate-calls-reversed-manifest.json` files are an independent audit of the
six controls. For each image path in either manifest, use a separate
`functions.exec` call containing exactly one display:

```javascript
text("image");
image((await tools.view_image({path: item.path})).image_url);
```

Persist that image's observation before the next image call. After the three
roles for one manifest item have each been inspected, persist the pair label
before moving to the next item. Save the two label files as
`separate-calls-labels.json` and `separate-calls-reversed-labels.json` using
the same exact label schema above. The follow-up report remains separate from
the original 24-view validation and cannot rehabilitate its result.
