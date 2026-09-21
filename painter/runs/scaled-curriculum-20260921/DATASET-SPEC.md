# Scaled curriculum: scaled-curriculum-20260921

This record prepares 2400 procedural training references and 240 procedural development references across the existing 12 families. It performs no rendering or model inference.

Procedural teacher programs use the existing expanded direct canvas-coordinate recipes. `source_sha256` identifies source text; rendered image hashes are pending until the Linux node. Rows imported from the inventory agent use exact `image_sha256` values when supplied and are deduplicated on that hash.

## Node commands

```bash
# On the controller, generate JS under an external writable directory:
python3 painter/scaled_curriculum.py build \
  --output painter/runs/scaled-curriculum-20260921 \
  --program-root /data/painter/scaled-curriculum-20260921/programs \
  --epochs 1 --shards 32

# On Linux, materialize final references only; the queue interleaves
# all families/compositions and resumes valid source/PNG receipts:
python3 painter/scaled_curriculum.py render-references \
  --program-root /data/painter/scaled-curriculum-20260921/programs \
  --renderer painter/vendor/integrations/watercolour/renderer.py \
  --python /data/painter/renderer-env/bin/python \
  --browsers-path /data/painter/browsers --workers 4
```

The helper reuses the renderer invocation and receipt checks from `render_demos.py`, but only renders final references and does not export SFT. The generated top-level manifest is for a smoke run; shard manifests are the resumable unit.
