# p5.brush centroid derivative

The pinned runtime in this directory remains the matched baseline. Its
`assets/p5.brush.js` is preserved byte-for-byte because existing replays and
foundation hashes depend on it.

To prepare a future runtime with the narrowly scoped zero-area centroid fix,
run this from the repository root:

```sh
python painter/patch_brush_centroid.py \
  --output-root painter/runtime/watercolour-p5-brush-2.2.1-centroid-fix-v1
```

The helper verifies the complete pinned source file set and hashes, copies the
renderer plus all assets, and performs exactly one guarded replacement inside
`function xn(t)`: `v[0]` becomes `t[0]`. It refuses changed, already-patched,
or existing output directories. The generated runtime contains
`PATCH-MANIFEST.json` with before/after SHA-256 values, upstream commit and
path, and the applicable license files. The generated `painter/runtime/`
directory is ignored so it cannot silently replace the frozen baseline in
Git; pass its renderer path explicitly when a future experiment is authorized
to use the derivative.

The pure-function regression is in `painter/test_patch_brush_centroid.py`.
It statically checks the guard and, when Node.js is available, exercises empty,
short, zero-area eight-vertex, and nondegenerate polygon inputs without a
renderer or model. The portable fixture can also be run directly with
`node painter/brush_centroid_regression.cjs`; it confirms the original
zero-area case raises `ReferenceError` and the derivative returns finite
centroids while preserving the nondegenerate result.
