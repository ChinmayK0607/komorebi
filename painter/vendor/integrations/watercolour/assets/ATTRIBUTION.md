Vendored unchanged from huggingface/huggingenvs commit
`35f7ccc096829ba2d1d7ac8d9feeed7fd882fa82`, directory
`02-watercolour/envs/watercolour/core/vendor`.

- p5.js 2.3.2: https://github.com/processing/p5.js (LGPL-2.1)
- p5.brush 2.2.1: https://github.com/acamposuribe/p5.brush (MIT)
- Renderer readiness, seeded Math.random initialization and capture strategy
  adapted from that project's `core/render.py` (source SPDX: BSD-3-Clause; repository-wide Apache-2.0 license
  reproduced in UPSTREAM-LICENSE). Local renderer adds process deadlines, fresh sandboxes,
  blocked network access and reproducibility metadata.

The actual vendored asset digests are saved with every rendering. Asset files
are used locally; no CDNs are contacted during painting.
