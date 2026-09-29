# Linux p5.brush renderer profile (2026-09-29)

The locked production renderer remains unchanged. Five saved, valid 600×600
p5.brush programs were replayed on a rented 1× RTX PRO 6000 Blackwell node
with 16 visible CPU cores. Input program SHA-256 values are in the ignored
probe manifest `local-evidence/render-probe-20260929/manifest.json`; the
experimental renderer and replay source are committed separately from the
locked renderer. The setup passed `foundation_lock.py` before replay.

| Program | Historical render | Current pinned-style SwiftShader | `toBlob` |
| --- | ---: | ---: | ---: |
| p005 | 4.8 s | 5.8 s | 5.2 s |
| p020 | 20.0 s | 21.0 s | 17.9 s |
| p060 | 57.3 s | 62.8 s | 61.3 s |
| p120 | 118.8 s | 104.2 s | 99.9 s |
| p175 | 175.4 s | 174.0 s | 158.4 s |

These are single replays, not stable speedup estimates. In the slowest
baseline, ~148 s were spent before capture finished and ~25 s in capture;
browser launch was ~0.25 s. On p060 the split was ~38 s drawing and ~23 s
capture. `toBlob` changes capture time little; its p175 timing gain is
mostly before capture and likely run-to-run variation. Screenshots of p005,
p020, and p060 took 5.7, 20.4, and 61.3 s, respectively. Mean absolute RGB
pixel differences against the baseline were ≤0.003/255 for `toBlob` and
≤0.001/255 for screenshots; PNG byte hashes differed. Neither capture change
is a demonstrated latency solution.

ANGLE `gl` and `vulkan` requests both silently selected SwiftShader on the
first program; the experimental renderer detected the fallback and rejected
them. This container has NVIDIA device nodes and EGL libraries but no NVIDIA
EGL vendor or Vulkan ICD configuration; GPU visibility to PyTorch alone does
not prove Chromium WebGL acceleration. No hardware-backed image passed.

Four concurrent copies of p060, rendered with the locked renderer and
identical source/seed, all succeeded in 117.7–125.1 s. Four sequential
renderings at the measured 62.8 s each would take about 251 s, so the
observed batch throughput was approximately **2×**, with roughly doubled
individual latency. This is a same-node throughput estimate, not a new
single-paint speed improvement. Browser path was explicitly set for the
unprivileged workers; an earlier invocation without that path failed before
rendering and is excluded from timing.

**Next:** provision a browser environment that can actually use NVIDIA
WebGL, or optimize the p5.brush draw path without changing pixels. Until
then, use four concurrent renderer workers where CPU headroom permits, and
measure end-to-end throughput alongside GPU utilization. Keep the production
renderer pin until a new backend has stable speed and pixel evidence. Avoid
making the reward depend on painting complexity or faster-but-lower-quality
output.
