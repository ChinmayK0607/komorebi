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

A follow-up with a temporary NVIDIA Vulkan ICD JSON produced a **black
600×600 canvas** and null WebGL vendor/renderer/version, even though the
experimental backend probe initially marked it valid. This was a probe-only
false positive, not a production render or training result. The experimental
renderer now rejects a hardware backend without a live nonsoftware WebGL
identity, with a regression check for the null-identity case. Neither the
temporary ICD setup nor its canvas will be used for evaluation.

Likely container-level cause: the pod exposed CUDA devices but no NVIDIA EGL
vendor or Vulkan ICD JSON. NVIDIA's [container runtime documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#driver-capabilities)
states that `graphics` must be requested for OpenGL/EGL/Vulkan; its unset
default is only `utility,compute`. Chromium's [headless GPU note](https://chromium.googlesource.com/chromium/src/+/HEAD/docs/gpu/using-gpu-hardware-in-headless-chrome.md)
also warns that Linux headless GPU detection may need explicit backend flags.
This is a diagnosis to test on a freshly provisioned graphics-capable
container, not a demonstrated speed fix on this node.

A second, low-cost RTX4090 pod was provisioned with the CLI's
`NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` request. Its running
container still reported the variable unset and lacked the NVIDIA EGL/Vulkan
ICD files. The backend prerequisite was therefore not met, no paintings
were benchmarked there, and the pod was removed promptly at approximately
$0.01. A successful provider/image-level graphics configuration is needed
before retesting hardware acceleration; setting an env var *inside* an
already running compute-only container is insufficient to mount missing
driver components.

Reading the installed Lium CLI showed that `--env` is applied to the
image-based template creation path, not to the default-template path. A
third short RTX4090 rental used the same image via that path and did receive
the requested variable. With a temporary EGL vendor JSON, `eglinfo` reported
the NVIDIA 4090 on the surfaceless platform. However, sandboxed Chromium
could not launch on that host because the unprivileged browser account could
not create a user namespace. Browser-only, no-sandbox identity probes were
also rejected: ANGLE fell back to SwiftShader, while full Chromium with
direct EGL/ANGLE flags failed to create a WebGL context. No accelerated
p5.brush output was produced, and the diagnostic browser without a sandbox
was never connected to training or evaluation. The graphics pod was removed
at approximately $0.05. Its local ignored diagnostic archive SHA-256 is
`460a8821087ae7fa8e171158d75b04020b8a57b2c9ab70314f82387dcb82a498`.

**Current conclusion:** four concurrent software renders give a measured
~2× batch-throughput gain, but single-paint latency and 180-second failures
are unresolved. Hardware acceleration cannot be claimed from `nvidia-smi`
or `eglinfo` alone; a sandboxed Chromium paint must pass live GPU identity,
pixel-fidelity, and timing checks before promotion. The pinned renderer
remains the production path.

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
