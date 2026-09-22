# Finite native RL launch

`rl_entrypoint.py` is the small controller for a node-side finite Prime run.
It does not schedule work or carry provider credentials. Prepare a concrete
photo profile first; the generated launch spec binds
the run to the exact Prime commit, base revision, config bytes, manifest path
and bytes, and initial adapter bytes before a process is started.

On the node, prepare the concrete photo profile once. `prepare` stages the
reviewed profile wrappers and copies the supplied manifest to the immutable
run-local `source-manifest.json`; it refuses to overwrite a different file:

```bash
python3 /root/painter/rl_entrypoint.py prepare \
  --profile photo \
  --data /root/painter/runs/photo-curriculum-rl-20260922/source-manifest.json \
  --run-id photo-curriculum-rl-20260922 \
  --run-root /root/painter/runs/photo-curriculum-rl-20260922 \
  --model /tmp/photo-curriculum-sft-20260922-models/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --adapter /root/painter/initial-adapter \
  --prime-root /root/painter/prime-rl \
  --validate-prime
```

The profile command derives the manifest row count and all file hashes, then
checks that the generated Prime taskset points at that exact staged file and
row count. It also generates the run-local Prime config and frozen-opponent
limits sidecar.

When the controller and node share a process host, `run` performs preparation,
starts the finite node supervisor, and starts a controller-side judge command
for the same invocation. The judge argv is runtime-only and is never written
to the launch spec or state:

```bash
python3 /root/painter/rl_entrypoint.py run \
  --profile photo \
  --data /root/painter/input/source-manifest.json \
  --run-id photo-curriculum-rl-20260922 \
  --run-root /root/painter/runs/photo-curriculum-rl-20260922 \
  --model /tmp/photo-curriculum-sft-20260922-models/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --adapter /root/painter/initial-adapter \
  --prime-root /root/painter/prime-rl \
  --judge-command python3 /root/painter/judge_bridge.py worker \
    --local-root /root/painter/runs/photo-curriculum-rl-20260922/judge-live \
    --host NODE_HOST --identity /controller/key --known-hosts /controller/known_hosts \
    --remote-root /root/painter/runs/photo-curriculum-rl-20260922 \
    --limit 8 --max-batches 8192 --deadline-seconds 28800
```

For a separate rented node, use `prepare` and `start` on the node and run the
same finite bridge command on the controller; this wrapper does not transport
controller credentials to the node.

The controller-side remote front end keeps that split explicit. From the
controller, a reviewed invocation is:

```bash
python3 -m painter.rl_remote run \
  --node POD_NAME \
  --data /root/painter/runs/photo-curriculum-rl-20260922/source-manifest.json \
  --run-id photo-curriculum-rl-remote-01 \
  --identity /controller/key \
  --known-hosts /controller/known_hosts \
  --prepare-only
```

It resolves the live pod with `lium ps --format json`, stages only the reviewed
launcher allowlist, links the prepared profile's source/render assets, and
reads the node-local model, adapter, UV, and token *paths*. It never reads or
copies the token, SSH key, provider address, or model/data artifacts. Remove
`--prepare-only` only when the node and controller are ready for the finite
run; the controller then starts the node supervisor and owns one local Luna
worker. The reviewed source is staged under the new run's isolated
`remote-code` directory so an active `/root/painter` checkout is not replaced.
`--dry-run` validates the run ID and remote paths without contacting Lium or
SSH.

`--prepare-only` also invokes the prepared launcher's `preflight` action with
`PAINTER_RL_PREFLIGHT_ONLY=1` and `PAINTER_RL_SKIP_EDITABLE_INSTALL=1`. This
checks the exact runtime serializer/import path and token/UV paths without
installing into the shared Prime environment or starting training.

The remote launch spec is immutable. A repeated invocation checks for its
existing `launch.json` before preparation and returns its node status without
starting another supervisor or judge. If the controller or worker fails after
the node starts, its cleanup path rechecks the supervisor PID and Linux process
start token before requesting a process-group stop.

Then start the finite node run:

```bash
python3 /root/painter/rl_entrypoint.py start --spec /root/painter/runs/photo-curriculum-rl-20260922/launch.json
```

The command starts one detached supervisor and returns a JSON status. Repeating
the same command attaches to the existing supervisor; it does not create a
second Prime tree. Inspect it with:

```bash
python3 /root/painter/rl_entrypoint.py status --spec /root/painter/runs/<run>/launch.json
```

To stop one controller-owned run, first revalidate its PID and Linux process
start token, then signal only that supervisor's process group:

```bash
python3 /root/painter/rl_entrypoint.py stop --spec /root/painter/runs/<run>/launch.json
```

The node launcher also supports an asset/import-only check. It runs the
installed VF wire round-trip and exits before telemetry, inference, or Prime:

```bash
PAINTER_RL_PREFLIGHT_ONLY=1 bash /root/painter/runs/<run>/launch_rl.sh /root/painter/runs/<run>
```

The controller writes only run state and private logs below the configured run
root. It never writes a token, SSH key, endpoint credential, or provider
connection detail to the spec or state. Put credential variable *names* in
`environment.passthrough`; do not put their values in JSON.

A minimal spec has this shape; the paths and hashes must be replaced with the
actual staged files:

```json
{
  "schema": "painter-native-rl-launch-v1",
  "run_id": "photo-curriculum-rl-20260922",
  "run_root": "/root/painter/runs/photo-curriculum-rl-20260922",
  "mode": "fresh",
  "identity": {
    "profile": "qwen38_27b",
    "base_model": "Qwen/Qwen3.8-27B",
    "base_revision": "<full-base-revision>",
    "prime_commit": "<full-prime-commit>",
    "config": {"path": "native-rl.toml", "sha256": "<64-lowercase-hex>"},
    "manifest": {"path": "source-manifest.json", "sha256": "<64-lowercase-hex>"},
    "adapter": {"path": "/root/painter/initial-adapter", "file": "adapter_model.safetensors", "sha256": "<64-lowercase-hex>"}
  },
  "commands": {
    "setup": ["bash", "{run_root}/setup_rl_runtime.sh", "{run_root}"],
    "fresh": ["bash", "{run_root}/launch_rl.sh", "{run_root}"]
  },
  "environment": {
    "set": {"PAINTER_NUM_TRAIN_GPUS": "1", "PAINTER_NUM_INFER_GPUS": "1"},
    "passthrough": ["HF_TOKEN", "HF_TOKEN_PATH"]
  }
}
```

The generated photo profile creates the run-local
`frozen-opponent-config.json` during config preparation and exports it through
`PAINTER_FROZEN_OPPONENT_CONFIG`. A start therefore runs setup, frozen-opponent
prewarm, and the Prime coordinator under one supervisor. Luna judging remains
on the controller machine: run the existing `judge_bridge.py worker` with the
operator's SSH and local Codex settings, using a finite deadline and the same
run root. The node spec contains no judge credentials and never pretends that
the node owns the Luna CLI. Stop that finite worker when the node writes
`rl.exit`; there is no watcher or recurring retry.

The supported run modes have different meanings:

- `fresh` loads the explicitly hashed adapter and creates a fresh optimizer.
- `recovered_policy` is an explicitly named new optimizer run from a supplied
  adapter; it is not optimizer continuation.
- Native optimizer resume is currently unsupported: the existing
  `train_from_sft.py` wrapper intentionally rejects `config.resume`, and the
  adapter-only public checkpoints do not contain optimizer state. The
  controller refuses resume requests rather than silently starting fresh.
  Prepare a new explicit `recovered_policy` run when restarting from a public
  adapter is intended.

A completed run is idempotently reported as `already_complete`. Failed or stale
fresh attempts require a new run root/run ID; the controller does not retry in
place and cannot accidentally reuse partial Prime output. A run root containing
prior Prime artifacts but no matching controller state is rejected.
