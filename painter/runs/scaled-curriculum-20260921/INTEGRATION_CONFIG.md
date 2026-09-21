# Expanded native RL integration config

This run is prepared for the hypothesis and matched baseline in
`PLAN.md`. It uses Prime revision
`26b3131d2716a4f8210b165df584b83b4bc54f61` and the installed shared
`native_painting_rl` taskset/environment plugin. The native implementation is
kept in `painter/native_painting.py` and `painter/native_painting_rl.py`; this
record does not modify either file.

## Concrete configuration

Run `painter/prepare_scaled_native_rl.py` on the Linux training node after the
pinned Prime checkout and staged model are available:

```bash
export PRIME_ROOT=/absolute/node/prime-rl
"$PRIME_ROOT/.venv/bin/python" painter/prepare_scaled_native_rl.py \
  --output /absolute/node/painter/native-rl.toml \
  --model /absolute/node/models/qwen38-27b \
  --run-root /absolute/node/painter \
  --gpus-per-node 2 \
  --num-train-gpus 1 \
  --num-infer-gpus 1 \
  --prime-root "$PRIME_ROOT" \
  --validate-prime
```

The output is a complete Prime RL TOML file. Its fixed learning and rollout
settings are:

| field | value |
| --- | ---: |
| `max_steps` | `128` |
| `orchestrator.batch_size` | `32` |
| `orchestrator.group_size` | `8` |
| `max_turns` | `6` |
| `orchestrator.train.sampling.max_completion_tokens` | `8192` |
| `seq_len` / total context | `32768` |
| `temperature` | `1.0` |
| `trainer.optim.lr` | `1e-6` |
| LoRA rank / alpha | `16` / `32` |
| `deployment.num_train_gpus` | `1` |
| `deployment.num_infer_gpus` | `1` |
| `inference.vllm.limit_mm_per_prompt.image` | `14` |

`deployment.gpus_per_node` is the node capacity argument. Prime maps these
role counts to physical devices at launch; the generated file contains no
`CUDA_VISIBLE_DEVICES`, device index, hostname, or node id. The role counts are
explicit generator arguments (`--num-train-gpus` and `--num-infer-gpus`), must
be positive, and must fit within `--gpus-per-node`. Defaults are one and one;
the pinned Prime schema is responsible for any supported inference parallelism
when a node supplies more role GPUs. The image limit is 14 because the shared
six-turn multimodal interaction can retain more than the previous six-image
limit.

The source is configured as:

```toml
[[orchestrator.train.source]]
name = "painter-native-rl"

[orchestrator.train.source.env]
id = "native_painting_rl"

[orchestrator.train.source.env.taskset]
id = "native_painting_rl"
root = "/absolute/node/painter"
manifest = "scaled-training-manifest.json"
max_turns = 6
policy_label = "training"
output_root = "rl-rollouts"
```

The finite manifest must be present under the run root before launch. Each
training row must be `split = "train"`, have a reference and a distinct fixed
same-source-group opponent, and carry the hashes expected by the shared
loader. The loader decides the measured number of usable rows; the config
does not invent or require a 512-row count when filtering reduces availability.
The run root must also contain the pinned `aesthetic_judge.py` and
`judge_queue.py` files used by the native terminal reward.

## Launch recipe

After the node bootstrap has staged the model, step-24 adapter, immutable
`model-receipt.json`, render/judge inputs, brush helper directories, and Prime
checkout, invoke the launch script explicitly:

```bash
export PAINTER_REPO_ROOT=/absolute/path/to/if-they-did-speed-painting-type
export PRIME_ROOT=/absolute/node/prime-rl
export MODEL_DIR=/absolute/node/models/qwen38-27b
export WATERCOLOUR_INIT_ADAPTER=/absolute/node/adapters/contract-quality-step24
export PAINTER_NUM_TRAIN_GPUS=1
export PAINTER_NUM_INFER_GPUS=1
bash "$PAINTER_REPO_ROOT/painter/runs/scaled-curriculum-20260921/launch_native_rl.sh" \
  /absolute/node/painter
```

The recipe exports `BRUSH_ROOT` from the run root and pins
`BRUSH_MODEL_PROFILE=qwen38_27b`; the trainer wrapper then checks that profile
before installing the adapter hooks.

The script verifies the Prime commit and the immutable model receipt fields:
base `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, SFT adapter revision
`b7103dd9923f19449ccf814f9b67bb2dcfed0909`, prefix
`contract-quality-20260918/step_24`, and adapter SHA-256
`d10eeb3b7b53fc539aa9d5e71eb98bcfe1c59d6a4878c7a69c93b8b09b59bcfa`. It also
hashes the staged adapter bytes, installs the dependency-free painter package
into the pinned environment, creates the TOML with Prime validation when it is
absent, and invokes the reviewed repository `run_from_sft.py` wrapper. The
wrapper uses `PAINTER_RUN_ROOT` for the staged brush helper bundle, preserves
the qwen38 profile verification, and changes only the trainer child command so
the portable adapter is installed at the start of a fresh RL optimizer; it
does not resume an optimizer or data checkpoint. No GPU launch, model
download, renderer execution, judge worker, or rental was performed while
preparing this record.

## Validation status

The generator and TOML were checked locally with Python's stdlib `tomllib`,
including the deployment role counts, taskset id, sequence length, and all
plan values. The pinned Prime source was inspected at the commit above,
including its `RLConfig` schema and local GPU-role mapping. Prime's installed
runtime parser was not available on this macOS workspace because the Prime
environment and `verifiers` dependency are Linux-node dependencies; therefore
`--validate-prime` and end-to-end native runtime behavior remain untested here.

Preparation source SHA-256 values at record time:

```text
painter/prepare_scaled_native_rl.py                         ec3d5473546a89bfdbeafee8282032cf0b8dd091f6e1dd46a9a9d1288683a9fe
painter/native_painting.py                                   dea9ea9cbba7da7249d2cbaaeb317dc3a645e34f72b50376677ea6e1fc51c43a
painter/native_painting_rl.py                                5bb6689833102af3a6096cabf6cb06782bf1f69718219c5daf1c9081e6f9ad7e
```
