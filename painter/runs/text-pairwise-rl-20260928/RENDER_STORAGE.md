# Public render storage for the eventual demo

The paintings and source evidence already reside in two **public,
anonymously hash-verified** Hugging Face archives. No new GPU render or
training run is needed to preserve them.

| Run | Public evidence archive | SHA-256 | Gallery contents |
| --- | --- | --- | --- |
| Margin RL, steps 4/12/20 | [margin-final-20260929.tar.gz](https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/blob/180ea85f472f39dd7506adff4c764a7e5fc6053f/runs/text-pairwise-rl-20260928/margin-final-20260929.tar.gz) | `de7a74c8a3c08096f7dec6b8942cde9c14c2c34b4ad70e88d248692e7746aee3` | Frozen SFT, fresh SFT, RL checkpoints, and reserve gallery. |
| Scaled RL, step 16 | [scaled-final-20260929.tar.gz](https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/blob/e39e6e58fc830eb7be0a77a8bdde7d906943d1d3/runs/text-pairwise-rl-20260928/scaled-final-20260929.tar.gz) | `ba4252251a54f9eeab16cff7cbf45939ee37e57730a8fa9191fe9121f235ccd3` | Frozen SFT, fresh SFT, and RL step-16 held-out paintings for 47 text prompts; raw turns and judge receipts. |

Local extracted galleries: [margin checkpoints](local-evidence/checkpoint-gallery/index.html),
[margin reserve](local-evidence/gallery/index.html), and
[scaled step 16](local-evidence/scaled-step16-public/evidence/gallery/index.html).
These are separate RL attempts, not one continuous checkpoint trajectory.
Selected paintings must not be presented as a global quality result without
the matched losses and invalid cases.

An optional per-PNG convenience upload was attempted on 2026-09-29 but did
not complete because this machine's Hugging Face Xet/LFS upload transport
failed. No separate per-PNG upload is claimed. Both earlier public archives
remain the source of record; verify their SHA-256 after download before
building the final demo.
