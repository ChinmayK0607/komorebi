"""Fetch the immutable public RL step-16 LoRA and verify its LFS digest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


REPO = "CK0607/qwen3.8-27b-brush-painting"
REVISION = "ac4d3eda8e5925e2d23992cfdc1a63620ebde573"
PREFIX = "text-pairwise-scaled-20260929/step_16"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError("remote revision did not resolve to pinned commit")
    weight_name = PREFIX + "/adapter_model.safetensors"
    siblings = {item.rfilename: item for item in info.siblings}
    expected = siblings[weight_name].lfs.sha256
    snapshot = Path(snapshot_download(REPO, revision=REVISION,
                                      allow_patterns=[PREFIX + "/*"]))
    adapter = snapshot / PREFIX
    actual = digest(adapter / "adapter_model.safetensors")
    if actual != expected or not (adapter / "adapter_config.json").is_file():
        raise ValueError("public RL adapter hash or config mismatch")
    receipt = {"repo": REPO, "revision": REVISION, "prefix": PREFIX,
               "adapter_path": str(adapter), "adapter_sha256": actual,
               "verified": True}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
