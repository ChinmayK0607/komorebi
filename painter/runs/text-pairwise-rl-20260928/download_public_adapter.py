"""Download and hash-check one pinned public Hugging Face LoRA adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


REPO = "CK0607/qwen3.8-27b-brush-painting"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(*, revision: str, prefix: str, expected_sha256: str,
             receipt: Path) -> dict:
    if len(revision) != 40 or len(expected_sha256) != 64:
        raise ValueError("pinned commit and adapter SHA-256 are required")
    if prefix.startswith("/") or ".." in Path(prefix).parts:
        raise ValueError("unsafe adapter prefix")
    info = HfApi().model_info(REPO, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError("remote revision did not resolve to pinned commit")
    weight_name = prefix + "/adapter_model.safetensors"
    siblings = {item.rfilename: item for item in info.siblings}
    remote_sha = siblings[weight_name].lfs.sha256
    if remote_sha != expected_sha256:
        raise ValueError("pinned adapter SHA differs from public LFS SHA")
    snapshot = Path(snapshot_download(REPO, revision=revision,
                                      allow_patterns=[prefix + "/*"]))
    adapter = snapshot / prefix
    if digest(adapter / "adapter_model.safetensors") != expected_sha256:
        raise ValueError("downloaded adapter SHA differs from pinned SHA")
    if not (adapter / "adapter_config.json").is_file():
        raise ValueError("adapter config missing")
    result = {"repo": REPO, "revision": revision, "prefix": prefix,
              "adapter_path": str(adapter), "adapter_sha256": expected_sha256,
              "verified": True}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(download(**vars(args)), sort_keys=True))
