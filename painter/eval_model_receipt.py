"""Download and verify the fixed base model and both comparison adapters.

This module is node-side setup code.  It writes only small receipts and path
pointers; model and adapter bytes stay in the node's Hugging Face cache.  The
two adapter hashes are pinned by the controlled-comparison experiment and are
checked after every download, including when the cache already contains the
requested revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download


MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
ADAPTER_REPO = "CK0607/qwen3.8-27b-brush-painting"

ADAPTERS = {
    "sft": {
        "label": "existing_sft_step24",
        "revision": "b7103dd9923f19449ccf814f9b67bb2dcfed0909",
        "prefix": "contract-quality-20260918/step_24",
        "sha256": "d10eeb3b7b53fc539aa9d5e71eb98bcfe1c59d6a4878c7a69c93b8b09b59bcfa",
    },
    "rl": {
        "label": "aesthetic_rl_step4",
        "revision": "2fab61ebb52a00f7e1c9c3829bad1668719d07b3",
        "prefix": "aesthetic-rl-20260921/step_4",
        "sha256": "fcbfb3c234b8a10ae3d948e90c778319359f1072f45823c7f90dd50bb69adba3",
    },
}

# These are the files needed by the Qwen multimodal processor.  Keeping the
# allowlist narrow makes it harder for an accidental repository artifact to be
# mistaken for model input.
MODEL_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.jinja",
    "*.txt",
    "*.model",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _api_info(api: HfApi, repo: str, revision: str):
    info = api.model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError(f"{repo} resolved to {info.sha}, expected {revision}")
    return info


def _verify_snapshot_files(snapshot: Path, info: Any, *, prefix: str = "") -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    remote = {
        item.rfilename: item
        for item in info.siblings
        if item.rfilename.startswith(prefix)
    }
    for name, item in sorted(remote.items()):
        path = snapshot / name
        if not path.is_file():
            continue
        record = _file_record(path)
        expected = getattr(getattr(item, "lfs", None), "sha256", None)
        if expected is not None and record["sha256"] != expected:
            raise RuntimeError(f"LFS hash mismatch for {name}: {record['sha256']} != {expected}")
        records[name] = record
    return records


def download_and_verify(root: Path) -> dict[str, Any]:
    if os.name != "posix" or sys.platform != "linux":
        raise RuntimeError("model setup must run on the Linux node")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()

    base_info = _api_info(api, MODEL_ID, MODEL_REVISION)
    base_snapshot = Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=list(MODEL_PATTERNS),
        )
    )
    base_files = _verify_snapshot_files(base_snapshot, base_info)
    for required in ("config.json", "model.safetensors.index.json"):
        if required not in base_files:
            raise RuntimeError(f"base snapshot is missing required file {required}")
    index = json.loads((base_snapshot / "model.safetensors.index.json").read_text())
    shards = set(index.get("weight_map", {}).values())
    missing_shards = sorted(shards - set(base_files))
    if missing_shards:
        raise RuntimeError(f"base snapshot is missing weight shards: {missing_shards}")

    adapter_receipts: dict[str, Any] = {}
    paths: dict[str, str] = {"base": str(base_snapshot)}
    for name, spec in ADAPTERS.items():
        info = _api_info(api, ADAPTER_REPO, spec["revision"])
        prefix = spec["prefix"].rstrip("/") + "/"
        snapshot = Path(
            snapshot_download(
                ADAPTER_REPO,
                revision=spec["revision"],
                allow_patterns=[prefix + "*"] ,
            )
        )
        files = _verify_snapshot_files(snapshot, info, prefix=prefix)
        weight_name = prefix + "adapter_model.safetensors"
        weight = snapshot / weight_name
        if not weight.is_file():
            raise RuntimeError(f"{name} adapter is missing {weight_name}")
        actual = _sha256(weight)
        if actual != spec["sha256"]:
            raise RuntimeError(f"{name} adapter hash mismatch: {actual} != {spec['sha256']}")
        config = snapshot / (prefix + "adapter_config.json")
        if not config.is_file():
            raise RuntimeError(f"{name} adapter is missing {prefix}adapter_config.json")
        config_data = json.loads(config.read_text())
        rank = config_data.get("r", config_data.get("rank"))
        if rank != 16:
            raise RuntimeError(f"{name} adapter rank is {rank!r}, expected 16")
        adapter_path = snapshot / spec["prefix"]
        paths[name] = str(adapter_path)
        adapter_receipts[name] = {
            **{key: spec[key] for key in ("label", "revision", "prefix", "sha256")},
            "files": files,
            "adapter_config_rank": rank,
            "resolved_revision": info.sha,
        }

    receipt = {
        "schema": "painter.controlled-comparison-model-receipt.v1",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "resolved_revision": base_info.sha, "files": base_files},
        "adapter_repo": ADAPTER_REPO,
        "adapters": adapter_receipts,
        "weights_in_receipt": False,
    }
    (root / "model-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (root / "model-path.txt").write_text(paths["base"] + "\n")
    (root / "sft-adapter-path.txt").write_text(paths["sft"] + "\n")
    (root / "rl-adapter-path.txt").write_text(paths["rl"] + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="node workspace for receipts and path pointers")
    args = parser.parse_args()
    print(json.dumps(download_and_verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
