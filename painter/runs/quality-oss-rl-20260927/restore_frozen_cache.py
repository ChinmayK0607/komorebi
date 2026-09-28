"""Restore only hash-bound frozen SFT opponents from a verified public archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path


KEY = re.compile(r"[0-9a-f]{64}")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def restore(*, archive: Path, expected_sha256: str, root: Path) -> dict:
    if sha_bytes(archive.read_bytes()) != expected_sha256:
        raise ValueError("public opponent archive SHA-256 mismatch")
    root = root.resolve()
    staged: dict[tuple[str, str], bytes] = {}
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            parts = Path(member.name).parts
            if len(parts) != 3 or parts[0] != "frozen-opponents":
                continue
            category, filename = parts[1:]
            suffix = ".json" if category == "receipts" else ".png" if category == "images" else None
            if suffix is None or not filename.endswith(suffix) or not KEY.fullmatch(filename[:-len(suffix)]):
                raise ValueError("invalid frozen opponent archive member")
            if not member.isfile() or member.size > 10_000_000:
                raise ValueError("unsafe frozen opponent archive member")
            payload = source.extractfile(member).read()
            key = (category, filename[:-len(suffix)])
            if key in staged:
                raise ValueError("duplicate frozen opponent archive member")
            staged[key] = payload
    receipts = 0
    images = 0
    for (category, key), payload in staged.items():
        if category != "receipts":
            continue
        receipt = json.loads(payload)
        if receipt.get("status") == "valid":
            image = staged.get(("images", key))
            if image is None or sha_bytes(image) != receipt.get("sha256"):
                raise ValueError(f"frozen opponent image hash mismatch: {key}")
            images += 1
        elif receipt.get("status") != "invalid_model_output":
            raise ValueError(f"unusable frozen opponent receipt: {key}")
        receipts += 1
    if receipts == 0:
        raise ValueError("archive has no frozen opponent receipts")
    if any(category == "images" and ("receipts", key) not in staged
           for category, key in staged):
        raise ValueError("orphan frozen opponent image")
    for (category, key), payload in staged.items():
        target = root / "frozen-opponents" / category / (key + (".json" if category == "receipts" else ".png"))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.is_symlink() or target.read_bytes() != payload:
                raise ValueError(f"existing frozen opponent differs: {key}")
        else:
            target.write_bytes(payload)
    return {"receipts": receipts, "valid_images": images,
            "archive_sha256": expected_sha256}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--hf-path")
    parser.add_argument("--archive-sha256", required=True)
    args = parser.parse_args()
    if bool(args.archive) == bool(args.hf_path):
        raise ValueError("provide exactly one of --archive or --hf-path")
    archive = args.archive
    if args.hf_path:
        if not args.hf_path.startswith("runs/quality-oss-rl-20260927/") or not args.hf_path.endswith(".tar.gz"):
            raise ValueError("unexpected public cache archive path")
        from huggingface_hub import hf_hub_download
        archive = Path(hf_hub_download(
            "CK0607/komorebi-painter-experiments", filename=args.hf_path,
            repo_type="dataset", token=False,
        ))
    print(json.dumps(restore(archive=archive, expected_sha256=args.archive_sha256,
                             root=args.root), sort_keys=True))


if __name__ == "__main__":
    main()
