#!/usr/bin/env python3
"""Package finite brush SFT data and pinned source for a Linux GPU node."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "painter/runs/quality-curriculum-20260924"))
from package_multiturn_node import SOURCES, BRUSH  # noqa: E402


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.data.resolve()
    mix = json.loads((data / "mix-manifest.json").read_text())
    if mix["schema"] != "painter.brush-sft-data.v1" or mix["optimizer_steps"] % 40:
        raise ValueError("wrong audited dataset")
    files = dict(SOURCES)
    for name in ("setup_brush_sft_node.sh", "launch_brush_sft.sh",
                 "launch_brush_sft_eval.sh", "run_brush_sft_campaign.sh"):
        files[name] = HERE / name
    for split in ("train", "validation"):
        path = data / f"{split}.jsonl"
        if sha(path.read_bytes()) != mix["output_sha256"][split]:
            raise ValueError(f"data hash mismatch: {split}")
        files[f"data/{split}.jsonl"] = path
    files["data/mix-manifest.json"] = data / "mix-manifest.json"
    for item in sorted(BRUSH.rglob("*")):
        if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc":
            files[f"source/brush-rl/{item.relative_to(BRUSH).as_posix()}"] = item
    for item in sorted((ROOT / "painter/vendor").rglob("*")):
        if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc":
            files[f"painter/vendor/{item.relative_to(ROOT / 'painter/vendor').as_posix()}"] = item
    references = ROOT / "painter/runs/photo-curriculum-sft-20260922/eval-prep/references"
    if not references.is_dir():
        references = ROOT.parents[1] / "painter/runs/photo-curriculum-sft-20260922/eval-prep/references"
    if not references.is_dir():
        raise FileNotFoundError("frozen 28-photo evaluation references missing")
    for item in sorted(references.iterdir()):
        if item.is_file():
            files[f"painter/eval-prep/references/{item.name}"] = item
    if any(not path.is_file() or path.is_symlink() for path in files.values()):
        raise ValueError("missing or symlinked package member")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {"schema": "painter.brush-sft-node-package.v1", "source_commit": commit,
                "base_model": "Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                "initializer": "CK0607/qwen3.8-27b-brush-painting step-512",
                "data_sha256": sha((data / "mix-manifest.json").read_bytes()),
                "credentials_included": False,
                "files": {name: {"sha256": sha(path.read_bytes()), "bytes": path.stat().st_size}
                          for name, path in sorted(files.items())}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        for name, path in sorted(files.items()):
            info = archive.gettarinfo(str(path), arcname=name)
            info.mtime = 0
            with path.open("rb") as stream:
                archive.addfile(info, stream)
        raw = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        info = tarfile.TarInfo("package-manifest.json")
        info.size = len(raw)
        info.mtime = 0
        archive.addfile(info, io.BytesIO(raw))
    with tarfile.open(args.output, "r:gz") as archive:
        if set(archive.getnames()) != set(files) | {"package-manifest.json"}:
            raise ValueError("unexpected package members")
        for name, metadata in manifest["files"].items():
            if sha(archive.extractfile(name).read()) != metadata["sha256"]:
                raise ValueError(f"package member changed: {name}")
    receipt = {"schema": "painter.brush-sft-node-package-receipt.v1",
               "archive_sha256": sha(args.output.read_bytes()),
               "archive_bytes": args.output.stat().st_size, "source_commit": commit,
               "data_sha256": manifest["data_sha256"], "member_count": len(files)}
    args.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
