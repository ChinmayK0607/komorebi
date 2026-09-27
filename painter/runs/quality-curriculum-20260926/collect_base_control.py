#!/usr/bin/env python3
"""Bundle finite base-control evidence, excluding model weights and credentials."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

RUN_NAME = "brush-base-control-20260927-v1"
POLICIES = ("base", "step40", "step160")
FILES = (
    "data/mix-manifest.json", "base-control.toml", "base-control.setup.json",
    "base-control-processor-audit.json", "base-control-batch-shapes.jsonl",
    "base-control-training.exit", "base-control-train.log",
    "base-control-gpu-telemetry.csv", "base-control-gpu-telemetry-summary.json",
    "painter/eval-prep/eval-manifest.json",
    "source/brush-rl/runs/brush-base-control-20260927-v1-setup/resolved.setup.json",
    f"train-output/{RUN_NAME}/artifacts/initial-adapter-audit.json",
)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bundle(root: Path, output: Path) -> dict:
    root = root.resolve(strict=True)
    if (root / "base-control-training.exit").read_text().strip() != "0":
        raise ValueError("trainer did not complete")
    paths = [root / name for name in FILES]
    if any(not path.is_file() for path in paths):
        raise ValueError("missing required training evidence")
    for policy in POLICIES:
        folder = root / "eval-base-control" / policy
        completion = json.loads((folder / "completion.json").read_text())
        if completion.get("status") != "completed" or completion.get("case_count") != 28:
            raise ValueError(f"incomplete {policy} evaluation")
        paths.extend(path for path in folder.rglob("*") if path.is_file() and "cache" not in path.parts)
        render_tree = root / "painter/evaluation-rollouts" / f"base-control-{policy}"
        if not render_tree.is_dir():
            raise ValueError(f"missing saved canvases and programs for {policy}")
        paths.extend(path for path in render_tree.rglob("*") if path.is_file())
    paths.extend(path for path in (root / "painter/eval-prep/references").iterdir() if path.is_file())
    adapters = root / "train-output" / RUN_NAME / "artifacts/adapters"
    for step in (0, 40, 80, 120, 160):
        folder = adapters / f"step_{step}"
        receipt = json.loads((folder / "hf-upload.json").read_text())
        if receipt.get("verified") is not True or receipt.get("private") is not False:
            raise ValueError(f"step {step} lacks a public verified checkpoint")
        for name in ("hf-upload.json", "validation.json", "adapter_config.json"):
            paths.append(folder / name)
    paths = sorted(set(paths))
    manifest = []
    for path in paths:
        relative = path.relative_to(root)
        if path.is_symlink() or any(part in {"private", "cache", ".venv", "browsers"} for part in relative.parts):
            raise ValueError(f"unsafe evidence path: {relative}")
        if path.name.endswith(".safetensors"):
            raise ValueError("model weights belong in the public checkpoint repository")
        manifest.append({"path": relative.as_posix(), "sha256": sha(path), "bytes": path.stat().st_size})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
        raw = (json.dumps({"schema": "painter.base-control-evidence.v1", "members": manifest}, sort_keys=True) + "\n").encode()
        info = tarfile.TarInfo("artifact-manifest.json")
        info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
    with tarfile.open(output, "r:gz") as archive:
        for member in manifest:
            if hashlib.sha256(archive.extractfile(member["path"]).read()).hexdigest() != member["sha256"]:
                raise ValueError(f"archive member mismatch: {member['path']}")
    receipt = {"schema": "painter.base-control-evidence-receipt.v1", "archive_sha256": sha(output),
               "archive_bytes": output.stat().st_size, "members": len(manifest), "policies": list(POLICIES)}
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(bundle(args.run, args.output)))
