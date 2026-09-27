#!/usr/bin/env python3
"""Bundle completed brush SFT/evaluation evidence, excluding keys and weights."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

RUN_NAME = "brush-sft-20260927-v1"
FILES = (
    "training.exit", "sft.toml", "sft.setup.json", "processor-audit.json",
    "gpu-telemetry.csv", "gpu-telemetry-summary.json", "data/mix-manifest.json",
    "train.log", "campaign-training.log", "campaign-baseline.log", "campaign-trained.log",
    "painter/eval-prep/eval-manifest.json",
    f"train-output/{RUN_NAME}/artifacts/initial-adapter-audit.json",
)
TREES = ("eval/baseline/results", "eval/midpoint/results", "eval/trained/results",
         "painter/eval-prep/references", "painter/evaluation-rollouts")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def bundle(root: Path, output: Path) -> dict:
    root = root.resolve()
    if (root / "training.exit").read_text().strip() != "0":
        raise ValueError("trainer did not finish successfully")
    for policy in ("baseline", "trained"):
        completion = json.loads((root / "eval" / policy / "completion.json").read_text())
        if completion.get("status") != "completed" or completion.get("case_count") != 28:
            raise ValueError(f"{policy} matched evaluation incomplete")
    manifest = json.loads((root / "data/mix-manifest.json").read_text())
    step = manifest["optimizer_steps"]
    receipt = root / f"train-output/{RUN_NAME}/artifacts/adapters/step_{step}/hf-upload.json"
    published = json.loads(receipt.read_text())
    if published.get("verified") is not True or published.get("private") is not False:
        raise ValueError("final adapter has no public hash-verified receipt")
    paths = [root / name for name in FILES if (root / name).is_file()]
    paths += [receipt]
    for name in TREES:
        tree = root / name
        if tree.is_dir():
            paths.extend(path for path in tree.rglob("*") if path.is_file() and not path.is_symlink())
    for policy in ("baseline", "midpoint", "trained"):
        if policy == "midpoint" and not (root / "eval/midpoint/completion.json").is_file():
            continue
        paths.extend(path for path in (root / "eval" / policy).iterdir()
                     if path.is_file() and path.name != "eval.lock")
    for item in (root / f"train-output/{RUN_NAME}/artifacts/adapters").glob("step_*/hf-upload.json"):
        paths.append(item)
    paths = sorted(set(paths))
    members = []
    for path in paths:
        relative = path.relative_to(root)
        if any(part in {"private", "cache", ".venv", "browsers"} for part in relative.parts):
            raise ValueError(f"unsafe evidence path: {relative}")
        members.append({"path": relative.as_posix(), "sha256": sha(path.read_bytes()), "bytes": path.stat().st_size})
    archive_manifest = {"schema": "painter.brush-sft-evidence.v1", "complete": True,
                        "model_run": RUN_NAME, "members": members}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
        raw = (json.dumps(archive_manifest, indent=2) + "\n").encode()
        info = tarfile.TarInfo("artifact-manifest.json")
        info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
    with tarfile.open(output, "r:gz") as archive:
        for member in members:
            if sha(archive.extractfile(member["path"]).read()) != member["sha256"]:
                raise ValueError(f"archive member mismatch: {member['path']}")
    record = {"schema": "painter.brush-sft-evidence-receipt.v1", "archive_sha256": sha(output.read_bytes()),
              "archive_bytes": output.stat().st_size, "members": len(members),
              "checkpoint_step": step}
    output.with_suffix(".receipt.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(bundle(a.run, a.output)))
