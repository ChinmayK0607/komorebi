#!/usr/bin/env python3
"""Bundle finite SFT/eval evidence without credentials, caches, or model weights."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path


FILES = (
    "campaign-completion.json", "campaign.log", "campaign-training.log",
    "campaign-baseline.log", "campaign-trained.log", "training.exit",
    "sft.toml", "sft.setup.json", "final-data-ready.json",
    "processor-audit.json", "gpu-telemetry.csv", "gpu-telemetry-summary.json",
    "eval/gpu-telemetry.csv", "eval/gpu-telemetry-summary.json",
    "data/mix-manifest.json", "train.log",
    "train-output/astra-firstpaint-sft-20260926-v1/artifacts/initial-adapter-audit.json",
    "eval/baseline/completion.json", "eval/baseline/eval.toml",
    "eval/baseline/eval.exit", "eval/baseline/eval.log",
    "eval/baseline/server.log", "eval/trained/completion.json",
    "eval/trained/eval.toml", "eval/trained/eval.exit",
    "eval/trained/eval.log", "eval/trained/server.log",
    "painter/eval-prep/eval-manifest.json",
)
TREES = (
    "eval/baseline/results", "eval/trained/results",
    "painter/eval-prep/references", "painter/evaluation-rollouts",
    "attempt1-failed-preflight",
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bundle(root: Path, output: Path, *, allow_partial: bool = False) -> dict:
    root = root.resolve()
    output = output.resolve()
    if not allow_partial:
        record = json.loads((root / "campaign-completion.json").read_text())
        if any(record.get(key) != 0 for key in ("training_exit", "baseline_eval_exit", "trained_eval_exit")):
            raise ValueError("campaign did not finish all three phases successfully")
        for policy in ("baseline", "trained"):
            if json.loads((root / "eval" / policy / "completion.json").read_text()).get("status") != "completed":
                raise ValueError(f"{policy} evaluation did not complete")
        final = root / "train-output/astra-firstpaint-sft-20260926-v1/artifacts/adapters/step_20/hf-upload.json"
        receipt = json.loads(final.read_text())
        if receipt.get("verified") is not True or receipt.get("private") is not False:
            raise ValueError("final adapter is not public and hash-verified")

    paths = [root / item for item in FILES if (root / item).is_file()]
    for name in TREES:
        tree = root / name
        if tree.is_dir():
            paths.extend(path for path in tree.rglob("*") if path.is_file() and not path.is_symlink())
    for receipt in (root / "train-output/astra-firstpaint-sft-20260926-v1/artifacts/adapters").glob("step_*/hf-upload.json"):
        paths.append(receipt)
    paths = sorted(set(paths))
    if not paths:
        raise ValueError("no evidence files found")
    members = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if any(part in {"private", "cache", ".venv", "browsers"} for part in Path(relative).parts):
            raise ValueError(f"unsafe evidence path: {relative}")
        data = path.read_bytes()
        members.append({"path": relative, "sha256": sha(data), "bytes": len(data)})
    manifest = {"schema": "painter.astra-firstpaint-evidence.v1", "complete": not allow_partial,
                "members": members}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
        raw = (json.dumps(manifest, indent=2) + "\n").encode()
        info = tarfile.TarInfo("artifact-manifest.json")
        info.size = len(raw)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(raw))
    with tarfile.open(output, "r:gz") as archive:
        for item in members:
            stream = archive.extractfile(item["path"])
            if stream is None or sha(stream.read()) != item["sha256"]:
                raise ValueError(f"archive member hash mismatch: {item['path']}")
    receipt = {"schema": "painter.astra-firstpaint-evidence-receipt.v1", "archive": output.name,
               "sha256": sha(output.read_bytes()), "bytes": output.stat().st_size,
               "members": len(members), "complete": not allow_partial}
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    print(json.dumps(bundle(args.run, args.output, allow_partial=args.allow_partial)))


if __name__ == "__main__":
    main()
