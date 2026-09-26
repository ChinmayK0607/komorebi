#!/usr/bin/env python3
"""Package reviewed Git source plus finite SFT data for one Linux GPU node.

The archive intentionally excludes credentials, rendered evidence, model
weights and cached environments.  Checkpoints are published by the trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
PHOTO = ROOT / "painter/runs/photo-curriculum-sft-20260922"
RL = ROOT / "painter/runs/photo-curriculum-rl-20260922"
BRUSH = ROOT / "painter/runs/contract-quality-20260918/source/brush-rl"

SOURCES = {
    "setup_multiturn_node.sh": RUN / "setup_multiturn_node.sh",
    "launch_multiturn_sft.sh": RUN / "launch_multiturn_sft.sh",
    "setup_multiturn_eval.sh": RUN / "setup_multiturn_eval.sh",
    "launch_multiturn_eval.sh": RUN / "launch_multiturn_eval.sh",
    "run_multiturn_campaign.sh": RUN / "run_multiturn_campaign.sh",
    "finalize_multiturn_data.sh": RUN / "finalize_multiturn_data.sh",
    "train_multiturn.py": RUN / "train_multiturn.py",
    "build_multiturn_mix.py": RUN / "build_multiturn_mix.py",
    "bootstrap.sh": PHOTO / "bootstrap.sh",
    "download_model.py": PHOTO / "download_model.py",
    "stage_model.py": PHOTO / "stage_model.py",
    "model_profiles.py": PHOTO / "model_profiles.py",
    "make_sft_config.py": PHOTO / "make_sft_config.py",
    "audit_dataset.py": PHOTO / "audit_dataset.py",
    "gpu_telemetry.py": PHOTO / "gpu_telemetry.py",
    "summarize_telemetry.py": PHOTO / "summarize_telemetry.py",
    "download_initial_adapter.py": RL / "download_initial_adapter.py",
    "painter/contract.py": ROOT / "painter/contract.py",
    "painter/native_painting.py": ROOT / "painter/native_painting.py",
    "painter/native_painting_eval.py": ROOT / "painter/native_painting_eval.py",
    "painter/native_painting_renderer.py": ROOT / "painter/native_painting_renderer.py",
    "painter/native_painting_report.py": ROOT / "painter/native_painting_report.py",
    "painter/eval-prep/eval-manifest.json": PHOTO / "eval-prep/eval-manifest.json",
}

FIRSTPAINT_SOURCES = {
    "setup_astra_firstpaint_node.sh": RUN / "setup_astra_firstpaint_node.sh",
    "launch_astra_firstpaint_sft.sh": RUN / "launch_astra_firstpaint_sft.sh",
    "launch_astra_firstpaint_eval.sh": RUN / "launch_astra_firstpaint_eval.sh",
    "run_astra_firstpaint_campaign.sh": RUN / "run_astra_firstpaint_campaign.sh",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def package(data_dir: Path, reviewed: Path, foundation: Path, foundation_validation: Path,
            output: Path, *, source_commit: str, firstpaint: bool = False) -> dict:
    files = dict(SOURCES)
    if firstpaint:
        files.update(FIRSTPAINT_SOURCES)
    for item in sorted(BRUSH.rglob("*")):
        if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc":
            files[f"source/brush-rl/{item.relative_to(BRUSH).as_posix()}"] = item
    for item in sorted((ROOT / "painter/vendor").rglob("*")):
        if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc":
            files[f"painter/vendor/{item.relative_to(ROOT / 'painter/vendor').as_posix()}"] = item
    references = PHOTO / "eval-prep/references"
    if not references.is_dir():
        references = ROOT.parents[1] / "painter/runs/photo-curriculum-sft-20260922/eval-prep/references"
    for item in sorted(references.iterdir()):
        if item.is_file():
            files[f"painter/eval-prep/references/{item.name}"] = item
    if firstpaint:
        files["data/train.jsonl"] = data_dir / "train.jsonl"
        files["data/validation.jsonl"] = data_dir / "validation.jsonl"
        files["data/mix-manifest.json"] = data_dir / "mix-manifest.json"
    else:
        files["data-input/reviewed.jsonl"] = reviewed
        files["data-input/foundation.jsonl"] = foundation
        files["data-input/foundation-validation.jsonl"] = foundation_validation
        files["expected-mix-manifest.json"] = data_dir / "mix-manifest.json"
    if any(not source.is_file() or source.is_symlink() for source in files.values()):
        raise ValueError("missing or symlinked package source")
    mix = json.loads((data_dir / "mix-manifest.json").read_text())
    for split in ("train", "validation"):
        if digest((data_dir / f"{split}.jsonl").read_bytes()) != mix["output_sha256"][split]:
            raise ValueError(f"{split} differs from mix manifest")
    if firstpaint:
        if mix.get("stage") != "first-paint" or mix.get("optimizer_steps") != 20:
            raise ValueError("not the reviewed first-paint pilot")
    else:
        for name, source in (("reviewed", reviewed), ("foundation", foundation),
                             ("foundation_validation", foundation_validation)):
            if digest(source.read_bytes()) != mix["input_sha256"][name]:
                raise ValueError(f"{name} differs from mix manifest")
    if not source_commit or len(source_commit) != 40:
        raise ValueError("record full reviewed source commit")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if source_commit != head:
        raise ValueError("source commit differs from current repository HEAD")
    manifest = {"schema": "painter.astra-firstpaint-node-package.v1" if firstpaint else "painter.multiturn-node-package.v1", "source_commit": source_commit,
                "model": "Qwen/Qwen3.8-27B", "initializer": "public photo SFT step-512",
                "credential_policy": "no credentials in archive; protected HF token transferred separately",
                "files": {name: {"sha256": digest(path.read_bytes()), "bytes": path.stat().st_size} for name, path in files.items()}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for name, path in sorted(files.items()):
            info = archive.gettarinfo(str(path), arcname=name)
            info.mtime = 0
            with path.open("rb") as stream:
                archive.addfile(info, stream)
        raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        info = tarfile.TarInfo("package-manifest.json")
        info.size = len(raw)
        info.mtime = 0
        archive.addfile(info, io.BytesIO(raw))
    with tarfile.open(output, "r:gz") as archive:
        if set(archive.getnames()) != set(files) | {"package-manifest.json"}:
            raise ValueError("archive member list differs from manifest")
        for name, details in manifest["files"].items():
            if digest(archive.extractfile(name).read()) != details["sha256"]:
                raise ValueError(f"packaged member mismatch: {name}")
    manifest.update(archive_sha256=digest(output.read_bytes()), archive_bytes=output.stat().st_size)
    output.with_suffix(".receipt.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"archive": str(output), "archive_sha256": manifest["archive_sha256"],
            "archive_bytes": manifest["archive_bytes"], "files": len(files)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--reviewed", type=Path, required=True)
    p.add_argument("--foundation", type=Path, required=True)
    p.add_argument("--foundation-validation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--firstpaint", action="store_true")
    args = p.parse_args()
    print(json.dumps(package(args.data_dir, args.reviewed, args.foundation,
                             args.foundation_validation, args.output, source_commit=args.source_commit,
                             firstpaint=args.firstpaint)))


if __name__ == "__main__":
    main()
