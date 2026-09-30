"""Recover immutable prior-control summaries and paintings from a public archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import requests

ARCHIVE_URL = (
    "https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/resolve/"
    "1cc58226c67fee23542a427d4ec937b3df1012c2/"
    "runs/text-pairwise-rl-20260928/continued-step-30-all-20260929.tar.gz"
)
ARCHIVE_SHA256 = "4bd5365d3123f53d02862b61b899ead9bcea157df77b93c22cfe4e59d359f801"
SFT_SUMMARY = "validation/scaled-sft-control/validation-summary.json"
STEP16_SUMMARY = "prior-step16/validation/scaled-step-16-all/comparison/summary.json"
IMAGES = "evidence/continued-gallery/images/"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def prepare(run: Path, cache: Path) -> dict:
    destination = run / "evidence"
    sft = destination / "frozen-sft-summary.json"
    step16 = destination / "frozen-step16-summary.json"
    images = destination / "frozen-control-images"
    cases = json.loads((run / "anchors-all/validation-manifest.json").read_text())["cases"]
    ids = {row["id"] for row in cases}
    if len(ids) != len(cases) or len(ids) != 47:
        raise ValueError("expected the immutable 47-case validation manifest")
    needed = {f"{ident}-{suffix}.png" for ident in ids for suffix in ("SFT", "RL-step-16")}
    if sft.is_file() and step16.is_file():
        summaries = (json.loads(path.read_text())["rows"] for path in (sft, step16))
        rows = list(summaries)
        if all(len(group) == len({row["id"] for row in group}) and
               ids.issubset({row["id"] for row in group}) for group in rows):
            present = all((images / f"{ident}-{label}.png").is_file()
                          for label, group in zip(("SFT", "RL-step-16"), rows)
                          for ident in ids
                          if next(row for row in group if row["id"] == ident)["status"] == "scored")
            if present:
                return {"cases": len(ids), "reused": True, "archive_sha256": ARCHIVE_SHA256}
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.is_file() or digest(cache) != ARCHIVE_SHA256:
        temporary = cache.with_suffix(cache.suffix + ".partial")
        with requests.get(ARCHIVE_URL, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for block in response.iter_content(1024 * 1024):
                    output.write(block)
        if digest(temporary) != ARCHIVE_SHA256:
            temporary.unlink(missing_ok=True)
            raise ValueError("prior-control public archive hash mismatch")
        temporary.replace(cache)
    destination.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    extracted = set()
    with tarfile.open(cache, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            if member.name in {SFT_SUMMARY, STEP16_SUMMARY}:
                target = sft if member.name == SFT_SUMMARY else step16
            elif member.name.startswith(IMAGES):
                name = member.name.removeprefix(IMAGES)
                if name not in needed:
                    continue
                target = images / name
            else:
                continue
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"unreadable archive member: {member.name}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted.add(member.name)
    if {SFT_SUMMARY, STEP16_SUMMARY} - extracted:
        raise ValueError("public archive lacks one or both frozen control summaries")
    for label, path in (("SFT", sft), ("RL-step-16", step16)):
        rows = json.loads(path.read_text())["rows"]
        by_id = {row["id"]: row for row in rows}
        if len(by_id) != len(rows) or not ids.issubset(by_id):
            raise ValueError(f"{label} control summary lacks frozen cases")
        for ident in ids:
            if by_id[ident]["status"] == "scored" and not (
                    images / f"{ident}-{label}.png").is_file():
                raise FileNotFoundError(f"{label} valid control painting: {ident}")
    return {"cases": len(ids), "reused": False, "archive_sha256": ARCHIVE_SHA256,
            "sft_summary_sha256": digest(sft), "step16_summary_sha256": digest(step16),
            "extracted_images": len(extracted) - 2}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.run_root, args.cache), sort_keys=True))


if __name__ == "__main__":
    main()
