#!/usr/bin/env python3
"""Restore only a hash-verified public teacher prefix into a fresh Cloud stage."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
BENCH_CLOUD = HERE.parents[1] / "benchmarks/openrouter-teachers-20260922/cloud"
sys.path.insert(0, str(BENCH_CLOUD))
from fetch_results import download, public_url  # noqa: E402

TOP_LEVEL = {"progress.json", "events.jsonl", "run-summary.json", "gallery.html",
             "restored-source.json", "restored-public-shard.json", "renderer-smoke-result.json"}
IDENTITY_FIELDS = ("shard", "mode", "tier", "model", "manifest_sha256",
                   "config_sha256", "prompt_sha256", "refs_sha256")


def same_workload(left: dict, right: dict) -> bool:
    # An archive can be repacked byte-for-byte equivalent at the manifest and
    # reference level. Episode request identity is bound to these source hashes,
    # not the compressed tar stream's incidental gzip bytes.
    return all(left.get(field) == right.get(field) for field in IDENTITY_FIELDS)


def retry(action):
    for attempt in range(7):
        try:
            return action()
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 6:
                raise
            time.sleep(min(45, 2 ** (attempt + 1)))
    raise AssertionError("unreachable retry state")


def exists(run_id: str) -> bool:
    try:
        retry(lambda: urlopen(Request(public_url("main", f"runs/{run_id}/receipt.json"),
                                      method="HEAD"), timeout=30).close())
        return True
    except HTTPError as exc:
        if exc.code == 404:
            return False
        raise


def restore(shard: str, stage: Path) -> dict:
    receipt = json.loads((HERE / "source-receipt.json").read_text())
    count = receipt["shards"].get(shard)
    if not isinstance(count, int) or count < 1:
        raise ValueError("unknown or empty shard")
    identity = json.loads((stage / "restored-source.json").read_text())
    if (identity.get("shard") != shard or
            identity.get("manifest_sha256") != receipt["manifest_sha256"]):
        raise ValueError("staged source identity mismatch")
    found = None
    for prefix in sorted({1, 4, 8, count}, reverse=True):
        if prefix > count:
            continue
        run_id = f"diverse-teacher-20260929-{shard}-n{prefix}"
        if exists(run_id):
            found = (run_id, prefix)
            break
    if found is None:
        raise ValueError("no publicly hash-verified canary; refusing paid resume")
    run_id, prefix = found
    with tempfile.TemporaryDirectory(prefix="painter-teacher-public-resume-") as temp:
        archive_path = Path(temp) / "prior.tar.gz"
        report = retry(lambda: download(run_id, archive_path))
        extracted_episodes = set()
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                name = Path(member.name)
                if (not member.isfile() or name.is_absolute() or ".." in name.parts or
                        not (member.name in TOP_LEVEL or
                             (len(name.parts) >= 3 and name.parts[0] == "episodes"))):
                    raise ValueError(f"unsafe public teacher member: {member.name}")
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"unreadable public teacher member: {member.name}")
                if member.name == "restored-source.json":
                    if not same_workload(json.load(source), identity):
                        raise ValueError("public prefix source identity changed")
                    continue
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                if len(name.parts) >= 3 and name.parts[0] == "episodes" and name.name == "episode.json":
                    extracted_episodes.add(name.parts[1])
    if len(extracted_episodes) != prefix:
        raise ValueError("public prefix episode count mismatch")
    return {"shard": shard, "restored_prefix": prefix, "source_rows": count,
            "run_id": run_id, "public_bundle_sha256": report["sha256"],
            "public_hash_verified": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard")
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    print(json.dumps(restore(args.shard, args.stage), sort_keys=True))


if __name__ == "__main__":
    main()
