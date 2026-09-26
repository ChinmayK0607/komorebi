#!/usr/bin/env python3
"""Verify already uploaded public Base64 result parts and publish a receipt.

Use only when the cloud renderer finished but anonymous part verification was
rate-limited before its normal receipt could be written. This never rerenders.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
from urllib.request import urlopen

from huggingface_hub import HfApi


DATASET = "CK0607/komorebi-painter-teachers"
ROOT = Path(__file__).resolve().parents[3]
POOL = ROOT / "painter/collected/quality-curriculum-20260926"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def recover(run_id: str, revision: str, batch: str, count: int) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,63}", run_id) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("unsafe run ID or revision")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,63}", batch):
        raise ValueError("unsafe batch name")
    prefix = f"runs/{run_id}/parts/"
    paths = sorted(path for path in HfApi().list_repo_files(DATASET, repo_type="dataset", revision=revision)
                   if path.startswith(prefix))
    if not paths or paths != [f"{prefix}part-{i:04d}.txt" for i in range(len(paths))]:
        raise ValueError("public part sequence is incomplete")
    archive = bytearray()
    for path in paths:
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{path}?download=true"
        with urlopen(url, timeout=180) as response:
            encoded = response.read()
        archive.extend(base64.b64decode(b"".join(encoded.split()), validate=True))
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tf:
        members = tf.getmembers()
        if not all(m.isfile() and not Path(m.name).is_absolute() and ".." not in Path(m.name).parts for m in members):
            raise ValueError("unsafe archive members")
        summary = json.load(tf.extractfile("run-summary.json"))
        source_receipt = json.loads((POOL / batch / "source-receipt.json").read_text())
        if (summary["run_id"] != run_id or summary["batch"] != batch
                or summary["count"] != count or len(summary["statuses"]) != count
                or summary["source_bundle_sha256"] != source_receipt["archive_sha256"]):
            raise ValueError("summary/source mismatch")
        for status in summary["statuses"]:
            if status["valid"]:
                canvas = tf.extractfile(f"episodes/{status['id']}/canvas.png").read()
                if sha(canvas) != status["canvas_sha256"]:
                    raise ValueError(f"canvas hash mismatch: {status['id']}")
    receipt = {"schema": "painter.hf-benchmark-result.v1", "run_id": run_id,
               "source_commit": summary["source_commit"], "dataset_repo": DATASET,
               "bundle_bytes": len(archive), "bundle_sha256": sha(archive),
               "file_count": len(members), "public_hash_verified": True,
               "dataset_commit": revision, "representation": "split-base64-text",
               "encoding": "base64-per-part", "part_order": "lexical", "part_paths": paths,
               "public_parts_anonymously_fetched": True,
               "recovery_note": "Receipt reconstructed after cloud-side anonymous verification was rate-limited; all parts and valid canvases reverified anonymously."}
    output = POOL / "rendered-cloud" / run_id
    output.mkdir(parents=True, exist_ok=True)
    (output / "recovered-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    HfApi(token=True).upload_file(path_or_fileobj=str(output / "recovered-receipt.json"),
                                  path_in_repo=f"runs/{run_id}/receipt.json",
                                  repo_id=DATASET, repo_type="dataset",
                                  commit_message=f"Record verified recovered painter result {run_id}")
    return {"run_id": run_id, "count": count, "valid": summary["valid"],
            "bundle_bytes": len(archive), "bundle_sha256": sha(archive),
            "public_parts": len(paths), "recovery_receipt": str(output / "recovered-receipt.json")}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_id")
    p.add_argument("revision")
    p.add_argument("batch")
    p.add_argument("count", type=int)
    a = p.parse_args()
    print(json.dumps(recover(a.run_id, a.revision, a.batch, a.count), sort_keys=True))
