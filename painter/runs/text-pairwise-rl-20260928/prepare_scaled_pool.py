"""Fetch and verify the public, immutable 290-brief v3 curriculum."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path


REVISION = "ff7f9877379cb538557911ffe7496830ca9391c3"
ARCHIVE_SHA256 = "482aa0d700ba42d5ae6b4dd8f09c4836863d38ccd24c4a287bbbb753c0c05c92"
POOL_SHA256 = "712349be327ca2f5cc800aa42eb5529d0d9b9d54a3508a3aa3ed8c8c9c26b12d"
URL = ("https://huggingface.co/datasets/CK0607/komorebi-painter-teachers/resolve/"
       f"{REVISION}/pools/text-paint-expanded-v3-20260929.tar.gz")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    with urllib.request.urlopen(URL, timeout=120) as response:
        archive = response.read()
    if sha(archive) != ARCHIVE_SHA256:
        raise ValueError("public archive hash mismatch")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        members = tar.getmembers()
        if len(members) != 291 or any(not m.isfile() or m.name.startswith("/") or
                                       ".." in Path(m.name).parts or
                                       Path(m.name).suffix not in {".txt", ".json"}
                                       for m in members):
            raise ValueError("unsafe or incomplete pool archive")
        output.mkdir(parents=True)
        for member in members:
            path = output / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError("missing archive payload")
            path.write_bytes(stream.read())
    manifest = output / "pool.json"
    if sha(manifest.read_bytes()) != POOL_SHA256:
        raise ValueError("pool manifest hash mismatch")
    pool = json.loads(manifest.read_text())
    rows = pool["cases"]
    if pool.get("schema") != "painter-expanded-text-pool-v3" or len(rows) != 290:
        raise ValueError("unexpected curriculum schema or size")
    for row in rows:
        if row.get("mode") != "fresh" or row.get("reference_kind") != "text":
            raise ValueError(f"unsupported native task contract: {row['id']}")
        prompt = output / row["reference"]
        if not prompt.is_file() or sha(prompt.read_bytes()) != row["reference_sha256"]:
            raise ValueError(f"prompt mismatch: {row['id']}")
    return {"pool_sha256": POOL_SHA256, "cases": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), sort_keys=True))


if __name__ == "__main__":
    main()
