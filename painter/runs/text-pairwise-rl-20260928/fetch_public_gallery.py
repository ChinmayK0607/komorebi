"""Stream the public, hash-verified run archive; keep only review evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path


class HashingReader:
    def __init__(self, source):
        self.source = source
        self.digest = hashlib.sha256()
        self.total = 0

    def read(self, size=-1):
        data = self.source.read(size)
        self.digest.update(data)
        self.total += len(data)
        return data


def wanted(name: str) -> bool:
    return name.startswith("evidence/gallery/") or name in {
        "NEXT_SCALED_RUN.md",
        "evidence/rollout-audit.json",
        "validation/scaled-step-16-all/analysis.json",
        "validation/scaled-step-16-all/direct-sft-versus-rl/summary.json",
        "validation/scaled-step-16-probe8/analysis.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    if receipt.get("public_hash_verified") is not True:
        raise ValueError("receipt is not publicly hash-verified")
    expected = receipt["sha256"]
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("invalid archive SHA-256")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    path = urllib.parse.quote(receipt["path"], safe="/")
    url = ("https://huggingface.co/datasets/" + receipt["repo_id"] +
           "/resolve/" + receipt["commit"] + "/" + path)
    request = urllib.request.Request(url, headers={"User-Agent": "komorebi-public-evidence/1"})
    with tempfile.TemporaryDirectory(prefix="painter-gallery-",
                                     dir=args.output.parent) as tmp:
        stage = Path(tmp) / "review"
        stage.mkdir()
        selected = 0
        selected_bytes = 0
        with urllib.request.urlopen(request, timeout=120) as response:
            stream = HashingReader(response)
            with tarfile.open(fileobj=stream, mode="r|gz") as archive:
                for member in archive:
                    if not wanted(member.name):
                        continue
                    parts = Path(member.name).parts
                    if (not member.isfile() or not parts or any(
                            part in {"", ".", ".."} for part in parts)):
                        raise ValueError("unsafe review archive member")
                    selected += 1
                    selected_bytes += member.size
                    if selected > 1000 or selected_bytes > 150_000_000:
                        raise ValueError("review extraction exceeds limit")
                    target = stage.joinpath(*parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = archive.extractfile(member)
                    if source is None:
                        raise ValueError("missing member bytes")
                    with source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
            while stream.read(8 * 1024 * 1024):
                pass
        if stream.digest.hexdigest() != expected:
            raise ValueError("public archive hash mismatch")
        if stream.total != receipt["bytes"]:
            raise ValueError("public archive length mismatch")
        gallery = stage / "evidence/gallery/index.html"
        if not gallery.is_file():
            raise ValueError("gallery missing from verified archive")
        (stage / "public-upload-receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        stage.rename(args.output)
    print(json.dumps({"files": selected, "bytes": selected_bytes,
                      "archive_sha256": expected,
                      "gallery": str(args.output / "evidence/gallery/index.html")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
