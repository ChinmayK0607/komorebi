"""Copy the verified base snapshot to local scratch before Prime mmap loads."""

from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEST_ROOT = Path("/tmp/photo-curriculum-sft-20260922-models")


def main() -> None:
    source = Path((ROOT / "model-path.txt").read_text().strip()).resolve(strict=True)
    receipt = json.loads((ROOT / "model-receipt.json").read_text())
    destination = DEST_ROOT / source.name
    required = sum(int(meta["bytes"]) for meta in receipt["files"].values())
    if shutil.disk_usage(destination.parent if destination.parent.exists() else Path("/tmp")).free < required + 5 * 1024**3:
        raise RuntimeError("insufficient local scratch for verified model copy")
    destination.mkdir(parents=True, exist_ok=True)

    def copy_one(item):
        name, meta = item
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".partial")
        shutil.copyfile(source / name, partial)
        with partial.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != meta["sha256"]:
            raise ValueError(f"staging hash mismatch: {name}")
        partial.replace(target)
        return name

    complete = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(copy_one, item) for item in receipt["files"].items()]
        for future in as_completed(futures):
            complete.append(future.result())
            (ROOT / "model-staging-progress.json").write_text(json.dumps({
                "completed": len(complete), "expected": len(receipt["files"]),
            }) + "\n")
    (ROOT / "model-encrypted-path.txt").write_text(str(source) + "\n")
    (ROOT / "model-path.txt").write_text(str(destination) + "\n")
    (ROOT / "model-staging-receipt.json").write_text(json.dumps({
        "source": str(source), "destination": str(destination), "verified": True,
        "files": len(complete), "bytes": required,
    }, indent=2) + "\n")
    print(json.dumps({"verified": True, "files": len(complete), "destination": str(destination)}))


if __name__ == "__main__":
    main()
