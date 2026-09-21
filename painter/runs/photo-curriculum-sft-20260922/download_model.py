"""Download and hash-verify the public Qwen base snapshot; never downloads an adapter."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from model_profiles import MODEL_ID, REVISION, verify_snapshot


ROOT = Path(__file__).resolve().parent


def token() -> str | None:
    path = os.environ.get("HF_TOKEN_PATH")
    if not path:
        return None
    candidate = Path(path)
    return candidate.read_text().strip() if candidate.is_file() else None


def main() -> None:
    if os.uname().sysname != "Linux":
        raise SystemExit("model download runs on the Linux node")
    access = token()
    api = HfApi(token=access)
    info = api.model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError(f"Hub resolved {info.sha}, expected immutable revision {REVISION}")
    local = Path(snapshot_download(
        MODEL_ID, revision=REVISION, token=access,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"],
    ))
    verify_snapshot(local)
    files = {}
    for sibling in info.siblings:
        path = local / sibling.rfilename
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            files[sibling.rfilename] = {
                "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                "bytes": path.stat().st_size,
            }
        if sibling.lfs and files[sibling.rfilename]["sha256"] != sibling.lfs.sha256:
            raise ValueError(f"Hub LFS hash mismatch: {sibling.rfilename}")
    index = local / "model.safetensors.index.json"
    if not index.is_file():
        raise ValueError("missing model.safetensors.index.json")
    weight_map = json.loads(index.read_text())["weight_map"]
    if not set(weight_map.values()) <= files.keys():
        raise ValueError("model index references an unverified weight file")
    (ROOT / "model-path.txt").write_text(str(local) + "\n")
    (ROOT / "model-receipt.json").write_text(json.dumps({
        "model_id": MODEL_ID, "revision": REVISION, "config_sha256": verify_snapshot(local)["config_sha256"],
        "files": files, "verified": True,
    }, indent=2) + "\n")
    print(json.dumps({"verified": True, "model_id": MODEL_ID, "revision": REVISION, "files": len(files)}))


if __name__ == "__main__":
    main()
