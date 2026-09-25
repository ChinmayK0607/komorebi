"""Download and hash-verify the public SFT step-512 adapter only."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


ROOT = Path(__file__).resolve().parent
REPO = "CK0607/qwen3.8-27b-brush-painting"
REVISION = "adf687e5b321514eda7be6eab050434fdc8e44ff"
PREFIX = "photo-curriculum-sft-20260922-balanced/step_512"
EXPECTED_SHA256 = "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740"


def _token() -> str | None:
    path = os.environ.get("HF_TOKEN_PATH")
    if not path:
        return None
    token_path = Path(path)
    return token_path.read_text().strip() if token_path.is_file() else None


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if os.uname().sysname != "Linux":
        raise SystemExit("adapter download runs on the Linux node")
    access = _token()
    api = HfApi(token=access)
    info = api.model_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError(f"Hub resolved {info.sha}, expected immutable revision {REVISION}")
    snapshot = Path(snapshot_download(
        REPO,
        revision=REVISION,
        token=access,
        allow_patterns=[PREFIX + "/*"],
    ))
    adapter = snapshot / PREFIX
    weight = adapter / "adapter_model.safetensors"
    config = adapter / "adapter_config.json"
    if not weight.is_file() or not config.is_file():
        raise ValueError(f"adapter snapshot is incomplete under {adapter}")
    digest = _sha(weight)
    if digest != EXPECTED_SHA256:
        raise ValueError(f"adapter SHA-256 mismatch: {digest}")
    files: dict[str, dict[str, int | str]] = {}
    for sibling in info.siblings:
        if not sibling.rfilename.startswith(PREFIX + "/"):
            continue
        path = snapshot / sibling.rfilename
        if not path.is_file():
            continue
        files[sibling.rfilename] = {"sha256": _sha(path), "bytes": path.stat().st_size}
    (ROOT / "initial-adapter-path.txt").write_text(str(adapter) + "\n")
    (ROOT / "downloaded-initial-adapter.json").write_text(json.dumps({
        "repo": REPO,
        "revision": REVISION,
        "prefix": PREFIX,
        "adapter_sha256": digest,
        "files": files,
        "verified": True,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verified": True, "repo": REPO, "revision": REVISION, "prefix": PREFIX}))


if __name__ == "__main__":
    main()
