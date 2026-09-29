"""Publish derived LoRA blends to the public model repo and verify anonymously."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPO = "CK0607/qwen3.8-27b-brush-painting"
PREFIX = "text-pairwise-merge-canary-20260929"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def upload(*, root: Path, token_file: Path, receipt: Path) -> list[dict]:
    if token_file.is_symlink() or not token_file.is_file() or token_file.stat().st_mode & 0o077:
        raise PermissionError("HF token file must be owner-only, regular and not a symlink")
    api = HfApi(token=token_file.read_text().strip())
    if api.model_info(REPO).private:
        raise ValueError("checkpoint repository must be public")
    rows = json.loads(receipt.read_text()) if receipt.is_file() else []
    if not isinstance(rows, list) or any(row.get("label") not in {"mix025", "mix050"}
                                         or row.get("public_download_hash_verified") is not True
                                         for row in rows):
        raise ValueError("invalid existing upload receipt")
    for label in ("mix025", "mix050"):
        folder = root / label
        weight = folder / "adapter_model.safetensors"
        for name in ("adapter_model.safetensors", "adapter_config.json", "merge_receipt.json"):
            if not (folder / name).is_file():
                raise FileNotFoundError(folder / name)
        local_sha = digest(weight)
        previous = next((row for row in rows if row["label"] == label), None)
        if previous is not None:
            if previous["sha256"] != local_sha:
                raise ValueError(f"uploaded {label} hash no longer matches local weight")
            continue
        prefix = f"{PREFIX}/{label}"
        commit = api.upload_folder(
            folder_path=str(folder), repo_id=REPO, path_in_repo=prefix,
            allow_patterns=["adapter_model.safetensors", "adapter_config.json", "merge_receipt.json"],
            commit_message=f"Add candidate {label} exact LoRA blend")
        filename = f"{prefix}/adapter_model.safetensors"
        public = Path(hf_hub_download(
            REPO, filename=filename, revision=commit.oid, token=False,
            force_download=True, local_dir=root / "public-verify"))
        remote_sha = digest(public)
        if remote_sha != local_sha:
            raise ValueError(f"public adapter hash mismatch: {label}")
        rows.append({"label": label, "repo": REPO, "commit": commit.oid,
                     "path": prefix, "sha256": local_sha,
                     "public_download_hash_verified": True})
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(upload(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
