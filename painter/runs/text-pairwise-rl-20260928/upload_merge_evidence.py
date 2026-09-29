"""Publish screened canary evidence and verify a public anonymous download."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPO = "CK0607/komorebi-painter-experiments"
REMOTE_PATH = "runs/text-pairwise-rl-20260928/merge-canary-20260929.tar.gz"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def upload(archive: Path, token_file: Path, receipt: Path) -> dict:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if token_file.is_symlink() or not token_file.is_file() or token_file.stat().st_mode & 0o077:
        raise PermissionError("HF token file must be owner-only, regular and not a symlink")
    local_sha = digest(archive)
    if receipt.exists():
        existing = json.loads(receipt.read_text())
        if existing.get("sha256") != local_sha or existing.get("public_download_hash_verified") is not True:
            raise ValueError("existing evidence receipt does not match archive")
        return existing
    api = HfApi(token=token_file.read_text().strip())
    if api.repo_info(REPO, repo_type="dataset").private:
        raise ValueError("evidence repository must be public")
    commit = api.upload_file(path_or_fileobj=str(archive), path_in_repo=REMOTE_PATH,
                             repo_id=REPO, repo_type="dataset",
                             commit_message="Add matched text-paint blend canary evidence")
    public = Path(hf_hub_download(REPO, filename=REMOTE_PATH, repo_type="dataset",
                                  revision=commit.oid, token=False,
                                  force_download=True, local_dir=archive.parent / "public-verify"))
    if digest(public) != local_sha:
        raise ValueError("public evidence archive hash mismatch")
    result = {"repo": REPO, "commit": commit.oid, "path": REMOTE_PATH,
              "sha256": local_sha, "bytes": archive.stat().st_size,
              "public_download_hash_verified": True}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(upload(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
