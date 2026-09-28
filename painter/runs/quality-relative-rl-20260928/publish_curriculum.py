"""Publish the nine matched four-turn anchors to a public HF dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "CK0607/komorebi-painter-experiments"
PREFIX = "four-turn-relative-curriculum-20260928"
REMOTE_PATH = f"runs/quality-relative-rl-20260928/{PREFIX}.tar.gz"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bundle(folder: Path, archive_path: Path) -> dict:
    folder = folder.resolve()
    manifest = json.loads((folder / "train-manifest.json").read_text())
    if len(manifest["rows"]) != 9:
        raise ValueError("expected nine reviewed four-turn anchors")
    selected = [folder / name for name in ("train-manifest.json", "provenance.json")]
    for row in manifest["rows"]:
        for role in ("reference", "opponent"):
            path = (folder / row[role]).resolve()
            if not path.is_relative_to(folder.resolve()) or path.is_symlink():
                raise ValueError("unsafe curriculum path")
            if digest(path) != row[role + "_sha256"]:
                raise ValueError(f"{row['id']} {role} digest mismatch")
            selected.append(path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(set(selected)):
            archive.add(path, arcname=f"{PREFIX}/{path.relative_to(folder)}", recursive=False)
    return {"bytes": archive_path.stat().st_size, "sha256": digest(archive_path),
            "files": len(set(selected)), "manifest_sha256": digest(folder / "train-manifest.json")}


def publish(folder: Path, archive_path: Path, verify_dir: Path) -> dict:
    summary = bundle(folder, archive_path)
    api = HfApi()
    if api.repo_info(REPO, repo_type="dataset").private:
        raise ValueError("curriculum destination must be public")
    commit = api.upload_file(path_or_fileobj=str(archive_path), path_in_repo=REMOTE_PATH,
                             repo_id=REPO, repo_type="dataset")
    remote = Path(hf_hub_download(REPO, filename=REMOTE_PATH, repo_type="dataset",
                                  revision=commit.oid, token=False, force_download=True,
                                  local_dir=verify_dir))
    if digest(remote) != summary["sha256"]:
        raise ValueError("anonymous remote curriculum hash mismatch")
    return {**summary, "repo": REPO, "path": REMOTE_PATH, "commit": commit.oid,
            "public": True, "verified": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--verify-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.folder, args.archive, args.verify_dir), sort_keys=True))


if __name__ == "__main__":
    main()
