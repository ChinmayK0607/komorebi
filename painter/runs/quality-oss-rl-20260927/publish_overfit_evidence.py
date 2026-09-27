"""Publish screened four-scene RL evidence to a public, hash-verified HF dataset."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "CK0607/komorebi-painter-experiments"
REMOTE_PATH = "runs/quality-oss-rl-20260927/overfit-four-20260928-evidence.tar.gz"
TEXT_SUFFIXES = {".js", ".json", ".jsonl", ".csv", ".html", ".toml"}
ALLOWED_SUFFIXES = TEXT_SUFFIXES | {".png", ".jpg", ".jpeg"}
FORBIDDEN_TEXT = (b"AI_GATEWAY_API_KEY", b"Authorization: Bearer", b"sk-or-v1-", b"hf_" )


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def select(root: Path, step: int) -> list[Path]:
    run = root / "async-output/gateway-overfit-four-20260928"
    matched = root / f"matched-eval-step-{step}"
    required = [root / "profile-data/overfit-four-manifest.json",
                root / "native-rl-overfit-four.toml",
                matched / "identity.json", matched / "cases.json",
                matched / "completion.json", matched / "review/index.html",
                matched / "review/report.json", matched / "review/judge.json"]
    for number in range(1, step + 1):
        required.append(run / f"artifacts/adapters/step_{number}/hf-upload.json")
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("required evidence missing: " + ", ".join(missing))
    optional = list((root / "logs").glob("gpu-telemetry-*.csv"))
    for folder in (root / "rl-rollouts", root / "judge-queue", matched / "review"):
        if folder.is_dir():
            optional.extend(path for path in folder.rglob("*")
                            if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES)
    return sorted(set(required + optional))


def package(root: Path, step: int, output: Path) -> dict:
    files = select(root, step)
    records = []
    for path in files:
        if path.is_symlink() or path.suffix.lower() not in ALLOWED_SUFFIXES:
            raise ValueError(f"unsafe evidence path: {path.relative_to(root)}")
        relative = path.relative_to(root)
        if any(part.startswith(".") or part in {"private", "credentials", "cache"} for part in relative.parts):
            raise ValueError(f"unsafe evidence path: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            payload = path.read_bytes()
            if any(marker in payload for marker in FORBIDDEN_TEXT):
                raise ValueError(f"credential-like text in evidence path: {relative}")
        records.append({"path": relative.as_posix(), "sha256": digest(path), "bytes": path.stat().st_size})
    manifest = json.dumps({"schema": "painter-overfit-public-evidence-v1", "files": records},
                          indent=2, sort_keys=True).encode() + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
        header = tarfile.TarInfo("evidence-manifest.json")
        header.size = len(manifest)
        header.mode = 0o644
        archive.addfile(header, io.BytesIO(manifest))
    return {"files": len(files), "bytes": output.stat().st_size, "sha256": digest(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    token_file = args.token_file
    if token_file.is_symlink() or not token_file.is_file() or token_file.stat().st_mode & 0o077:
        raise PermissionError("HF token file must be owner-only and not a symlink")
    output = root / "final-evidence.tar.gz"
    summary = package(root, args.step, output)
    token = token_file.read_text().strip()
    api = HfApi(token=token)
    info = api.repo_info(REPO_ID, repo_type="dataset")
    if info.private:
        raise ValueError("evidence dataset is private")
    commit = api.upload_file(path_or_fileobj=str(output), path_in_repo=REMOTE_PATH,
                             repo_id=REPO_ID, repo_type="dataset", token=token)
    # Verify with no credential. A private or incomplete upload cannot pass.
    downloaded = Path(hf_hub_download(REPO_ID, filename=REMOTE_PATH, repo_type="dataset",
                                      revision="main", token=False, force_download=True,
                                      local_dir=root / "public-verification"))
    if digest(downloaded) != summary["sha256"]:
        raise ValueError("public remote evidence hash mismatch")
    result = {**summary, "repo_id": REPO_ID, "path": REMOTE_PATH,
              "commit": getattr(commit, "oid", None), "private": False, "verified": True}
    (root / "final-evidence-upload.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
