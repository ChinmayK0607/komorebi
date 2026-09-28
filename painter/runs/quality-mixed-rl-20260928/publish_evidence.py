"""Publish screened mixed-RL evidence with anonymous remote hash verification."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "CK0607/komorebi-painter-experiments"
REMOTE_PATH = "runs/quality-mixed-rl-20260928/evidence.tar.gz"
RUN_NAME = "quality-mixed-anchors-20260928"
TEXT_SUFFIXES = {".js", ".json", ".jsonl", ".csv", ".html", ".toml"}
ALLOWED_SUFFIXES = TEXT_SUFFIXES | {".png", ".jpg", ".jpeg"}
FORBIDDEN_TEXT = (b"AI_GATEWAY_API_KEY", b"Authorization: Bearer", b"sk-or-v1-", b"hf_")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def select(old: Path, evaluation: Path, pack: Path, telemetry: Path,
           step: int) -> list[tuple[Path, str]]:
    if step not in (4, 8, 12):
        raise ValueError("evaluation step must be a planned public checkpoint: 4, 8, or 12")
    run = old / "async-output" / RUN_NAME
    matched = evaluation / f"matched-eval-step-{step}"
    required: list[tuple[Path, str]] = [
        (old / "native-rl-mixed-anchors.toml", "native-rl-mixed-anchors.toml"),
        (pack / "train-manifest.json", "curriculum/train-manifest.json"),
        (pack / "heldout-cases.json", "curriculum/heldout-cases.json"),
        (pack / "provenance.json", "curriculum/provenance.json"),
        (matched / "identity.json", "matched-eval/identity.json"),
        (matched / "cases.json", "matched-eval/cases.json"),
        (matched / "completion.json", "matched-eval/completion.json"),
        (matched / "review/index.html", "matched-eval/review/index.html"),
        (matched / "review/report.json", "matched-eval/review/report.json"),
        (matched / "review/judge.json", "matched-eval/review/judge.json"),
        (telemetry, "telemetry/gpu-telemetry.csv"),
    ]
    for checkpoint in (4, 8, 12):
        if checkpoint > step:
            continue
        required.append((run / f"artifacts/adapters/step_{checkpoint}/hf-upload.json",
                         f"checkpoints/step_{checkpoint}/hf-upload.json"))
    missing = [name for path, name in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("required evidence missing: " + ", ".join(missing))
    optional: list[tuple[Path, str]] = []
    for folder, name in ((old / "rl-rollouts", "rl-rollouts"),
                         (old / "judge-queue", "judge-queue"),
                         (matched / "review", "matched-eval/review")):
        if folder.is_dir():
            optional.extend((path, f"{name}/{path.relative_to(folder).as_posix()}")
                            for path in folder.rglob("*")
                            if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES)
    indexed = {name: path for path, name in required + optional}
    return sorted(((path, name) for name, path in indexed.items()), key=lambda x: x[1])


def package(old: Path, evaluation: Path, pack: Path, telemetry: Path,
            output: Path, step: int) -> dict:
    files = select(old, evaluation, pack, telemetry, step)
    records = []
    for path, name in files:
        if path.is_symlink() or path.suffix.lower() not in ALLOWED_SUFFIXES:
            raise ValueError(f"unsafe evidence file: {name}")
        if any(part.startswith(".") or part in {"private", "credentials", "cache"}
               for part in Path(name).parts):
            raise ValueError(f"unsafe archive path: {name}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            payload = path.read_bytes()
            if any(marker in payload for marker in FORBIDDEN_TEXT):
                raise ValueError(f"credential-like text in evidence: {name}")
        records.append({"path": name, "sha256": digest(path), "bytes": path.stat().st_size})
    manifest = (json.dumps({"schema": "painter-mixed-rl-evidence-v1", "files": records},
                           indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for path, name in files:
            archive.add(path, arcname=name, recursive=False)
        header = tarfile.TarInfo("evidence-manifest.json")
        header.size = len(manifest)
        header.mode = 0o644
        archive.addfile(header, io.BytesIO(manifest))
    return {"files": len(files), "bytes": output.stat().st_size,
            "sha256": digest(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--step", type=int, default=12)
    args = parser.parse_args()
    token_file = args.token_file
    if token_file.is_symlink() or not token_file.is_file() or token_file.stat().st_mode & 0o077:
        raise PermissionError("HF token file must be owner-only and not a symlink")
    output = args.eval_root / "collected-evidence.tar.gz"
    summary = package(args.old_root, args.eval_root, args.pack_root,
                      args.telemetry, output, args.step)
    token = token_file.read_text().strip()
    api = HfApi(token=token)
    if api.repo_info(REPO_ID, repo_type="dataset").private:
        raise ValueError("evidence dataset is private")
    remote_path = REMOTE_PATH.replace("evidence.tar.gz", f"step-{args.step}-evidence.tar.gz")
    commit = api.upload_file(path_or_fileobj=str(output), path_in_repo=remote_path,
                             repo_id=REPO_ID, repo_type="dataset", token=token)
    remote = Path(hf_hub_download(REPO_ID, filename=remote_path, repo_type="dataset",
                                  revision=getattr(commit, "oid"), token=False,
                                  force_download=True,
                                  local_dir=args.eval_root / "public-verification"))
    if digest(remote) != summary["sha256"]:
        raise ValueError("anonymous remote evidence hash mismatch")
    receipt = {**summary, "repo_id": REPO_ID, "path": remote_path,
               "commit": getattr(commit, "oid"), "private": False, "verified": True}
    (args.eval_root / "collected-evidence-upload.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
