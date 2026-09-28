"""Publish screened text-paint RL evidence without model weights or secrets."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "CK0607/komorebi-painter-experiments"
TEXT_EXT = {".json", ".jsonl", ".js", ".txt", ".toml", ".log", ".csv", ".md", ".html"}
IMAGE_EXT = {".png", ".jpg", ".jpeg"}
SECRET_PATTERNS = (
    re.compile(rb"AI_GATEWAY_API_KEY"),
    re.compile(rb"Authorization:\s*Bearer", re.I),
    re.compile(rb"sk-or-v1-[A-Za-z0-9]{20,}"),
    re.compile(rb"vck_[A-Za-z0-9]{20,}"),
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _add_dir(found: dict[str, Path], root: Path, folder: str) -> None:
    source = root / folder
    if not source.is_dir():
        return
    for path in source.rglob("*"):
        if path.is_file() and path.suffix.lower() in TEXT_EXT | IMAGE_EXT:
            found[path.relative_to(root).as_posix()] = path


def select(run: Path, *, run_name: str, eval_dir: str) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for name in (
        "EXPERIMENT.md", "native-rl.toml", "rl-config-receipt-initial.json",
        "anchors-frozen/train-manifest.json",
        "anchors-frozen/validation-manifest.json",
        "prepared/pool.json",
        f"async-output/{run_name}/monitors/file/traces/stream/00000.jsonl",
        f"async-output/{run_name}/logs/attempt_1/orchestrator.log",
        f"async-output/{run_name}/logs/attempt_1/trainer.log",
        f"async-output/{run_name}/artifacts/adapters/step_4/hf-upload.json",
        "validation/sft-step512-control/comparison/summary.json",
        f"validation/{eval_dir}/comparison/summary.json",
    ):
        path = run / name
        if not path.is_file():
            raise FileNotFoundError(name)
        files[name] = path
    for folder in (
        "prepared/prompts", "anchors-frozen/anchors", "rl-rollouts",
        "validation/sft-step512-control/comparison",
        f"validation/{eval_dir}/comparison",
    ):
        _add_dir(files, run, folder)
    for path in sorted((run / "logs").glob("gpu-telemetry*.csv")):
        # The first telemetry file belongs to the first attempt; later files
        # represent separately preserved attempts.
        files[path.relative_to(run).as_posix()] = path
        break
    return files


def package(files: dict[str, Path], output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with tarfile.open(output, "w:gz") as archive:
        for name, path in sorted(files.items()):
            if path.is_symlink() or any(part.startswith(".") or part in
                                       {"private", "credentials", "cache"}
                                       for part in Path(name).parts):
                raise ValueError(f"unsafe evidence path: {name}")
            if path.suffix.lower() not in TEXT_EXT | IMAGE_EXT:
                raise ValueError(f"unsafe evidence extension: {name}")
            if path.suffix.lower() in TEXT_EXT:
                content = path.read_bytes()
                if any(pattern.search(content) for pattern in SECRET_PATTERNS):
                    raise ValueError(f"credential-like text in evidence: {name}")
            rows.append({"path": name, "sha256": digest(path),
                         "bytes": path.stat().st_size})
            archive.add(path, arcname=name, recursive=False)
        manifest = (json.dumps({"schema": "painter-text-rl-evidence-v1",
                                "files": rows}, sort_keys=True, indent=2) + "\n").encode()
        entry = tarfile.TarInfo("evidence-manifest.json")
        entry.size, entry.mode = len(manifest), 0o644
        archive.addfile(entry, io.BytesIO(manifest))
    return {"files": len(rows), "bytes": output.stat().st_size,
            "sha256": digest(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-path", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()
    if args.remote_path.startswith("/") or ".." in Path(args.remote_path).parts:
        raise ValueError("remote path must be a safe repository path")
    token_file = args.token_file
    if token_file.is_symlink() or not token_file.is_file() or token_file.stat().st_mode & 0o077:
        raise PermissionError("HF token file must be owner-only and not a symlink")
    summary = package(select(args.run_root, run_name=args.run_name,
                             eval_dir=args.eval_dir), args.output)
    api = HfApi(token=token_file.read_text().strip())
    if api.repo_info(REPO_ID, repo_type="dataset").private:
        raise ValueError("evidence dataset must be public")
    commit = api.upload_file(path_or_fileobj=str(args.output),
                             path_in_repo=args.remote_path, repo_id=REPO_ID,
                             repo_type="dataset")
    public = Path(hf_hub_download(REPO_ID, filename=args.remote_path,
                                  repo_type="dataset", revision=commit.oid,
                                  token=False, force_download=True,
                                  local_dir=args.output.parent / "public-verify"))
    if digest(public) != summary["sha256"]:
        raise ValueError("public evidence hash mismatch")
    receipt = {**summary, "repo_id": REPO_ID, "path": args.remote_path,
               "commit": commit.oid, "public_hash_verified": True}
    args.output.with_suffix(".upload.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
