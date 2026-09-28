"""Publish the relative-RL audit before a node is released."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "CK0607/komorebi-painter-experiments"
REMOTE = "runs/quality-relative-rl-20260928/step-4-evidence.tar.gz"
RUN = "quality-relative-eight-20260928"
TEXT = {".js", ".json", ".jsonl", ".csv", ".html", ".toml", ".log", ".md"}
ALLOWED = TEXT | {".png", ".jpg", ".jpeg"}
FORBIDDEN = (b"AI_GATEWAY_API_KEY", b"Authorization: Bearer", b"sk-or-v1-", b"hf_")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def interrupted_files(interrupted: Path) -> list[tuple[Path, str]]:
    """Keep the stopped diagnostic's evidence, excluding duplicate step-zero weights."""
    folders = ("group-credit", "rl-rollouts", "judge-queue", "reward-batches",
               "async-output/logs", "async-output/monitors")
    selected = []
    for folder in folders:
        source = interrupted / folder
        if source.is_dir():
            selected.extend((path, f"interrupted/{path.relative_to(interrupted).as_posix()}")
                            for path in source.rglob("*")
                            if path.is_file() and path.suffix.lower() in ALLOWED)
    for name in ("launch.log", "gateway-worker.log", "gpu-telemetry.csv"):
        path = interrupted / name
        if path.is_file():
            selected.append((path, f"interrupted/{name}"))
    return selected


def select(old: Path, root: Path, pack: Path, telemetry: Path,
           interrupted: Path | None = None) -> list[tuple[Path, str]]:
    run = old / "async-output" / RUN
    matched = root / "matched-eval-step-4"
    step_two_receipt = run / "artifacts/adapters/step_2/hf-upload.json"
    if not step_two_receipt.is_file():
        # Prime may prune the local step-two directory after saving step four.
        step_two_receipt = old / "checkpoint-receipts/step_2/hf-upload.json"
    required = [
        (old / "native-rl-relative-eight.toml", "config/native-rl-relative-eight.toml"),
        (pack / "train-manifest.json", "curriculum/train-manifest.json"),
        (pack / "provenance.json", "curriculum/provenance.json"),
        (matched / "identity.json", "matched-eval/identity.json"),
        (matched / "cases.json", "matched-eval/cases.json"),
        (matched / "completion.json", "matched-eval/completion.json"),
        (matched / "review/index.html", "matched-eval/review/index.html"),
        (matched / "review/report.json", "matched-eval/review/report.json"),
        (matched / "review/judge.json", "matched-eval/review/judge.json"),
        (telemetry, "telemetry/gpu-telemetry.csv"),
        (run / "monitors/file/traces/stream/00000.jsonl", "prime/traces/stream/00000.jsonl"),
        (run / "monitors/file/metrics.jsonl", "prime/metrics.jsonl"),
        (run / "logs/attempt_1/orchestrator.log", "prime/orchestrator.log"),
        (run / "logs/attempt_1/trainer.log", "prime/trainer.log"),
        (step_two_receipt, "checkpoints/step_2/hf-upload.json"),
        (run / "artifacts/adapters/step_4/hf-upload.json", "checkpoints/step_4/hf-upload.json"),
    ]
    absent = [name for path, name in required if not path.is_file()]
    if absent:
        raise FileNotFoundError("missing required evidence: " + ", ".join(absent))
    optional: list[tuple[Path, str]] = []
    for folder, name in (
        (old / "group-credit", "group-credit"),
        (old / "rl-rollouts", "rl-rollouts"),
        (old / "judge-queue", "judge-queue"),
        (old / "reward-batches", "reward-batches"),
        (run / "monitors/file/traces", "prime/traces"),
        (matched / "review", "matched-eval/review"),
    ):
        if folder.is_dir():
            optional += [(path, f"{name}/{path.relative_to(folder).as_posix()}")
                         for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in ALLOWED]
    if not any(name.startswith("group-credit/") and name.endswith("/audit.json")
               for _, name in optional):
        raise FileNotFoundError("no per-group token-credit audit was retained")
    for name in ("gateway-worker-relative-eight.log", "gateway-worker-retry2.log"):
        path = old / "logs" / name
        if path.is_file():
            optional.append((path, f"worker/{name}"))
    if interrupted is not None:
        if not any((interrupted / "group-credit").rglob("audit.json")):
            raise FileNotFoundError("interrupted attempt lacks a group-credit audit")
        optional.extend(interrupted_files(interrupted))
    return sorted({name: path for path, name in required + optional}.items())


def package(old: Path, root: Path, pack: Path, telemetry: Path, output: Path,
            interrupted: Path | None = None) -> dict:
    selected = select(old, root, pack, telemetry, interrupted)
    records = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for name, path in selected:
            if path.is_symlink() or path.suffix.lower() not in ALLOWED:
                raise ValueError(f"unsafe evidence file: {name}")
            if any(part.startswith(".") or part in {"private", "credentials", "cache"}
                   for part in Path(name).parts):
                raise ValueError(f"unsafe evidence path: {name}")
            if path.suffix.lower() in TEXT and any(marker in path.read_bytes() for marker in FORBIDDEN):
                raise ValueError(f"credential-like text in evidence: {name}")
            records.append({"path": name, "sha256": sha(path), "bytes": path.stat().st_size})
            archive.add(path, arcname=name, recursive=False)
        data = (json.dumps({"schema": "painter-relative-rl-evidence-v1", "files": records},
                           indent=2, sort_keys=True) + "\n").encode()
        entry = tarfile.TarInfo("evidence-manifest.json")
        entry.size, entry.mode = len(data), 0o644
        archive.addfile(entry, io.BytesIO(data))
    return {"files": len(records), "bytes": output.stat().st_size, "sha256": sha(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--interrupted-root", type=Path)
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()
    token_file = args.token_file
    if token_file.is_symlink() or not token_file.is_file() or token_file.stat().st_mode & 0o077:
        raise PermissionError("HF token file must be owner-only and not a symlink")
    output = args.root / "matched-eval-step-4/collected-evidence.tar.gz"
    summary = package(args.old_root, args.root, args.pack_root, args.telemetry,
                      output, args.interrupted_root)
    token = token_file.read_text().strip()
    api = HfApi(token=token)
    if api.repo_info(REPO_ID, repo_type="dataset").private:
        raise ValueError("evidence dataset is private")
    commit = api.upload_file(path_or_fileobj=str(output), path_in_repo=REMOTE,
                             repo_id=REPO_ID, repo_type="dataset", token=token)
    remote = Path(hf_hub_download(REPO_ID, filename=REMOTE, repo_type="dataset",
                                  revision=commit.oid, token=False, force_download=True,
                                  local_dir=args.root / "matched-eval-step-4/public-verification"))
    if sha(remote) != summary["sha256"]:
        raise ValueError("anonymous public evidence hash mismatch")
    receipt = {**summary, "repo_id": REPO_ID, "path": REMOTE, "commit": commit.oid,
               "private": False, "verified": True}
    (args.root / "matched-eval-step-4/collected-evidence-upload.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
