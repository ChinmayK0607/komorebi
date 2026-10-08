#!/usr/bin/env python3
"""Restore a pinned public native-anchor evidence archive into one run.

The restore is deliberately opt-in and evidence-only.  It accepts the
curriculum anchor tree and its native-generation traces/manifests, but never
restores model weights, Python source, browser assets, or arbitrary run files.
Existing files are immutable: an identical byte stream is reused, while a
different byte stream aborts before anything is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request


SCHEMA = "structured-flow-anchor-evidence-restore-v1"
RUN_ID = "structured-flow-rl-20261009"
ANCHOR_REL = Path("curriculum/anchors")
MAX_MEMBER_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
MAX_MEMBERS = 20_000
ALLOWED_SUFFIXES = {
    ".json", ".jsonl", ".ndjson", ".log", ".png", ".toml", ".txt", ".js", ".p5",
}
SHA256_HEX = set("0123456789abcdef")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA256_HEX


def _normalized_member(name: str) -> tuple[str, Path] | None:
    """Return an evidence kind and run-relative destination for a tar name."""
    if not isinstance(name, str) or not name or "\x00" in name:
        raise ValueError("archive member has an invalid name")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    parts = path.parts
    # Public run archives commonly carry one of these harmless directory
    # wrappers.  The destination is selected only by the final evidence tree;
    # no archive path can select a parent outside the run.
    anchor_index = None
    for index in range(len(parts) - 1):
        if parts[index:index + 2] == ("curriculum", "anchors"):
            anchor_index = index
            break
    if anchor_index is not None:
        relative = Path(*parts[anchor_index + 2:])
        kind = "curriculum-anchors"
    elif "anchors" in parts:
        index = parts.index("anchors")
        relative = Path(*parts[index + 1:])
        kind = "anchors"
    elif "native-generation" in parts:
        index = parts.index("native-generation")
        relative = Path(*parts[index:])
        kind = "native-generation"
    else:
        return None
    if not relative.parts:
        return None
    # Tar directory entries often omit a trailing slash.  They are accepted
    # as structural entries and never extracted as files; a file without an
    # extension is rejected below.
    if not relative.suffix and (name.endswith("/") or relative.name in {"anchors", "native-generation"}):
        return None
    if relative.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(f"archive member is outside the evidence file allowlist: {name!r}")
    if any(part in {"model", "weights", "checkpoint", "private", "credentials"} for part in relative.parts):
        raise ValueError(f"archive member names a forbidden subtree: {name!r}")
    return kind, ANCHOR_REL / relative


def _read_member(stream, size: int) -> bytes:
    if size < 0 or size > MAX_MEMBER_BYTES:
        raise ValueError(f"archive member exceeds the per-file limit: {size}")
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(min(1024 * 1024, remaining))
        if not chunk:
            raise ValueError("archive member ended before its declared size")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _archive_path(*, archive: Path | None, url: str | None, run_root: Path, expected: str) -> tuple[Path, str, tempfile.TemporaryDirectory[str] | None]:
    if archive is not None and url is not None:
        raise ValueError("choose one local archive or archive URL")
    if archive is None and url is None:
        raise ValueError("an archive path or URL is required")
    if archive is not None:
        path = archive.expanduser().resolve()
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"anchor evidence archive is missing or symlinked: {path}")
        return path, "local_archive", None
    parsed = urllib.parse.urlparse(str(url))
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise ValueError("anchor evidence URL must be an absolute HTTP(S) URL")
    temporary = tempfile.TemporaryDirectory(prefix="anchor-restore-", dir=str(run_root / ".restore-tmp"))
    path = Path(temporary.name) / "evidence.tar.gz"
    request = urllib.request.Request(str(url), headers={"User-Agent": "structured-flow-anchor-restore/1"})
    with urllib.request.urlopen(request, timeout=180) as response, path.open("wb") as output:
        total = 0
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > MAX_TOTAL_BYTES:
                raise ValueError("anchor evidence archive exceeds the total byte limit")
            output.write(block)
    return path, "public_http_archive", temporary


def _verify_optional_painting_hashes(anchor_root: Path) -> None:
    """Check explicit hashes in native ``painting.json`` evidence when present."""
    for painting in sorted(anchor_root.rglob("painting.json")):
        if painting.is_symlink() or not painting.is_file():
            raise ValueError(f"painting evidence is not a regular file: {painting}")
        try:
            value = json.loads(painting.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid painting evidence JSON: {painting}") from exc
        if not isinstance(value, dict):
            continue
        # Invalid/program-timeout receipts are preserved as raw evidence, but
        # their hash fields may intentionally be absent or placeholders.  The
        # admission contract only binds hashes for a valid painting receipt.
        if value.get("valid") is not True:
            continue
        for digest_key, path_keys, fallback_suffixes in (
            ("png_sha256", ("png", "canvas", "image", "painting", "final_canvas"), (".png",)),
            ("canvas_sha256", ("canvas", "final_canvas", "png"), (".png",)),
            ("source_sha256", ("source", "source_path", "program", "program_path"), (".js", ".p5")),
        ):
            expected = value.get(digest_key)
            if expected is None:
                continue
            if not _valid_sha(expected):
                raise ValueError(f"invalid {digest_key} in {painting}")
            candidate: Path | None = None
            for key in path_keys:
                reference = value.get(key)
                if isinstance(reference, str) and reference:
                    reference_path = Path(reference)
                    candidate = (painting.parent / reference_path.name).resolve() if reference_path.is_absolute() else (painting.parent / reference_path).resolve()
                    if not candidate.is_file():
                        candidate = None
                    break
            if candidate is None:
                siblings = [
                    path for suffix in fallback_suffixes
                    for path in sorted(painting.parent.glob(f"*{suffix}"))
                    if path.is_file() and not path.is_symlink()
                ]
                if len(siblings) == 1:
                    candidate = siblings[0].resolve()
            if candidate is None or candidate.is_symlink() or not candidate.is_file():
                raise ValueError(f"{digest_key} in {painting} has no regular evidence target")
            try:
                candidate.relative_to(anchor_root.resolve())
            except ValueError as exc:
                raise ValueError(f"{digest_key} target escapes restored anchors: {candidate}") from exc
            if sha256_file(candidate) != expected:
                raise ValueError(f"{digest_key} mismatch in {painting}")


def _case_seed(source_group: str, reference_sha256: str, retry_index: int) -> int:
    material = (
        f"20261009:{source_group}:{reference_sha256}"
        if retry_index == 0
        else f"20261009:retry-{retry_index}:{source_group}:{reference_sha256}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def _verify_native_manifests(anchor_root: Path) -> list[dict[str, object]]:
    manifests: list[dict[str, object]] = []
    for manifest in sorted((anchor_root / "native-generation").glob("batch-*/native-eval-manifest.json")):
        if manifest.is_symlink() or not manifest.is_file():
            raise ValueError(f"native manifest is not a regular file: {manifest}")
        value = json.loads(manifest.read_text(encoding="utf-8"))
        cases = value.get("cases") if isinstance(value, dict) else None
        if not isinstance(cases, list):
            raise ValueError(f"native manifest has no cases: {manifest}")
        config = manifest.parent / "native-eval.toml"
        if config.is_symlink() or not config.is_file():
            raise ValueError(f"native manifest has no regular config: {manifest}")
        config_text = config.read_text(encoding="utf-8")
        seed_match = re.search(r"(?m)^seed\s*=\s*(\d+)\s*$", config_text)
        if seed_match is None:
            raise ValueError(f"native manifest config has no batch seed: {config}")
        batch_seed = int(seed_match.group(1))
        retry_index = batch_seed - 20261009
        if retry_index not in {0, 1}:
            raise ValueError(f"native manifest config has an unsupported batch seed: {config}")
        for case in cases:
            if not isinstance(case, dict):
                raise ValueError(f"native manifest contains a non-object case: {manifest}")
            try:
                explicit_retry = case.get("retry_index")
                if explicit_retry is not None and int(explicit_retry) != retry_index:
                    raise ValueError(f"native manifest case retry index mismatch: {case.get('id')}")
                expected = _case_seed(str(case["source_group"]), str(case["reference_sha256"]), retry_index)
                actual = int(case["case_seed"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"native manifest case lacks canonical seed fields: {manifest}") from exc
            if actual != expected:
                raise ValueError(f"native manifest case seed mismatch: {case.get('id')}")
        manifests.append({"path": str(manifest.relative_to(anchor_root)), "cases": len(cases)})
    return manifests


def restore(*, run_root: Path, expected_sha256: str, archive: Path | None = None, url: str | None = None) -> dict[str, object]:
    run_root = run_root.expanduser().resolve()
    if not run_root.is_dir():
        raise ValueError(f"run root is missing: {run_root}")
    if not _valid_sha(expected_sha256):
        raise ValueError("expected archive SHA-256 must be lowercase hexadecimal")
    (run_root / ".restore-tmp").mkdir(mode=0o700, exist_ok=True)
    archive_path, source_kind, temporary = _archive_path(
        archive=archive, url=url, run_root=run_root, expected=expected_sha256,
    )
    try:
        actual_archive_sha = sha256_file(archive_path)
        if actual_archive_sha != expected_sha256:
            raise ValueError(f"source evidence archive hash mismatch: {actual_archive_sha}")
        stage_parent = Path(tempfile.mkdtemp(prefix="anchor-restore-stage-", dir=str(run_root / ".restore-tmp")))
        try:
            stage_root = stage_parent / ANCHOR_REL
            records: list[dict[str, object]] = []
            seen: set[Path] = set()
            total_bytes = 0
            with tarfile.open(archive_path, "r:*") as archive_file:
                for index, member in enumerate(archive_file):
                    if index >= MAX_MEMBERS:
                        raise ValueError("anchor evidence archive has too many members")
                    if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                        raise ValueError(f"archive contains an unsupported non-regular member: {member.name!r}")
                    if member.isdir():
                        directory = PurePosixPath(member.name)
                        if directory.is_absolute() or any(part in {"", ".", ".."} for part in directory.parts):
                            raise ValueError(f"unsafe archive member path: {member.name!r}")
                        continue
                    selected = _normalized_member(member.name)
                    if selected is None:
                        continue
                    _kind, target_relative = selected
                    if target_relative in seen:
                        raise ValueError(f"duplicate evidence member: {member.name!r}")
                    seen.add(target_relative)
                    total_bytes += int(member.size)
                    if total_bytes > MAX_TOTAL_BYTES:
                        raise ValueError("anchor evidence exceeds the total byte limit")
                    source = archive_file.extractfile(member)
                    if source is None:
                        raise ValueError(f"cannot read archive member: {member.name!r}")
                    payload = _read_member(source, int(member.size))
                    stage_path = stage_root / target_relative.relative_to(ANCHOR_REL)
                    stage_path.parent.mkdir(parents=True, exist_ok=True)
                    stage_path.write_bytes(payload)
                    stage_path.chmod(0o600)
                    records.append({
                        "archive_member": member.name,
                        "target": str(target_relative),
                        "bytes": len(payload),
                        "sha256": sha256_bytes(payload),
                    })
            if not records:
                raise ValueError("source archive contains no permitted anchor evidence")
            stage_anchor = stage_root
            _verify_optional_painting_hashes(stage_anchor)
            manifest_records = _verify_native_manifests(stage_anchor)
            target_anchor = run_root / ANCHOR_REL
            for record in records:
                target = run_root / str(record["target"])
                if target.is_symlink() or (target.exists() and not target.is_file()):
                    raise ValueError(f"restore target is not a regular file: {target}")
                if target.is_file() and sha256_file(target) != record["sha256"]:
                    raise ValueError(f"refusing to overwrite existing evidence: {target}")
            reused = 0
            installed = 0
            for record in records:
                target = run_root / str(record["target"])
                source = stage_root / target.relative_to(target_anchor)
                if target.is_file():
                    reused += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source.replace(target)
                target.chmod(0o600)
                installed += 1
            receipt = {
                "schema": SCHEMA,
                "run_id": RUN_ID,
                "status": "restored_verified_public_evidence",
                "source_kind": source_kind,
                "source_archive_sha256": expected_sha256,
                "source_url_host": urllib.parse.urlparse(url).netloc if url else None,
                "anchor_root": str(target_anchor),
                "member_count": len(records),
                "installed_count": installed,
                "reused_count": reused,
                "native_manifests": manifest_records,
                "members": records,
                "limitations": [
                    "Evidence was restored from a hash-verified public archive; it is not relabeled as newly generated.",
                    "Model weights, browser assets, source code, and credentials were excluded by the restore allowlist.",
                ],
            }
            receipt_path = run_root / "anchor-restore-receipt.json"
            encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
            if receipt_path.exists():
                if receipt_path.is_symlink():
                    raise ValueError(f"existing restore receipt is symlinked: {receipt_path}")
                try:
                    existing = json.loads(receipt_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"existing restore receipt is invalid: {receipt_path}") from exc
                # Counts are run-local bookkeeping and differ on a replay
                # (installed on the first call, reused on the second).  The
                # archive hash and per-member byte identities are immutable;
                # preserve the original receipt rather than rewriting paid
                # evidence facts with a new replay count.
                if (
                    existing.get("schema") != SCHEMA
                    or existing.get("source_archive_sha256") != expected_sha256
                    or existing.get("members") != records
                ):
                    raise ValueError(f"existing restore receipt differs: {receipt_path}")
                return existing
            else:
                receipt_path.write_text(encoded, encoding="utf-8")
                receipt_path.chmod(0o600)
            return receipt
        finally:
            shutil.rmtree(stage_parent, ignore_errors=True)
    finally:
        if temporary is not None:
            temporary.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--url")
    args = parser.parse_args()
    receipt = restore(
        run_root=args.run_root,
        expected_sha256=args.sha256,
        archive=args.archive,
        url=args.url,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
