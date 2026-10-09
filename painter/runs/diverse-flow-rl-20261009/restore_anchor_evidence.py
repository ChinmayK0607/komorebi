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
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
from typing import Mapping


SCHEMA = "Diverse-flow-anchor-evidence-restore-v1"
RUN_ID = "diverse-flow-rl-20261009"
LEGACY_RUN_ID = "structured-flow-rl-20261009"
# One public archive is the only permitted source for a same-run retry.  Keep
# the revision in the URL and bind it again to the downloaded bytes; callers
# must not substitute a local path, a moving ``main`` URL, or another run's
# evidence archive.
SAME_RUN_ARCHIVE_REVISION = "c35b77c6d3a858ab224f0812f1ed73e0150d1525"
SAME_RUN_ARCHIVE_URL = (
    "https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/resolve/"
    f"{SAME_RUN_ARCHIVE_REVISION}/runs/{RUN_ID}/evidence.tar.gz"
)
SAME_RUN_ARCHIVE_SHA256 = "ddd90296e2c6562cb24bc8f8033f439c4925e77002ad8eabca2723c38323eaac"
ANCHOR_REL = Path("curriculum/anchors")
MAX_MEMBER_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
MAX_MEMBERS = 20_000
ALLOWED_SUFFIXES = {
    ".json", ".jsonl", ".ndjson", ".log", ".png", ".toml", ".txt", ".js", ".p5",
}
SHA256_HEX = set("0123456789abcdef")

# Keep the restore validator coupled to the schema and pilot sizes emitted by
# the run's curriculum builder.  Loading this small, side-effect-free module
# avoids a second copy of the native manifest identity in the restore path.
_CURRICULUM_SPEC = importlib.util.spec_from_file_location(
    "diverse_flow_prepare_curriculum_contract", Path(__file__).with_name("prepare_curriculum.py")
)
if _CURRICULUM_SPEC is None or _CURRICULUM_SPEC.loader is None:
    raise RuntimeError("cannot load prepare_curriculum.py for the native manifest contract")
_CURRICULUM_MODULE = importlib.util.module_from_spec(_CURRICULUM_SPEC)
_CURRICULUM_SPEC.loader.exec_module(_CURRICULUM_MODULE)
NATIVE_MANIFEST_SCHEMA = str(_CURRICULUM_MODULE.NATIVE_SCHEMA)
PILOT_TRAIN_COUNT = int(_CURRICULUM_MODULE.PILOT_TRAIN_COUNT)
PILOT_HELDOUT_COUNT = int(_CURRICULUM_MODULE.PILOT_HELDOUT_COUNT)
DISABLED_CASEBRIEF_SYSTEM = "Structured-flow hint is disabled for this run."
DISABLED_CASEBRIEF_PREFIX = f"[SYSTEM]\n{DISABLED_CASEBRIEF_SYSTEM}\n\n[USER]\n"


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


def validate_same_run_archive_pin(*, url: str, expected_sha256: str) -> None:
    """Reject every archive coordinate except the reviewed public retry pin."""
    if url != SAME_RUN_ARCHIVE_URL:
        raise ValueError("same-run archive URL does not match the pinned public revision")
    if expected_sha256 != SAME_RUN_ARCHIVE_SHA256:
        raise ValueError("same-run archive SHA-256 does not match the pinned public archive")
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "huggingface.co"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("same-run archive URL must be credential-free HTTPS Hugging Face")


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
    request = urllib.request.Request(str(url), headers={"User-Agent": "Diverse-flow-anchor-restore/1"})
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
    if url is not None:
        validate_same_run_archive_pin(url=url, expected_sha256=expected_sha256)
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



# The supervision archive is a separate, explicit restore mode.  It is kept
# out of the anchor allowlist because it contains authored programs, PNGs, and
# Flash decisions rather than native opponent traces.  A caller must provide
# the renderer identity expected by the current node; the archive is never
# admitted merely because its outer tarball hash matches.
SUPERVISION_SCHEMA = "Diverse-flow-supervision-evidence-restore-v1"
SUPERVISION_REL = Path("code-supervision")
SUPERVISION_METADATA = {
    "manifest.json", "source-manifest.json", "prompt-manifest.json",
    "render-manifest.json", "render-manifest.jsonl", "manifest.jsonl",
    "approval-candidates.json", "supervision-phase.json", "supervision.json",
    "sft-approved.jsonl", "train.jsonl", "dev.jsonl",
}
SUPERVISION_ALLOWED_SUFFIXES = {".json", ".jsonl", ".ndjson", ".png", ".js", ".txt"}
SUPERVISION_RENDERER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
# The archive carries the historical filtered export for audit, but the new
# run regenerates these derived files with its current phase receipt.  They
# may therefore differ on an idempotent preparation retry; immutable programs,
# renders, manifests, and decisions still require byte identity.
SUPERVISION_DERIVED = {"supervision-phase.json", "supervision.json", "sft-approved.jsonl", "train.jsonl", "dev.jsonl"}

# Matched Flash12 evaluation is a separate restore mode.  It never shares the
# anchor allowlist: baseline rollout evidence includes invalid/censored turns
# that must remain auditable, while native anchor restoration is a different
# admission boundary.
BASELINE_SCHEMA = "Diverse-flow-matched-baseline-evidence-restore-v1"
LEGACY_CANDIDATE_SCHEMA = "painter.structured-flow-candidates.v1"
LEGACY_PROGRAM_SCHEMA = "painter.structured-flow-program.v1"
LEGACY_RENDER_SCHEMA = "painter.structured-flow-render.v1"
LEGACY_APPROVAL_SCHEMA = "painter.structured-flow-approval-candidates.v1"
BASELINE_REL = Path("matched-baseline/pre-step12")
BASELINE_METADATA = {
    "baseline-receipt.json", "eval.toml", "Diverse-flow-hook-receipt.json",
    "baseline-reuse-receipt.json", "config.json",
}
BASELINE_ALLOWED_SUFFIXES = {".json", ".jsonl", ".ndjson", ".log", ".png", ".js", ".txt", ".toml"}
BASELINE_POLICY_REVISION = "1227a2427a8ae16014d2fdca4d1adb6855d3fa1c"
BASELINE_POLICY_PREFIX = "rl-judge-matched-20261004-flash/step_12"
BASELINE_POLICY_SHA256 = "c70c47ec0d6d7172b3bc25c4f54178919cf7c079cd9652fff3bc76db9e1eca79"
BASELINE_SEED = 918210
BASELINE_MAX_TURNS = 3
BASELINE_CASES = 32
BASELINE_MAX_TOKENS = 8192


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _normalized_supervision_member(name: str) -> Path | None:
    if not isinstance(name, str) or not name or "\x00" in name:
        raise ValueError("archive member has an invalid name")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    try:
        index = path.parts.index("code-supervision")
    except ValueError:
        return None
    relative = Path(*path.parts[index + 1:])
    if not relative.parts:
        return None
    if relative.suffix.lower() not in SUPERVISION_ALLOWED_SUFFIXES:
        raise ValueError(f"archive member is outside the supervision allowlist: {name!r}")
    parts = relative.parts
    allowed = (
        (len(parts) == 1 and relative.name in SUPERVISION_METADATA)
        or (len(parts) == 2 and parts[0] == "supervision-briefs" and relative.suffix == ".txt")
        or (len(parts) == 3 and parts[0] == "programs" and relative.suffix == ".js" and parts[2] in {"block_in.js", "improved_final.js"})
        or (len(parts) == 3 and parts[0] == "renders" and relative.suffix in {".png", ".json"} and parts[2] in {"block_in.png", "improved_final.png", "block_in.json", "improved_final.json"})
    )
    if not allowed:
        raise ValueError(f"archive member is outside the supervision allowlist: {name!r}")
    return SUPERVISION_REL / relative


def _normalized_baseline_member(name: str) -> Path | None:
    """Map only the cached matched-baseline tree into the run root."""
    if not isinstance(name, str) or not name or "\x00" in name:
        raise ValueError("archive member has an invalid name")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    try:
        index = next(i for i in range(len(path.parts) - 1) if path.parts[i:i + 2] == ("matched-baseline", "pre-step12"))
    except StopIteration:
        return None
    relative = Path(*path.parts[index + 2:])
    if not relative.parts:
        return None
    if relative.suffix.lower() not in BASELINE_ALLOWED_SUFFIXES:
        raise ValueError(f"archive member is outside the matched-baseline allowlist: {name!r}")
    parts = relative.parts
    allowed = (
        (len(parts) == 1 and relative.name in BASELINE_METADATA)
        or (len(parts) >= 2 and parts[0] in {"evaluation-rollouts", "logs"})
    )
    if not allowed:
        raise ValueError(f"archive member is outside the matched-baseline allowlist: {name!r}")
    if any(part.lower() in {"model", "weights", "checkpoint", "private", "credentials", "cache"} for part in parts):
        raise ValueError(f"archive member names a forbidden subtree: {name!r}")
    return BASELINE_REL / relative


def _supervision_json(root: Path, name: str) -> dict[str, object]:
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"supervision evidence is missing: {name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid supervision evidence JSON: {name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"supervision evidence must be an object: {name}")
    return value


def _require_supervision_hash(value: object, label: str) -> str:
    if not _valid_sha(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return str(value)


def _validate_supervision_tree(supervision_root: Path, expected_renderer_sha256: str) -> dict[str, object]:
    """Validate the complete paid supervision tree before it can be reused."""
    expected_renderer_sha256 = _require_supervision_hash(expected_renderer_sha256, "renderer SHA-256")
    manifest = _supervision_json(supervision_root, "manifest.json")
    if not (
        (manifest.get("schema") == "painter.Diverse-flow-candidates.v1" and manifest.get("run_id") == RUN_ID)
        or (manifest.get("schema") == LEGACY_CANDIDATE_SCHEMA and manifest.get("run_id") == LEGACY_RUN_ID)
    ):
        raise ValueError("supervision manifest is not the Diverse-flow candidate manifest")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 16 or manifest.get("program_count") != 32:
        raise ValueError("supervision manifest must contain exactly sixteen candidates and 32 programs")
    source_manifest = _supervision_json(supervision_root, "source-manifest.json")
    if not (
        (source_manifest.get("schema") == "painter.Diverse-flow-program.v1" and source_manifest.get("run_id") == RUN_ID)
        or (source_manifest.get("schema") == LEGACY_PROGRAM_SCHEMA and source_manifest.get("run_id") == LEGACY_RUN_ID)
    ):
        raise ValueError("supervision source manifest is not the pinned program manifest")
    source_manifest_sha = sha256_file(supervision_root / "source-manifest.json")
    if manifest.get("source_manifest_sha256") != source_manifest_sha:
        raise ValueError("candidate manifest source-manifest hash mismatch")
    source_entries = source_manifest.get("sources")
    if not isinstance(source_entries, list) or len(source_entries) != 32:
        raise ValueError("supervision source manifest must contain 32 programs")
    source_by_path: dict[str, dict[str, object]] = {}
    for entry in source_entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("supervision source manifest has a malformed program entry")
        rel = Path(str(entry["path"]))
        if rel.is_absolute() or rel.parts[:1] != ("programs",) or rel.parts[-1] not in {"block_in.js", "improved_final.js"}:
            raise ValueError("supervision source manifest contains an unsafe program path")
        target = supervision_root / rel
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"supervision program is missing: {rel}")
        actual = sha256_file(target)
        if actual != entry.get("sha256"):
            raise ValueError(f"supervision program hash mismatch: {rel}")
        source_by_path[rel.as_posix()] = entry
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("supervision candidate row is malformed")
        programs = row.get("programs")
        if not isinstance(programs, list) or len(programs) != 2 or any(str(path) not in source_by_path for path in programs):
            raise ValueError(f"candidate {row.get('id')} has unbound programs")

    render_manifest = _supervision_json(supervision_root, "render-manifest.json")
    if not (
        (render_manifest.get("schema") == "painter.Diverse-flow-render.v1" and render_manifest.get("run_id") == RUN_ID)
        or (render_manifest.get("schema") == LEGACY_RENDER_SCHEMA and render_manifest.get("run_id") == LEGACY_RUN_ID)
    ):
        raise ValueError("supervision render manifest is not the pinned render protocol")
    if render_manifest.get("source_manifest_sha256") != source_manifest_sha:
        raise ValueError("render manifest source-manifest hash mismatch")
    renderer = render_manifest.get("renderer")
    if not isinstance(renderer, dict) or renderer.get("renderer_sha256") != expected_renderer_sha256:
        raise ValueError("supervision render manifest renderer hash does not match the current renderer")
    results = render_manifest.get("results")
    if not isinstance(results, list) or len(results) != 32 or render_manifest.get("program_count") != 32 or render_manifest.get("valid_program_count") != 32 or render_manifest.get("complete_hash_verified") is not True:
        raise ValueError("supervision render manifest is incomplete")
    render_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("supervision render result is malformed")
        candidate_id, version = str(result.get("candidate_id")), str(result.get("version"))
        if version not in {"block_in", "improved_final"} or (candidate_id, version) in render_by_key:
            raise ValueError("supervision render result identity is invalid or duplicated")
        source_rel = Path("programs") / candidate_id / f"{version}.js"
        png_rel = Path("renders") / candidate_id / f"{version}.png"
        source = supervision_root / source_rel
        png = supervision_root / png_rel
        if result.get("source_sha256") != sha256_file(source) or result.get("png_sha256") != sha256_file(png):
            raise ValueError(f"supervision render hash mismatch: {candidate_id}/{version}")
        receipt = result.get("receipt")
        if not isinstance(receipt, dict) or receipt.get("renderer_sha256") != expected_renderer_sha256 or receipt.get("valid") is not True:
            raise ValueError(f"supervision renderer receipt is not bound: {candidate_id}/{version}")
        render_by_key[(candidate_id, version)] = result

    approval = _supervision_json(supervision_root, "approval-candidates.json")
    if not (
        (approval.get("schema") == "painter.Diverse-flow-approval-candidates.v1" and approval.get("run_id") == RUN_ID)
        or (approval.get("schema") == LEGACY_APPROVAL_SCHEMA and approval.get("run_id") == LEGACY_RUN_ID)
    ) or approval.get("status") != "reviewed" or approval.get("review_protocol") != "painter.text_pairwise_consistent_side_judge_v2_generic" or approval.get("judge_model") != "xiaomi/mimo-v2.6-flash":
        raise ValueError("supervision approval manifest protocol is not the pinned generic Flash review")
    candidates = approval.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 16 or approval.get("candidate_count") != 16:
        raise ValueError("supervision approval manifest must contain all sixteen decisions")
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("supervision approval candidate is malformed")
        candidate_id = str(candidate.get("id"))
        if candidate_id in seen:
            raise ValueError("supervision approval candidate is duplicated")
        seen.add(candidate_id)
        brief = Path(str(candidate.get("brief", "")))
        if brief.is_absolute() or brief.parts[:1] != ("supervision-briefs",):
            raise ValueError(f"supervision brief path is unsafe: {candidate_id}")
        brief_path = supervision_root / brief
        if not brief_path.is_file() or sha256_file(brief_path) != candidate.get("brief_sha256"):
            raise ValueError(f"supervision brief hash mismatch: {candidate_id}")
        for version, field in (("block_in", "initial_png_sha256"), ("improved_final", "final_png_sha256")):
            record = render_by_key.get((candidate_id, version))
            if record is None or candidate.get(field) != record.get("png_sha256"):
                raise ValueError(f"supervision approval/render hash mismatch: {candidate_id}/{version}")
        if candidate.get("decision") not in {"approved", "censored"}:
            raise ValueError(f"supervision candidate decision is invalid: {candidate_id}")
    if seen != {str(row.get("id")) for row in rows}:
        raise ValueError("supervision approval IDs do not match candidate manifest")
    return {
        "schema": SUPERVISION_SCHEMA,
        "source_run_id": str(manifest.get("run_id")),
        "source_manifest_schema": str(manifest.get("schema")),
        "candidate_count": len(rows),
        "program_count": len(source_entries),
        "render_count": len(results),
        "approved_count": sum(item.get("decision") == "approved" for item in candidates),
        "source_manifest_sha256": source_manifest_sha,
        "renderer_sha256": expected_renderer_sha256,
        "approval_manifest_sha256": sha256_file(supervision_root / "approval-candidates.json"),
    }


def _baseline_json(root: Path, name: str) -> dict[str, object]:
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"matched-baseline evidence is missing: {name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid matched-baseline JSON: {name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"matched-baseline JSON must be an object: {name}")
    return value


def _toml_int(text: str, key: str) -> int | None:
    match = re.search(rf"(?m)^\s*{re.escape(key)}\s*=\s*(\d+)\s*$", text)
    return int(match.group(1)) if match else None


def _toml_string(text: str, key: str) -> str | None:
    match = re.search(rf"(?m)^\s*{re.escape(key)}\s*=\s*[\"']([^\"']+)[\"']\s*$", text)
    return match.group(1) if match else None


def _flow_prompt_hashes(flow_source: Path) -> dict[str, str]:
    """Load current source constants without starting native evaluation."""
    spec = importlib.util.spec_from_file_location("diverse_flow_baseline_prompt", flow_source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import flow source: {flow_source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    system = getattr(module, "NATIVE_SYSTEM", None)
    hint = getattr(module, "FLOW_HINT", None)
    flow_system = getattr(module, "flow_system", None)
    if not isinstance(system, str) or not isinstance(hint, str) or not callable(flow_system):
        raise ValueError("current flow source does not expose native system and hint")
    return {
        "flow_source_sha256": sha256_file(flow_source),
        "native_system_sha256": sha256_bytes(system.encode("utf-8")),
        "flow_hint_sha256": sha256_bytes(hint.encode("utf-8")),
        "flow_system_sha256": sha256_bytes(str(flow_system()).encode("utf-8")),
    }


def _validate_current_flow_manifest(manifest_path: Path, prompt_hashes: Mapping[str, str]) -> dict[str, object]:
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"current evaluation manifest is missing: {manifest_path}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = value.get("cases") or value.get("rows") if isinstance(value, dict) else value
    if not isinstance(rows, list) or len(rows) != BASELINE_CASES:
        raise ValueError("current evaluation manifest must contain exactly sixteen cases")
    hint = None
    casebrief_mode = None
    disabled_hint = DISABLED_CASEBRIEF_SYSTEM
    casebriefs = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("current evaluation manifest has a malformed case")
        # Bind the paid baseline to the same frozen prompt and opponent PNG
        # identities that the current native taskset will load.  This catches
        # a regenerated manifest even when its casebrief text is unchanged.
        for path_key, hash_key in (("reference", "reference_sha256"), ("opponent", "opponent_sha256")):
            reference_value = row.get(path_key)
            digest_value = row.get(hash_key)
            if not isinstance(reference_value, str) or not reference_value or not _valid_sha(digest_value):
                raise ValueError(f"current evaluation manifest {path_key} identity is missing: {row.get('id')}")
            evidence_path = Path(reference_value).expanduser()
            if not evidence_path.is_absolute():
                evidence_path = manifest_path.parent / evidence_path
            evidence_path = evidence_path.absolute()
            if evidence_path.is_symlink() or not evidence_path.is_file() or sha256_file(evidence_path) != digest_value:
                raise ValueError(f"current evaluation manifest {path_key} hash mismatch: {row.get('id')}")
        casebrief = row.get("casebrief")
        if not isinstance(casebrief, str) or not casebrief:
            raise ValueError("current evaluation manifest casebrief is missing")
        path = Path(casebrief).expanduser()
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.absolute()
        if path.is_symlink() or not path.is_file() or sha256_file(path) != row.get("casebrief_sha256"):
            raise ValueError(f"current evaluation casebrief hash mismatch: {row.get('id')}")
        text = path.read_text(encoding="utf-8")
        if not text.startswith("[SYSTEM]\n") or "\n\n[USER]\n" not in text:
            raise ValueError(f"current evaluation casebrief markers are malformed: {row.get('id')}")
        system, user = text[len("[SYSTEM]\n"):].split("\n\n[USER]\n", 1)
        system = system.strip()
        if not system or not user.strip():
            raise ValueError(f"current evaluation casebrief is empty: {row.get('id')}")
        hint = hint or system
        if system != hint:
            raise ValueError("current evaluation casebrief system texts differ across cases")
        this_mode = "flow_hint" if sha256_bytes(system.encode("utf-8")) == prompt_hashes["flow_hint_sha256"] else None
        if system == disabled_hint:
            this_mode = "disabled_casebrief"
        elif sha256_bytes(system.encode("utf-8")) != prompt_hashes["flow_hint_sha256"]:
            raise ValueError("current evaluation casebrief system differs from the canonical flow hint")
        if casebrief_mode is None:
            casebrief_mode = this_mode
        elif casebrief_mode != this_mode:
            raise ValueError("current evaluation casebrief modes differ across cases")
        expected_hint_sha = sha256_bytes((system + "\n").encode("utf-8"))
        if row.get("system_hint_sha256") not in {None, expected_hint_sha}:
            raise ValueError(f"current evaluation flow-hint hash mismatch: {row.get('id')}")
        casebriefs += 1
    if hint is None or casebrief_mode is None:
        raise ValueError("current evaluation casebrief has no recognized system mode")
    return {"manifest_sha256": sha256_file(manifest_path), "case_count": len(rows), "casebrief_count": casebriefs, "casebrief_mode": casebrief_mode, **prompt_hashes}


def _curriculum_member(name: str) -> tuple[tuple[str, ...], str] | None:
    """Return a safe curriculum-relative member and its limited evidence kind."""
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe cached pilot archive member: {name!r}")
    if path.parts[-1] == "flow_demonstrations.py" and (len(path.parts) == 1 or RUN_ID in path.parts):
        return path.parts, "flow_source"
    try:
        curriculum_index = path.parts.index("curriculum")
    except ValueError:
        return None
    relative = path.parts[curriculum_index + 1:]
    if not relative:
        return None
    basename = relative[-1]
    if basename in {"pilot-train-manifest.json", "pilot-heldout-manifest.json", "matched32-manifest.json"}:
        if len(relative) != 2 or not relative[0].startswith(".materialized-pilot"):
            return None
        return relative, "manifest"
    if len(relative) >= 3 and relative[0].startswith(".materialized-pilot") and relative[1] == "casebriefs" and PurePosixPath(basename).suffix == ".txt":
        return relative, "casebrief"
    return None


def _manifest_rows(value: Mapping[str, object], *, name: str, expected_count: int) -> list[dict[str, object]]:
    if value.get("schema") != NATIVE_MANIFEST_SCHEMA or value.get("run_id") != RUN_ID:
        raise ValueError(f"cached {name} manifest is not the pinned native Diverse-flow manifest")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f"cached {name} manifest must contain {expected_count} rows")
    if name == "train" and value.get("split") != "train":
        raise ValueError("cached train manifest split mismatch")
    if name != "train" and value.get("split") != "heldout":
        raise ValueError(f"cached {name} manifest split mismatch")
    if name == "train" and value.get("native_rl_compatible") is not True:
        raise ValueError("cached train manifest is not native RL compatible")
    result: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError(f"cached {name} manifest contains a malformed row")
        result.append(row)
    return result


def _rebase_curriculum_path(value: object, *, run_root: Path, field: str) -> str:
    """Rebase one archived absolute path using only its known curriculum suffix."""
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError(f"cached pilot {field} path is not an absolute archived path")
    parts = PurePosixPath(value).parts
    try:
        curriculum_index = max(index for index, part in enumerate(parts) if part == "curriculum")
    except ValueError as exc:
        raise ValueError(f"cached pilot {field} path has no curriculum suffix: {value}") from exc
    suffix = parts[curriculum_index + 1:]
    if not suffix or any(part in {"", ".", ".."} for part in suffix):
        raise ValueError(f"cached pilot {field} path has an unsafe curriculum suffix: {value}")
    return str((run_root / "curriculum" / Path(*suffix)).absolute())


def _rebase_cached_row(
    row: Mapping[str, object],
    proposal_row: Mapping[str, object],
    *,
    proposal_path: Path,
    anchor_root: Path,
    run_root: Path,
) -> dict[str, object]:
    """Make archived admitted row paths portable while preserving all hashes."""
    rebased = dict(row)
    prompt_path = _proposal_file(proposal_path, proposal_row)
    rebased["reference"] = str(prompt_path)
    if "source_prompt" in rebased:
        rebased["source_prompt"] = str(prompt_path)
    casebrief = _rebase_curriculum_path(rebased.get("casebrief"), run_root=run_root, field="casebrief")
    rebased["casebrief"] = casebrief
    rebased["opponent"] = str((anchor_root / f"{rebased['id']}.png").absolute())
    if "anchor_provenance_file" in rebased:
        rebased["anchor_provenance_file"] = str((anchor_root / "anchor-provenance.json").absolute())
    return rebased


def _validate_cached_casebrief_text(text: str, expected_user: str) -> None:
    """Require the exact archived disabled marker and current prompt text."""
    if not text.startswith(DISABLED_CASEBRIEF_PREFIX):
        raise ValueError("cached casebrief does not preserve the canonical disabled flow marker")
    if text.split("\n\n[USER]\n", 1)[1].strip() != expected_user:
        raise ValueError("cached casebrief user brief differs from the current proposal")


def _proposal_file(proposal: Path, row: Mapping[str, object]) -> Path:
    reference = row.get("reference")
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"proposal reference is missing for {row.get('id')}")
    path = Path(reference).expanduser()
    if not path.is_absolute():
        path = proposal.parent / path
    path = path.absolute()
    if path.is_symlink() or not path.is_file() or not _valid_sha(row.get("reference_sha256")) or sha256_file(path) != row.get("reference_sha256"):
        raise ValueError(f"proposal reference hash mismatch for {row.get('id')}")
    return path


def _validate_cached_pilot_row(
    row: Mapping[str, object],
    proposal_row: Mapping[str, object],
    *,
    proposal_path: Path,
    anchor_root: Path,
    run_root: Path,
) -> list[tuple[Path, str, str]]:
    if str(row.get("id")) != str(proposal_row.get("id")):
        raise ValueError("cached pilot row ID differs from the current proposal")
    if row.get("admission_status") != "ready_for_native_painting_rl" or row.get("anchor_status") != "verified_file":
        raise ValueError(f"cached pilot row is not an admitted frozen-anchor row: {row.get('id')}")
    for key in ("source_group", "reference_sha256", "stage", "base_family_id", "split"):
        if row.get(key) != proposal_row.get(key):
            raise ValueError(f"cached pilot row {row.get('id')} differs from the current proposal: {key}")
    _proposal_file(proposal_path, proposal_row)
    opponent_sha = row.get("opponent_sha256")
    if not _valid_sha(opponent_sha):
        raise ValueError(f"cached pilot row has no opponent hash: {row.get('id')}")
    anchor = anchor_root / f"{row['id']}.png"
    if anchor.is_symlink() or not anchor.is_file() or sha256_file(anchor) != opponent_sha:
        raise ValueError(f"cached pilot opponent differs from the restored current anchor: {row.get('id')}")
    frozen = row.get("frozen_sft512_checkpoint_sha256")
    if frozen != "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740":
        raise ValueError(f"cached pilot checkpoint identity differs: {row.get('id')}")
    casebrief_value = row.get("casebrief")
    casebrief_sha = row.get("casebrief_sha256")
    if not isinstance(casebrief_value, str) or not casebrief_value.startswith("/") or not _valid_sha(casebrief_sha):
        raise ValueError(f"cached pilot casebrief identity is missing: {row.get('id')}")
    casebrief_path = Path(casebrief_value).expanduser().absolute()
    try:
        casebrief_path.relative_to(run_root)
    except ValueError as exc:
        raise ValueError(f"cached pilot casebrief escapes the current run root: {casebrief_path}") from exc
    expected_reference = _proposal_file(proposal_path, proposal_row).read_text(encoding="utf-8").strip()
    return [(casebrief_path, str(casebrief_sha), expected_reference)]


def restore_cached_pilot_manifests(
    *,
    run_root: Path,
    expected_sha256: str,
    train_proposal: Path,
    heldout_proposal: Path,
    anchor_root: Path,
    train_manifest: Path,
    heldout_manifest: Path,
    flow_source: Path,
    archive: Path | None = None,
    url: str | None = None,
) -> dict[str, object]:
    """Restore the exact admitted 96/32 manifests and referenced casebriefs.

    The archived casebriefs intentionally retain their disabled marker, and
    the current hook appends that archived hint at task load.  Canonical flow
    guidance remains pending for this matched baseline.  Absolute paths from
    the archived node are rebased only beneath the current run's curriculum
    root; all row hashes and semantic identities remain pinned.
    """
    run_root = run_root.expanduser().absolute()
    train_proposal = train_proposal.expanduser().absolute()
    heldout_proposal = heldout_proposal.expanduser().absolute()
    anchor_root = anchor_root.expanduser().absolute()
    train_manifest = train_manifest.expanduser().absolute()
    heldout_manifest = heldout_manifest.expanduser().absolute()
    flow_source = flow_source.expanduser().absolute()
    if not _valid_sha(expected_sha256):
        raise ValueError("expected cached pilot archive SHA-256 must be lowercase hexadecimal")
    if url is not None:
        validate_same_run_archive_pin(url=url, expected_sha256=expected_sha256)
    (run_root / ".restore-tmp").mkdir(mode=0o700, exist_ok=True)
    archive_path, source_kind, temporary = _archive_path(archive=archive, url=url, run_root=run_root, expected=expected_sha256)
    try:
        if sha256_file(archive_path) != expected_sha256:
            raise ValueError("cached pilot archive hash mismatch")
        members: dict[tuple[str, ...], tuple[str, bytes]] = {}
        flow_members: list[tuple[str, bytes]] = []
        with tarfile.open(archive_path, "r:*") as archive_file:
            for member in archive_file:
                if not member.isfile() or member.issym() or member.islnk():
                    continue
                selected = _curriculum_member(member.name)
                if selected is None:
                    continue
                relative, kind = selected
                if member.size > MAX_MEMBER_BYTES:
                    raise ValueError(f"cached pilot member exceeds the per-file limit: {member.name}")
                stream = archive_file.extractfile(member)
                if stream is None:
                    raise ValueError(f"cannot read cached pilot member: {member.name}")
                payload = _read_member(stream, int(member.size))
                if kind == "flow_source":
                    flow_members.append((member.name, payload))
                else:
                    members[relative] = (member.name, payload)
        manifest_names = ("pilot-train-manifest.json", "pilot-heldout-manifest.json", "matched32-manifest.json")
        candidates: dict[tuple[str, ...], dict[str, object]] = {}
        for relative, (member_name, payload) in members.items():
            if relative[-1] not in manifest_names:
                continue
            try:
                value = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"cached pilot manifest is invalid: {member_name}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"cached pilot manifest is not an object: {member_name}")
            candidates[relative[:-1]] = {**candidates.get(relative[:-1], {}), relative[-1]: value, "_bytes_" + relative[-1]: payload, "_member_" + relative[-1]: member_name}
        chosen = [stage for stage, value in candidates.items() if all(name in value for name in manifest_names)]
        if len(chosen) != 1:
            raise ValueError(f"cached pilot archive must contain exactly one complete admitted manifest set; found {len(chosen)}")
        stage = chosen[0]
        candidate = candidates[stage]
        train_value = candidate["pilot-train-manifest.json"]
        heldout_value = candidate["pilot-heldout-manifest.json"]
        matched_value = candidate["matched32-manifest.json"]
        if not isinstance(train_value, dict) or not isinstance(heldout_value, dict) or not isinstance(matched_value, dict):
            raise ValueError("cached pilot manifest set is malformed")
        train_rows = _manifest_rows(train_value, name="train", expected_count=PILOT_TRAIN_COUNT)
        heldout_rows = _manifest_rows(heldout_value, name="heldout", expected_count=PILOT_HELDOUT_COUNT)
        matched_rows = _manifest_rows(matched_value, name="matched32", expected_count=PILOT_HELDOUT_COUNT)
        if sha256_bytes(candidate["_bytes_pilot-heldout-manifest.json"]) != sha256_bytes(candidate["_bytes_matched32-manifest.json"]):
            raise ValueError("cached matched32 manifest differs from the archived heldout manifest")
        train_doc = json.loads(train_proposal.read_text(encoding="utf-8"))
        heldout_doc = json.loads(heldout_proposal.read_text(encoding="utf-8"))
        train_by_id = {str(row.get("id")): row for row in train_doc.get("rows", []) if isinstance(row, dict)}
        heldout_by_id = {str(row.get("id")): row for row in heldout_doc.get("rows", []) if isinstance(row, dict)}
        if len(train_by_id) != PILOT_TRAIN_COUNT or len(heldout_by_id) != PILOT_HELDOUT_COUNT:
            raise ValueError(f"current Diverse-flow proposals are not the pinned {PILOT_TRAIN_COUNT}/{PILOT_HELDOUT_COUNT} pool")
        rebased_train_rows: list[dict[str, object]] = []
        rebased_heldout_rows: list[dict[str, object]] = []
        needed_casebriefs: dict[Path, tuple[str, str]] = {}
        for rows, proposals, proposal_path in ((train_rows, train_by_id, train_proposal), (heldout_rows, heldout_by_id, heldout_proposal)):
            rebased_rows = rebased_train_rows if rows is train_rows else rebased_heldout_rows
            for row in rows:
                proposal_row = proposals.get(str(row["id"]))
                if proposal_row is None:
                    raise ValueError(f"cached pilot row is absent from the current proposal: {row['id']}")
                portable_row = _rebase_cached_row(
                    row, proposal_row, proposal_path=proposal_path,
                    anchor_root=anchor_root, run_root=run_root,
                )
                rebased_rows.append(portable_row)
                for casebrief_path, digest, expected_user in _validate_cached_pilot_row(portable_row, proposal_row, proposal_path=proposal_path, anchor_root=anchor_root, run_root=run_root):
                    prior = needed_casebriefs.get(casebrief_path)
                    if prior is not None and prior != (digest, expected_user):
                        raise ValueError(f"cached casebrief identity is inconsistent: {casebrief_path}")
                    needed_casebriefs[casebrief_path] = (digest, expected_user)
        rebased_matched_rows: list[dict[str, object]] = []
        for row in matched_rows:
            proposal_row = heldout_by_id.get(str(row["id"]))
            if proposal_row is None:
                raise ValueError(f"cached matched32 row is absent from the current proposal: {row['id']}")
            portable_row = _rebase_cached_row(
                row, proposal_row, proposal_path=heldout_proposal,
                anchor_root=anchor_root, run_root=run_root,
            )
            rebased_matched_rows.append(portable_row)
            if str(row["id"]) not in {str(item["id"]) for item in heldout_rows}:
                raise ValueError("cached matched32 IDs differ from the archived heldout IDs")
        def _portable_manifest(
            value: Mapping[str, object], rows: list[dict[str, object]], original_payload: bytes,
        ) -> tuple[bytes, bool]:
            portable = dict(value)
            portable["rows"] = rows
            if "cases" in portable:
                portable["cases"] = rows
            encoded = json.dumps(portable, indent=2, sort_keys=True) + "\n"
            payload = encoded.encode("utf-8")
            return payload, payload != original_payload

        train_payload, train_rebased = _portable_manifest(
            train_value, rebased_train_rows, candidate["_bytes_pilot-train-manifest.json"],
        )
        heldout_payload, heldout_rebased = _portable_manifest(
            heldout_value, rebased_heldout_rows, candidate["_bytes_pilot-heldout-manifest.json"],
        )
        matched_payload, matched_rebased = _portable_manifest(
            matched_value, rebased_matched_rows, candidate["_bytes_matched32-manifest.json"],
        )
        manifest_payloads = {
            "pilot-train-manifest.json": train_payload,
            "pilot-heldout-manifest.json": heldout_payload,
            "matched32-manifest.json": matched_payload,
        }
        selected_casebriefs: dict[Path, bytes] = {}
        for target, (digest, expected_user) in needed_casebriefs.items():
            parts = PurePosixPath(target).parts
            curriculum_index = parts.index("curriculum")
            suffix = parts[curriculum_index + 1:]
            matches = [(name, payload) for relative, (name, payload) in members.items() if relative == suffix]
            if len(matches) != 1 or sha256_bytes(matches[0][1]) != digest:
                raise ValueError(f"cached casebrief is missing or hash-mismatched: {target}")
            selected_casebriefs[target] = matches[0][1]
            text = matches[0][1].decode("utf-8")
            try:
                _validate_cached_casebrief_text(text, expected_user)
            except ValueError as exc:
                raise ValueError(f"{exc}: {target}") from exc
        current_flow_sha = sha256_file(flow_source)
        matching_flow = [payload for name, payload in flow_members if name.endswith("/flow_demonstrations.py") or name == "flow_demonstrations.py"]
        if len(matching_flow) != 1 or sha256_bytes(matching_flow[0]) != current_flow_sha:
            raise ValueError("cached archive flow_demonstrations.py does not match the current canonical source")
        staging_prefix = run_root / "curriculum" / Path(*stage)
        install_items = [
            (train_manifest, train_payload),
            (heldout_manifest, matched_payload),
            (staging_prefix / "pilot-train-manifest.json", train_payload),
            (staging_prefix / "pilot-heldout-manifest.json", heldout_payload),
            (staging_prefix / "matched32-manifest.json", matched_payload),
            *selected_casebriefs.items(),
        ]
        manifest_targets = {Path(train_manifest), Path(heldout_manifest), staging_prefix / "pilot-train-manifest.json", staging_prefix / "pilot-heldout-manifest.json", staging_prefix / "matched32-manifest.json"}
        for target, payload in install_items:
            target = Path(target)
            if target.is_symlink() or (target.exists() and (not target.is_file() or (target not in manifest_targets and target.read_bytes() != payload))):
                raise ValueError(f"refusing to overwrite changed cached pilot evidence: {target}")
        for target, payload in install_items:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target in manifest_targets:
                target.write_bytes(payload)
                target.chmod(0o600)
        reuse = {
            "schema": "Diverse-flow-cached-anchor-reuse-v1",
            "status": "reused_archived_admitted_pilot",
            "source_archive_sha256": expected_sha256,
            "source_kind": source_kind,
            "generator_calls": 0,
            "train_rows": PILOT_TRAIN_COUNT,
            "heldout_cases": PILOT_HELDOUT_COUNT,
            "casebrief_count": len(selected_casebriefs),
            "casebrief_mode": "disabled_casebrief_with_flow_hook",
            "flow_source_sha256": current_flow_sha,
            "train_manifest_sha256": sha256_file(train_manifest),
            "heldout_manifest_sha256": sha256_file(heldout_manifest),
            "manifest_rebased": bool(train_rebased or heldout_rebased or matched_rebased),
            "manifest_hashes": {
                "archived": {
                    "pilot-train-manifest.json": sha256_bytes(candidate["_bytes_pilot-train-manifest.json"]),
                    "pilot-heldout-manifest.json": sha256_bytes(candidate["_bytes_pilot-heldout-manifest.json"]),
                    "matched32-manifest.json": sha256_bytes(candidate["_bytes_matched32-manifest.json"]),
                },
                "installed": {
                    name: sha256_bytes(payload) for name, payload in manifest_payloads.items()
                },
            },
            "manifest_members": [candidate["_member_pilot-train-manifest.json"], candidate["_member_pilot-heldout-manifest.json"], candidate["_member_matched32-manifest.json"]],
            "limitations": ["Exact archived admitted manifests and referenced casebriefs were reused; no rematerialization, model, renderer, or judge call was made."],
        }
        _write_json(run_root / "cached-anchor-reuse.json", reuse)
        return reuse
    finally:
        if temporary is not None:
            temporary.cleanup()


def _validate_baseline_rollouts(root: Path, expected_renderer_sha256: str) -> dict[str, object]:
    rollout_root = root / "evaluation-rollouts"
    if not rollout_root.is_dir() or rollout_root.is_symlink():
        raise ValueError("matched-baseline evaluation-rollouts evidence is missing")
    receipts = sorted(path for path in rollout_root.rglob("painting.json") if path.is_file() and not path.is_symlink())
    if not receipts:
        raise ValueError("matched-baseline contains no painting receipts")
    valid = invalid = missing_png = hash_mismatch = 0
    renderer_bound = 0
    for receipt_path in receipts:
        value = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"matched-baseline painting receipt is malformed: {receipt_path}")
        if value.get("valid") is not True:
            invalid += 1
            continue
        valid += 1
        observed_renderer = value.get("renderer_sha256")
        if observed_renderer is not None:
            if observed_renderer != expected_renderer_sha256:
                raise ValueError(f"matched-baseline renderer hash mismatch: {receipt_path}")
            renderer_bound += 1
        png = receipt_path.with_name("painting.png")
        if png.is_symlink() or not png.is_file():
            missing_png += 1
            continue
        if value.get("png_sha256") != sha256_file(png):
            hash_mismatch += 1
    if missing_png or hash_mismatch:
        raise ValueError(f"matched-baseline valid PNG evidence is not hash verified: missing={missing_png}, mismatch={hash_mismatch}")
    if renderer_bound != valid:
        raise ValueError("matched-baseline valid receipts are not all bound to the current renderer hash")
    return {
        "receipt_count": len(receipts), "valid_receipts": valid, "invalid_receipts": invalid,
        "missing_png": missing_png, "hash_mismatch": hash_mismatch,
        "renderer_bound_receipts": renderer_bound,
    }


def _validate_cached_manifest_rebase(
    run_root: Path, *, archived_manifest_sha256: object, installed_manifest_sha256: str,
) -> dict[str, str]:
    """Bind an archived baseline to both sides of a portable manifest rewrite."""
    receipt_path = run_root / "cached-anchor-reuse.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("cached baseline requires the cached-anchor-reuse receipt")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cached-anchor-reuse receipt is invalid") from exc
    if receipt.get("schema") != "Diverse-flow-cached-anchor-reuse-v1":
        raise ValueError("cached-anchor-reuse receipt has the wrong schema")
    hashes = receipt.get("manifest_hashes")
    if not isinstance(hashes, dict):
        raise ValueError("cached-anchor-reuse receipt has no archived/installed manifest hashes")
    archived = hashes.get("archived")
    installed = hashes.get("installed")
    archived_hash = archived.get("matched32-manifest.json") if isinstance(archived, dict) else None
    installed_hash = installed.get("matched32-manifest.json") if isinstance(installed, dict) else None
    if not _valid_sha(archived_hash) or not _valid_sha(installed_hash):
        raise ValueError("cached-anchor-reuse matched32 manifest hashes are invalid")
    if archived_hash != archived_manifest_sha256:
        raise ValueError("cached baseline receipt is not bound to the archived matched32 manifest")
    if installed_hash != installed_manifest_sha256:
        raise ValueError("cached baseline current manifest hash differs from the installed matched32 manifest")
    return {"archived": str(archived_hash), "installed": str(installed_hash)}


def _validate_baseline_tree(
    baseline_root: Path,
    *,
    run_root: Path,
    expected_policy_sha256: str,
    expected_policy_revision: str,
    expected_policy_prefix: str,
    expected_seed: int,
    expected_renderer_sha256: str,
    eval_manifest: Path,
    flow_source: Path,
    renderer_path: Path,
) -> dict[str, object]:
    policy_sha = _require_supervision_hash(expected_policy_sha256, "policy SHA-256")
    renderer_sha = _require_supervision_hash(expected_renderer_sha256, "renderer SHA-256")
    if renderer_path.is_symlink() or not renderer_path.is_file() or sha256_file(renderer_path) != renderer_sha:
        raise ValueError("current matched-baseline renderer hash does not match the pinned renderer")
    receipt = _baseline_json(baseline_root, "baseline-receipt.json")
    if receipt.get("schema") != "Diverse-flow-matched-flash12-baseline-v1" or receipt.get("status") != "complete":
        raise ValueError("matched-baseline receipt is not complete")
    policy = receipt.get("policy")
    if not isinstance(policy, dict) or policy.get("adapter_sha256") != policy_sha or policy.get("revision") != expected_policy_revision or policy.get("prefix") != expected_policy_prefix or policy.get("label") != "flash12-pre-warmup":
        raise ValueError("matched-baseline policy identity differs from the current Flash12 pin")
    if receipt.get("manifest_cases") != BASELINE_CASES or receipt.get("max_turns") != BASELINE_MAX_TURNS or receipt.get("seed") != expected_seed:
        raise ValueError("matched-baseline receipt count, turns, or seed differs from the current contract")
    if receipt.get("judge_deferred") is not True or receipt.get("quality_scoring") != "offline_only":
        raise ValueError("matched-baseline receipt does not preserve deferred offline quality scoring")
    eval_path = baseline_root / "eval.toml"
    if eval_path.is_symlink() or not eval_path.is_file():
        raise ValueError("matched-baseline eval.toml is missing")
    eval_text = eval_path.read_text(encoding="utf-8")
    if _toml_string(eval_text, "model") != "flash12-baseline" or _toml_int(eval_text, "num_tasks") != BASELINE_CASES or _toml_int(eval_text, "num_rollouts") != 1 or _toml_int(eval_text, "max_tokens") != BASELINE_MAX_TOKENS or _toml_int(eval_text, "seed") != expected_seed or _toml_int(eval_text, "max_turns") != BASELINE_MAX_TURNS:
        raise ValueError("matched-baseline eval.toml is not the pinned 32-case/3-turn/8192-token native policy")
    if "policy_label = \"flash12-pre-warmup\"" not in eval_text or "id = \"native_painting\"" not in eval_text or "push = false" not in eval_text:
        raise ValueError("matched-baseline eval.toml is not the native policy evaluation")
    if not re.search(r"(?m)^\s*renderer\s*=\s*\"[^\"]+\"", eval_text) or not re.search(r"(?m)^\s*renderer_python\s*=\s*\"[^\"]+\"", eval_text):
        raise ValueError("matched-baseline eval.toml has no explicit renderer binding")
    prompt_hashes = _flow_prompt_hashes(flow_source)
    current_manifest = _validate_current_flow_manifest(eval_manifest, prompt_hashes)
    manifest_binding = _validate_cached_manifest_rebase(
        run_root,
        archived_manifest_sha256=receipt.get("manifest_sha256"),
        installed_manifest_sha256=str(current_manifest["manifest_sha256"]),
    )
    hook = _baseline_json(baseline_root, "Diverse-flow-hook-receipt.json")
    if hook.get("schema") != "Diverse-flow-prompt-hook-v1" or hook.get("installed") is not True or hook.get("enabled") is not True or hook.get("evaluation_taskset_patched") is not True or hook.get("eval_manifest_sha256") != manifest_binding["archived"] or hook.get("eval_rows") != BASELINE_CASES:
        raise ValueError("matched-baseline flow hook receipt is not bound to the current evaluation manifest")
    rollouts = _validate_baseline_rollouts(baseline_root, renderer_sha)
    return {
        "schema": BASELINE_SCHEMA, "status": "validated_cached_baseline",
        "policy": {"adapter_sha256": policy_sha, "revision": expected_policy_revision, "prefix": expected_policy_prefix},
        "eval": {"manifest_sha256": current_manifest["manifest_sha256"], "seed": expected_seed, "max_turns": BASELINE_MAX_TURNS, "max_tokens": BASELINE_MAX_TOKENS},
        "prompt": current_manifest, "renderer_sha256": renderer_sha, "rollouts": rollouts,
        "manifest_binding": manifest_binding,
        "invalid_or_censored_retained": rollouts["invalid_receipts"] > 0,
    }


def restore_baseline(
    *,
    run_root: Path,
    expected_sha256: str,
    expected_policy_sha256: str = BASELINE_POLICY_SHA256,
    expected_policy_revision: str = BASELINE_POLICY_REVISION,
    expected_policy_prefix: str = BASELINE_POLICY_PREFIX,
    expected_seed: int = BASELINE_SEED,
    renderer_sha256: str,
    renderer_path: Path,
    eval_manifest: Path,
    flow_source: Path,
    archive: Path | None = None,
    url: str | None = None,
) -> dict[str, object]:
    """Restore and validate a cached matched baseline without evaluation."""
    run_root = run_root.expanduser().resolve()
    if not run_root.is_dir():
        raise ValueError(f"run root is missing: {run_root}")
    if not _valid_sha(expected_sha256):
        raise ValueError("expected archive SHA-256 must be lowercase hexadecimal")
    if url is not None:
        validate_same_run_archive_pin(url=url, expected_sha256=expected_sha256)
    (run_root / ".restore-tmp").mkdir(mode=0o700, exist_ok=True)
    archive_path, source_kind, temporary = _archive_path(archive=archive, url=url, run_root=run_root, expected=expected_sha256)
    try:
        if sha256_file(archive_path) != expected_sha256:
            raise ValueError("source baseline archive hash mismatch")
        stage_parent = Path(tempfile.mkdtemp(prefix="baseline-restore-stage-", dir=str(run_root / ".restore-tmp")))
        try:
            stage_root = stage_parent / BASELINE_REL
            records: list[dict[str, object]] = []
            seen: set[Path] = set()
            total_bytes = 0
            with tarfile.open(archive_path, "r:*") as archive_file:
                for index, member in enumerate(archive_file):
                    if index >= MAX_MEMBERS:
                        raise ValueError("baseline archive has too many members")
                    if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                        raise ValueError(f"archive contains an unsupported non-regular member: {member.name!r}")
                    if member.isdir():
                        directory = PurePosixPath(member.name)
                        if directory.is_absolute() or any(part in {"", ".", ".."} for part in directory.parts):
                            raise ValueError(f"unsafe archive member path: {member.name!r}")
                        continue
                    target_relative = _normalized_baseline_member(member.name)
                    if target_relative is None:
                        continue
                    if target_relative in seen:
                        raise ValueError(f"duplicate baseline member: {member.name!r}")
                    seen.add(target_relative)
                    total_bytes += int(member.size)
                    if total_bytes > MAX_TOTAL_BYTES:
                        raise ValueError("baseline evidence exceeds the total byte limit")
                    source = archive_file.extractfile(member)
                    if source is None:
                        raise ValueError(f"cannot read archive member: {member.name!r}")
                    payload = _read_member(source, int(member.size))
                    stage_path = stage_root / target_relative.relative_to(BASELINE_REL)
                    stage_path.parent.mkdir(parents=True, exist_ok=True)
                    stage_path.write_bytes(payload)
                    stage_path.chmod(0o600)
                    records.append({"archive_member": member.name, "target": str(target_relative), "bytes": len(payload), "sha256": sha256_bytes(payload)})
            if not records:
                raise ValueError("source archive contains no permitted matched-baseline evidence")
            validation = _validate_baseline_tree(
                stage_root,
                run_root=run_root,
                expected_policy_sha256=expected_policy_sha256,
                expected_policy_revision=expected_policy_revision,
                expected_policy_prefix=expected_policy_prefix,
                expected_seed=expected_seed,
                expected_renderer_sha256=renderer_sha256,
                eval_manifest=eval_manifest,
                flow_source=flow_source,
                renderer_path=renderer_path,
            )
            target_root = run_root / BASELINE_REL
            for record in records:
                target = run_root / str(record["target"])
                if target.is_symlink() or (target.exists() and not target.is_file()):
                    raise ValueError(f"baseline restore target is not a regular file: {target}")
                if target.is_file() and sha256_file(target) != record["sha256"]:
                    raise ValueError(f"refusing to overwrite existing baseline evidence: {target}")
            reused = installed = 0
            for record in records:
                target = run_root / str(record["target"])
                source = stage_root / target.relative_to(target_root)
                if target.is_file():
                    reused += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source.replace(target)
                target.chmod(0o600)
                installed += 1
            receipt = {
                "schema": BASELINE_SCHEMA, "run_id": RUN_ID,
                "status": "restored_verified_cached_baseline",
                "source_kind": source_kind, "source_archive_sha256": expected_sha256,
                "source_url_host": urllib.parse.urlparse(url).netloc if url else None,
                "baseline_root": str(target_root), "member_count": len(records),
                "installed_count": installed, "reused_count": reused,
                "validation": validation, "members": records,
                "limitations": [
                    "Baseline programs, canvases, and invalid/censored receipts were restored from a hash-verified archive; no new model or renderer call was made.",
                    "Model weights, browser assets, source code, and credentials were excluded by the restore allowlist.",
                ],
            }
            receipt_path = run_root / "baseline-restore-receipt.json"
            encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
            if receipt_path.exists():
                if receipt_path.is_symlink():
                    raise ValueError(f"existing baseline restore receipt is symlinked: {receipt_path}")
                existing = json.loads(receipt_path.read_text(encoding="utf-8"))
                if existing.get("schema") != BASELINE_SCHEMA or existing.get("source_archive_sha256") != expected_sha256 or existing.get("members") != records or existing.get("validation") != validation:
                    raise ValueError(f"existing baseline restore receipt differs: {receipt_path}")
                return existing
            receipt_path.write_text(encoded, encoding="utf-8")
            receipt_path.chmod(0o600)
            return receipt
        finally:
            shutil.rmtree(stage_parent, ignore_errors=True)
    finally:
        if temporary is not None:
            temporary.cleanup()



def restore_supervision(*, run_root: Path, expected_sha256: str, renderer_sha256: str, archive: Path | None = None, url: str | None = None) -> dict[str, object]:
    """Restore only a hash-bound paid supervision tree from a public archive."""
    run_root = run_root.expanduser().resolve()
    if not run_root.is_dir():
        raise ValueError(f"run root is missing: {run_root}")
    if not _valid_sha(expected_sha256):
        raise ValueError("expected archive SHA-256 must be lowercase hexadecimal")
    _require_supervision_hash(renderer_sha256, "renderer SHA-256")
    (run_root / ".restore-tmp").mkdir(mode=0o700, exist_ok=True)
    archive_path, source_kind, temporary = _archive_path(archive=archive, url=url, run_root=run_root, expected=expected_sha256)
    try:
        if sha256_file(archive_path) != expected_sha256:
            raise ValueError("source supervision archive hash mismatch")
        stage_parent = Path(tempfile.mkdtemp(prefix="supervision-restore-stage-", dir=str(run_root / ".restore-tmp")))
        try:
            stage_root = stage_parent / SUPERVISION_REL
            records: list[dict[str, object]] = []
            seen: set[Path] = set()
            total_bytes = 0
            with tarfile.open(archive_path, "r:*") as archive_file:
                for index, member in enumerate(archive_file):
                    if index >= MAX_MEMBERS:
                        raise ValueError("supervision archive has too many members")
                    if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                        raise ValueError(f"archive contains an unsupported non-regular member: {member.name!r}")
                    if member.isdir():
                        directory = PurePosixPath(member.name)
                        if directory.is_absolute() or any(part in {"", ".", ".."} for part in directory.parts):
                            raise ValueError(f"unsafe archive member path: {member.name!r}")
                        continue
                    target_relative = _normalized_supervision_member(member.name)
                    if target_relative is None:
                        continue
                    if target_relative in seen:
                        raise ValueError(f"duplicate supervision member: {member.name!r}")
                    seen.add(target_relative)
                    total_bytes += int(member.size)
                    if total_bytes > MAX_TOTAL_BYTES:
                        raise ValueError("supervision evidence exceeds the total byte limit")
                    source = archive_file.extractfile(member)
                    if source is None:
                        raise ValueError(f"cannot read archive member: {member.name!r}")
                    payload = _read_member(source, int(member.size))
                    stage_path = stage_root / target_relative.relative_to(SUPERVISION_REL)
                    stage_path.parent.mkdir(parents=True, exist_ok=True)
                    stage_path.write_bytes(payload)
                    stage_path.chmod(0o600)
                    records.append({"archive_member": member.name, "target": str(target_relative), "bytes": len(payload), "sha256": sha256_bytes(payload)})
            if not records:
                raise ValueError("source archive contains no permitted code-supervision evidence")
            validation = _validate_supervision_tree(stage_root, renderer_sha256)
            target_root = run_root / SUPERVISION_REL
            for record in records:
                target = run_root / str(record["target"])
                if target.is_symlink() or (target.exists() and not target.is_file()):
                    raise ValueError(f"restore target is not a regular file: {target}")
                if target.is_file() and sha256_file(target) != record["sha256"]:
                    relative = target.relative_to(target_root)
                    prior_restore = run_root / "supervision-restore-receipt.json"
                    prior_identity = None
                    if prior_restore.is_file() and not prior_restore.is_symlink():
                        try:
                            prior_identity = json.loads(prior_restore.read_text(encoding="utf-8"))
                        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                            prior_identity = None
                    if relative.name not in SUPERVISION_DERIVED or not isinstance(prior_identity, dict) or prior_identity.get("source_archive_sha256") != expected_sha256:
                        raise ValueError(f"refusing to overwrite existing supervision evidence: {target}")
            reused = installed = 0
            for record in records:
                target = run_root / str(record["target"])
                source = stage_root / target.relative_to(target_root)
                if target.is_file():
                    reused += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source.replace(target)
                target.chmod(0o600)
                installed += 1
            receipt = {
                "schema": SUPERVISION_SCHEMA,
                "run_id": RUN_ID,
                "status": "restored_verified_public_supervision_evidence",
                "source_kind": source_kind,
                "source_archive_sha256": expected_sha256,
                "source_url_host": urllib.parse.urlparse(url).netloc if url else None,
                "supervision_root": str(target_root),
                "member_count": len(records),
                "installed_count": installed,
                "reused_count": reused,
                "validation": validation,
                "members": records,
                "limitations": [
                    "Paid programs, renders, and raw Flash decisions were restored from a hash-verified public archive; they are not relabeled as newly generated.",
                    "Model weights, browser assets, native anchor traces, and credentials were excluded by the supervision allowlist.",
                ],
            }
            receipt_path = run_root / "supervision-restore-receipt.json"
            encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
            if receipt_path.exists():
                if receipt_path.is_symlink():
                    raise ValueError(f"existing supervision restore receipt is symlinked: {receipt_path}")
                try:
                    existing = json.loads(receipt_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"existing supervision restore receipt is invalid: {receipt_path}") from exc
                if existing.get("schema") != SUPERVISION_SCHEMA or existing.get("source_archive_sha256") != expected_sha256 or existing.get("members") != records or existing.get("validation") != validation:
                    raise ValueError(f"existing supervision restore receipt differs: {receipt_path}")
                return existing
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
    parser.add_argument("--mode", choices=("anchor", "supervision", "baseline"), default="anchor")
    parser.add_argument("--renderer-sha256")
    parser.add_argument("--renderer", type=Path)
    parser.add_argument("--eval-manifest", type=Path)
    parser.add_argument("--flow-source", type=Path)
    parser.add_argument("--policy-sha256", default=BASELINE_POLICY_SHA256)
    parser.add_argument("--policy-revision", default=BASELINE_POLICY_REVISION)
    parser.add_argument("--policy-prefix", default=BASELINE_POLICY_PREFIX)
    parser.add_argument("--seed", type=int, default=BASELINE_SEED)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--url")
    args = parser.parse_args()
    if args.mode == "baseline":
        if not args.renderer_sha256 or not args.renderer or not args.eval_manifest or not args.flow_source:
            parser.error("--mode baseline requires --renderer-sha256, --renderer, --eval-manifest, and --flow-source")
        receipt = restore_baseline(
            run_root=args.run_root, expected_sha256=args.sha256,
            expected_policy_sha256=args.policy_sha256,
            expected_policy_revision=args.policy_revision,
            expected_policy_prefix=args.policy_prefix,
            expected_seed=args.seed,
            renderer_sha256=args.renderer_sha256, renderer_path=args.renderer,
            eval_manifest=args.eval_manifest, flow_source=args.flow_source,
            archive=args.archive, url=args.url,
        )
        print(json.dumps(receipt, sort_keys=True))
        return
    if args.mode == "supervision":
        if not args.renderer_sha256:
            parser.error("--renderer-sha256 is required for --mode supervision")
        receipt = restore_supervision(
            run_root=args.run_root, expected_sha256=args.sha256,
            renderer_sha256=args.renderer_sha256, archive=args.archive, url=args.url,
        )
    else:
        receipt = restore(
            run_root=args.run_root, expected_sha256=args.sha256,
            archive=args.archive, url=args.url,
        )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
