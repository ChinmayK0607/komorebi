"""Run one finite neutral aesthetic-judge batch through ``codex exec``.

This is a bounded handoff worker for files emitted by ``judge_queue.stage``.
It validates every neutral image and its declared hash before attaching the
three images in ``reference, A, B`` order.  Each successful invocation writes
the exact queue-compatible ``labels-*.json`` document plus a separate
execution record containing raw output and provenance.  It never builds or
installs receipts; the existing queue remains the review-controlled adapter
for importing labels.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import re
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from typing import Mapping


DISPLAY_ROLES = ("reference", "A", "B")
HEX64 = frozenset("0123456789abcdef")
PAIR_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
JUDGE_MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "low"
EXECUTION_SCHEMA = "painter.luna-cli-judge.execution.v1"
SUMMARY_SCHEMA = "painter.luna-cli-judge.summary.v1"
DEFAULT_MAX_CONCURRENCY = 2
DEFAULT_EXEC_TIMEOUT_SECONDS = 1800.0
DEFAULT_PROTOCOL_DIR = Path(__file__).resolve().parent / "runs" / "aesthetic-rl-20260921"


def _attachment_instruction(names: Mapping[str, str]) -> str:
    return (
        "The attached neutral files are named "
        f"{names['reference']}, {names['A']}, and {names['B']}, "
        "in this exact order: "
        f"REFERENCE ({names['reference']}), A ({names['A']}), B ({names['B']}). "
        "Inspect the attached images directly. Do not call tools or inspect local files.\n\n"
    )


DEFAULT_ATTACHMENT_NAMES = {role: f"{role}.png" for role in DISPLAY_ROLES}
ATTACHMENT_PREAMBLE = _attachment_instruction(DEFAULT_ATTACHMENT_NAMES)


class OutputCollision(ValueError):
    """A prior batch artifact exists and must not be overwritten."""


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _load_judge_module(protocol_dir: Path):
    path = protocol_dir / "aesthetic_judge.py"
    if not path.is_file():
        raise FileNotFoundError(f"aesthetic judge schema is missing: {path}")
    name = "luna_cli_aesthetic_judge_" + sha(path)[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load aesthetic judge schema: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_sha(value: Any, role: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in HEX64 for char in value):
        raise ValueError(f"invalid image_sha256 for {role}")
    return value


def _validate_pair_id(value: Any) -> str:
    if not isinstance(value, str) or not value or not PAIR_ID_PATTERN.fullmatch(value):
        raise ValueError("comparison id must be a safe filename component")
    return value


def _attachment_extension(path: Path, payload: bytes) -> str:
    """Choose a neutral attachment suffix from bytes, with a safe fallback."""

    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if payload.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return ".webp"
    suffix = path.suffix.lower()
    if suffix == ".jpeg" or suffix == ".jpg":
        return ".jpg"
    if suffix in {".png", ".webp"}:
        return suffix
    raise ValueError(f"unsupported image type for attachment: {path}")


def _resolve_input(batch_dir: Path, value: Any, role: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing image path for {role}")
    path = Path(value)
    if not path.is_absolute():
        path = batch_dir / path
    if path.is_symlink():
        raise ValueError(f"image path is a symlink for {role}: {path}")
    if not path.is_file():
        raise ValueError(f"image is missing for {role}: {path}")
    return path.resolve()


@dataclass(frozen=True)
class PairInput:
    document: Path
    document_sha256: str
    pair_id: str
    protocol_version: str
    images: dict[str, Path]
    image_sha256: dict[str, str]


def validate_pair_document(path: Path, batch_dir: Path, judge: Any) -> PairInput:
    """Validate one staged ``judge-*.json`` and all three immutable images."""

    raw_bytes = path.read_bytes()
    document = json.loads(raw_bytes)
    required_document = {"protocol_version", "judge_model", "reasoning_effort", "comparisons"}
    if not isinstance(document, dict) or set(document) != required_document:
        raise ValueError(f"invalid judge document fields: {path.name}")
    if document["protocol_version"] != judge.PROTOCOL_VERSION:
        raise ValueError(f"protocol mismatch in {path.name}")
    if document["judge_model"] != JUDGE_MODEL or document["reasoning_effort"] != REASONING_EFFORT:
        raise ValueError(f"judge provenance mismatch in {path.name}")
    comparisons = document["comparisons"]
    if not isinstance(comparisons, list) or len(comparisons) != 1:
        raise ValueError(f"one comparison is required per judge document: {path.name}")
    pair = comparisons[0]
    required_pair = {"id", "reference", "A", "B", "image_sha256", "protocol_version"}
    if not isinstance(pair, dict) or set(pair) != required_pair:
        raise ValueError(f"invalid comparison fields in {path.name}")
    pair_id = pair["id"]
    try:
        pair_id = _validate_pair_id(pair_id)
    except ValueError as exc:
        raise ValueError(f"invalid comparison id in {path.name}: {exc}") from exc
    expected_filename = f"judge-{pair_id}.json"
    if path.name != expected_filename:
        raise ValueError(f"judge filename does not match comparison id: {path.name}")
    if pair["protocol_version"] != judge.PROTOCOL_VERSION:
        raise ValueError(f"comparison protocol mismatch in {path.name}")
    hashes = pair["image_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != set(DISPLAY_ROLES):
        raise ValueError(f"image_sha256 must contain exactly {DISPLAY_ROLES}: {path.name}")
    image_hashes = {role: _validate_sha(hashes[role], role) for role in DISPLAY_ROLES}
    images = {
        role: _resolve_input(batch_dir, pair[role], role)
        for role in DISPLAY_ROLES
    }
    for role in DISPLAY_ROLES:
        actual = sha(images[role])
        if actual != image_hashes[role]:
            raise ValueError(
                f"changed image hash for {pair_id}:{role}: expected {image_hashes[role]}, found {actual}"
            )
    return PairInput(
        document=path,
        document_sha256=sha_bytes(raw_bytes),
        pair_id=pair_id,
        protocol_version=judge.PROTOCOL_VERSION,
        images=images,
        image_sha256=image_hashes,
    )


def output_schema() -> dict[str, Any]:
    """Return the strict four-field schema passed to ``codex exec``."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["winner", "reason", "weaknesses_A", "weaknesses_B"],
        "properties": {
            "winner": {"type": "string", "enum": ["A", "B", "tie", "uncertain"]},
            "reason": {"type": "string", "minLength": 1},
            "weaknesses_A": {"type": "array", "items": {"type": "string"}},
            "weaknesses_B": {"type": "array", "items": {"type": "string"}},
        },
    }


def build_prompt(prompt: str, attachment_names: Mapping[str, str] | None = None) -> str:
    """Add the per-pair attachment mapping used by the successful CLI probe."""

    names = dict(attachment_names or DEFAULT_ATTACHMENT_NAMES)
    instruction = _attachment_instruction(names)
    if prompt.startswith(instruction):
        return prompt
    return instruction + prompt


def build_command(
    pair: PairInput,
    *,
    schema_path: Path,
    last_message_path: Path,
    image_paths: Mapping[str, Path] | None = None,
    codex_bin: str = "codex",
) -> list[str]:
    """Build an argument vector; callers pass the approved prompt on stdin."""

    command = [codex_bin, "exec"]
    image_paths = image_paths or pair.images
    for role in DISPLAY_ROLES:
        command.extend(["--image", str(image_paths[role])])
    command.extend([
        "--model", JUDGE_MODEL,
        "-c", 'model_reasoning_effort="low"',
        "--output-schema", str(schema_path),
        "--output-last-message", str(last_message_path),
        "--json",
        "--ephemeral",
        "--ignore-user-config",
        "--sandbox", "read-only",
        "--skip-git-repo-check",
        "-",
    ])
    return command


def _default_runner(
    command: list[str], prompt: str, cwd: Path, last_message_path: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        input=prompt,
        text=True,
        capture_output=True,
        cwd=str(cwd),
        check=False,
        timeout=DEFAULT_EXEC_TIMEOUT_SECONDS,
    )


def _text_bytes(value: Any) -> bytes:
    if value is None:
        return b""
    return value if isinstance(value, bytes) else str(value).encode("utf-8")


def _parse_usage(events: bytes) -> dict[str, Any] | None:
    """Extract the actual ``turn.completed`` usage event, when present."""

    try:
        text = events.decode("utf-8")
    except UnicodeDecodeError:
        return None
    usage: dict[str, Any] | None = None
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(event, dict)
            and event.get("type") == "turn.completed"
            and isinstance(event.get("usage"), dict)
        ):
            usage = dict(event["usage"])
    return usage


def _input_metadata(
    pair: PairInput,
    attached_images: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    return {
        "judge_document": str(pair.document),
        "judge_document_sha256": pair.document_sha256,
        "image_inputs": [
            {
                "role": role,
                "path": str(pair.images[role]),
                "sha256": pair.image_sha256[role],
                "attachment_status": "attached",
                **(
                    {
                        "attachment_path": str(attached_images[role]),
                        "attachment_sha256": sha(attached_images[role]),
                    }
                    if attached_images is not None
                    else {}
                ),
            }
            for role in DISPLAY_ROLES
        ],
        "image_sha256": dict(pair.image_sha256),
    }


def _base_execution(
    pair: PairInput,
    *,
    prompt_sha256: str | None,
    prompt_source_sha256: str,
    schema_sha256: str,
    command: list[str] | None,
) -> dict[str, Any]:
    return {
        "schema": EXECUTION_SCHEMA,
        "pair_id": pair.pair_id,
        "protocol_version": pair.protocol_version,
        "judge_model": JUDGE_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "input_hashes": _input_metadata(pair),
        "prompt_sha256": prompt_sha256,
        "prompt_source_sha256": prompt_source_sha256,
        "output_schema_sha256": schema_sha256,
        "command": command,
        "image_input_mode": "cli_attachment",
        "working_directory_policy": "dedicated temporary neutral directory",
        "usage": None,
        "usage_available": False,
        "exit_code": None,
        "stdout_sha256": None,
        "stderr_sha256": None,
        "events_sha256": None,
        "stdout_path": None,
        "stderr_path": None,
        "events_path": None,
        "last_message_path": None,
        "last_message_sha256": None,
        "raw_label": None,
        "validated_label": None,
        "status": "failed",
        "error": None,
    }


def _run_pair(
    pair: PairInput,
    *,
    output_dir: Path,
    prompt_text: str,
    prompt_source_sha256: str,
    schema_path: Path,
    schema_sha256: str,
    scratch_dir: Path,
    judge: Any,
    runner: Callable[[list[str], str, Path, Path], Any],
    codex_bin: str,
) -> dict[str, Any]:
    label_path = output_dir / f"labels-{pair.pair_id}.json"
    execution_path = output_dir / f"execution-{pair.pair_id}.json"
    command: list[str] | None = None
    execution = _base_execution(
        pair,
        prompt_sha256=None,
        prompt_source_sha256=prompt_source_sha256,
        schema_sha256=schema_sha256,
        command=None,
    )
    stdout_path = output_dir / f"stdout-{pair.pair_id}.log"
    stderr_path = output_dir / f"stderr-{pair.pair_id}.log"
    events_path = output_dir / f"events-{pair.pair_id}.jsonl"
    durable_last_message_path = output_dir / f"last-message-{pair.pair_id}.json"
    execution.update(
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        events_path=str(events_path),
        last_message_path=str(durable_last_message_path),
    )

    def persist_streams(stdout_bytes: bytes, stderr_bytes: bytes) -> None:
        # ``--json`` emits the event stream on stdout. Keep all three raw files
        # so a reviewer can inspect the exact CLI evidence after the temp cwd is
        # removed; hashes alone are insufficient for an initial probe.
        _atomic_bytes(stdout_path, stdout_bytes)
        _atomic_bytes(stderr_path, stderr_bytes)
        _atomic_bytes(events_path, stdout_bytes)
        execution["stdout_sha256"] = sha_bytes(stdout_bytes)
        execution["stderr_sha256"] = sha_bytes(stderr_bytes)
        execution["events_sha256"] = sha_bytes(stdout_bytes)

    def persist_last_message(path: Path) -> str | None:
        if not path.is_file():
            return None
        value = path.read_bytes()
        _atomic_bytes(durable_last_message_path, value)
        execution["last_message_sha256"] = sha_bytes(value)
        return value.decode("utf-8")

    try:
        if label_path.exists() or execution_path.exists():
            raise OutputCollision(f"output already exists for {pair.pair_id}")
        # Recheck mutable input bytes immediately before attachment.
        pair_scratch = scratch_dir / f"pair-{pair.pair_id}"
        pair_scratch.mkdir(parents=True, exist_ok=False)
        attached_images: dict[str, Path] = {}
        for role in DISPLAY_ROLES:
            source = pair.images[role]
            payload = source.read_bytes()
            actual = sha_bytes(payload)
            if actual != pair.image_sha256[role]:
                raise ValueError(f"changed image hash for {pair.pair_id}:{role}")
            destination = pair_scratch / f"{role}{_attachment_extension(source, payload)}"
            _atomic_bytes(destination, payload)
            attached = sha(destination)
            if attached != pair.image_sha256[role]:
                raise ValueError(f"attachment hash mismatch for {pair.pair_id}:{role}")
            attached_images[role] = destination
        execution["input_hashes"] = _input_metadata(pair, attached_images)
        protocol_prompt = build_prompt(
            prompt_text,
            {role: attached_images[role].name for role in DISPLAY_ROLES},
        )
        execution["prompt_sha256"] = sha_bytes(protocol_prompt.encode("utf-8"))
        last_message_path = pair_scratch / "last-message.json"
        command = build_command(
            pair,
            schema_path=schema_path,
            last_message_path=last_message_path,
            image_paths=attached_images,
            codex_bin=codex_bin,
        )
        execution["command"] = command
        started = time.time()
        try:
            completed = runner(command, protocol_prompt, pair_scratch, last_message_path)
        except subprocess.TimeoutExpired as exc:
            stdout_bytes = _text_bytes(getattr(exc, "stdout", ""))
            stderr_bytes = _text_bytes(getattr(exc, "stderr", ""))
            persist_streams(stdout_bytes, stderr_bytes)
            execution["started_at"] = started
            execution["ended_at"] = time.time()
            execution["error"] = (
                f"execution timeout after {DEFAULT_EXEC_TIMEOUT_SECONDS:g} seconds"
            )
            _atomic_json(execution_path, execution)
            return {"pair_id": pair.pair_id, "status": "failed", "error": execution["error"]}
        except Exception as exc:
            execution["error"] = f"execution failure: {type(exc).__name__}: {exc}"
            execution["started_at"] = started
            execution["ended_at"] = time.time()
            _atomic_json(execution_path, execution)
            return {"pair_id": pair.pair_id, "status": "failed", "error": execution["error"]}
        execution["started_at"] = started
        execution["ended_at"] = time.time()
        stdout_bytes = _text_bytes(getattr(completed, "stdout", ""))
        stderr_bytes = _text_bytes(getattr(completed, "stderr", ""))
        persist_streams(stdout_bytes, stderr_bytes)
        execution["exit_code"] = getattr(completed, "returncode", None)
        raw_label = persist_last_message(last_message_path)
        if raw_label is not None:
            execution["raw_label"] = raw_label
        execution["usage"] = _parse_usage(stdout_bytes)
        execution["usage_available"] = execution["usage"] is not None
        if execution["exit_code"] != 0:
            raise ValueError(f"codex exec exited with {execution['exit_code']}")
        if raw_label is None:
            raise ValueError("codex exec did not produce --output-last-message")
        validated = judge.parse_judgment(raw_label)
        execution["validated_label"] = validated
        execution["status"] = "validated"
        # Existing judge_queue._label_documents accepts this exact metadata +
        # judgments shape. Execution provenance stays in its sibling record.
        _atomic_json(
            label_path,
            {
                "protocol_version": pair.protocol_version,
                "judge_model": JUDGE_MODEL,
                "reasoning_effort": REASONING_EFFORT,
                "judgments": [{"id": pair.pair_id, **validated}],
            },
        )
        _atomic_json(execution_path, execution)
        return {"pair_id": pair.pair_id, "status": "validated", "label_path": str(label_path)}
    except OutputCollision:
        # run_batch preflights all outputs. If another process races us, do not
        # overwrite its evidence with a synthetic failure record.
        return {"pair_id": pair.pair_id, "status": "failed", "error": f"output already exists for {pair.pair_id}"}
    except Exception as exc:
        execution["command"] = command
        execution["error"] = str(exc)
        _atomic_json(execution_path, execution)
        return {"pair_id": pair.pair_id, "status": "failed", "error": str(exc)}


def _invalid_pair_record(
    path: Path,
    output_dir: Path,
    *,
    prompt_sha256: str | None,
    prompt_source_sha256: str,
    schema_sha256: str,
    error: str,
) -> dict[str, Any]:
    raw_pair_id = path.stem.removeprefix("judge-")
    pair_id = _record_pair_id(path) if not PAIR_ID_PATTERN.fullmatch(raw_pair_id or "") else raw_pair_id
    try:
        document_sha256: str | None = sha(path)
    except OSError:
        document_sha256 = None
    execution = {
        "schema": EXECUTION_SCHEMA,
        "pair_id": pair_id,
        "protocol_version": None,
        "judge_model": JUDGE_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "input_hashes": {"judge_document": str(path), "judge_document_sha256": document_sha256},
        "prompt_sha256": prompt_sha256,
        "prompt_source_sha256": prompt_source_sha256,
        "output_schema_sha256": schema_sha256,
        "command": None,
        "image_input_mode": "cli_attachment",
        "working_directory_policy": "dedicated temporary neutral directory",
        "usage": None,
        "usage_available": False,
        "exit_code": None,
        "stdout_sha256": None,
        "stderr_sha256": None,
        "events_sha256": None,
        "stdout_path": None,
        "stderr_path": None,
        "events_path": None,
        "last_message_path": None,
        "last_message_sha256": None,
        "raw_label": None,
        "validated_label": None,
        "status": "failed",
        "error": error,
    }
    path_out = output_dir / f"execution-{pair_id}.json"
    _atomic_json(path_out, execution)
    return {"pair_id": pair_id, "status": "failed", "error": error}


def _record_pair_id(path: Path) -> str:
    """Return a safe deterministic output stem for malformed documents."""

    try:
        digest = sha(path)[:16]
    except OSError:
        digest = sha_bytes(str(path).encode("utf-8"))[:16]
    return "invalid-" + digest


def _assert_no_output_collisions(documents: list[Path], output_dir: Path) -> None:
    artifacts: list[Path] = []
    for path in documents:
        raw_pair_id = path.stem.removeprefix("judge-")
        pair_id = raw_pair_id if PAIR_ID_PATTERN.fullmatch(raw_pair_id or "") else _record_pair_id(path)
        artifacts.extend(
            output_dir / name
            for name in (
                f"labels-{pair_id}.json",
                f"execution-{pair_id}.json",
                f"stdout-{pair_id}.log",
                f"stderr-{pair_id}.log",
                f"events-{pair_id}.jsonl",
                f"last-message-{pair_id}.json",
            )
        )
    existing = [path for path in artifacts if path.exists()]
    if existing:
        raise OutputCollision(
            "batch output already exists; refusing to overwrite: "
            + ", ".join(str(path) for path in existing)
        )


def run_batch(
    batch_dir: Path,
    *,
    output_dir: Path | None = None,
    protocol_dir: Path = DEFAULT_PROTOCOL_DIR,
    prompt_path: Path | None = None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    codex_bin: str = "codex",
    runner: Callable[[list[str], str, Path, Path], Any] = _default_runner,
    scratch_parent: Path | None = None,
) -> dict[str, Any]:
    """Process exactly one finite batch and return a reviewable summary."""

    batch_dir = Path(batch_dir).resolve()
    if not batch_dir.is_dir():
        raise FileNotFoundError(f"batch directory is missing: {batch_dir}")
    if isinstance(max_concurrency, bool) or max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    if max_concurrency > 64:
        raise ValueError("max_concurrency is unreasonably large")
    output_dir = (Path(output_dir) if output_dir is not None else batch_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_dir = Path(protocol_dir).resolve()
    prompt_file = (Path(prompt_path) if prompt_path is not None else protocol_dir / "JUDGE_PROMPT.md").resolve()
    prompt_source_digest = sha(prompt_file)
    prompt_text = prompt_file.read_text()
    judge = _load_judge_module(protocol_dir)
    schema_value = output_schema()

    with tempfile.TemporaryDirectory(prefix="painter-luna-cli-", dir=str(scratch_parent) if scratch_parent else None) as scratch_name:
        scratch_dir = Path(scratch_name)
        schema_path = scratch_dir / "output-schema.json"
        _atomic_json(schema_path, schema_value)
        schema_digest = sha(schema_path)
        documents = sorted(path for path in batch_dir.glob("judge-*.json") if path.is_file())
        if not documents:
            raise ValueError(f"no finite judge-*.json inputs found in {batch_dir}")
        _assert_no_output_collisions(documents, output_dir)
        valid: list[PairInput] = []
        invalid: list[tuple[Path, str]] = []
        for path in documents:
            try:
                valid.append(validate_pair_document(path, batch_dir, judge))
            except Exception as exc:
                invalid.append((path, str(exc)))
        results: list[dict[str, Any]] = [
            _invalid_pair_record(
                path,
                output_dir,
                prompt_sha256=None,
                prompt_source_sha256=prompt_source_digest,
                schema_sha256=schema_digest,
                error=error,
            )
            for path, error in invalid
        ]
        if valid:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as pool:
                futures = [
                    pool.submit(
                        _run_pair,
                        pair,
                        output_dir=output_dir,
                        prompt_text=prompt_text,
                        prompt_source_sha256=prompt_source_digest,
                        schema_path=schema_path,
                        schema_sha256=schema_digest,
                        scratch_dir=scratch_dir,
                        judge=judge,
                        runner=runner,
                        codex_bin=codex_bin,
                    )
                    for pair in valid
                ]
                results.extend(future.result() for future in futures)
    results.sort(key=lambda row: row["pair_id"])
    summary = {
        "schema": SUMMARY_SCHEMA,
        "batch_dir": str(batch_dir),
        "output_dir": str(output_dir),
        "pairs": len(documents),
        "validated": sum(row["status"] == "validated" for row in results),
        "failed": sum(row["status"] == "failed" for row in results),
        "max_concurrency": max_concurrency,
        "results": results,
        "receipt_installation": "not performed",
    }
    _atomic_json(output_dir / "cli-judge-summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    parser.add_argument("--prompt", type=Path, default=None)
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    parser.add_argument("--codex-bin", default="codex")
    args = parser.parse_args(argv)
    summary = run_batch(
        args.batch_dir,
        output_dir=args.output_dir,
        protocol_dir=args.protocol_dir,
        prompt_path=args.prompt,
        max_concurrency=args.max_concurrency,
        codex_bin=args.codex_bin,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
