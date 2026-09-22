"""Controller-side front end for a finite native Prime RL run on a Lium node.

The node owns model, renderer, and Prime processes.  The controller resolves a
live node address with ``lium ps --format json``, stages a small code allowlist,
reads model/adapter pointers from the prepared node profile, and invokes the
existing node-side ``rl_entrypoint.py``.  Luna credentials and SSH arguments
remain local to the finite ``judge_bridge.py worker`` process and are never
written to the node launch spec or a run receipt.

This module does not create pods, schedule work, copy model/data artifacts, or
start a run during import.  ``--dry-run`` performs no provider or SSH call;
``--prepare-only`` prepares the remote spec and runtime links but does not
start Prime or the judge.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import time
from typing import Any, Callable, Sequence


RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
REMOTE_ABSOLUTE = re.compile(r"^/[A-Za-z0-9._+@%/:=-]+$")
TERMINAL = frozenset({"complete", "failed", "stale"})
DEFAULT_PROFILE_ROOT = Path("/root/painter/runs/photo-curriculum-rl-20260922")
DEFAULT_PRIME_ROOT = Path("/root/painter/prime-rl")
DEFAULT_BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DEFAULT_PRIME_COMMIT = "26b3131d2716a4f8210b165df584b83b4bc54f61"
STAGE_FILES = (
    "painter/rl_entrypoint.py",
    "painter/prepare_scaled_native_rl.py",
    "painter/contract.py",
    "painter/native_painting.py",
    "painter/native_painting_rl.py",
    "painter/native_painting_renderer.py",
    "painter/native_painting_eval.py",
    "painter/frozen_opponent.py",
    "painter/native_painting_report.py",
    "painter/pyproject.toml",
    "painter/runs/photo-curriculum-rl-20260922/launch_rl.sh",
    "painter/runs/photo-curriculum-rl-20260922/setup_rl_runtime.sh",
    "painter/runs/photo-curriculum-rl-20260922/prepare_rl_config.py",
    "painter/runs/photo-curriculum-rl-20260922/run_from_sft.py",
    "painter/runs/photo-curriculum-rl-20260922/train_from_sft.py",
    "painter/runs/photo-curriculum-rl-20260922/judge_queue.py",
    "painter/runs/photo-curriculum-rl-20260922/aesthetic_judge.py",
)
ASSET_LINKS = ("source", "vendor", "renderer-env", "browsers", "bootstrap")


class RemoteLaunchError(RuntimeError):
    """The node identity, staged code, or finite run contract is invalid."""


@dataclass(frozen=True)
class NodeConnection:
    """Ephemeral SSH details obtained from one live ``lium ps`` record."""

    selector: str
    pod_id: str
    host: str
    port: int
    user: str = "root"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_json_prefix(raw: str) -> Any:
    try:
        value, end = json.JSONDecoder().raw_decode(raw.lstrip())
    except json.JSONDecodeError as exc:
        raise RemoteLaunchError("lium ps did not return JSON") from exc
    if raw[end:].strip():
        raise RemoteLaunchError("lium ps returned trailing non-JSON output")
    return value


def _pod_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("pods"), list):
        records = payload["pods"]
    elif isinstance(payload, dict):
        records = [payload]
    else:
        raise RemoteLaunchError("lium ps JSON must contain pod objects")
    if not all(isinstance(item, dict) for item in records):
        raise RemoteLaunchError("lium ps returned a non-object pod record")
    return records


def connection_from_ps(payload: Any, selector: str, *, user: str = "root") -> NodeConnection:
    """Select a running pod by id, name, huid, or node label."""
    records = _pod_records(payload)
    matches = [
        pod for pod in records
        if selector in {str(pod.get(key)) for key in ("id", "name", "huid", "node") if pod.get(key) is not None}
    ]
    if len(matches) != 1:
        raise RemoteLaunchError(f"node selector {selector!r} matched {len(matches)} active pods")
    pod = matches[0]
    if str(pod.get("status", "")).upper() not in {"", "RUNNING", "READY", "ACTIVE"}:
        raise RemoteLaunchError(f"selected node is not running: {pod.get('status')!r}")
    host = pod.get("ip") or pod.get("host")
    ports = pod.get("ports") if isinstance(pod.get("ports"), dict) else {}
    raw_port = ports.get("22") or ports.get(22) or pod.get("ssh_port") or 22
    if not isinstance(host, str) or not host or any(char.isspace() for char in host):
        raise RemoteLaunchError("lium ps record has no safe SSH address")
    try:
        port = int(raw_port)
    except (TypeError, ValueError) as exc:
        raise RemoteLaunchError("lium ps record has an invalid SSH port") from exc
    if not 1 <= port <= 65535:
        raise RemoteLaunchError("lium ps record has an out-of-range SSH port")
    pod_id = str(pod.get("id") or selector)
    return NodeConnection(selector=selector, pod_id=pod_id, host=host, port=port, user=user)


def discover_node(
    selector: str,
    *,
    user: str = "root",
    lium_bin: str = "lium",
    run: Callable[..., Any] = subprocess.run,
) -> NodeConnection:
    completed = run(
        [lium_bin, "ps", "--format", "json"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return connection_from_ps(_parse_json_prefix(completed.stdout), selector, user=user)


def validate_run_id(value: str) -> str:
    if not isinstance(value, str) or not RUN_ID.fullmatch(value):
        raise RemoteLaunchError("run-id must be a short filename-safe identifier")
    return value


def validate_remote_path(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not REMOTE_ABSOLUTE.fullmatch(value):
        raise RemoteLaunchError(f"{name} must be an absolute remote path without shell metacharacters")
    return value


def _ssh_argv(connection: NodeConnection, *, identity: Path | None, known_hosts: Path | None) -> list[str]:
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
    if identity is not None:
        command.extend(["-i", str(identity)])
    if known_hosts is not None:
        command.extend(["-o", f"UserKnownHostsFile={known_hosts}"])
    command.extend(["-p", str(connection.port), f"{connection.user}@{connection.host}"])
    return command


class SSHRemote:
    """Small shell-free SSH transport, injectable in tests."""

    def __init__(self, connection: NodeConnection, *, identity: Path | None, known_hosts: Path | None):
        self.connection = connection
        self.identity = identity
        self.known_hosts = known_hosts

    def run(self, argv: Sequence[str], *, input_bytes: bytes | None = None, check: bool = True) -> subprocess.CompletedProcess:
        if not argv or not all(isinstance(item, str) and item for item in argv):
            raise RemoteLaunchError("remote command must be a non-empty argv list")
        # ssh passes its trailing arguments to the remote login shell.  A raw
        # argv append therefore turns Python source and paths into shell
        # syntax.  Quote the complete remote argv as one command string.
        remote_command = shlex.join(list(argv))
        return subprocess.run(
            _ssh_argv(self.connection, identity=self.identity, known_hosts=self.known_hosts) + [remote_command],
            input=input_bytes,
            capture_output=True,
            check=check,
            timeout=300,
        )


def _stdout(result: subprocess.CompletedProcess[Any]) -> str:
    value = result.stdout
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", "replace")
    return "" if value is None else str(value)


def _tar_bytes(repo_root: Path) -> tuple[bytes, dict[str, str]]:
    files: dict[str, Path] = {}
    for relative in STAGE_FILES:
        source = repo_root / relative
        if not source.is_file() or source.is_symlink():
            raise RemoteLaunchError(f"reviewed launcher source is missing or symlinked: {relative}")
        files[relative] = source
    manifest = {name: _sha(path) for name, path in sorted(files.items())}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, path in files.items():
            info = archive.gettarinfo(str(path), arcname=name)
            info.mode = 0o644 if path.suffix != ".sh" else 0o755
            with path.open("rb") as stream:
                archive.addfile(info, stream)
        manifest_info = tarfile.TarInfo(".rl-remote-code-manifest.json")
        payload = (json.dumps(manifest, sort_keys=True) + "\n").encode()
        manifest_info.size = len(payload)
        manifest_info.mode = 0o644
        archive.addfile(manifest_info, io.BytesIO(payload))
    return buffer.getvalue(), manifest


def stage_reviewed_code(remote: SSHRemote, *, repo_root: Path, remote_root: Path) -> dict[str, Any]:
    """Stage only reviewed source files and reject conflicting destinations."""
    payload, manifest = _tar_bytes(repo_root)
    stage = remote_root / f".rl-remote-stage-{hashlib.sha256(payload).hexdigest()[:16]}"
    remote.run(["mkdir", "-p", str(stage)])
    try:
        remote.run(["tar", "-xzf", "-", "-C", str(stage)], input_bytes=payload)
        verifier = r'''
import hashlib, json, os, shutil, sys
from pathlib import Path
stage, root = map(Path, sys.argv[1:3])
expected = json.loads(sys.argv[3])
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()
for relative, digest in expected.items():
    source = stage / relative
    target = root / relative
    if not source.is_file() or sha(source) != digest:
        raise SystemExit(f'stage hash mismatch: {relative}')
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file() or sha(target) != digest:
            raise SystemExit(f'conflicting reviewed source on node: {target}')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if target.suffix == '.sh':
            target.chmod(0o755)
print(json.dumps({'status': 'staged', 'files': len(expected), 'root': str(root)}))
'''
        result = remote.run(["python3", "-c", verifier, str(stage), str(remote_root), json.dumps(manifest, sort_keys=True)])
        return json.loads(_stdout(result).strip().splitlines()[-1])
    finally:
        remote.run(["rm", "-rf", str(stage)], check=False)


def reuse_profile_assets(remote: SSHRemote, *, profile_root: Path, run_root: Path) -> dict[str, Any]:
    """Link prepared source/render assets into a fresh sibling run root."""
    script = r'''
import json, os, sys
from pathlib import Path
profile, run = map(Path, sys.argv[1:3])
links = %s
run.mkdir(parents=True, exist_ok=True)
for name in links:
    source, target = profile / name, run / name
    if not source.exists() and not source.is_symlink():
        raise SystemExit(f'prepared profile asset is missing: {source}')
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise SystemExit(f'run asset points elsewhere: {target}')
    elif target.exists():
        if target.resolve() != source.resolve():
            raise SystemExit(f'run asset already exists with a different target: {target}')
    else:
        target.symlink_to(source, target_is_directory=source.is_dir())
print(json.dumps({'status': 'assets_ready', 'assets': list(links)}))
''' % repr(list(ASSET_LINKS))
    result = remote.run(["python3", "-c", script, str(profile_root), str(run_root)])
    return json.loads(_stdout(result).strip().splitlines()[-1])


def profile_runtime(remote: SSHRemote, *, profile_root: Path) -> dict[str, str]:
    """Resolve prepared runtime paths without reading or copying credentials."""
    script = r'''
import json, os, sys
from pathlib import Path
profile = Path(sys.argv[1])
token = profile / "private" / "hf-token"
if not token.is_file() or not os.access(token, os.R_OK):
    raise SystemExit(f"prepared HF token path is not readable: {token}")
uv = profile / "bootstrap" / "bin" / "uv"
uv_path = str(uv) if uv.is_file() and os.access(uv, os.X_OK) else "uv"
print(json.dumps({"HF_TOKEN_PATH": str(token), "UV_BIN": uv_path}, sort_keys=True))
'''
    result = remote.run(["python3", "-c", script, str(profile_root)])
    value = _json_last(_stdout(result), name="prepared runtime")
    if not all(isinstance(value.get(key), str) and value[key] for key in ("HF_TOKEN_PATH", "UV_BIN")):
        raise RemoteLaunchError("prepared runtime returned invalid paths")
    return {key: value[key] for key in ("HF_TOKEN_PATH", "UV_BIN")}


def read_remote_pointer(remote: SSHRemote, path: Path, *, name: str) -> str:
    result = remote.run(["cat", str(path)])
    value = _stdout(result).strip()
    return validate_remote_path(value, name=name)


def _json_last(stdout: str, *, name: str) -> dict[str, Any]:
    if not stdout.strip():
        raise RemoteLaunchError(f"{name} returned no JSON")
    decoder = json.JSONDecoder()
    candidates: list[tuple[int, dict[str, Any], bool]] = []
    # Node helpers may print a compact receipt followed by the entrypoint's
    # pretty-printed multiline receipt. Search from the end and accept the
    # last complete object whose suffix is only whitespace.
    for index, char in enumerate(stdout):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(stdout[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        candidates.append((index, value, not stdout[index + end :].strip()))
    if not candidates:
        raise RemoteLaunchError(f"{name} returned invalid JSON")
    clean = [candidate for candidate in candidates if candidate[2]]
    return max(clean or candidates, key=lambda candidate: candidate[0])[1]


def _node_entrypoint(code_root: Path) -> Path:
    return code_root / "painter" / "rl_entrypoint.py"


def prepare_remote(
    remote: SSHRemote,
    *,
    repo_root: Path,
    profile_root: Path,
    remote_root: Path,
    remote_code_root: Path,
    data: str,
    run_id: str,
    prime_root: Path,
    max_steps: int,
    base_revision: str,
    prime_commit: str,
    validate_prime: bool,
) -> dict[str, Any]:
    """Stage code/assets and prepare a new node-side launch spec."""
    validate_run_id(run_id)
    validate_remote_path(data, name="data")
    if max_steps < 8:
        raise RemoteLaunchError("max-steps must be at least 8 for the pinned photo profile")
    code = stage_reviewed_code(remote, repo_root=repo_root, remote_root=remote_code_root)
    assets = reuse_profile_assets(remote, profile_root=profile_root, run_root=remote_root)
    runtime = profile_runtime(remote, profile_root=profile_root)
    model = read_remote_pointer(remote, profile_root / "model-path.txt", name="model-path.txt")
    adapter = read_remote_pointer(remote, profile_root / "initial-adapter-path.txt", name="initial-adapter-path.txt")
    remote_python = prime_root / ".venv" / "bin" / "python"
    command = [
        "env", *[f"{key}={value}" for key, value in sorted(runtime.items())],
        "python3", str(_node_entrypoint(remote_code_root)), "prepare",
        "--profile", "photo", "--data", data, "--run-id", run_id,
        "--run-root", str(remote_root), "--model", model, "--adapter", adapter,
        "--prime-root", str(prime_root), "--python", str(remote_python),
        "--max-steps", str(max_steps), "--base-revision", base_revision,
        "--prime-commit", prime_commit,
    ]
    if validate_prime:
        command.append("--validate-prime")
    prepared = _json_last(_stdout(remote.run(command)), name="node prepare")
    return {"code": code, "assets": assets, "runtime": runtime, "prepare": prepared, "run_root": str(remote_root)}


def start_remote(
    remote: SSHRemote,
    *,
    remote_root: Path,
    remote_code_root: Path,
    runtime: dict[str, str] | None = None,
) -> dict[str, Any]:
    spec = remote_root / "launch.json"
    command: list[str] = ["python3", str(_node_entrypoint(remote_code_root)), "start", "--spec", str(spec)]
    if runtime:
        command = ["env"] + [f"{key}={value}" for key, value in sorted(runtime.items())] + command
    return _json_last(
        _stdout(remote.run(command)),
        name="node start",
    )


def status_remote(remote: SSHRemote, *, remote_root: Path, remote_code_root: Path) -> dict[str, Any]:
    spec = remote_root / "launch.json"
    return _json_last(
        _stdout(remote.run(["python3", str(_node_entrypoint(remote_code_root)), "status", "--spec", str(spec)])),
        name="node status",
    )


def preflight_remote(
    remote: SSHRemote,
    *,
    remote_root: Path,
    remote_code_root: Path,
    runtime: dict[str, str],
) -> dict[str, Any]:
    """Run the launcher's no-GPU wire/import preflight on the prepared node."""
    spec = remote_root / "launch.json"
    command = [
        "env",
        "PAINTER_RL_PREFLIGHT_ONLY=1",
        "PAINTER_RL_SKIP_EDITABLE_INSTALL=1",
        *[f"{key}={value}" for key, value in sorted(runtime.items())],
        "python3", str(_node_entrypoint(remote_code_root)), "preflight", "--spec", str(spec),
    ]
    return _json_last(_stdout(remote.run(command)), name="node preflight")


def remote_run_exists(remote: SSHRemote, *, remote_root: Path) -> bool:
    """Check the immutable launch spec without invoking the node launcher."""
    result = remote.run(["test", "-f", str(remote_root / "launch.json")], check=False)
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise RemoteLaunchError(f"could not inspect existing remote run (exit {result.returncode})")


def stop_remote(remote: SSHRemote, *, remote_root: Path) -> dict[str, Any]:
    """Stop only the supervisor whose PID and Linux start token are recorded."""
    state_path = remote_root / ".rl-launch-state.json"
    script = r'''
import json, os, signal, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    print(json.dumps({"status": "not_started"}, sort_keys=True))
    raise SystemExit(0)
try:
    state = json.loads(path.read_text())
except (OSError, json.JSONDecodeError):
    print(json.dumps({"status": "stop_guard_failed", "reason": "invalid state"}, sort_keys=True))
    raise SystemExit(0)
pid, expected = state.get("supervisor_pid"), state.get("supervisor_start_token")
if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 or not isinstance(expected, str) or not expected:
    print(json.dumps({"status": "stop_guard_failed", "reason": "missing supervisor identity"}, sort_keys=True))
    raise SystemExit(0)
try:
    fields = Path(f"/proc/{pid}/stat").read_text().split()
    actual = fields[21]
    zombie = len(fields) > 2 and fields[2] == "Z"
except (OSError, IndexError):
    print(json.dumps({"status": "already_stopped", "pid": pid}, sort_keys=True))
    raise SystemExit(0)
if zombie or actual != expected:
    print(json.dumps({"status": "stop_guard_failed", "reason": "pid start token mismatch", "pid": pid}, sort_keys=True))
    raise SystemExit(0)
try:
    if os.getpgid(pid) == pid:
        os.killpg(pid, signal.SIGTERM)
    else:
        os.kill(pid, signal.SIGTERM)
except ProcessLookupError:
    pass
print(json.dumps({"status": "stop_requested", "pid": pid}, sort_keys=True))
'''
    result = remote.run(["python3", "-c", script, str(state_path)], check=False)
    try:
        return _json_last(_stdout(result), name="remote stop")
    except RemoteLaunchError:
        return {"status": "stop_failed", "exit_code": result.returncode}


def judge_command(
    *,
    repo_root: Path,
    connection: NodeConnection,
    identity: Path,
    known_hosts: Path,
    remote_root: Path,
    local_root: Path,
    protocol_dir: Path,
    codex_bin: str,
    deadline_seconds: float,
    poll_seconds: float,
    max_batches: int,
    remote_code_root: Path,
) -> list[str]:
    bridge = repo_root / "painter" / "runs" / "photo-curriculum-rl-20260922" / "judge_bridge.py"
    queue = remote_code_root / "painter" / "runs" / "photo-curriculum-rl-20260922" / "judge_queue.py"
    return [
        sys.executable, str(bridge), "worker", "--host", connection.host,
        "--port", str(connection.port), "--user", connection.user,
        "--identity", str(identity), "--known-hosts", str(known_hosts),
        "--remote-root", str(remote_root), "--queue-script", str(queue),
        "--local-root", str(local_root), "--protocol-dir", str(protocol_dir),
        "--codex-bin", codex_bin, "--deadline-seconds", str(deadline_seconds),
        "--poll-seconds", str(poll_seconds), "--max-batches", str(max_batches),
    ]


def _dry_plan(args: argparse.Namespace) -> dict[str, Any]:
    run_id = validate_run_id(args.run_id)
    data = validate_remote_path(args.data, name="data")
    profile = Path(args.profile_root)
    remote_root = profile.parent / run_id
    return {
        "status": "dry_run",
        "node_selector": args.node,
        "profile_root": str(profile),
        "remote_run_root": str(remote_root),
        "remote_manifest": data,
        "actions": ["resolve_lived_node_with_lium_ps", "stage_reviewed_code", "reuse_profile_assets", "read_model_and_adapter_pointers", "prepare_node_spec", "start_node_run", "run_local_luna_worker"],
        "provider_details_persisted": False,
        "credentials_staged": False,
    }


def run(args: argparse.Namespace, *, repo_root: Path | None = None) -> dict[str, Any]:
    if args.dry_run:
        return _dry_plan(args)
    if args.identity is None or args.known_hosts is None:
        raise RemoteLaunchError("run requires --identity and --known-hosts; credentials stay controller-local")
    repo_root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    connection = discover_node(args.node, user=args.user, lium_bin=args.lium_bin)
    remote = SSHRemote(connection, identity=args.identity, known_hosts=args.known_hosts)
    profile_root = Path(args.profile_root)
    remote_root = profile_root.parent / validate_run_id(args.run_id)
    # Keep the reviewed source beside this run.  The prepared profile and any
    # currently active run under /root/painter are left untouched.  An
    # explicit override remains available for a pre-created isolated root.
    remote_code_root = Path(args.remote_code_root) if args.remote_code_root else remote_root / "remote-code"
    # A launch spec is immutable. Inspect it before staging or preparing so a
    # repeated invocation can transition a prepared-but-not-started run into
    # start, while every active/terminal run remains attach-free.
    existing = None
    if remote_run_exists(remote, remote_root=remote_root):
        existing = status_remote(remote, remote_root=remote_root, remote_code_root=remote_code_root)
        if args.prepare_only and existing.get("status") == "not_started":
            runtime = profile_runtime(remote, profile_root=profile_root)
            preflight = preflight_remote(
                remote, remote_root=remote_root, remote_code_root=remote_code_root, runtime=runtime,
            )
            return {
                "status": "prepared",
                "run_id": args.run_id,
                "remote_run_root": str(remote_root),
                "existing": existing,
                "preflight": preflight,
                "duplicate_start": False,
                "provider_details_persisted": False,
            }
        if existing.get("status") != "not_started":
            return {
                "status": existing.get("status", "existing"),
                "run_id": args.run_id,
                "remote_run_root": str(remote_root),
                "existing": existing,
                "duplicate_start": True,
                "provider_details_persisted": False,
            }
        prepared = {"runtime": profile_runtime(remote, profile_root=profile_root), "prepare": existing}
    else:
        prepared = prepare_remote(
            remote, repo_root=repo_root, profile_root=profile_root, remote_root=remote_root,
            remote_code_root=remote_code_root,
            data=args.data, run_id=args.run_id, prime_root=Path(args.prime_root),
            max_steps=args.max_steps, base_revision=args.base_revision,
            prime_commit=args.prime_commit, validate_prime=args.validate_prime,
        )
        if args.prepare_only:
            preflight = preflight_remote(
                remote, remote_root=remote_root, remote_code_root=remote_code_root,
                runtime=prepared["runtime"],
            )
            return {
                "status": "prepared",
                "run_id": args.run_id,
                "remote_run_root": str(remote_root),
                "prepare": prepared["prepare"],
                "preflight": preflight,
                "provider_details_persisted": False,
            }
    started = start_remote(remote, remote_root=remote_root, remote_code_root=remote_code_root, runtime=prepared.get("runtime"))
    if started.get("status") != "started":
        # rl_entrypoint is the idempotency authority; never attach a second
        # judge or spawn a second run for an existing state.
        return {"status": started.get("status"), "run_id": args.run_id, "remote_run_root": str(remote_root), "duplicate_start": True}
    local_root = Path(args.local_root).expanduser().resolve() / args.run_id
    local_root.mkdir(parents=True, exist_ok=True)
    worker_log = local_root / "judge-worker.log"
    worker_log_handle = worker_log.open("ab")
    try:
        worker = subprocess.Popen(
            judge_command(
                repo_root=repo_root, connection=connection, identity=args.identity,
                known_hosts=args.known_hosts, remote_root=remote_root, local_root=local_root,
                protocol_dir=Path(args.protocol_dir).expanduser().resolve(), codex_bin=args.codex_bin,
                deadline_seconds=args.judge_deadline_seconds, poll_seconds=args.judge_poll_seconds,
                max_batches=args.judge_max_batches, remote_code_root=remote_code_root,
            ),
            stdout=worker_log_handle, stderr=subprocess.STDOUT,
        )
    except BaseException:
        worker_log_handle.close()
        stop_remote(remote, remote_root=remote_root)
        raise
    deadline = time.monotonic() + args.wait_seconds
    last_status: dict[str, Any] = started
    try:
        while time.monotonic() < deadline:
            last_status = status_remote(remote, remote_root=remote_root, remote_code_root=remote_code_root)
            if last_status.get("status") in TERMINAL:
                break
            if worker.poll() is not None:
                raise RemoteLaunchError("local Luna judge worker exited before the node run completed")
            time.sleep(args.poll_seconds)
        else:
            raise RemoteLaunchError("remote run exceeded --wait-seconds")
        worker.wait(timeout=max(30.0, args.judge_poll_seconds + 30.0))
        if worker.returncode != 0:
            raise RemoteLaunchError(f"local Luna judge worker exited {worker.returncode}")
    except BaseException:
        # The node supervisor must not outlive a failed controller/worker.  The
        # remote helper rechecks its recorded PID start token before signalling.
        stop_remote(remote, remote_root=remote_root)
        raise
    finally:
        if worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=20)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
        worker_log_handle.close()
    return {"status": last_status.get("status"), "run_id": args.run_id, "remote_run_root": str(remote_root), "node": started, "final": last_status, "provider_details_persisted": False}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run_parser = sub.add_parser("run", help="prepare/start one finite node run and own its Luna worker")
    run_parser.add_argument("--node", required=True, help="active Lium pod id/name/huid")
    run_parser.add_argument("--data", required=True, help="absolute manifest path already present on the node")
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--profile-root", default=str(DEFAULT_PROFILE_ROOT))
    run_parser.add_argument("--remote-code-root", default=None)
    run_parser.add_argument("--prime-root", default=str(DEFAULT_PRIME_ROOT))
    run_parser.add_argument("--identity", type=Path)
    run_parser.add_argument("--known-hosts", type=Path)
    run_parser.add_argument("--user", default="root")
    run_parser.add_argument("--lium-bin", default="lium")
    run_parser.add_argument("--max-steps", type=int, default=32)
    run_parser.add_argument("--base-revision", default=DEFAULT_BASE_REVISION)
    run_parser.add_argument("--prime-commit", default=DEFAULT_PRIME_COMMIT)
    run_parser.add_argument("--validate-prime", action="store_true")
    run_parser.add_argument("--prepare-only", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--local-root", default="/tmp/painter-rl-judge")
    run_parser.add_argument("--protocol-dir", default=str(Path(__file__).resolve().parent / "runs" / "aesthetic-rl-20260921"))
    run_parser.add_argument("--codex-bin", default="codex")
    run_parser.add_argument("--judge-deadline-seconds", type=float, default=48 * 60 * 60)
    run_parser.add_argument("--judge-poll-seconds", type=float, default=20.0)
    run_parser.add_argument("--judge-max-batches", type=int, default=1024)
    run_parser.add_argument("--wait-seconds", type=float, default=48 * 60 * 60)
    run_parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "run":
        result = run(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1 if result.get("status") in {"failed", "stale", "stopped"} else 0
    raise RemoteLaunchError(f"unsupported action: {args.action}")


if __name__ == "__main__":
    raise SystemExit(main())
