"""Finite, config-driven controller for a pinned native Prime RL run.

This controller owns process identity and evidence paths; Prime, vLLM, the
renderer, and the judge remain supplied by the launch specification.  It does
not use a shell, scheduler, watcher, or remote transport.  ``start`` launches
one detached supervisor, ``status`` is read-only apart from marking a dead
supervisor stale, and every new attempt must pass the same immutable identity
checks before a process is started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from typing import Any, Iterable

try:
    import fcntl
except ImportError:  # pragma: no cover - the production node is Linux
    fcntl = None  # type: ignore[assignment]


SCHEMA = "painter-native-rl-launch-v1"
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")
TERMINAL_STATES = frozenset({"complete", "failed", "stale", "stopped"})
_DETACHED_CHILDREN: dict[int, Any] = {}


class LaunchError(RuntimeError):
    """A launch specification or durable run state is unsafe to use."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise LaunchError(f"JSON object required: {path}")
    return value


def _path(spec_path: Path, value: Any, *, name: str, must_exist: bool = True) -> Path:
    if not isinstance(value, str) or not value:
        raise LaunchError(f"{name} must be a non-empty path")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = spec_path.parent / candidate
    try:
        result = candidate.resolve(strict=must_exist)
    except OSError as exc:
        raise LaunchError(f"{name} does not resolve: {candidate}") from exc
    if must_exist and not result.is_file():
        raise LaunchError(f"{name} is not a regular file: {result}")
    return result


def _digest(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise LaunchError(f"{name} must be a lowercase SHA-256")
    return value


def _command(value: Any, *, name: str) -> list[str]:
    if isinstance(value, str):
        value = shlex.split(value)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise LaunchError(f"{name} must be a non-empty argv list or shell-free string")
    return list(value)


def _identity_payload(spec: dict[str, Any], *, paths: dict[str, Path], digests: dict[str, str]) -> dict[str, Any]:
    identity = spec["identity"]
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "run_id": spec["run_id"],
        "mode": spec["mode"],
        "profile": identity["profile"],
        "base_model": identity["base_model"],
        "base_revision": identity["base_revision"],
        "prime_commit": identity["prime_commit"],
        "config_path": str(paths["config"]),
        "config_sha256": digests["config"],
        "manifest_path": str(paths["manifest"]),
        "manifest_sha256": digests["manifest"],
    }
    if "adapter" in paths:
        payload.update(
            adapter_path=str(paths["adapter"]),
            adapter_weight_path=str(paths["adapter_weight"]),
            adapter_sha256=digests["adapter"],
        )
    return payload


def validate_spec(spec_path: Path) -> "ValidatedSpec":
    spec_path = spec_path.expanduser().resolve(strict=True)
    spec = _read_json(spec_path)
    if spec.get("schema") != SCHEMA:
        raise LaunchError(f"unsupported launch spec schema: {spec.get('schema')!r}")
    run_id = spec.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise LaunchError("run_id must be a short filename-safe identifier")
    mode = spec.get("mode", "fresh")
    if mode == "native_resume":
        raise LaunchError(
            "native optimizer resume is not implemented for this Prime adapter wrapper; "
            "prepare a new explicit recovered_policy run instead"
        )
    if mode not in {"fresh", "recovered_policy"}:
        raise LaunchError("mode must be fresh or recovered_policy")
    identity = spec.get("identity")
    if not isinstance(identity, dict):
        raise LaunchError("identity object is required")
    for key in ("profile", "base_model", "base_revision", "prime_commit"):
        if not isinstance(identity.get(key), str) or not identity[key]:
            raise LaunchError(f"identity.{key} is required")
    paths: dict[str, Path] = {
        "run_root": _path(spec_path, spec.get("run_root"), name="run_root", must_exist=False),
        "config": _path(spec_path, (identity.get("config") or {}).get("path"), name="identity.config.path"),
        "manifest": _path(spec_path, (identity.get("manifest") or {}).get("path"), name="identity.manifest.path"),
    }
    digests = {
        "config": _digest((identity.get("config") or {}).get("sha256"), name="identity.config.sha256"),
        "manifest": _digest((identity.get("manifest") or {}).get("sha256"), name="identity.manifest.sha256"),
    }
    for key in ("config", "manifest"):
        actual = sha256_file(paths[key])
        if actual != digests[key]:
            raise LaunchError(f"{key} SHA-256 mismatch: expected {digests[key]}, got {actual}")
    adapter_identity = identity.get("adapter")
    if not isinstance(adapter_identity, dict):
        raise LaunchError(f"{mode} requires identity.adapter")
    adapter_dir = _path(spec_path, adapter_identity.get("path"), name="identity.adapter.path", must_exist=False)
    if not adapter_dir.is_dir() or adapter_dir.is_symlink():
        raise LaunchError(f"identity.adapter.path must be a regular directory: {adapter_dir}")
    adapter_file = adapter_identity.get("file", "adapter_model.safetensors")
    if not isinstance(adapter_file, str) or not adapter_file or Path(adapter_file).name != adapter_file:
        raise LaunchError("identity.adapter.file must be a single filename")
    adapter_weight = adapter_dir / adapter_file
    # Hugging Face snapshot files are commonly symlinks into the snapshot's
    # blob store.  ``is_file`` follows that link; the content hash below is
    # the binding, so rejecting a verified snapshot link would reject valid
    # public adapters.
    if not adapter_weight.is_file():
        raise LaunchError(f"adapter weight is missing: {adapter_weight}")
    paths["adapter"] = adapter_dir
    paths["adapter_weight"] = adapter_weight
    digests["adapter"] = _digest(adapter_identity.get("sha256"), name="identity.adapter.sha256")
    if sha256_file(adapter_weight) != digests["adapter"]:
        raise LaunchError("adapter SHA-256 mismatch")
    commands = spec.get("commands")
    if not isinstance(commands, dict):
        raise LaunchError("commands object is required")
    selected = mode
    if selected not in commands:
        raise LaunchError(f"commands.{selected} is required for mode {mode}")
    command = _command(commands[selected], name=f"commands.{selected}")
    setup = _command(commands["setup"], name="commands.setup") if "setup" in commands else None
    judge = spec.get("judge") or {}
    if not isinstance(judge, dict):
        raise LaunchError("judge must be an object")
    judge_command = _command(judge["command"], name="judge.command") if judge.get("command") is not None else None
    if judge.get("required", False) and judge_command is None:
        raise LaunchError("judge.required=true requires judge.command")
    environment = spec.get("environment") or {}
    if not isinstance(environment, dict):
        raise LaunchError("environment must be an object")
    fixed_env = environment.get("set") or {}
    passthrough = environment.get("passthrough") or []
    if not isinstance(fixed_env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in fixed_env.items()):
        raise LaunchError("environment.set must map string names to string values")
    if not isinstance(passthrough, list) or not all(isinstance(k, str) and k for k in passthrough):
        raise LaunchError("environment.passthrough must be a list of variable names")
    identity_payload = _identity_payload(spec, paths=paths, digests=digests)
    identity_digest = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ValidatedSpec(
        path=spec_path,
        raw=spec,
        run_id=run_id,
        mode=mode,
        run_root=paths["run_root"],
        paths=paths,
        digests=digests,
        identity_payload=identity_payload,
        identity_digest=identity_digest,
        command=command,
        setup_command=setup,
        judge_command=judge_command,
        judge_required=bool(judge.get("required", False)),
        fixed_env=fixed_env,
        passthrough=passthrough,
    )


class ValidatedSpec:
    def __init__(self, *, path: Path, raw: dict[str, Any], run_id: str, mode: str, run_root: Path,
                 paths: dict[str, Path], digests: dict[str, str], identity_payload: dict[str, Any],
                 identity_digest: str, command: list[str], setup_command: list[str] | None,
                 judge_command: list[str] | None, judge_required: bool, fixed_env: dict[str, str],
                 passthrough: list[str]):
        self.path = path
        self.raw = raw
        self.run_id = run_id
        self.mode = mode
        self.run_root = run_root
        self.paths = paths
        self.digests = digests
        self.identity_payload = identity_payload
        self.identity_digest = identity_digest
        self.command = command
        self.setup_command = setup_command
        self.judge_command = judge_command
        self.judge_required = judge_required
        self.fixed_env = fixed_env
        self.passthrough = passthrough

    @property
    def state_path(self) -> Path:
        return self.run_root / ".rl-launch-state.json"

    @property
    def lock_path(self) -> Path:
        return self.run_root / ".rl-launch.lock"

    def context(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "run_root": str(self.run_root),
            "config_path": str(self.paths["config"]),
            "manifest_path": str(self.paths["manifest"]),
            "adapter_path": str(self.paths.get("adapter", "")),
            "checkpoint_dir": str(self.paths.get("checkpoint", "")),
            "checkpoint_receipt": str(self.paths.get("checkpoint_receipt", "")),
        }


def render_command(command: Iterable[str], spec: ValidatedSpec) -> list[str]:
    context = spec.context()
    rendered: list[str] = []
    for item in command:
        value = item
        for key, replacement in context.items():
            value = value.replace("{" + key + "}", replacement)
        unknown = PLACEHOLDER.findall(value)
        if unknown:
            raise LaunchError(f"unknown command placeholder: {unknown[0]}")
        rendered.append(value)
    return rendered


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _load_state(spec: ValidatedSpec) -> dict[str, Any] | None:
    if not spec.state_path.is_file():
        return None
    state = _read_json(spec.state_path)
    if state.get("identity_digest") != spec.identity_digest:
        raise LaunchError("existing run state belongs to a different immutable identity")
    return state


def _pid_start_token(pid: Any) -> str | None:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        return fields[21]
    except (OSError, IndexError):
        return None


def _pid_alive(pid: Any, start_token: str | None = None) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # A detached supervisor that has exited can remain as a zombie until its
    # parent reaps it; it is not an attachable live controller.
    try:
        if Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z":
            return False
    except (OSError, IndexError):
        pass
    return start_token is None or _pid_start_token(pid) == start_token


def _lock(spec: ValidatedSpec):
    spec.run_root.mkdir(parents=True, exist_ok=True)
    handle = spec.lock_path.open("a+")
    if fcntl is not None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except OSError:
            handle.close()
            raise
    return handle


def _environment(spec: ValidatedSpec, *, attempt: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(spec.fixed_env)
    for name in spec.passthrough:
        if name in os.environ:
            environment[name] = os.environ[name]
    environment.update({
        "PAINTER_RL_RUN_ID": spec.run_id,
        "PAINTER_RL_RUN_ROOT": str(spec.run_root),
        "PAINTER_RL_CONFIG": str(spec.paths["config"]),
        "PAINTER_RL_MANIFEST": str(spec.paths["manifest"]),
        "PAINTER_RL_MANIFEST_SHA256": spec.digests["manifest"],
        "PAINTER_RL_CONFIG_SHA256": spec.digests["config"],
        "PAINTER_RL_LAUNCH_IDENTITY_SHA256": spec.identity_digest,
        "PAINTER_RL_ATTEMPT": str(attempt),
        "PAINTER_RL_MODE": spec.mode,
    })
    if "adapter" in spec.paths:
        environment["WATERCOLOUR_INIT_ADAPTER"] = str(spec.paths["adapter"])
    if "checkpoint" in spec.paths:
        environment["PAINTER_RL_RESUME_CHECKPOINT"] = str(spec.paths["checkpoint"])
    return environment


def _safe_run(command: list[str], *, env: dict[str, str], log, cwd: Path) -> int:
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return process.wait()


def _terminate_process(process: Any) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except OSError:
        try:
            process.terminate()
        except OSError:
            return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            try:
                process.kill()
            except OSError:
                pass
        process.wait()


def _runtime_judge_from_env() -> tuple[list[str] | None, bool]:
    """Read a one-invocation judge command without putting it in the spec.

    The command is supplied through the parent process environment by the
    foreground ``run`` action.  It is deliberately absent from the durable
    launch JSON/state so SSH keys, controller paths, and provider arguments
    cannot become experiment metadata.
    """
    raw = os.environ.get("PAINTER_RL_RUNTIME_JUDGE_JSON")
    if not raw:
        return None, False
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LaunchError("PAINTER_RL_RUNTIME_JUDGE_JSON is not valid JSON") from exc
    if not isinstance(value, list):
        raise LaunchError("runtime judge command must be an argv list")
    return _command(value, name="runtime judge command"), True


def _supervise(spec: ValidatedSpec, *, attempt: int, setup_requested: bool) -> int:
    spec.run_root.mkdir(parents=True, exist_ok=True)
    log_dir = spec.run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"rl-launch-{attempt}.log"
    env = _environment(spec, attempt=attempt)
    state = {
        "schema": SCHEMA,
        "run_id": spec.run_id,
        "identity_digest": spec.identity_digest,
        "mode": spec.mode,
        "attempt": attempt,
        "status": "running",
        "supervisor_pid": os.getpid(),
        "supervisor_start_token": _pid_start_token(os.getpid()),
        "started_at_unix": time.time(),
        "log": str(log_path),
    }
    _atomic_json(spec.state_path, state)
    judge = None
    runner = None
    run_status = 1
    interrupted = False

    def interrupt(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    previous_term = signal.signal(signal.SIGTERM, interrupt)
    previous_int = signal.signal(signal.SIGINT, interrupt)
    with log_path.open("ab") as log:
        try:
            runtime_judge, runtime_judge_required = _runtime_judge_from_env()
            judge_command = runtime_judge if runtime_judge is not None else spec.judge_command
            judge_required = runtime_judge_required or spec.judge_required
            if setup_requested and spec.setup_command:
                setup_status = _safe_run(render_command(spec.setup_command, spec), env=env, log=log, cwd=spec.run_root)
                if setup_status != 0:
                    state.update(status="failed", phase="setup", exit_code=setup_status, finished_at_unix=time.time())
                    _atomic_json(spec.state_path, state)
                    return setup_status
            if judge_command:
                judge = subprocess.Popen(
                    render_command(judge_command, spec),
                    cwd=spec.run_root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                state["judge_pid"] = judge.pid
                _atomic_json(spec.state_path, state)
            runner = subprocess.Popen(
                render_command(spec.command, spec),
                cwd=spec.run_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            state["runner_pid"] = runner.pid
            _atomic_json(spec.state_path, state)
            while True:
                run_status = runner.poll()
                if run_status is not None:
                    break
                if interrupted:
                    _terminate_process(runner)
                    run_status = 143
                    break
                if judge is not None and judge.poll() is not None and judge_required:
                    # A required local judge must remain alive for the entire
                    # finite Prime run, including a zero exit code.
                    _terminate_process(runner)
                    run_status = 143
                    state.update(status="failed", phase="judge", exit_code=judge.returncode, runner_exit_code=run_status)
                    _atomic_json(spec.state_path, state)
                    return 1
                time.sleep(1.0)
            _terminate_process(judge)
            state.update(
                status="complete" if run_status == 0 and not interrupted else "failed",
                exit_code=run_status,
                finished_at_unix=time.time(),
            )
            _atomic_json(spec.state_path, state)
            return int(run_status)
        except BaseException as exc:
            _terminate_process(runner)
            _terminate_process(judge)
            state.update(status="failed", phase="controller", error=f"{type(exc).__name__}: {exc}", finished_at_unix=time.time())
            _atomic_json(spec.state_path, state)
            return 1
        finally:
            signal.signal(signal.SIGTERM, previous_term)
            signal.signal(signal.SIGINT, previous_int)


def start(
    spec: ValidatedSpec,
    *,
    retry: bool = False,
    setup: bool = True,
    runtime_judge: list[str] | None = None,
) -> dict[str, Any]:
    with _lock(spec):
        state = _load_state(spec)
        if state is not None:
            if state.get("status") in {"starting", "running"} and _pid_alive(
                state.get("supervisor_pid"), state.get("supervisor_start_token")
            ):
                return {"status": "already_running", "run_id": spec.run_id, "pid": state.get("supervisor_pid"), "attempt": state.get("attempt")}
            if state.get("status") == "complete":
                return {"status": "already_complete", "run_id": spec.run_id, "attempt": state.get("attempt")}
            if not retry:
                raise LaunchError("run has a prior failed/stale attempt; pass --retry explicitly")
            raise LaunchError(
                "in-place retry is unsupported; prepare a new run_id/run_root to avoid reusing partial Prime output"
            )
        else:
            # The immutable config/manifest are required inputs and may already
            # exist in a prepared run root. Only runtime/output state blocks a
            # fresh attempt without an identity receipt.
            sentinels = ("async-output", "checkpoints", "rl-rollouts", "rl.exit")
            existing = [name for name in sentinels if (spec.run_root / name).exists()]
            if existing:
                raise LaunchError("run root contains prior artifacts without matching controller state: " + ", ".join(existing))
            attempt = 1
        state = {
            "schema": SCHEMA,
            "run_id": spec.run_id,
            "identity_digest": spec.identity_digest,
            "mode": spec.mode,
            "attempt": attempt,
            "status": "starting",
            "started_at_unix": time.time(),
        }
        _atomic_json(spec.state_path, state)
        supervisor_env = os.environ.copy()
        if runtime_judge is not None:
            supervisor_env["PAINTER_RL_RUNTIME_JUDGE_JSON"] = json.dumps(runtime_judge)
        else:
            supervisor_env.pop("PAINTER_RL_RUNTIME_JUDGE_JSON", None)
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "_run", "--spec", str(spec.path), "--attempt", str(attempt), *( ["--setup"] if setup else [])],
            cwd=spec.run_root,
            env=supervisor_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        # Keep the Popen handle alive while the detached supervisor runs. This
        # avoids Python's ResourceWarning and lets status() reap the handle
        # after the finite child exits.
        _DETACHED_CHILDREN[process.pid] = process
        # The detached child can finish a very short run before this parent
        # records its PID.  Merge into the latest state instead of restoring
        # the provisional ``starting`` record over a terminal result.
        latest = _read_json(spec.state_path) if spec.state_path.is_file() else state
        if latest.get("identity_digest") != spec.identity_digest:
            raise LaunchError("supervisor changed launch identity during startup")
        latest["supervisor_pid"] = process.pid
        latest["supervisor_start_token"] = _pid_start_token(process.pid)
        _atomic_json(spec.state_path, latest)
        return {"status": "started", "run_id": spec.run_id, "pid": process.pid, "attempt": attempt}


def status(spec: ValidatedSpec) -> dict[str, Any]:
    state = _load_state(spec)
    if state is None:
        return {"status": "not_started", "run_id": spec.run_id, "identity_digest": spec.identity_digest}
    if state.get("status") in {"starting", "running"} and not _pid_alive(
        state.get("supervisor_pid"), state.get("supervisor_start_token")
    ):
        state.update(status="stale", finished_at_unix=time.time(), reason="supervisor is no longer alive")
        _atomic_json(spec.state_path, state)
    pid = state.get("supervisor_pid")
    handle = _DETACHED_CHILDREN.get(pid)
    if handle is not None and handle.poll() is not None:
        _DETACHED_CHILDREN.pop(pid, None)
    return {key: state[key] for key in state if key not in {"error"} or not state.get("error")}


def preflight(spec: ValidatedSpec) -> dict[str, Any]:
    """Run the launcher import/wire checks without setup, telemetry, or Prime."""
    environment = _environment(spec, attempt=0)
    environment["PAINTER_RL_PREFLIGHT_ONLY"] = "1"
    environment["PAINTER_RL_SKIP_EDITABLE_INSTALL"] = "1"
    command = render_command(spec.raw["commands"]["fresh"], spec)
    completed = subprocess.run(
        command,
        cwd=spec.run_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "preflight failed").strip().splitlines()[-1]
        raise LaunchError(f"launcher preflight failed ({completed.returncode}): {detail}")
    return {"status": "preflight_passed", "run_id": spec.run_id, "training_started": False}


def _signal_supervisor(pid: int, start_token: str | None, signum: int) -> None:
    """Signal only the controller process group that passed identity checks."""
    if not _pid_alive(pid, start_token):
        return
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, signum)
        else:
            os.kill(pid, signum)
    except OSError:
        pass


def stop(spec: ValidatedSpec, *, wait_seconds: float = 30.0) -> dict[str, Any]:
    """Stop an owned controller tree after verifying PID and start identity."""
    if wait_seconds <= 0:
        raise LaunchError("stop wait timeout must be positive")
    with _lock(spec):
        state = _load_state(spec)
        if state is None:
            return {"status": "not_started", "run_id": spec.run_id}
        if state.get("status") in TERMINAL_STATES:
            return status(spec)
        pid = state.get("supervisor_pid")
        token = state.get("supervisor_start_token")
        if not _pid_alive(pid, token):
            state.update(status="stale", reason="supervisor identity is no longer alive", finished_at_unix=time.time())
            _atomic_json(spec.state_path, state)
            return status(spec)
        _signal_supervisor(pid, token, signal.SIGTERM)

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        current = _load_state(spec)
        if current is not None and current.get("status") in TERMINAL_STATES:
            return status(spec)
        if not _pid_alive(pid, token):
            break
        time.sleep(min(0.25, max(0.01, deadline - time.monotonic())))

    if _pid_alive(pid, token):
        _signal_supervisor(pid, token, signal.SIGKILL)
    with _lock(spec):
        current = _load_state(spec) or state
        if current.get("status") not in TERMINAL_STATES:
            current.update(status="stopped", reason="controller stop requested", finished_at_unix=time.time())
            _atomic_json(spec.state_path, current)
        return status(spec)


def setup(spec: ValidatedSpec) -> dict[str, Any]:
    if spec.setup_command is None:
        return {"status": "no_setup_command", "run_id": spec.run_id}
    setup_state = spec.run_root / ".rl-setup-state.json"
    if setup_state.is_file():
        previous = _read_json(setup_state)
        if previous.get("identity_digest") == spec.identity_digest and previous.get("status") == "complete":
            return {"status": "already_ready", "run_id": spec.run_id}
    spec.run_root.mkdir(parents=True, exist_ok=True)
    log_path = spec.run_root / "logs" / "rl-setup.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        code = _safe_run(render_command(spec.setup_command, spec), env=_environment(spec, attempt=0), log=log, cwd=spec.run_root)
    result = {"schema": SCHEMA, "run_id": spec.run_id, "identity_digest": spec.identity_digest,
              "status": "complete" if code == 0 else "failed", "exit_code": code, "finished_at_unix": time.time()}
    _atomic_json(setup_state, result)
    if code != 0:
        raise LaunchError(f"setup command failed with exit code {code}")
    return {"status": "ready", "run_id": spec.run_id}


def _manifest_count(path: Path) -> int:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return sum(1 for line in path.read_text().splitlines() if line.strip())
    value = json.loads(path.read_text())
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        for key in ("training_rows", "rows", "tasks", "training_states"):
            rows = value.get(key)
            if isinstance(rows, list):
                return len(rows)
    raise LaunchError(f"cannot count rows in manifest: {path}")


def _assert_config_manifest(config: Path, manifest: Path, expected_count: int) -> None:
    """Require the generated taskset to point at this exact staged manifest."""
    try:
        document = tomllib.loads(config.read_text())
        taskset = document["orchestrator"]["train"]["source"][0]["env"]["taskset"]
        configured = Path(taskset["manifest"])
        configured = (config.parent / configured).resolve() if not configured.is_absolute() else configured.resolve()
    except (KeyError, IndexError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise LaunchError("generated native RL config has no readable taskset manifest") from exc
    if configured != manifest.resolve():
        raise LaunchError(f"native RL taskset manifest mismatch: {configured} != {manifest.resolve()}")
    if taskset.get("expected_count") != expected_count:
        raise LaunchError(
            f"native RL expected_count mismatch: {taskset.get('expected_count')!r} != {expected_count}"
        )


def _stage_profile_file(source: Path, destination: Path) -> None:
    """Install one reviewed profile file without silently replacing a run input."""
    if not source.is_file():
        raise LaunchError(f"profile file is missing: {source}")
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != sha256_file(source):
            raise LaunchError(f"run root already contains a different profile file: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


_ASSET_KEYS = frozenset({
    "reference", "reference_path", "source_path", "initial_canvas", "initial_program",
    "canvas", "canvas_path", "opponent", "opponent_png", "fixed_opponent", "anchor",
    "anchor_png", "baseline", "baseline_png", "source_image", "image", "png", "file",
})


def _normalise_manifest_assets(value: Any, *, source_dir: Path, staged_dir: Path, asset_context: bool = False) -> Any:
    """Keep portable refs when possible; absolutize refs that would break on staging."""
    if isinstance(value, list):
        return [
            _normalise_manifest_assets(item, source_dir=source_dir, staged_dir=staged_dir, asset_context=asset_context)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _normalise_manifest_assets(
                item,
                source_dir=source_dir,
                staged_dir=staged_dir,
                asset_context=asset_context or key in _ASSET_KEYS,
            )
            for key, item in value.items()
        }
    if not asset_context or not isinstance(value, str) or not value or Path(value).is_absolute():
        return value
    candidate = source_dir / value
    if not candidate.is_file():
        return value
    staged_candidate = staged_dir / value
    if staged_candidate.is_file():
        try:
            if candidate.samefile(staged_candidate):
                return value
        except OSError:
            pass
    return str(candidate.resolve())


def _stage_manifest(source: Path, destination: Path) -> None:
    """Stage JSON/JSONL without losing format or breaking relative assets."""
    if destination.exists():
        if not destination.is_file():
            raise LaunchError(f"staged manifest is not a regular file: {destination}")
        # An existing identical destination is safe and makes prepare
        # idempotent up to the later immutable-config check.
        if sha256_file(destination) == sha256_file(source):
            return
        raise LaunchError(f"run root already contains a different manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in {".jsonl", ".ndjson"}:
        lines = [line for line in source.read_text().splitlines() if line.strip()]
        values = [json.loads(line) for line in lines]
        normalised = [
            _normalise_manifest_assets(item, source_dir=source.parent, staged_dir=destination.parent)
            for item in values
        ]
        if normalised == values:
            shutil.copy2(source, destination)
        else:
            destination.write_text("".join(json.dumps(item, separators=(",", ":")) + "\n" for item in normalised))
        return
    document = json.loads(source.read_text())
    normalised = _normalise_manifest_assets(document, source_dir=source.parent, staged_dir=destination.parent)
    if normalised == document:
        shutil.copy2(source, destination)
    else:
        destination.write_text(json.dumps(normalised, indent=2) + "\n")


def _staged_manifest_path(run_root: Path, source: Path) -> Path:
    suffix = source.suffix.lower() if source.suffix.lower() in {".jsonl", ".ndjson"} else ".json"
    return run_root / f"source-manifest{suffix}"


def _stage_photo_profile(run_root: Path) -> None:
    """Copy the small, reviewed node wrapper set into a fresh run directory."""
    source_root = Path(__file__).resolve().parent / "runs" / "photo-curriculum-rl-20260922"
    for name in (
        "launch_rl.sh",
        "setup_rl_runtime.sh",
        "prepare_rl_config.py",
        "run_from_sft.py",
        "train_from_sft.py",
    ):
        _stage_profile_file(source_root / name, run_root / name)


def _prepare_photo(args: argparse.Namespace) -> dict[str, Any]:
    if args.profile != "photo":
        raise LaunchError("only the pinned photo profile is available")
    if isinstance(args.max_steps, bool) or not isinstance(args.max_steps, int) or args.max_steps < 8:
        raise LaunchError("photo profile requires max_steps >= 8 for checkpoint interval 8")
    run_root = args.run_root.expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    data_source = args.data.expanduser().resolve(strict=True)
    data = _staged_manifest_path(run_root, data_source)
    if data_source != data:
        _stage_manifest(data_source, data)
    elif not data.is_file():
        raise LaunchError(f"data manifest is not a regular file: {data}")
    if not data.is_file():
        raise LaunchError(f"data manifest is not a regular file: {data}")
    _stage_photo_profile(run_root)
    adapter = args.adapter.expanduser().resolve(strict=True)
    if not adapter.is_dir() or not (adapter / "adapter_model.safetensors").is_file():
        raise LaunchError("--adapter must be a portable LoRA directory containing adapter_model.safetensors")
    model = args.model.expanduser().resolve(strict=True)
    if not model.is_dir():
        raise LaunchError("--model must be the staged base model directory")
    prime = args.prime_root.expanduser().resolve(strict=True)
    if not (prime / ".git").exists():
        raise LaunchError("--prime-root must be the pinned Prime checkout")
    count = _manifest_count(data)
    config = run_root / "native-rl.toml"
    spec_path = run_root / "launch.json"
    if config.exists() or spec_path.exists():
        raise LaunchError("prepared config/spec already exists; choose a new run_root/run_id")
    prepare_script = run_root / "prepare_rl_config.py"
    relative_manifest = data.relative_to(run_root).as_posix()
    python = args.python or os.environ.get("PAINTER_RL_PYTHON") or sys.executable
    command = [
        python,
        str(prepare_script),
        "--output", str(config),
        "--model", str(model),
        "--run-root", str(run_root),
        "--manifest", relative_manifest,
        "--max-steps", str(args.max_steps),
        "--expected-count", str(count),
        "--run-name", args.run_id,
    ]
    if args.validate_prime:
        command.extend(["--prime-root", str(prime), "--validate-prime"])
    prepare_env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[1])
    prepare_env["PYTHONPATH"] = repo_root + (os.pathsep + prepare_env["PYTHONPATH"] if prepare_env.get("PYTHONPATH") else "")
    subprocess.run(command, check=True, cwd=run_root, env=prepare_env)
    _assert_config_manifest(config, data, count)
    adapter_weight = adapter / "adapter_model.safetensors"
    environment = {
        "PRIME_ROOT": str(prime),
        "MODEL_DIR": str(model),
        "PAINTER_REPO_ROOT": str(Path(__file__).resolve().parents[1]),
        "PAINTER_RL_MANIFEST": str(data),
        "PAINTER_RL_CONFIG": str(config),
        "PAINTER_RL_MAX_STEPS": str(args.max_steps),
        "WATERCOLOUR_INIT_ADAPTER": str(adapter),
        "PAINTER_RL_RUN_NAME": args.run_id,
        "PAINTER_RL_EXPECTED_COUNT": str(count),
    }
    # These are node-local paths, not credential values. Persisting the paths
    # lets the detached start/preflight reconstruct the prepared runtime while
    # keeping the token itself outside the launch spec and logs.
    for name in ("HF_TOKEN_PATH", "UV_BIN"):
        if os.environ.get(name):
            environment[name] = os.environ[name]
    if args.render_root:
        environment["PAINTER_RENDER_ROOT"] = str(args.render_root.expanduser().resolve())
    spec = {
        "schema": SCHEMA,
        "run_id": args.run_id,
        "run_root": str(run_root),
        "mode": "fresh",
        "identity": {
            "profile": "qwen38_27b",
            "base_model": "Qwen/Qwen3.8-27B",
            "base_revision": args.base_revision,
            "prime_commit": args.prime_commit,
            "config": {"path": str(config), "sha256": sha256_file(config)},
            "manifest": {"path": str(data), "sha256": sha256_file(data)},
            "adapter": {
                "path": str(adapter),
                "file": "adapter_model.safetensors",
                "sha256": sha256_file(adapter_weight),
            },
        },
        "commands": {
            "setup": ["bash", str(run_root / "setup_rl_runtime.sh"), str(run_root)],
            "fresh": ["bash", str(run_root / "launch_rl.sh"), str(run_root)],
        },
        "judge": {
            "required": False,
            "coordination": "supply a controller-side bridge command at run time; credentials never enter the node spec",
        },
        "environment": {
            "set": environment,
            "passthrough": ["HF_TOKEN", "HF_TOKEN_PATH", "PAINTER_LOCAL_API_KEY"],
        },
    }
    _atomic_json(spec_path, spec)
    validated = validate_spec(spec_path)
    return {
        "status": "prepared",
        "profile": args.profile,
        "run_id": args.run_id,
        "spec": str(spec_path),
        "identity_sha256": validated.identity_digest,
        "manifest_path": str(data),
        "manifest_sha256": validated.digests["manifest"],
        "manifest_rows": count,
        "config_sha256": validated.digests["config"],
        "adapter_sha256": validated.digests["adapter"],
    }


def _run_photo(args: argparse.Namespace) -> dict[str, Any]:
    """Prepare and run one finite profile with an explicitly supplied judge.

    ``judge_command`` is intentionally runtime-only.  This action is useful
    when the controller and node share a filesystem/process host.  For a
    separate rented node, invoke the same generated spec on the node and run
    the existing bridge on the controller; this wrapper does not smuggle SSH
    credentials into durable node state.
    """
    if not args.judge_command:
        raise LaunchError(
            "run requires --judge-command for a finite controller-side judge; "
            "use prepare/start when the bridge runs on a separate controller"
        )
    prepared = _prepare_photo(args)
    spec = validate_spec(Path(prepared["spec"]))
    result = start(spec, setup=not args.no_setup, runtime_judge=list(args.judge_command))
    if result.get("status") != "started":
        raise LaunchError(f"cannot attach a new controller-side judge to {result.get('status')}")
    deadline = time.monotonic() + float(args.wait_seconds)
    while True:
        current = status(spec)
        if current.get("status") in TERMINAL_STATES:
            result = dict(prepared)
            result.update(current)
            if current.get("status") != "complete":
                raise LaunchError(f"finite RL run ended {current.get('status')}: {json.dumps(result, sort_keys=True)}")
            return result
        if time.monotonic() >= deadline:
            raise LaunchError("finite run exceeded --wait-seconds; inspect status and logs")
        time.sleep(args.poll_seconds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preflight", "run", "start", "stop", "status", "setup", "_run"))
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--profile", default="photo")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--prime-root", type=Path)
    parser.add_argument("--render-root", type=Path)
    parser.add_argument("--python")
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--base-revision", default="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
    parser.add_argument("--prime-commit", default="26b3131d2716a4f8210b165df584b83b4bc54f61")
    parser.add_argument("--validate-prime", action="store_true")
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--no-setup", action="store_true")
    parser.add_argument("--attempt", type=int)
    parser.add_argument("--setup", action="store_true")
    parser.add_argument(
        "--judge-command", nargs="+",
        help="runtime-only controller-side judge argv; required by `run`, never written to launch.json",
    )
    parser.add_argument("--wait-seconds", type=float, default=24 * 60 * 60)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--stop-wait-seconds", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action in {"prepare", "run"}:
        for name in ("data", "run_id", "run_root", "model", "adapter", "prime_root"):
            if getattr(args, name) in (None, ""):
                raise LaunchError(f"{args.action} requires --{name.replace('_', '-')}")
        if args.wait_seconds <= 0 or args.poll_seconds <= 0:
            raise LaunchError("--wait-seconds and --poll-seconds must be positive")
        result = _run_photo(args) if args.action == "run" else _prepare_photo(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.spec is None:
        raise LaunchError(f"{args.action} requires --spec")
    spec = validate_spec(args.spec)
    if args.action == "preflight":
        print(json.dumps(preflight(spec), sort_keys=True))
        return 0
    if args.action == "start":
        print(json.dumps(start(spec, retry=args.retry, setup=not args.no_setup), sort_keys=True))
        return 0
    if args.action == "stop":
        print(json.dumps(stop(spec, wait_seconds=args.stop_wait_seconds), indent=2, sort_keys=True))
        return 0
    if args.action == "status":
        print(json.dumps(status(spec), indent=2, sort_keys=True))
        return 0
    if args.action == "setup":
        print(json.dumps(setup(spec), sort_keys=True))
        return 0
    if args.attempt is None or args.attempt < 1:
        raise SystemExit("_run requires a positive --attempt")
    return _supervise(spec, attempt=args.attempt, setup_requested=args.setup)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LaunchError as exc:
        print(f"launch error: {exc}", file=sys.stderr)
        raise SystemExit(2)
