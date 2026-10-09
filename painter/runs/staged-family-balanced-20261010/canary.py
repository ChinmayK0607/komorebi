#!/usr/bin/env python3
"""Run the finite, node-local Prime/runtime-binding canary.

The canary imports the pinned Prime checkout, installs the staged
``sitecustomize`` hook, and runs one bounded deterministic renderer probe. It
never constructs an orchestrator, starts an env server, launches a trainer,
imports CUDA workers, or contacts a model/judge. Native Prime setup,
TrainSink, task dispatch, and tail-purge API remain explicit runtime gates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tomllib
from typing import Any


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[2]
SPEC = importlib.util.spec_from_file_location("staged_prepare_node_for_canary", RUN_ROOT / "prepare_node.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("prepare_node.py is unavailable")
PREP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREP
SPEC.loader.exec_module(PREP)
CONFIG = PREP.CONFIG


RUN_ID = CONFIG.RUN_ID
PRIME_COMMIT = CONFIG.PRIME_COMMIT
PROTOCOL = "staged-family-balanced-protocol-v1"
RENDERER_SHA256 = PREP.RENDERER_SHA256
RENDER_TIMEOUT_SECONDS = 90


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _renderer_invocation(
    *, renderer: Path, renderer_python: Path, source: Path, output: Path,
    browser_root: Path | None, probe_root: Path,
) -> tuple[list[str], dict[str, str], bool]:
    """Build the bounded renderer command and the reviewed root boundary."""

    base_command = [
        str(renderer_python), str(renderer), str(source),
        "--output", str(output), "--seed", "20261010",
        "--timeout", str(RENDER_TIMEOUT_SECONDS), "--backend", "swiftshader",
    ]
    runtime_home = probe_root / "home"
    runtime_tmp = probe_root / "tmp"
    probe_root.mkdir(parents=True, exist_ok=True)
    runtime_home.mkdir(exist_ok=False)
    runtime_tmp.mkdir(exist_ok=False)
    runtime_home.chmod(0o700)
    runtime_tmp.chmod(0o700)
    runuser = shutil.which("runuser")
    unprivileged = os.geteuid() == 0 and runuser is not None
    if unprivileged:
        painter_uid = subprocess.run(["id", "-u", "painter"], capture_output=True, text=True, check=False)
        painter_gid = subprocess.run(["id", "-g", "painter"], capture_output=True, text=True, check=False)
        if painter_uid.returncode != 0 or painter_gid.returncode != 0:
            raise RuntimeError("root renderer canary requires the unprivileged painter account")
        os.chown(runtime_home, int(painter_uid.stdout.strip()), int(painter_gid.stdout.strip()))
        os.chown(runtime_tmp, int(painter_uid.stdout.strip()), int(painter_gid.stdout.strip()))
    environment = {
        "PATH": os.pathsep.join(dict.fromkeys((str(renderer_python.parent), "/usr/local/bin", "/usr/bin", "/bin"))),
        "HOME": str(runtime_home), "TMPDIR": str(runtime_tmp), "XDG_RUNTIME_DIR": str(runtime_tmp),
        "LANG": "C", "LC_ALL": "C", "CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "void",
        "PAINTER_RENDER_BACKEND": "swiftshader", "PAINTER_RENDER_TIMEOUT_SECONDS": str(RENDER_TIMEOUT_SECONDS),
    }
    if browser_root is not None:
        environment["PAINTER_BROWSER_ROOT"] = str(browser_root)
        environment["PLAYWRIGHT_BROWSERS_PATH"] = str(browser_root)
    if unprivileged:
        command = [str(runuser), "-u", "painter", "--", "env", "-i"]
        command.extend(f"{key}={value}" for key, value in environment.items())
        command.extend(base_command)
    else:
        command = base_command
    return command, environment, unprivileged


def _renderer_probe(
    *,
    renderer: Path,
    renderer_python: Path,
    node_root: Path,
    browser_root: Path | None = None,
) -> dict[str, Any]:
    """Render one fixed p5.brush line through the verified renderer boundary."""

    renderer = PREP._file(renderer, "prepared renderer")
    renderer_python = PREP._runtime_executable(renderer_python, "prepared renderer Python", renderer_python.parent.parent)
    actual_renderer_sha = sha256_file(renderer)
    if actual_renderer_sha != RENDERER_SHA256:
        raise ValueError(f"renderer SHA-256 is {actual_renderer_sha}, expected {RENDERER_SHA256}")
    probe_root = node_root / "renderer-canary"
    previous_evidence: Path | None = None
    if probe_root.is_symlink():
        raise FileExistsError(f"renderer canary path cannot be a symlink: {probe_root}")
    if probe_root.exists():
        # Keep every failed attempt for diagnosis while giving the bounded
        # retry a fresh, deterministic path. The unique suffix avoids
        # replacing prior evidence if a retry happens in the same nanosecond.
        previous_evidence = node_root / f"renderer-canary.previous-{time.time_ns()}-{os.getpid()}"
        if previous_evidence.exists() or previous_evidence.is_symlink():
            raise FileExistsError(f"renderer canary evidence destination already exists: {previous_evidence}")
        probe_root.rename(previous_evidence)
    probe_root.mkdir(parents=True)
    if os.geteuid() == 0 and shutil.which("runuser") is not None:
        # The probe contains no credentials.  Its browser child must be able
        # to traverse the prepared root and create the PNG/renderer receipt.
        os.chmod(node_root, 0o711)
        os.chmod(probe_root, 0o733)
    source = probe_root / "probe.js"
    output = probe_root / "probe.png"
    renderer_receipt = output.with_suffix(".json")
    receipt_path = probe_root / "renderer-canary-receipt.json"
    source.write_text(
        "function setup(){createCanvas(600,600,WEBGL);}\n"
        "function draw(){translate(-width/2,-height/2);background('#f5efe4');"
        "brush.set('cpencil','#554d44',0.8);brush.stroke('#554d44');"
        "brush.line(80,80,520,520);noLoop();}\n",
        encoding="utf-8",
    )
    source_hash = sha256_file(source)
    if browser_root is not None:
        browser_root = PREP._directory(browser_root, "renderer browser root")
    command, environment, unprivileged = _renderer_invocation(
        renderer=renderer, renderer_python=renderer_python, source=source,
        output=output, browser_root=browser_root, probe_root=probe_root,
    )
    started = time.monotonic()
    timed_out = False
    returncode: int | None = None
    try:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=environment,
            capture_output=True,
            text=True,
            timeout=RENDER_TIMEOUT_SECONDS,
            check=False,
        )
        returncode = completed.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
    except OSError as exc:
        raise RuntimeError(f"renderer canary could not start: {type(exc).__name__}") from exc
    elapsed = time.monotonic() - started
    png_hash = sha256_file(output) if output.is_file() else None
    native = None
    try:
        native = json.loads(renderer_receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    valid = bool(
        not timed_out
        and returncode == 0
        and elapsed <= RENDER_TIMEOUT_SECONDS
        and output.is_file()
        and output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        and isinstance(native, dict)
        and native.get("valid") is True
        and native.get("renderer_sha256") == actual_renderer_sha
        and native.get("source_sha256") == source_hash
        and native.get("png_sha256") == png_hash
    )
    receipt = {
        "schema": "staged-family-balanced-renderer-canary-v1",
        "status": "verified" if valid else "failed",
        "valid": valid,
        "renderer": str(renderer),
        "renderer_sha256": actual_renderer_sha,
        "renderer_python": str(renderer_python),
        "source": str(source),
        "source_sha256": source_hash,
        "output": str(output),
        "png_sha256": png_hash,
        "renderer_receipt": str(renderer_receipt),
        "renderer_receipt_sha256": sha256_file(renderer_receipt) if renderer_receipt.is_file() else None,
        "renderer_result": native,
        "timeout_seconds": RENDER_TIMEOUT_SECONDS,
        "elapsed_seconds": round(elapsed, 3),
        "returncode": returncode,
        "timed_out": timed_out,
        "unprivileged_renderer_boundary": unprivileged,
        "renderer_command": command[:8] + ["..."] if len(command) > 8 else command,
        "previous_evidence": str(previous_evidence) if previous_evidence is not None else None,
        "cuda_visibility": "CUDA_VISIBLE_DEVICES empty; no model/GPU allocation attempted",
    }
    _atomic_json(receipt_path, receipt)
    if not valid:
        raise RuntimeError(f"renderer canary failed; evidence preserved at {receipt_path}")
    return receipt


def _read_config(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Prime config is not a regular file: {path}")
    value = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Prime config is not a TOML table")
    return value


def _probe_code() -> str:
    # The probe intentionally does not instantiate or call Orchestrator.setup.
    # Validate the complete pinned config before importing the patched class;
    # this catches schema drift without starting Prime or any worker process.
    return """
import json
import sys
import tomllib
from pathlib import Path
from prime_rl.configs.rl import RLConfig
from prime_rl.orchestrator.orchestrator import Orchestrator
from runtime_binding import install_from_environment, INSTALL_SENTINEL
config = tomllib.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
RLConfig.model_validate(config)
result = install_from_environment()
if result.get("run_id") != "staged-family-balanced-20261010":
    raise RuntimeError("staged hook returned the wrong run ID")
if not result.get("installed") and not result.get("already_installed"):
    raise RuntimeError("staged hook did not install")
marker = getattr(Orchestrator, INSTALL_SENTINEL, None)
if not marker:
    raise RuntimeError("staged hook did not patch pinned Prime Orchestrator.setup")
print(json.dumps({
    "status": "import_and_hook_passed",
    "rl_config_validation": "passed",
    "prime_module": str(Orchestrator.__module__),
    "hook_marker": INSTALL_SENTINEL,
    "already_installed": bool(result.get("already_installed")),
}, sort_keys=True))
"""


def _runtime_environment(*, prepare_receipt: dict[str, Any], source_root: Path, node_root: Path) -> dict[str, str]:
    """Build the non-secret environment that the launcher must forward."""

    policy = prepare_receipt.get("policy_initializer")
    frozen = prepare_receipt.get("frozen_sft")
    if not isinstance(policy, dict) or not isinstance(frozen, dict):
        raise ValueError("preparation receipt lacks both adapter identities")
    if policy.get("weights_sha256") != CONFIG.POLICY["sha256"] or frozen.get("weights_sha256") != CONFIG.FROZEN_SFT["sha256"]:
        raise ValueError("preparation receipt adapter hashes are not pinned")
    policy_path = PREP._directory(Path(str(policy.get("path", ""))), "policy initializer")
    frozen_path = PREP._directory(Path(str(frozen.get("path", ""))), "frozen SFT adapter")
    policy_weights = PREP._hf_snapshot_file(policy_path / "adapter_model.safetensors", "policy initializer weights")
    PREP._hf_snapshot_file(policy_path / "adapter_config.json", "policy initializer config")
    frozen_weights = PREP._hf_snapshot_file(frozen_path / "adapter_model.safetensors", "frozen SFT adapter weights")
    PREP._hf_snapshot_file(frozen_path / "adapter_config.json", "frozen SFT adapter config")
    if PREP.sha256_file(policy_weights) != CONFIG.POLICY["sha256"]:
        raise ValueError("policy initializer bytes changed after preparation")
    if PREP.sha256_file(frozen_weights) != CONFIG.FROZEN_SFT["sha256"]:
        raise ValueError("frozen SFT adapter bytes changed after preparation")
    return {
        "WATERCOLOUR_INIT_ADAPTER": str(policy_path),
        "PAINTER_FROZEN_SFT_ADAPTER": str(frozen_path),
        "PAINTER_FROZEN_SFT_ALIAS": CONFIG.FROZEN_SFT["alias"],
        "PAINTER_STAGED_CURRICULUM_ENABLE": "1",
        "PAINTER_STAGED_CURRICULUM_RUN_ID": RUN_ID,
        "PAINTER_STAGED_CURRICULUM_PROTOCOL": PROTOCOL,
        "PAINTER_STAGED_CURRICULUM_ADMISSION": "1",
        "PAINTER_STAGED_CURRICULUM_ROOT": str(source_root.resolve()),
        "PAINTER_STAGED_CURRICULUM_STATE": str((node_root / "runtime-state.json").resolve()),
    }


def _prime_console_scripts(*, prime_root: Path, prime_python: Path) -> tuple[Path, dict[str, Path]]:
    """Require the three Prime entry points in the pinned venv bin directory."""

    prime_bin = prime_root / ".venv/bin"
    expected_python = prime_bin / "python"
    if prime_python.absolute() != expected_python.absolute():
        raise ValueError("pinned Prime Python is outside the exact .venv/bin path")
    prime_bin = PREP._directory(prime_bin, "pinned Prime venv bin")
    scripts: dict[str, Path] = {}
    for name in ("inference", "trainer", "orchestrator"):
        path = prime_bin / name
        if path.is_symlink() or not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"pinned Prime console script is missing or unsafe: {path}")
        scripts[name] = path
    return prime_bin, scripts


def run_probe(*, prime_python: Path, prime_root: Path, config_path: Path, source_root: Path, repo_root: Path, state_path: Path, runtime_env: dict[str, str]) -> dict[str, Any]:
    prime_python = PREP._runtime_executable(prime_python, "pinned Prime Python", prime_python.parent.parent)
    prime_bin, scripts = _prime_console_scripts(prime_root=prime_root, prime_python=prime_python)
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"HF_TOKEN", "OPENAI_API_KEY", "WANDB_API_KEY"}
    }
    env.update({
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "void",
        "PAINTER_PRIME_ROLE": "orchestrator",
        "PAINTER_STAGED_CURRICULUM_ENABLE": "1",
        "PAINTER_STAGED_CURRICULUM_RUN_ID": RUN_ID,
        "PAINTER_STAGED_CURRICULUM_PROTOCOL": PROTOCOL,
        "PAINTER_STAGED_CURRICULUM_ADMISSION": "1",
        "PAINTER_STAGED_CURRICULUM_ROOT": str(source_root.resolve()),
        "PAINTER_STAGED_CURRICULUM_STATE": str(state_path.resolve()),
        "PYTHONPATH": os.pathsep.join(
            [
                str(source_root),
                str(repo_root),
                str(repo_root / "painter"),
                str(source_root.parent.parent.parent),
                *[item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item],
            ]
        ),
    })
    env.update(runtime_env)
    # Keep the probe's command lookup independent of ambient or receipt-derived
    # PATH values.  The exact pinned venv bin is always first and the fallback
    # directories are deliberately finite.
    env["PATH"] = os.pathsep.join((str(prime_bin), "/usr/local/bin", "/usr/bin", "/bin"))
    for name, path in scripts.items():
        if shutil.which(name, path=env["PATH"]) != str(path):
            raise RuntimeError(f"pinned Prime PATH does not resolve {name} to {path}")
    try:
        completed = subprocess.run(
            [str(prime_python), "-c", _probe_code(), str(config_path)],
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"pinned Prime import probe could not run: {type(exc).__name__}") from exc
    if completed.returncode != 0:
        # Do not persist subprocess stderr: it may contain provider paths or
        # environment details.  The return code remains auditable in receipt.
        raise RuntimeError(f"pinned Prime/runtime hook probe failed with exit code {completed.returncode}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("pinned Prime/runtime hook probe produced no receipt")
    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError("pinned Prime/runtime hook probe did not emit JSON") from exc
    if not isinstance(result, dict) or result.get("status") != "import_and_hook_passed" or result.get("rl_config_validation") != "passed":
        raise RuntimeError("pinned Prime/runtime hook probe returned an unverified receipt")
    result["prime_bin"] = str(prime_bin)
    result["prime_console_scripts"] = {name: str(path) for name, path in scripts.items()}
    result["path_prefix"] = str(prime_bin)
    return result


def canary(*, source_root: Path, node_root: Path, materialized_root: Path, renderer_root: Path, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    PREP.require_linux()
    source_root = PREP._directory(source_root, "staged source root")
    node_root = PREP._directory(node_root, "node root")
    renderer_root = PREP._directory(renderer_root, "renderer root")
    repo_root = PREP._directory(repo_root, "repository root")
    materialized = PREP._validate_materialized(materialized_root)
    prepare_receipt_path = PREP._file(node_root / "prepare-receipt.json", "node preparation receipt")
    prepare_receipt = json.loads(prepare_receipt_path.read_text(encoding="utf-8"))
    if prepare_receipt.get("status") != "prepared_no_process_started" or prepare_receipt.get("run_id") != RUN_ID:
        raise ValueError("node preparation receipt is not the staged finite preparation")
    runtime_env = _runtime_environment(
        prepare_receipt=prepare_receipt,
        source_root=source_root,
        node_root=node_root,
    )
    config_path = PREP._file(node_root / "native.toml", "prepared Prime config")
    config_receipt_path = PREP._file(node_root / "config-receipt.json", "Prime config receipt")
    config = _read_config(config_path)
    CONFIG.validate_config(config, node_root=node_root, materialized_root=Path(materialized["root"]))
    if prepare_receipt.get("config", {}).get("sha256") != sha256_file(config_path):
        raise ValueError("prepared config changed after node preparation")
    prime_root = PREP._directory(Path(str(prepare_receipt.get("prime_root", ""))), "prepared Prime root")
    prime_python = PREP._runtime_executable(prime_root / ".venv/bin/python", "prepared Prime Python", prime_root / ".venv")
    revision = PREP._git_revision(prime_root)
    state_path = node_root / "runtime-state.json"
    probe = run_probe(
        prime_python=prime_python,
        prime_root=prime_root,
        config_path=config_path,
        source_root=source_root,
        repo_root=repo_root,
        state_path=state_path,
        runtime_env=runtime_env,
    )
    prepared_renderer = prepare_receipt.get("renderer")
    if not isinstance(prepared_renderer, dict):
        raise ValueError("node preparation receipt lacks the verified renderer identity")
    if prepared_renderer.get("root") != str(renderer_root):
        raise ValueError("node preparation renderer root does not match the canary root")
    renderer = PREP._file(Path(str(prepared_renderer.get("path", ""))), "prepared renderer")
    renderer_python = PREP._runtime_executable(
        Path(str(prepared_renderer.get("python", ""))),
        "prepared renderer Python",
        Path(str(prepared_renderer.get("python", ""))).parent.parent,
    )
    if prepared_renderer.get("sha256") != RENDERER_SHA256:
        raise ValueError("node preparation receipt renderer hash is not pinned")
    renderer_probe = _renderer_probe(
        renderer=renderer,
        renderer_python=renderer_python,
        node_root=node_root,
        browser_root=renderer_root / "browsers",
    )
    receipt = {
        "schema": "staged-family-balanced-prime-canary-v1",
        "status": "passed_import_hook_and_renderer",
        "training_started": False,
        "processes_started": False,
        "orchestrator_setup_called": False,
        "run_id": RUN_ID,
        "prime_revision": revision,
        "prime_python": str(prime_python),
        "source_root": str(source_root),
        "node_root": str(node_root),
        "renderer_root": str(renderer_root),
        "materialized": materialized,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path), "receipt_sha256": sha256_file(config_receipt_path)},
        "probe": probe,
        "prime_console_scripts": probe.get("prime_console_scripts"),
        "prime_config_validation": {
            "status": probe.get("rl_config_validation"),
            "model": "prime_rl.configs.rl.RLConfig",
        },
        "renderer_probe": renderer_probe,
        "cuda_visibility": "CUDA_VISIBLE_DEVICES empty; no CUDA/runtime allocation attempted",
        "runtime_environment": {
            key: value for key, value in runtime_env.items()
            if key in {"WATERCOLOUR_INIT_ADAPTER", "PAINTER_FROZEN_SFT_ADAPTER", "PAINTER_FROZEN_SFT_ALIAS"}
        },
        "scope": [
            "pinned Prime Python imported prime_rl.orchestrator.orchestrator",
            "sitecustomize/runtime_binding installed the staged patch target on Orchestrator.setup",
            "verified renderer interpreter produced a deterministic PNG through the bounded CLI probe",
            "Orchestrator.setup was not called; no Orchestrator instance, env server, trainer, inference server, judge, or provider was started (only the deterministic renderer probe ran)",
        ],
        "unknown_runtime_assumptions": [
            "Prime's installed rl entrypoint must retain the reviewed `@ config` CLI shape.",
            "Prime's Orchestrator.setup, TrainSource, Dispatcher, TrainSink, and WeightWatcher shapes still require a canary that reaches native setup before finite training.",
        ],
    }
    _atomic_json(node_root / "canary-receipt.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--node-root", type=Path, required=True)
    parser.add_argument("--renderer-root", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    receipt = canary(**vars(args))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
