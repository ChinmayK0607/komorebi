"""Run Prime's native RL coordinator with the reviewed SFT adapter as policy v0.

Prime's fresh filesystem path starts inference from the base model.  Before the
orchestrator is allowed to start, this wrapper waits for the owned inference
server and loads the same adapter that the trainer initializer will install
under Prime's stable served base-model name.  Subsequent policy updates remain
Prime's native filesystem broadcasts, which reload that name in place; no
adapter reset or optimizer resume is introduced here.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable
import urllib.error
import urllib.request


DEFAULT_INFERENCE_URL = "http://127.0.0.1:8000"
READY_TIMEOUT_SECONDS = 600.0
MAX_READY_TIMEOUT_SECONDS = 1200.0
LOAD_TIMEOUT_SECONDS = 120.0


def _ready_timeout_seconds() -> float:
    """Return the bounded model-server startup allowance.

    The historical 10-minute default remains unchanged.  A fresh ablation
    node may opt into the ten-minute larger bound while loading a 27B model,
    but the environment cannot turn this readiness wait into an unbounded
    rental or silently exceed the reviewed 20-minute ceiling.
    """

    raw = os.environ.get("PAINTER_INFERENCE_READY_TIMEOUT_SECONDS")
    if raw is None:
        return READY_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("PAINTER_INFERENCE_READY_TIMEOUT_SECONDS must be numeric") from exc
    if not math.isfinite(value) or value <= 0 or value > MAX_READY_TIMEOUT_SECONDS:
        raise ValueError(
            "PAINTER_INFERENCE_READY_TIMEOUT_SECONDS must be finite, positive, and at most "
            f"{MAX_READY_TIMEOUT_SECONDS:g} seconds"
        )
    return value


def trainer_command(command: list[str], wrapper: Path) -> list[str]:
    result = list(command)
    target = ["-m", "prime_rl.trainer.rl.train"]
    matches = [i for i in range(len(result) - 1) if result[i : i + 2] == target]
    if not matches:
        return result
    if len(matches) != 1 or Path(result[0]).name != "torchrun":
        raise ValueError("Unexpected Prime trainer invocation")
    index = matches[0]
    return result[:index] + [str(wrapper)] + result[index + 2 :]


def _adapter_path() -> Path:
    value = os.environ.get("WATERCOLOUR_INIT_ADAPTER")
    if not value:
        raise ValueError("WATERCOLOUR_INIT_ADAPTER is required")
    path = Path(value).resolve(strict=True)
    if not (path / "adapter_model.safetensors").is_file():
        raise ValueError("WATERCOLOUR_INIT_ADAPTER lacks adapter_model.safetensors")
    return path


def _frozen_adapter_path(*, required: bool) -> Path | None:
    value = os.environ.get("PAINTER_FROZEN_SFT_ADAPTER")
    if not value:
        if required:
            raise ValueError("PAINTER_FROZEN_SFT_ADAPTER is required for a recovered-policy run")
        return None
    path = Path(value).resolve(strict=True)
    if not (path / "adapter_model.safetensors").is_file():
        raise ValueError("PAINTER_FROZEN_SFT_ADAPTER lacks adapter_model.safetensors")
    return path


def _request(url: str, *, payload: dict | None = None, timeout: float = 10.0) -> tuple[int, bytes]:
    data = None
    headers = {}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        # Consume the response without logging its body; server output can
        # contain paths or provider details that do not belong in the receipt.
        try:
            exc.read()
        except Exception:
            pass
        return int(exc.code), b""


def _initial_aliases(live_alias: str, frozen_alias: str) -> tuple[str, ...]:
    if not live_alias or not frozen_alias or live_alias == frozen_alias:
        raise ValueError("initial adapter aliases must be non-empty and distinct")
    return live_alias, frozen_alias


def _rl_config_path(argv: list[str]) -> Path:
    """Resolve the same config path passed to Prime's ``@`` CLI argument."""
    value = os.environ.get("PAINTER_RL_CONFIG")
    if value:
        return Path(value).expanduser().resolve(strict=True)
    for index, argument in enumerate(argv[:-1]):
        if argument == "@":
            return Path(argv[index + 1]).expanduser().resolve(strict=True)
    raise ValueError("RL config path is required via PAINTER_RL_CONFIG or Prime's @ argument")


def _load_first_rl_task(config_path: Path) -> Any:
    """Load the first real task through the installed native RL taskset."""
    import tomllib

    raw = config_path.read_bytes()
    if config_path.suffix.lower() == ".json" or raw.lstrip().startswith(b"{"):
        config = json.loads(raw.decode("utf-8"))
    else:
        config = tomllib.loads(raw.decode("utf-8"))
    taskset_values = config["orchestrator"]["train"]["source"][0]["env"]["taskset"]
    try:
        from native_painting_rl import NativePaintingRLConfig, NativePaintingRLTaskset
    except ImportError:
        from painter.native_painting_rl import NativePaintingRLConfig, NativePaintingRLTaskset
    taskset_config = NativePaintingRLConfig.model_validate(taskset_values)
    return next(iter(NativePaintingRLTaskset(taskset_config).load()))


def _prewarm_frozen_opponent(
    config_path: Path,
    *,
    task_loader: Callable[[], Any] | None = None,
    provider_factory: Callable[[Any], Any] | None = None,
    spec_type: type | None = None,
) -> Any:
    """Verify a fixed anchor or resolve one lazy opponent before rollouts."""
    task = task_loader() if task_loader is not None else _load_first_rl_task(config_path)
    data = task.data
    if data.opponent is not None:
        anchor = Path(data.opponent)
        expected = data.opponent_sha256
        if not anchor.is_file() or not expected:
            raise ValueError("fixed opponent is missing or lacks a SHA-256")
        actual = hashlib.sha256(anchor.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("fixed opponent SHA-256 changed after task loading")
        return {"status": "fixed_verified", "path": str(anchor), "sha256": actual}
    required = (
        data.opponent_model_alias,
        data.opponent_checkpoint_sha256,
        data.opponent_protocol_version,
        data.opponent_render_protocol,
        data.opponent_decoding_config,
        data.opponent_seed,
    )
    if any(value is None for value in required):
        raise ValueError("first RL task is missing lazy frozen-opponent identity fields")
    if provider_factory is None:
        try:
            from native_painting_rl import _frozen_provider
        except ImportError:
            from painter.native_painting_rl import _frozen_provider
        provider_factory = _frozen_provider
    if spec_type is None:
        try:
            from frozen_opponent import OpponentSpec
        except ImportError:
            from painter.frozen_opponent import OpponentSpec
        spec_type = OpponentSpec
    spec = spec_type(
        source_group=str(data.source_group),
        reference=Path(data.reference),
        reference_sha256=str(data.reference_sha256),
        model_alias=str(data.opponent_model_alias),
        checkpoint_sha256=str(data.opponent_checkpoint_sha256),
        protocol_version=str(data.opponent_protocol_version),
        render_protocol=str(data.opponent_render_protocol),
        decoding_config=str(data.opponent_decoding_config),
        seed=data.opponent_seed,
        max_turns=int(data.max_turns),
        max_completion_tokens=int(data.opponent_max_completion_tokens),
        context_length=int(data.opponent_context_length),
    )
    result = asyncio.run(provider_factory(data).resolve(spec))
    status = getattr(result, "status", None)
    if status == "unavailable":
        raise RuntimeError(
            "frozen-opponent prewarm unavailable: "
            + str(getattr(result, "reason", None) or "unknown service failure")
        )
    if status not in {"valid", "invalid_model_output"}:
        raise RuntimeError(f"frozen-opponent prewarm returned unexpected status: {status!r}")
    print(json.dumps({
        "status": "frozen_opponent_prewarm",
        "task_id": str(data.task_id),
        "source_group": str(data.source_group),
        "opponent_status": status,
        "cache_key": str(getattr(result, "cache_key", "")),
    }, sort_keys=True))
    return result


def _wait_and_load(
    process,
    adapter: Path,
    *,
    base_url: str,
    aliases: tuple[str, ...],
    frozen_adapter: Path | None = None,
) -> None:
    if not aliases or any(not alias for alias in aliases) or len(set(aliases)) != len(aliases):
        raise ValueError("initial adapter aliases must be non-empty and distinct")
    deadline = time.monotonic() + _ready_timeout_seconds()
    last_status = None
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"owned inference server exited before readiness (status {process.returncode})")
            try:
                status, _ = _request(base_url.rstrip("/") + "/liveness", timeout=5.0)
                last_status = status
                if status == 200:
                    break
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            time.sleep(2.0)
        else:
            raise TimeoutError(f"owned inference server did not become ready (last status {last_status!r})")

        for alias in aliases:
            alias_adapter = frozen_adapter if alias == aliases[1] and frozen_adapter is not None else adapter
            load_deadline = time.monotonic() + LOAD_TIMEOUT_SECONDS
            while time.monotonic() < load_deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"owned inference server exited while loading initial adapter (status {process.returncode})")
                try:
                    status, _ = _request(
                        base_url.rstrip("/") + "/load_lora_adapter",
                        payload={"lora_name": alias, "lora_path": alias_adapter.as_posix()},
                        timeout=30.0,
                    )
                    if status == 200:
                        break
                    # Prime's endpoint can transiently return 404/500 while
                    # vLLM finishes loading workers. Other statuses are
                    # configuration failures and fail fast without retrying.
                    if status not in {404, 500, 502, 503, 504}:
                        raise RuntimeError(f"initial adapter load rejected with HTTP {status} for {alias}")
                except (urllib.error.URLError, TimeoutError, OSError):
                    pass
                time.sleep(2.0)
            else:
                raise TimeoutError(f"initial adapter load did not complete within 120 seconds for {alias}")
    except Exception:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except Exception:
                process.kill()
        raise


def _verify_orchestrator_policy(config_path: Path, expected_model: str) -> None:
    """Check that the pinned Prime policy name remains the base model name.

    Prime 26b uses the served base model name as its stable LoRA alias and
    reloads that alias in place on every filesystem broadcast.  Prime may
    materialize the child orchestrator config as TOML or JSON; rewriting it
    to a separate alias would make startup policy checks and later broadcasts
    disagree.
    """

    import tomllib

    raw = config_path.read_bytes()
    if config_path.suffix.lower() == ".json" or raw.lstrip().startswith(b"{"):
        config = json.loads(raw.decode("utf-8"))
    else:
        config = tomllib.loads(raw.decode("utf-8"))
    actual = config.get("model", {}).get("name")
    if actual != expected_model:
        raise ValueError(f"orchestrator policy model changed from the pinned base name: {actual!r}")


def _cache_env(root: Path, role: str, environment: dict[str, str]) -> dict[str, str]:
    cache = root / "cache" / role
    values = dict(environment)
    values.update(
        VLLM_CACHE_ROOT=str(cache / "vllm"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "torchinductor"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        # vLLM's data-parallel workers inherit this marker and the run-local
        # sitecustomize hook gives each worker rank (or process fallback) its
        # own compiler cache. Compilation remains reusable within a process.
        PAINTER_PRIME_ROLE=role,
        PAINTER_COMPILE_CACHE_ROOT=str(cache),
        PAINTER_COMPILE_CACHE_PROCESS_ISOLATION="1" if role == "inference" else "0",
    )
    if role == "inference":
        # vLLM documents this environment variable for its worker context;
        # spawn makes every EngineCore worker execute sitecustomize before
        # importing Torch/Triton. The process-token fallback still protects
        # workers whose rank is assigned after Python startup.
        values["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    for path in (cache / "vllm", cache / "torchinductor", cache / "triton"):
        path.mkdir(parents=True, exist_ok=True)
    return values


def _orchestrator_env(environment: dict[str, str]) -> dict[str, str]:
    """Forward painter's optional group scorer through Prime's role env.

    Prime supplies a separate child environment to ``Popen``. Keep only the
    run's explicit role values rather than inheriting unrelated secrets. The
    gateway variable is a path to an owner-only credential file, not the
    credential itself; the group-admission values are non-secret protocol
    controls forwarded to the orchestrator child.
    """
    values = dict(environment)
    for name in (
        "PAINTER_GROUP_RANK_MODULE",
        "PAINTER_GATEWAY_ENV_FILE",
        "PAINTER_WHOLE_GROUP_ADMISSION",
        "PAINTER_WHOLE_GROUP_SIZE",
        "PAINTER_CONSISTENT_GUARD_RECEIPT",
        "PAINTER_CONSISTENT_GROUP_TELEMETRY",
        "PAINTER_REQUIRE_VALID_FINAL_ACTION",
        "PAINTER_JS_SYNTAX_PREFLIGHT",
        "PAINTER_P5_BRUSH_API_REFERENCE",
        "PAINTER_INSPECT_CHOICE_FEEDBACK",
        "PAINTER_RENDER_WORKERS",
        "PAINTER_RENDERER_PATH",
        "PAINTER_RENDERER_PYTHON_PATH",
        "PAINTER_JUDGE_PROTOCOL",
        "PAINTER_JUDGE_FORMAT",
        "PAINTER_JUDGE_MODEL",
        "PLAYWRIGHT_BROWSERS_PATH",
        "PYTHONPATH",
    ):
        value = os.environ.get(name)
        if value is not None:
            values[name] = value
    # This helper is used only for Prime's orchestrator child.  The
    # run-local sitecustomize hook uses the role marker to install the
    # whole-group guard after Prime imports its scorer classes.
    values["PAINTER_PRIME_ROLE"] = "orchestrator"
    return values


def main() -> None:
    import prime_rl.entrypoints.rl as entrypoint

    adapter = _adapter_path()
    mode = os.environ.get("PAINTER_RL_MODE", "fresh")
    frozen_adapter = _frozen_adapter_path(required=mode == "recovered_policy")
    root = Path(__file__).resolve().parent
    wrapper = Path(__file__).with_name("train_from_sft.py").resolve(strict=True)
    rl_config_path = _rl_config_path(sys.argv[1:])
    base_url = os.environ.get("PAINTER_RL_INFERENCE_URL", DEFAULT_INFERENCE_URL)
    live_alias = os.environ.get("MODEL_DIR")
    frozen_alias = os.environ.get("PAINTER_FROZEN_SFT_ALIAS", "sft-step512")
    if not live_alias:
        raise ValueError("MODEL_DIR is required so Prime's stable live LoRA alias is explicit")
    aliases = _initial_aliases(live_alias, frozen_alias)
    original = entrypoint.Popen
    inference_started = False
    orchestrator_started = False
    trainer_started = False

    def launch(command, *args, **kwargs):
        nonlocal inference_started, orchestrator_started, trainer_started
        result = list(command)
        basename = Path(str(result[0])).name if result else ""
        environment = dict(kwargs.get("env", os.environ))
        environment["WATERCOLOUR_INIT_ADAPTER"] = str(adapter)
        if basename == "inference":
            if inference_started:
                raise RuntimeError("Prime attempted to start duplicate inference server")
            # Validate the reviewed startup bound before creating the owned
            # server, so a malformed override cannot start a billable process
            # and only fail after the first readiness poll.
            _ready_timeout_seconds()
            inference_started = True
            kwargs["env"] = _cache_env(root, "inference", environment)
            process = original(command, *args, **kwargs)
            _wait_and_load(
                process,
                adapter,
                base_url=base_url,
                aliases=aliases,
                frozen_adapter=frozen_adapter,
            )
            try:
                _prewarm_frozen_opponent(rl_config_path)
            except Exception:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except Exception:
                        process.kill()
                raise
            return process
        if basename == "orchestrator":
            if orchestrator_started:
                raise RuntimeError("Prime attempted to start duplicate orchestrator")
            orchestrator_started = True
            if len(result) < 3 or result[1] != "@":
                raise ValueError("Unexpected Prime orchestrator invocation")
            _verify_orchestrator_policy(Path(result[2]).resolve(), live_alias)
            kwargs["env"] = _orchestrator_env(environment)
            return original(command, *args, **kwargs)
        modified = trainer_command(result, wrapper)
        if modified != result:
            if trainer_started:
                raise RuntimeError("Prime attempted to start duplicate trainer")
            trainer_started = True
            kwargs["env"] = _cache_env(root, "trainer", environment)
            return original(modified, *args, **kwargs)
        return original(command, *args, **kwargs)

    entrypoint.Popen = launch
    entrypoint.main()
    if not inference_started:
        raise RuntimeError("Prime entrypoint did not start inference")
    if not orchestrator_started or not trainer_started:
        raise RuntimeError("Prime entrypoint did not start both orchestrator and trainer")


if __name__ == "__main__":
    main()
