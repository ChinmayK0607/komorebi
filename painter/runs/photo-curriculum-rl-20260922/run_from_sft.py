"""Run Prime's native RL coordinator with the reviewed SFT adapter as policy v0.

Prime's fresh filesystem path starts inference from the base model.  Before the
orchestrator is allowed to start, this wrapper waits for the owned inference
server and loads the same adapter that the trainer initializer will install
under Prime's stable served base-model name.  Subsequent policy updates remain
Prime's native filesystem broadcasts, which reload that name in place; no
adapter reset or optimizer resume is introduced here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


DEFAULT_INFERENCE_URL = "http://127.0.0.1:8000"
READY_TIMEOUT_SECONDS = 600.0
LOAD_TIMEOUT_SECONDS = 120.0


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


def _wait_and_load(process, adapter: Path, *, base_url: str, aliases: tuple[str, ...]) -> None:
    if not aliases or any(not alias for alias in aliases) or len(set(aliases)) != len(aliases):
        raise ValueError("initial adapter aliases must be non-empty and distinct")
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
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
            load_deadline = time.monotonic() + LOAD_TIMEOUT_SECONDS
            while time.monotonic() < load_deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"owned inference server exited while loading initial adapter (status {process.returncode})")
                try:
                    status, _ = _request(
                        base_url.rstrip("/") + "/load_lora_adapter",
                        payload={"lora_name": alias, "lora_path": adapter.as_posix()},
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
    )
    for path in (cache / "vllm", cache / "torchinductor", cache / "triton"):
        path.mkdir(parents=True, exist_ok=True)
    return values


def main() -> None:
    import prime_rl.entrypoints.rl as entrypoint

    adapter = _adapter_path()
    root = Path(__file__).resolve().parent
    wrapper = Path(__file__).with_name("train_from_sft.py").resolve(strict=True)
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
            inference_started = True
            kwargs["env"] = _cache_env(root, "inference", environment)
            process = original(command, *args, **kwargs)
            _wait_and_load(process, adapter, base_url=base_url, aliases=aliases)
            return process
        if basename == "orchestrator":
            if orchestrator_started:
                raise RuntimeError("Prime attempted to start duplicate orchestrator")
            orchestrator_started = True
            if len(result) < 3 or result[1] != "@":
                raise ValueError("Unexpected Prime orchestrator invocation")
            _verify_orchestrator_policy(Path(result[2]).resolve(), live_alias)
            kwargs["env"] = environment
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
