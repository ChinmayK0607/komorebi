"""Run pinned Prime RL while initializing the trainer from a portable SFT adapter.

The wrapper only replaces Prime's trainer child command.  Prime's coordinator,
orchestrator, inference server, and optimizer remain the pinned native paths.
"""

from __future__ import annotations

import os
from pathlib import Path


def trainer_command(command: list[str], wrapper: Path) -> list[str]:
    result = list(command)
    target = ["-m", "prime_rl.trainer.rl.train"]
    matches = [i for i in range(len(result) - 1) if result[i : i + 2] == target]
    if not matches:
        return result
    if len(matches) != 1 or Path(result[0]).name != "torchrun":
        raise ValueError("unexpected Prime trainer invocation")
    index = matches[0]
    return result[:index] + [str(wrapper)] + result[index + 2 :]


def main() -> None:
    import prime_rl.entrypoints.rl as entrypoint

    adapter = os.environ.get("WATERCOLOUR_INIT_ADAPTER")
    if not adapter or not (Path(adapter) / "adapter_model.safetensors").is_file():
        raise ValueError("an explicit portable SFT adapter is required")
    wrapper = Path(__file__).with_name("train_from_sft.py").resolve(strict=True)
    original = entrypoint.Popen
    trainer_started = False

    def launch(command, *args, **kwargs):
        nonlocal trainer_started
        modified = trainer_command(command, wrapper)
        if modified != list(command):
            if trainer_started:
                raise RuntimeError("unexpected duplicate trainer start")
            trainer_started = True
            environment = dict(kwargs.get("env", os.environ))
            environment["WATERCOLOUR_INIT_ADAPTER"] = str(Path(adapter).resolve())
            kwargs["env"] = environment
        return original(modified, *args, **kwargs)

    entrypoint.Popen = launch
    entrypoint.main()


if __name__ == "__main__":
    main()
