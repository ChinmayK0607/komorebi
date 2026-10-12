#!/usr/bin/env python3
"""Prime trainer wrapper with distributed staged-adapter initialization."""

from __future__ import annotations

import os
import sys
import json
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


RUN = Path(__file__).resolve().parent


def _rank_guard_paths() -> tuple[Path, Path] | None:
    rank_dir = os.environ.get("PAINTER_RANK_GUARD_DIR")
    failure_file = os.environ.get("PAINTER_DISTRIBUTED_FAILURE_FILE")
    if not rank_dir or not failure_file:
        return None
    return Path(rank_dir).expanduser().resolve(), Path(failure_file).expanduser().resolve()


def _rank_number() -> str:
    return os.environ.get("RANK", os.environ.get("LOCAL_RANK", "unknown"))


class _RankHeartbeat:
    def __init__(self, rank_dir: Path) -> None:
        self.rank_dir = rank_dir
        self.rank = _rank_number()
        self.path = rank_dir / f"rank-{self.rank}.heartbeat"
        self.started = rank_dir / f"rank-{self.rank}.started.json"
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.rank_dir.mkdir(parents=True, exist_ok=True)
        self.started.write_text(json.dumps({
            "schema": "staged-family-balanced-rank-start-v1",
            "rank": self.rank,
            "pid": os.getpid(),
            "world_size": os.environ.get("WORLD_SIZE"),
            "recorded_at_unix": time.time(),
        }, sort_keys=True) + "\n", encoding="utf-8")
        self.path.touch()
        self.thread = threading.Thread(target=self._run, name="painter-rank-heartbeat", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop.wait(5.0):
            try:
                self.path.touch()
            except OSError:
                return

    def close(self) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)


def _record_rank_failure(failure_file: Path, exc: BaseException) -> None:
    failure_file.parent.mkdir(parents=True, exist_ok=True)
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    status = "rank_oom" if "out of memory" in lowered or "cuda oom" in lowered else "rank_failed"
    payload = {
        "schema": "staged-family-balanced-distributed-failure-v1",
        "status": status,
        "rank": _rank_number(),
        "pid": os.getpid(),
        "world_size": os.environ.get("WORLD_SIZE"),
        "reason": text,
        "traceback": traceback.format_exc(limit=12),
        "recorded_at_unix": time.time(),
    }
    temporary = failure_file.with_name(f".{failure_file.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(failure_file)
    except OSError:
        # Preserve the original trainer exception even if the diagnostic path
        # is unavailable; the foreground supervisor still sees the rank exit.
        temporary.unlink(missing_ok=True)


def _require_resource_preflight() -> None:
    """Refuse a real trainer until both memory-saving hooks are proven."""

    path = os.environ.get("PAINTER_RESOURCE_PREFLIGHT")
    if not path:
        raise ValueError("resource preflight receipt is required before the staged trainer launch")
    try:
        receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("resource preflight receipt is unreadable") from exc
    report = receipt.get("resource_preflight") if isinstance(receipt, dict) else None
    if not isinstance(report, dict) or report.get("schema") != "staged-family-balanced-resource-preflight-v1":
        raise ValueError("resource preflight report is missing or unsupported")
    if report.get("launch_permitted") is not True:
        token = report.get("token_length_balance", {})
        vision = report.get("text_only_vision_memory", {})
        raise ValueError(
            "staged trainer is fail-closed until token-length balancing and text-only vision memory hooks are supported: "
            f"token={token.get('reason')}; vision={vision.get('reason')}"
        )


def _configure_real_backward_canary(config: object) -> float | None:
    if os.environ.get("PAINTER_CANARY_MODE") != "real_backward":
        return None
    # Only the bounded canary may override the prepared 20-update target.
    # The trainer entrypoint receives TrainerConfig directly: max_steps and
    # ckpt.interval are top-level fields. Fail closed if that API shape changes
    # rather than silently running a full job.
    if not hasattr(config, "max_steps") or not hasattr(config, "ckpt") or config.ckpt is None:
        raise ValueError("real-backward canary cannot prove mutable max_steps/checkpoint config")
    config.max_steps = 1
    if not hasattr(config.ckpt, "interval"):
        raise ValueError("real-backward canary cannot prove checkpoint interval support")
    config.ckpt.interval = 1
    return time.monotonic()


def _write_real_backward_receipt(config: object, started: float) -> None:
    if os.environ.get("PAINTER_CANARY_MODE") != "real_backward" or _rank_number() not in {"0", "unknown"}:
        return
    output_dir = Path(str(getattr(config, "output_dir", ""))).expanduser().resolve()
    checkpoint = output_dir / "checkpoints" / "step_1" / "trainer"
    checkpoint_metadata = checkpoint / ".metadata"
    checkpoint_files = (
        sorted(path for path in checkpoint.rglob("*") if path.is_file() and path != checkpoint_metadata)
        if checkpoint.is_dir() else []
    )
    checkpoint_produced = checkpoint.is_dir() and checkpoint_metadata.is_file() and bool(checkpoint_files)
    update_receipt_path = Path(os.environ.get("PAINTER_STAGED_CURRICULUM_STATE", ""))
    update_receipt: dict[str, Any] | None = None
    try:
        state = json.loads(update_receipt_path.read_text(encoding="utf-8"))
        confirmations = state.get("update_confirmations", [])
        exact = [
            value for value in confirmations
            if isinstance(value, dict) and value.get("controller_recorded") is True and value.get("optimizer_update") == 1
        ]
        controller = state.get("controller", {})
        if len(exact) == 1 and controller.get("optimizer_updates_observed") == 1:
            update_receipt = exact[0]
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
        update_receipt = None
    hourly_raw = os.environ.get("PAINTER_CANARY_NODE_HOURLY_USD")
    try:
        hourly = float(hourly_raw) if hourly_raw is not None else None
    except ValueError:
        hourly = None
    cost_source = os.environ.get("PAINTER_CANARY_COST_SOURCE")
    try:
        time_bound = float(os.environ["PAINTER_CANARY_MAX_SECONDS"])
        cost_bound = float(os.environ["PAINTER_CANARY_COST_BOUND_USD"])
    except (KeyError, ValueError):
        time_bound = -1.0
        cost_bound = -1.0
    elapsed = time.monotonic() - started
    cost = hourly * elapsed / 3600.0 if hourly is not None and hourly > 0 else None
    receipt_path = Path(os.environ.get("PAINTER_CANARY_RECEIPT", str(output_dir / "real-backward-canary-receipt.json")))
    receipt = {
        "schema": "staged-family-balanced-real-backward-canary-v1",
        "status": "passed" if update_receipt is not None and checkpoint_produced and cost is not None and cost_source == "node_hourly_price_input" and 0 <= elapsed <= time_bound <= 2400 and 0 <= cost <= cost_bound <= 9 else "failed",
        "optimizer_updates": 1 if update_receipt is not None else 0,
        "update_receipt": update_receipt,
        "update_receipt_path": str(update_receipt_path) if update_receipt is not None else None,
        "checkpoint_produced": checkpoint_produced,
        "checkpoint_path": str(checkpoint) if checkpoint_produced else None,
        "checkpoint_metadata": str(checkpoint_metadata) if checkpoint_produced else None,
        "checkpoint_file_count": len(checkpoint_files),
        "world_size": int(os.environ.get("WORLD_SIZE", "0")),
        "elapsed_seconds": round(elapsed, 3),
        "cost_usd": cost,
        "node_hourly_price_usd": hourly,
        "cost_source": cost_source if cost is not None else None,
        "cost_bound_usd": cost_bound,
        "time_bound_seconds": time_bound,
        "processes_left_running": False,
        "training_started": True,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if receipt["status"] != "passed":
        raise ValueError("real-backward canary lacks exact update, checkpoint-layout, or observed-cost evidence")


def _trainer_dependency_paths() -> tuple[Path, Path, Path]:
    root = RUN
    brush_root = Path(os.environ.get("BRUSH_ROOT", root.parents[2] / "rl_helpers")).resolve(strict=True)
    helpers = brush_root / "integrations" / "prime" / "watercolour_sft"
    override = root / "prime_runtime" / "integrations" / "prime" / "watercolour_sft"
    if not (override / "init_adapter.py").is_file():
        raise FileNotFoundError(f"staged adapter override is missing under {override}")
    for name in ("init_adapter.py", "export_adapter.py", "model_profiles.py"):
        if not (helpers / name).is_file():
            raise FileNotFoundError(f"reviewed Prime helper is missing under {helpers}")
    return root, brush_root, helpers


def _run() -> None:
    from prime_rl.configs.trainer import TrainerConfig
    from prime_rl.utils.config import cli
    from prime_rl.utils.process import set_proc_title

    config = cli(TrainerConfig)
    _require_resource_preflight()
    canary_started = _configure_real_backward_canary(config)
    root, brush_root, helpers = _trainer_dependency_paths()
    override = root / "prime_runtime" / "integrations" / "prime" / "watercolour_sft"
    mixed = brush_root / "integrations" / "speedpainting" / "mixed_sft"
    sys.path[:0] = [str(override), str(helpers), str(mixed), str(root)]

    from model_profiles import profile, verify_snapshot

    verify_snapshot(Path(config.model.name))
    if profile()["name"] != "qwen38_27b":
        raise ValueError("Unexpected model profile")
    if config.resume is not None or config.model.lora is None:
        raise ValueError("RL must use a fresh optimizer with the explicit SFT LoRA initializer")
    if config.ckpt is not None and getattr(config.ckpt, "resume_step", None) is not None:
        raise ValueError("RL launch does not resume native optimizer/checkpoint state")
    from init_adapter import validate_trainer_world_size, install_init_hook

    expected_world = int(os.environ.get("PAINTER_NUM_TRAIN_GPUS", "4"))
    validate_trainer_world_size(int(os.environ.get("WORLD_SIZE", "1")), expected_world)
    if expected_world != 4:
        raise ValueError(f"staged-family-balanced RL requires exactly four trainer ranks; got {expected_world}")
    if config.model.optim_cpu_offload or config.model.full_offload:
        raise ValueError("staged 4-rank FSDP keeps optimizer/model offload disabled")

    adapter = os.environ.get("WATERCOLOUR_INIT_ADAPTER")
    if not adapter:
        raise ValueError("WATERCOLOUR_INIT_ADAPTER is required")
    adapter_path = Path(adapter).resolve(strict=True)
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise ValueError("WATERCOLOUR_INIT_ADAPTER lacks adapter_model.safetensors")

    proxy = SimpleNamespace(model=config.model, resume=config.resume, run_dir=config.output_dir, max_steps=config.max_steps)
    install_init_hook(proxy, str(adapter_path))
    from export_adapter import install_export_hook

    install_export_hook(proxy)
    import prime_rl.trainer.rl.train as rl_train

    set_proc_title("PainterRLTrainer")
    rl_train.train(config)
    if canary_started is not None:
        _write_real_backward_receipt(config, canary_started)


def main() -> None:
    guard = _rank_guard_paths()
    heartbeat: _RankHeartbeat | None = None
    if guard is not None:
        heartbeat = _RankHeartbeat(guard[0])
        heartbeat.start()
    try:
        _run()
    except BaseException as exc:
        if guard is not None:
            _record_rank_failure(guard[1], exc)
        raise
    finally:
        if heartbeat is not None:
            heartbeat.close()


if __name__ == "__main__":
    main()
