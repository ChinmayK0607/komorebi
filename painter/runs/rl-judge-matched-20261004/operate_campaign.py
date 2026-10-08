#!/usr/bin/env python3
"""Finite detached controller for the matched three-arm campaign.

The reviewed single-run controller owns bounded SSH/SFTP observation, durable
exit receipts, diagnostics scrubbing, and exact-pod release.  This adapter
changes only the remote command, receipt paths, campaign budget, and the
campaign-specific readiness check; it does not duplicate transport logic.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import shlex
import sys
from typing import Any, Callable, Sequence


RUN = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "matched_base_operation", RUN.parent / "throughput-recovery-20261003" / "operate_single.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("reviewed single-run controller is unavailable")
base = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)

RUN_NAME = "rl-judge-matched-20261004"
EXPERIMENT_NAME = "rl-judge-matched-20261004"
REMOTE_ROOT = Path("/root/painter")
REMOTE_REPO = REMOTE_ROOT / "repo"
REMOTE_RUN = REMOTE_REPO / "painter/runs/rl-judge-matched-20261004"
REMOTE_TOKEN = REMOTE_ROOT / "private/hf-token"
REMOTE_GATEWAY = REMOTE_ROOT / "private/gateway.env"
REMOTE_BUNDLE = REMOTE_ROOT / "source.bundle"
REMOTE_JOB_DIR = REMOTE_ROOT / "private"
REMOTE_JOB_START = REMOTE_JOB_DIR / "rl-judge-matched-remote-job-start.json"
REMOTE_JOB_PID = REMOTE_JOB_DIR / "rl-judge-matched-remote-job.pid"
REMOTE_JOB_EXIT = REMOTE_JOB_DIR / "rl-judge-matched-remote-job-exit.json"
REMOTE_JOB_STDOUT = REMOTE_JOB_DIR / "rl-judge-matched-remote-job.stdout.log"
REMOTE_JOB_STDERR = REMOTE_JOB_DIR / "rl-judge-matched-remote-job.stderr.log"
GPU_COUNT = 2
MIN_GPU_MEMORY_MIB = 80_000
_SUCCESS_STEP = re.compile(r"SUCCESS[^\n]*?Step\s+(\d+)\s+\|")
_TRAINER_METRIC_KEYS = frozenset({"optim/grad_norm", "time/step", "optimizer/step"})


def _completed_optimizer_steps_from_metrics(raw: bytes | None, *, max_step: int = 12) -> set[int]:
    """Extract only trainer-produced completion rows from a bounded metrics tail."""
    completed: set[int] = set()
    if raw is None:
        return completed
    for line in raw.decode("utf-8", "replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        step = row.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or not 1 <= step <= max_step:
            continue
        producer = row.get("producer")
        trainer_metric = bool(_TRAINER_METRIC_KEYS.intersection(row))
        if (producer == "trainer" and trainer_metric) or (
            producer is None and "optim/grad_norm" in row
        ):
            completed.add(step)
    return completed


def _completed_optimizer_steps_from_trainer_log(raw: bytes | None, *, max_step: int = 12) -> set[int]:
    """Extract only explicit trainer SUCCESS rows from a bounded log tail."""
    if raw is None:
        return set()
    return {
        int(value)
        for value in _SUCCESS_STEP.findall(raw.decode("utf-8", "replace"))
        if 1 <= int(value) <= max_step
    }


def _state_step(state: dict[str, Any]) -> int | None:
    """Use a state counter when present, preserving an explicit zero."""
    for key in ("completed_optimizer_steps", "last_optimizer_step"):
        value = state.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _state_admitted_batch(state: dict[str, Any]) -> int | None:
    for key in ("last_admitted_batch", "admitted_batch_counter"):
        value = state.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _campaign_job_command(*, source_commit: str, rental_start_epoch: float,
                          price_usd_per_hour: float, cap_usd: float) -> str:
    env = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in {
            "PAINTER_SOURCE_COMMIT": source_commit,
            "PAINTER_NODE_ROOT": str(REMOTE_ROOT),
            "HF_TOKEN_PATH": str(REMOTE_TOKEN),
            "PAINTER_GATEWAY_ENV_FILE": str(REMOTE_GATEWAY),
            "PAINTER_GPUS_PER_NODE": "2",
            "PAINTER_NUM_TRAIN_GPUS": "1",
            "PAINTER_NUM_INFER_GPUS": "1",
        }.items()
    )
    execute = shlex.quote(str(REMOTE_RUN / "campaign.sh"))
    args = " ".join(shlex.quote(str(value)) for value in (rental_start_epoch, price_usd_per_hour, cap_usd))
    exit_tmp = shlex.quote(str(REMOTE_JOB_EXIT) + ".tmp")
    inner = (
        f"set +e; env {env} bash {execute} {args} >{shlex.quote(str(REMOTE_JOB_STDOUT))} "
        f"2>{shlex.quote(str(REMOTE_JOB_STDERR))}; rc=$?; "
        f"printf '{{\"schema\":\"matched-judge-remote-job-v1\",\"exit_code\":%s}}\\n' \"$rc\" >{exit_tmp}; "
        f"mv -f {exit_tmp} {shlex.quote(str(REMOTE_JOB_EXIT))}; exit \"$rc\""
    )
    return (
        "set -Eeuo pipefail; "
        f"rm -f {shlex.quote(str(REMOTE_JOB_START))} {shlex.quote(str(REMOTE_JOB_PID))} "
        f"{shlex.quote(str(REMOTE_JOB_EXIT))} {exit_tmp}; "
        f"nohup setsid bash -c {shlex.quote(inner)} </dev/null >/dev/null 2>&1 & pid=$!; "
        f"printf '%s\\n' \"$pid\" >{shlex.quote(str(REMOTE_JOB_PID))}; "
        f"printf '%s\\n' '{{\"schema\":\"matched-judge-remote-job-v1\",\"status\":\"started\"}}' >{shlex.quote(str(REMOTE_JOB_START))}"
    )


def _campaign_summary(sftp: Any, *, timeout: float) -> dict[str, Any]:
    deadline = base.time.monotonic() + max(0.0, float(timeout))

    def read(path: Path) -> bytes | None:
        remaining = deadline - base.time.monotonic()
        if remaining <= 0:
            raise base.OperationError("campaign status observation budget exhausted")
        return base._sftp_read(sftp, str(path), timeout=remaining)

    def obj(path: Path) -> dict[str, Any]:
        return base._json_bytes(read(path)) or {}

    status = obj(REMOTE_RUN / "campaign/status.json")
    materialization = obj(REMOTE_RUN / "materialization-receipt.json")
    terminal = obj(REMOTE_RUN / "campaign/terminal-receipt.json")
    arms_raw = read(REMOTE_RUN / "campaign/arms.jsonl") or b""
    arms = []
    for line in arms_raw.decode("utf-8", "replace").splitlines()[-3:]:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            arms.append(item)
    job = obj(REMOTE_JOB_START)
    job.update(obj(REMOTE_JOB_EXIT))
    reason = status.get("reason") if isinstance(status.get("reason"), str) else ""
    active_arm = reason.removeprefix("arm_").removesuffix("_running")
    metrics = None
    config_receipt = {}
    active_state = {}
    receipt_arm = active_arm if active_arm in {"flash", "pro", "luna"} else "flash"
    if materialization.get("schema") == "matched-three-judge-materialization-v1":
        arm_root = REMOTE_RUN.parent / f"{RUN_NAME}-{receipt_arm}"
        config_receipt = obj(arm_root / "rl-config-receipt.json")
    if active_arm in {"flash", "pro", "luna"}:
        arm_root = REMOTE_RUN.parent / f"{RUN_NAME}-{active_arm}"
        active_state = obj(arm_root / "supervisor-state.json")
        metrics = base._sftp_read_metrics_tail(
            sftp,
            str(arm_root / f"async-output/{RUN_NAME}-{active_arm}/monitors/file/metrics.jsonl"),
            timeout=max(0.0, deadline - base.time.monotonic()),
        )
    ready = bool(materialization.get("schema") == "matched-three-judge-materialization-v1" and config_receipt)
    finished_steps = sum(
        int(row.get("completed_optimizer_steps", 0) or 0)
        for row in arms
        if row.get("arm") != active_arm
    )
    active_steps = _state_step(active_state)
    if active_steps is None and active_arm in {"flash", "pro", "luna"}:
        trainer_log = read(
            REMOTE_RUN.parent
            / f"{RUN_NAME}-{active_arm}/async-output/{RUN_NAME}-{active_arm}/logs/attempt_1/trainer.log"
        )
        completed = _completed_optimizer_steps_from_metrics(metrics)
        completed.update(_completed_optimizer_steps_from_trainer_log(trainer_log))
        active_steps = max(completed) if completed else 0
    if active_steps is None:
        active_steps = 0
    active_admitted_batch = _state_admitted_batch(active_state)
    return {
        "state": {"status": status.get("status"), "reason": status.get("reason"),
                  "completed_optimizer_steps": finished_steps + active_steps,
                  "active_arm": active_arm if active_arm in {"flash", "pro", "luna"} else None,
                  "active_completed_optimizer_steps": active_steps,
                  "last_admitted_batch": active_admitted_batch},
        "terminal": terminal,
        "job": job,
        "phase_ready": ready,
        "config": materialization.get("normalized_common_native_config", {}),
        "config_receipt": config_receipt,
        "materialization": materialization,
        "preparation": {"profile": "matched-three-judge-v1", "arms": len(materialization.get("arms", []))},
        "render": {
            "scope": "active arm recent bounded metrics tail and campaign receipt tail",
            "arm_count": len(arms), "active_arm": active_arm if active_arm in {"flash", "pro", "luna"} else None,
            "queue": base._queue_summary(metrics),
        },
    }


def _validate_campaign_summary(summary: dict[str, Any]) -> None:
    if summary.get("preparation", {}).get("profile") != "matched-three-judge-v1":
        raise base.OperationError("campaign materialization receipt is not ready")
    config = summary.get("config", {})
    if not isinstance(config, dict):
        raise base.OperationError("campaign common config receipt is malformed")
    orchestrator = config.get("orchestrator", {})
    concurrency = orchestrator.get("concurrency", {})
    trainer = config.get("trainer", {})
    offload = trainer.get("model", {}).get("ac_offloading")
    deployment = config.get("deployment")
    optim = trainer.get("optim", {})
    source = (orchestrator.get("train", {}).get("source") or [{}])[0]
    taskset = source.get("env", {}).get("taskset", {})
    expected_deployment = {"type": "single_node", "gpus_per_node": 2,
                           "num_train_gpus": 1, "num_infer_gpus": 1}
    if (
        config.get("max_steps") != 12 or config.get("ckpt", {}).get("interval") != 4 or
        orchestrator.get("max_steps") != 12 or orchestrator.get("batch_size") != 16 or
        orchestrator.get("group_size") != 4 or orchestrator.get("max_off_policy_steps") != 1 or
        concurrency.get("initial_inflight") != 32 or concurrency.get("max_inflight") != 32 or
        trainer.get("max_steps") != 12 or float(optim.get("lr", -1)) != 1e-6 or
        offload != {"pin_memory": True, "max_inflight_activations": 1} or
        deployment != expected_deployment or source.get("name") != "text-paint-fixed-sft-pairwise" or
        taskset.get("expected_count") != 40 or taskset.get("max_turns") != 3 or
        taskset.get("opponent_checkpoint_sha256") != "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740"
    ):
        raise base.OperationError("campaign common config receipt does not match the fixed twelve-update recipe")
    materialization = summary.get("materialization", {})
    initializer = materialization.get("initializer", {})
    frozen = materialization.get("frozen_sft", {})
    if (
        materialization.get("prime_revision") != "26b3131d2716a4f8210b165df584b83b4bc54f61" or
        materialization.get("protocol_sha256") is None or
        initializer.get("sha256") != "e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c" or
        initializer.get("prefix") != "text-pairwise-scaled-20260929/step_16" or
        frozen.get("sha256") != "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740"
    ):
        raise base.OperationError("campaign materialization receipt is not bound to the registered initializer")
    receipt = summary.get("config_receipt", {})
    if (
        not isinstance(receipt, dict) or receipt.get("adapter_revision") != "ac4d3eda8e5925e2d23992cfdc1a63620ebde573" or
        receipt.get("adapter_prefix") != "text-pairwise-scaled-20260929/step_16" or
        receipt.get("adapter_sha256") != "e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c" or
        receipt.get("mode") != "recovered_policy" or receipt.get("optimizer_resume") is not False or
        receipt.get("checkpoint_interval") != 4 or receipt.get("batch_size") != 16 or
        receipt.get("group_size") != 4 or receipt.get("initial_inflight") != 32 or
        receipt.get("max_inflight") != 32 or receipt.get("max_off_policy_steps") != 1 or
        receipt.get("activation_cpu_offload") is not True or
        receipt.get("reward") != "text-paint-sft-pairwise-consistent-v1"
    ):
        raise base.OperationError("active arm config receipt is not bound to the matched campaign")


# Bind the reviewed transport/controller to this campaign without changing its
# historical single-arm defaults.
base.RUN_NAME = RUN_NAME
base.EXPERIMENT_NAME = EXPERIMENT_NAME
base.REMOTE_ROOT = REMOTE_ROOT
base.REMOTE_REPO = REMOTE_REPO
base.REMOTE_RUN = REMOTE_RUN
base.REMOTE_TOKEN = REMOTE_TOKEN
base.REMOTE_GATEWAY = REMOTE_GATEWAY
base.REMOTE_BUNDLE = REMOTE_BUNDLE
base.REMOTE_JOB_DIR = REMOTE_JOB_DIR
base.REMOTE_JOB_START = REMOTE_JOB_START
base.REMOTE_JOB_PID = REMOTE_JOB_PID
base.REMOTE_JOB_EXIT = REMOTE_JOB_EXIT
base.REMOTE_JOB_STDOUT = REMOTE_JOB_STDOUT
base.REMOTE_JOB_STDERR = REMOTE_JOB_STDERR
base.GPU_COUNT = GPU_COUNT
base.MIN_GPU_MEMORY_MIB = MIN_GPU_MEMORY_MIB
base._detached_job_command = _campaign_job_command
base._remote_summary = _campaign_summary
base._validate_config_summary = _validate_campaign_summary


def make_campaign_window(*, pod_id: str, rental_start_epoch: float,
                         price_usd_per_hour: float, cap_usd: float) -> Any:
    if not isinstance(pod_id, str) or not base.POD_ID_RE.fullmatch(pod_id) or pod_id.lower() in {"all", "*"}:
        raise ValueError("pod-id must be one exact provider ID")
    start = base._positive_finite("rental-start-epoch", rental_start_epoch)
    price = base._positive_finite("price-usd-hour", price_usd_per_hour)
    cap = base._positive_finite("cap-usd", cap_usd)
    if cap > 58.5:
        raise ValueError("campaign cap must be at most 58.50")
    window = base.RentalWindow(pod_id, start, price, cap)
    if window.release_deadline_epoch <= start:
        raise ValueError("release margin leaves no finite operating window")
    return window


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--pod-id", required=True)
    value.add_argument("--rental-start-epoch", type=float, required=True)
    value.add_argument("--price-usd-hour", type=float, required=True)
    value.add_argument("--cap-usd", type=float, required=True)
    value.add_argument("--source-bundle", type=Path, required=True)
    value.add_argument("--source-commit", required=True)
    value.add_argument("--hf-token-file", type=Path, default=base.default_hf_token_file())
    value.add_argument("--gateway-env-file", type=Path, default=base.default_gateway_env_file())
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--plan-only", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    window = make_campaign_window(pod_id=args.pod_id, rental_start_epoch=args.rental_start_epoch,
                                  price_usd_per_hour=args.price_usd_hour, cap_usd=args.cap_usd)
    base._regular_file(args.source_bundle, "source-bundle")
    base._regular_file(args.hf_token_file, "HF token file", owner_only=True)
    base._regular_file(args.gateway_env_file, "Gateway environment file", owner_only=True)
    if not base.COMMIT_RE.fullmatch(args.source_commit):
        raise SystemExit("source-commit must be a full immutable Git commit")
    if args.plan_only:
        print(json.dumps({"schema": "matched-three-judge-operation-plan-v1", "run_name": RUN_NAME,
                          "campaign_cap_usd": window.cap_usd,
                          "per_arm_cap_usd": window.cap_usd / 3,
                          "topology": {"gpus_per_node": 2, "num_train_gpus": 1, "num_infer_gpus": 1},
                          "max_updates": 12, "arms": ["flash", "pro", "luna"]}, sort_keys=True))
        return 0
    return base.operate(window=window, source_bundle=args.source_bundle,
                        source_commit=args.source_commit, hf_token_file=args.hf_token_file,
                        gateway_env_file=args.gateway_env_file, output_root=args.output_root)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (base.OperationError, OSError, ValueError) as exc:
        print(f"matched campaign operation: {exc}", file=sys.stderr)
        raise SystemExit(2)
