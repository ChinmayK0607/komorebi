#!/usr/bin/env python3
"""Operate one bounded Diverse-flow run on an already rented pod.

Transport, private process ownership, bounded polling, and exact-pod release
come from the reviewed single-arm controller.  This adapter changes only the
run identity, remote command, and structured config checks.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import shlex
import shutil
import sys
import subprocess
import time
from typing import Any
import urllib.parse


RUN = Path(__file__).resolve().parent
if str(RUN) not in sys.path:
    sys.path.insert(0, str(RUN))
import hardware_profile
SOURCE = RUN.parent / "throughput-recovery-20261003" / "operate_single.py"
spec = importlib.util.spec_from_file_location("diverse_flow_operation_base", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"single operation controller unavailable: {SOURCE}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

base.RUN_NAME = "diverse-flow-rl-20261009"
base.EXPERIMENT_NAME = "diverse-flow-rl-20261009-flash"
base.REMOTE_RUN = base.REMOTE_REPO / "painter/runs/diverse-flow-rl-20261009"
base.REMOTE_BUNDLE = base.REMOTE_ROOT / "source.bundle"
base.REMOTE_JOB_START = base.REMOTE_JOB_DIR / "Diverse-flow-remote-job-start.json"
base.REMOTE_JOB_PID = base.REMOTE_JOB_DIR / "Diverse-flow-remote-job.pid"
base.REMOTE_JOB_EXIT = base.REMOTE_JOB_DIR / "Diverse-flow-remote-job-exit.json"
base.REMOTE_JOB_STDOUT = base.REMOTE_JOB_DIR / "Diverse-flow-remote-job.stdout.log"
base.REMOTE_JOB_STDERR = base.REMOTE_JOB_DIR / "Diverse-flow-remote-job.stderr.log"
EVALUATION_ONLY = os.environ.get("PAINTER_EVALUATION_ONLY", "0")
if EVALUATION_ONLY not in {"0", "1"}:
    raise ValueError("PAINTER_EVALUATION_ONLY must be 0 or 1")
EVALUATION_ONLY = EVALUATION_ONLY == "1"
HARDWARE_PROFILE = hardware_profile.selected_name()
PROFILE = hardware_profile.get(HARDWARE_PROFILE)
base.GPU_COUNT = 1 if EVALUATION_ONLY else PROFILE["gpu_count"]
base.TRAIN_GPU_COUNT = 0 if EVALUATION_ONLY else PROFILE["num_train_gpus"]
base.INFER_GPU_COUNT = 1 if EVALUATION_ONLY else PROFILE["num_infer_gpus"]
base.MIN_GPU_MEMORY_MIB = PROFILE["min_gpu_memory_mib"]

HF_REFRESH_INTERVAL_SECONDS = 1_800.0
HF_REFRESH_TIMEOUT_SECONDS = 20.0
HF_REMOTE_ROTATION_TMP = base.REMOTE_ROOT / "private/.hf-token.rotate"
_BASE_OBSERVE_SUMMARY = base._observe_summary
_BASE_OPERATE = base.operate
_HF_REFRESH_CONTEXT: dict[str, Any] | None = None


def _safe_token_bytes(path: Path) -> bytes:
    """Read an owner-only token without ever returning it to a log path."""
    candidate = path.expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("HF token path is not a regular file")
    if candidate.stat().st_mode & 0o077:
        raise ValueError("HF token path is not owner-only")
    value = candidate.read_bytes().strip()
    if not value:
        raise ValueError("HF token path is empty")
    return value


def _refresh_local_hf_auth(*, authorized: Path, cache: Path) -> dict[str, Any]:
    """Ask the normal local HF client to refresh OAuth-backed auth.

    The Lium SDK environment may not carry ``huggingface_hub``.  Run the
    normal local ``python3`` as a bounded subprocess in that case, with
    ``HF_TOKEN`` removed so it consults the user's regular HF cache.  Only the
    resulting authorized token file is copied onward; OAuth client material
    never enters the rental or remote filesystem.
    """
    python = shutil.which("python3") or sys.executable
    script = "from huggingface_hub import HfApi; HfApi().whoami()"
    child_env = dict(os.environ)
    child_env.pop("HF_TOKEN", None)
    child_env.pop("HF_TOKEN_PATH", None)
    subprocess.run(
        [python, "-c", script],
        env=child_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
        timeout=HF_REFRESH_TIMEOUT_SECONDS,
    )
    cache_bytes = _safe_token_bytes(cache)
    authorized_bytes = _safe_token_bytes(authorized)
    changed = cache_bytes != authorized_bytes
    if cache.resolve() != authorized.resolve() and cache_bytes != authorized_bytes:
        # Atomic replacement keeps the controller's authorized path valid if
        # the next poll races this refresh.  The temporary file is private and
        # never included in a receipt or diagnostic.
        temporary = authorized.with_name(f".{authorized.name}.refresh-{os.getpid()}")
        temporary.write_bytes(cache_bytes)
        temporary.chmod(0o600)
        temporary.replace(authorized)
        authorized_bytes = cache_bytes
    return {
        "status": "refreshed",
        "authorized_sha256": hashlib.sha256(authorized_bytes).hexdigest(),
        "changed": changed,
    }


def _rotate_remote_hf_token(*, sftp: Any, local: Path, timeout: float) -> str:
    """Atomically replace the exact remote credential path over SFTP."""
    before = _safe_token_bytes(local)
    digest = hashlib.sha256(before).hexdigest()
    base._sftp_put(sftp, local, str(HF_REMOTE_ROTATION_TMP), timeout=timeout)
    # posix_rename replaces the destination atomically on the same protected
    # filesystem.  Do not fall back to remove+rename, which would expose a
    # missing token window to a checkpoint upload.
    rename = getattr(sftp, "posix_rename", None)
    if not callable(rename):
        raise RuntimeError("SFTP server lacks atomic posix_rename")
    base._bounded_call(
        lambda: rename(str(HF_REMOTE_ROTATION_TMP), str(base.REMOTE_TOKEN)),
        timeout,
    )
    after = _safe_token_bytes(local)
    if hashlib.sha256(after).hexdigest() != digest:
        raise RuntimeError("authorized token changed during SFTP rotation")
    return digest


def _record_hf_refresh(status: dict[str, Any], *, now, context: dict[str, Any]) -> None:
    """Record a redacted refresh outcome without token-derived digests."""
    safe = {
        key: value
        for key, value in status.items()
        if key not in {"authorized_sha256", "remote_sha256"}
    }
    base._safe_event(
        context["events"], "hf_token_refresh", context["started"],
        clock=now, **safe,
    )


def _maybe_refresh_remote_hf_token(*, sftp: Any | None, timeout: float, now, window) -> None:
    context = _HF_REFRESH_CONTEXT
    if context is None:
        return
    current = time.monotonic()
    local = context["authorized"]
    try:
        observed = _safe_token_bytes(local)
        observed_sha = hashlib.sha256(observed).hexdigest()
    except BaseException as exc:
        # A transient local replacement must never stop the finite training
        # process; the next poll can retry after the owner repairs the file.
        _record_hf_refresh(
            {"status": "failed", "stage": "read_authorized", "failure_type": type(exc).__name__},
            now=now, context=context,
        )
        context["last_observed_sha"] = None
        context["last_attempt"] = current
        return

    due_to_change = observed_sha != context.get("last_observed_sha")
    due_to_interval = current - context.get("last_attempt", 0.0) >= context["interval"]
    if not due_to_change and not due_to_interval:
        return
    context["last_observed_sha"] = observed_sha
    context["last_attempt"] = current
    refresh_status: dict[str, Any] = {"status": "unchanged"}
    if due_to_interval and context["enabled"]:
        try:
            refresh_status = _refresh_local_hf_auth(authorized=local, cache=context["cache"])
            observed = _safe_token_bytes(local)
            observed_sha = hashlib.sha256(observed).hexdigest()
            context["last_observed_sha"] = observed_sha
        except BaseException as exc:
            # Record only the exception class.  In particular, do not include
            # subprocess stderr, token paths, or provider responses.
            refresh_status = {"status": "failed", "stage": "local_sdk_refresh", "failure_type": type(exc).__name__}
    if sftp is None or observed_sha == context.get("last_remote_sha"):
        refresh_status.setdefault("remote_rotated", False)
        _record_hf_refresh(refresh_status, now=now, context=context)
        return
    try:
        remaining = max(1.0, min(float(timeout), window.release_deadline_epoch - now()))
        remote_sha = _rotate_remote_hf_token(sftp=sftp, local=local, timeout=remaining)
        context["last_remote_sha"] = remote_sha
        refresh_status["remote_rotated"] = True
        refresh_status["remote_sha256"] = remote_sha
    except BaseException as exc:
        refresh_status.update({"remote_rotated": False, "stage": "sftp_atomic_rotate", "failure_type": type(exc).__name__})
    _record_hf_refresh(refresh_status, now=now, context=context)


def _observe_summary_with_hf_refresh(ssh: Any, sftp: Any | None, *, timeout: float, now, window, events: Path, started: float, phase: str):
    _maybe_refresh_remote_hf_token(sftp=sftp, timeout=timeout, now=now, window=window)
    return _BASE_OBSERVE_SUMMARY(
        ssh, sftp, timeout=timeout, now=now, window=window,
        events=events, started=started, phase=phase,
    )


base._observe_summary = _observe_summary_with_hf_refresh


def operate(*, window, source_bundle: Path, source_commit: str, hf_token_file: Path,
            gateway_env_file: Path, output_root: Path, api=None, now=time.time,
            sleep=time.sleep, clock_stall_timeout=None) -> int:
    """Run the reviewed controller with a bounded owner-token refresh hook."""
    global _HF_REFRESH_CONTEXT
    authorized = Path(hf_token_file).expanduser().resolve()
    try:
        interval = float(os.environ.get("PAINTER_HF_REFRESH_INTERVAL_SECONDS", str(HF_REFRESH_INTERVAL_SECONDS)))
    except (TypeError, ValueError):
        interval = HF_REFRESH_INTERVAL_SECONDS
    if not interval > 0:
        interval = HF_REFRESH_INTERVAL_SECONDS
    enabled = os.environ.get("PAINTER_HF_REFRESH_ENABLED", "1") == "1"
    cache = Path(
        os.environ.get("PAINTER_HF_AUTH_CACHE_PATH", str(Path.home() / ".cache/huggingface/token"))
    ).expanduser()
    initial = _safe_token_bytes(authorized)
    initial_sha = hashlib.sha256(initial).hexdigest()
    _HF_REFRESH_CONTEXT = {
        "authorized": authorized,
        "cache": cache,
        "interval": interval,
        "enabled": enabled,
        "last_attempt": 0.0,
        "last_observed_sha": initial_sha,
        # The initial transfer is performed by the reviewed base controller;
        # avoid a duplicate rotation on the first status poll.
        "last_remote_sha": initial_sha,
        "events": Path(output_root).expanduser().resolve() / "operations-single.jsonl",
        "started": time.time(),
    }
    try:
        return _BASE_OPERATE(
            window=window, source_bundle=source_bundle, source_commit=source_commit,
            hf_token_file=hf_token_file, gateway_env_file=gateway_env_file,
            output_root=output_root, api=api, now=now, sleep=sleep,
            clock_stall_timeout=clock_stall_timeout,
        )
    finally:
        _HF_REFRESH_CONTEXT = None


# allocate_single.py calls the controller through ``operation.base`` to retain
# the proven release path. Replace that entrypoint with this thin wrapper while
# keeping the original implementation in _BASE_OPERATE.
base.operate = operate


def _detached_job_command(*, source_commit: str, rental_start_epoch: float, price_usd_per_hour: float, cap_usd: float) -> str:
    if not EVALUATION_ONLY:
        hardware_profile.require_resume(HARDWARE_PROFILE)
    topology = hardware_profile.get(HARDWARE_PROFILE)
    values = {
        "PAINTER_SOURCE_COMMIT": source_commit,
        "PAINTER_NODE_ROOT": str(base.REMOTE_ROOT),
        "HF_TOKEN_PATH": str(base.REMOTE_TOKEN),
        "PAINTER_GATEWAY_ENV_FILE": str(base.REMOTE_GATEWAY),
        "PAINTER_RUN_ROOT": str(base.REMOTE_RUN),
        "PAINTER_RUN_NAME": base.RUN_NAME,
        "PAINTER_HARDWARE_PROFILE": HARDWARE_PROFILE,
        "PAINTER_GPUS_PER_NODE": "1" if EVALUATION_ONLY else str(topology["gpus_per_node"]),
        "PAINTER_NUM_TRAIN_GPUS": "0" if EVALUATION_ONLY else str(topology["num_train_gpus"]),
        "PAINTER_NUM_INFER_GPUS": "1" if EVALUATION_ONLY else str(topology["num_infer_gpus"]),
        "PAINTER_TRAIN_GPU_IDS": "" if EVALUATION_ONLY else ",".join(topology["train_gpu_ids"]),
        "PAINTER_INFER_GPU_IDS": "0" if EVALUATION_ONLY else ",".join(topology["infer_gpu_ids"]),
        "PAINTER_INFERENCE_TENSOR_PARALLEL": "1" if EVALUATION_ONLY else str(topology["inference_tensor_parallel"]),
        "PAINTER_STRUCTURED_FLOW_POLICY": "1",
        "PAINTER_CODE_SUPERVISION_ENABLED": "1",
        "PAINTER_RUN_SUPERVISION": "1",
        "PAINTER_RUN_POST_WARMUP_EVAL": "1",
        # The evaluator owns a shorter archive/release tail; the training
        # branch retains its historical restart reserve.
        "PAINTER_RESERVE_SECONDS": "120" if EVALUATION_ONLY else "2700",
        "HF_CHECKPOINT_REPO": "CK0607/qwen3.8-27b-brush-painting",
        "HF_CHECKPOINT_RUN": "diverse-flow-rl-20261009-sft-warmup",
    }
    if EVALUATION_ONLY:
        values.update({
            "PAINTER_EVALUATION_ONLY": "1",
            "PAINTER_EVAL_PREFERRED_STEP": "4",
            "PAINTER_EVAL_SCOPE": "all",
            "PAINTER_EVAL_EXPECTED_COUNT": "32",
            "PAINTER_EVAL_SEED": "918210",
            "PAINTER_EVAL_MAX_CONCURRENT": "8",
            "PAINTER_MAX_TURNS": "3",
            "PAINTER_JUDGE_MODEL": "xiaomi/mimo-v2.6-flash",
            "PAINTER_JUDGE_EFFORT": "medium",
            "PAINTER_JUDGE_FORMAT": "identity-joint-side-schema-v2-generic-v1",
            "PAINTER_JUDGE_PROTOCOL": "text-paint-sft-pairwise-consistent-v1",
            "PAINTER_RUN_WARMUP": "0",
            "PAINTER_WARMUP_REQUIRED": "0",
            "PAINTER_RUN_SUPERVISION": "0",
            "PAINTER_RUN_POST_WARMUP_EVAL": "0",
            "PAINTER_RESTORE_SUPERVISION_EVIDENCE": "0",
            "PAINTER_CODE_SUPERVISION_ENABLED": "0",
            "PAINTER_STRUCTURED_FLOW_POLICY": "1",
            # Evaluation recovery must consume this run's fresh 32-case panel;
            # prior 16-case archives are never a fallback.
            "PAINTER_RESTORE_ANCHOR_EVIDENCE": "0",
            "PAINTER_REUSE_MATCHED_BASELINE_EVIDENCE": "0",
            # Both the success and failure archive paths use this protected
            # remote token file; its contents never enter the command string.
            "PAINTER_RECOVERY_TOKEN_FILE": str(base.REMOTE_TOKEN),
        })
    # Warmup is a controller-level requirement for training.  Evaluation
    # recovery is the only role allowed to carry both flags as zero.
    run_warmup = os.environ.get("PAINTER_RUN_WARMUP", "0" if EVALUATION_ONLY else "1")
    warmup_required = os.environ.get("PAINTER_WARMUP_REQUIRED", "0" if EVALUATION_ONLY else "1")
    if run_warmup not in {"0", "1"} or warmup_required not in {"0", "1"}:
        raise ValueError("PAINTER_RUN_WARMUP and PAINTER_WARMUP_REQUIRED must be 0 or 1")
    if run_warmup != warmup_required:
        raise ValueError(
            "PAINTER_RUN_WARMUP and PAINTER_WARMUP_REQUIRED must be paired; "
            "use both 0 for the explicit RL-only fallback"
        )
    if not EVALUATION_ONLY and (run_warmup, warmup_required) != ("1", "1"):
        raise ValueError("Diverse-flow training controller requires the four-update code-flow warmup")
    values.update({
        "PAINTER_RUN_WARMUP": run_warmup,
        "PAINTER_WARMUP_REQUIRED": warmup_required,
    })
    # Same-run resume is one atomic opt-in. Both legacy phase bits are kept in
    # the command for node-side compatibility, but their coordinates are
    # always replaced by this exact credential-free public URL/SHA pin.
    same_run_resume = os.environ.get("PAINTER_RESUME_SAME_RUN_ARCHIVE", "0")
    restore_anchor = os.environ.get("PAINTER_RESTORE_ANCHOR_EVIDENCE", "0")
    reuse_baseline = os.environ.get("PAINTER_REUSE_MATCHED_BASELINE_EVIDENCE", "0")
    if same_run_resume not in {"0", "1"}:
        raise ValueError("PAINTER_RESUME_SAME_RUN_ARCHIVE must be 0 or 1")
    if restore_anchor not in {"0", "1"}:
        raise ValueError("PAINTER_RESTORE_ANCHOR_EVIDENCE must be 0 or 1")
    if reuse_baseline not in {"0", "1"}:
        raise ValueError("PAINTER_REUSE_MATCHED_BASELINE_EVIDENCE must be 0 or 1")
    if same_run_resume == "1":
        restore_anchor = reuse_baseline = "1"
    elif restore_anchor == "1" or reuse_baseline == "1":
        if restore_anchor != "1" or reuse_baseline != "1":
            raise ValueError("anchor restore and matched-baseline reuse must be enabled together for a same-run retry")
        same_run_resume = "1"
    values["PAINTER_RESUME_SAME_RUN_ARCHIVE"] = same_run_resume
    values["PAINTER_RESTORE_ANCHOR_EVIDENCE"] = restore_anchor
    values["PAINTER_REUSE_MATCHED_BASELINE_EVIDENCE"] = reuse_baseline
    if same_run_resume == "1":
        archive_sha = os.environ.get(
            "PAINTER_SAME_RUN_ARCHIVE_SHA256",
            os.environ.get("PAINTER_ANCHOR_EVIDENCE_SHA256", "41f527d104e1e5fe52ef83369e3fd089ba153ce2e3112ae25d4da25f682881ef"),
        )
        archive_url = os.environ.get(
            "PAINTER_SAME_RUN_ARCHIVE_URL",
            os.environ.get(
                "PAINTER_ANCHOR_EVIDENCE_URL",
                "https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/resolve/"
                "2de12b59875fbdd5cc0f206f79eed12c8ee9f43b/runs/diverse-flow-rl-20261009/evidence.tar.gz",
            ),
        )
        pinned_sha = "41f527d104e1e5fe52ef83369e3fd089ba153ce2e3112ae25d4da25f682881ef"
        pinned_url = (
            "https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/resolve/"
            "2de12b59875fbdd5cc0f206f79eed12c8ee9f43b/runs/diverse-flow-rl-20261009/evidence.tar.gz"
        )
        for name in (
            "PAINTER_SAME_RUN_ARCHIVE_SHA256",
            "PAINTER_ANCHOR_EVIDENCE_SHA256",
            "PAINTER_MATCHED_BASELINE_EVIDENCE_SHA256",
        ):
            value = os.environ.get(name)
            if value and value != pinned_sha:
                raise ValueError("same-run resume requires the pinned public archive URL and SHA-256")
        for name in (
            "PAINTER_SAME_RUN_ARCHIVE_URL",
            "PAINTER_ANCHOR_EVIDENCE_URL",
            "PAINTER_MATCHED_BASELINE_EVIDENCE_URL",
        ):
            value = os.environ.get(name)
            if value and value != pinned_url:
                raise ValueError("same-run resume requires the pinned public archive URL and SHA-256")
        if archive_sha != pinned_sha or archive_url != pinned_url:
            raise ValueError("same-run resume requires the pinned public archive URL and SHA-256")
        parsed = urllib.parse.urlparse(archive_url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "huggingface.co"
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("same-run archive URL must be a credential-free pinned HTTPS Hugging Face URL")
        if any(os.environ.get(name) for name in (
            "PAINTER_ANCHOR_EVIDENCE_ARCHIVE",
            "PAINTER_MATCHED_BASELINE_EVIDENCE_ARCHIVE",
        )):
            raise ValueError("same-run resume accepts only the pinned public URL; local archives are disabled")
        values.update({
            "PAINTER_SAME_RUN_ARCHIVE_SHA256": pinned_sha,
            "PAINTER_SAME_RUN_ARCHIVE_URL": pinned_url,
            "PAINTER_ANCHOR_EVIDENCE_SHA256": pinned_sha,
            "PAINTER_ANCHOR_EVIDENCE_URL": pinned_url,
            "PAINTER_MATCHED_BASELINE_EVIDENCE_SHA256": pinned_sha,
            "PAINTER_MATCHED_BASELINE_EVIDENCE_URL": pinned_url,
        })
    # Keep this opt-in: the controller's environment can select a node-local
    # filesystem for the large model and Prime checkout, while credentials
    # remain under REMOTE_ROOT/private.  An unset value preserves the reviewed
    # node-root defaults.
    local_scratch = os.environ.get("PAINTER_LOCAL_SCRATCH_ROOT", "")
    if local_scratch:
        if not local_scratch.startswith("/"):
            raise ValueError("PAINTER_LOCAL_SCRATCH_ROOT must be absolute")
        values["PAINTER_LOCAL_SCRATCH_ROOT"] = local_scratch
    prime_seed = os.environ.get("PAINTER_PRIME_SEED_ENABLED", "0")
    if prime_seed not in {"0", "1"}:
        raise ValueError("PAINTER_PRIME_SEED_ENABLED must be 0 or 1")
    if prime_seed == "1":
        seed_root = os.environ.get("PAINTER_PRIME_SEED_ROOT") or (
            str(Path(local_scratch) / "prime") if local_scratch else "/root/painter/prime"
        )
        if not seed_root.startswith("/"):
            raise ValueError("PAINTER_PRIME_SEED_ROOT must be absolute")
        values["PAINTER_PRIME_SEED_ENABLED"] = "1"
        values["PAINTER_PRIME_SEED_ROOT"] = seed_root
    restore = os.environ.get("PAINTER_RESTORE_ANCHOR_EVIDENCE", "0")
    if restore not in {"0", "1"}:
        raise ValueError("PAINTER_RESTORE_ANCHOR_EVIDENCE must be 0 or 1")
    if restore == "1":
        restore_sha = os.environ.get("PAINTER_ANCHOR_EVIDENCE_SHA256", "")
        if len(restore_sha) != 64 or any(char not in "0123456789abcdef" for char in restore_sha):
            raise ValueError("PAINTER_ANCHOR_EVIDENCE_SHA256 must be lowercase hexadecimal")
        restore_url = os.environ.get("PAINTER_ANCHOR_EVIDENCE_URL", "")
        parsed_url = urllib.parse.urlparse(restore_url)
        if parsed_url.scheme != "https" or parsed_url.netloc != "huggingface.co" or parsed_url.username or parsed_url.password or parsed_url.query or parsed_url.fragment:
            raise ValueError("PAINTER_ANCHOR_EVIDENCE_URL must be a credential-free pinned HTTPS Hugging Face URL")
        values.update({
            "PAINTER_RESTORE_ANCHOR_EVIDENCE": "1",
            "PAINTER_ANCHOR_EVIDENCE_SHA256": restore_sha,
            "PAINTER_ANCHOR_EVIDENCE_URL": restore_url,
        })
    supervision_restore = os.environ.get("PAINTER_RESTORE_SUPERVISION_EVIDENCE", "1")
    if supervision_restore not in {"0", "1"}:
        raise ValueError("PAINTER_RESTORE_SUPERVISION_EVIDENCE must be 0 or 1")
    if supervision_restore == "1":
        supervision_sha = os.environ.get(
            "PAINTER_SUPERVISION_EVIDENCE_SHA256",
            "ab2531f9f12032b9c11e75cbeb681bdf08d084a7b1458864db050493292ccb67",
        )
        if len(supervision_sha) != 64 or any(char not in "0123456789abcdef" for char in supervision_sha):
            raise ValueError("PAINTER_SUPERVISION_EVIDENCE_SHA256 must be lowercase hexadecimal")
        supervision_url = os.environ.get(
            "PAINTER_SUPERVISION_EVIDENCE_URL",
            "https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/resolve/"
            "ca0a838605dfd8cfb75a54dde2d480a94bcbef66/runs/structured-flow-rl-20261009/evidence.tar.gz",
        )
        parsed_url = urllib.parse.urlparse(supervision_url)
        if parsed_url.scheme != "https" or parsed_url.netloc != "huggingface.co" or parsed_url.username or parsed_url.password or parsed_url.query or parsed_url.fragment:
            raise ValueError("PAINTER_SUPERVISION_EVIDENCE_URL must be a credential-free pinned HTTPS Hugging Face URL")
        values.update({
            "PAINTER_RESTORE_SUPERVISION_EVIDENCE": "1",
            "PAINTER_SUPERVISION_EVIDENCE_SHA256": supervision_sha,
            "PAINTER_SUPERVISION_EVIDENCE_URL": supervision_url,
        })
    env = " ".join(
        f"{key}={shlex.quote(value)}"
        for key, value in values.items()
    )
    execute_name = "recover_evaluation.sh" if EVALUATION_ONLY else "execute_single.sh"
    execute = shlex.quote(str(base.REMOTE_RUN / execute_name))
    args = " ".join(shlex.quote(str(value)) for value in (rental_start_epoch, price_usd_per_hour, cap_usd))
    exit_tmp = shlex.quote(str(base.REMOTE_JOB_EXIT) + ".tmp")
    inner = (
        f"set +e; env {env} bash {execute} {args} >{shlex.quote(str(base.REMOTE_JOB_STDOUT))} "
        f"2>{shlex.quote(str(base.REMOTE_JOB_STDERR))}; rc=$?; "
        f"printf '{{\"schema\":\"Diverse-flow-remote-job-v1\",\"exit_code\":%s}}\\n' \"$rc\" >{exit_tmp}; "
        f"mv -f {exit_tmp} {shlex.quote(str(base.REMOTE_JOB_EXIT))}; exit \"$rc\""
    )
    return (
        "set -Eeuo pipefail; "
        f"rm -f {shlex.quote(str(base.REMOTE_JOB_START))} {shlex.quote(str(base.REMOTE_JOB_PID))} "
        f"{shlex.quote(str(base.REMOTE_JOB_EXIT))} {exit_tmp}; "
        f"nohup setsid bash -c {shlex.quote(inner)} </dev/null >/dev/null 2>&1 & pid=$!; "
        f"printf '%s\\n' \"$pid\" >{shlex.quote(str(base.REMOTE_JOB_PID))}; "
        f"printf '%s\\n' '{{\"schema\":\"Diverse-flow-remote-job-v1\",\"status\":\"started\"}}' >{shlex.quote(str(base.REMOTE_JOB_START))}"
    )


base._detached_job_command = _detached_job_command
_base_summary = base._remote_summary


def _remote_summary(*args, **kwargs):
    value = _base_summary(*args, **kwargs)
    # The shared controller summary intentionally exposes only generic shape
    # fields. Read the structured receipt once more so the controller can
    # reject a stale 8-update config before training/evaluation proceeds.
    config = value.get("config")
    sftp = args[0] if args else kwargs.get("sftp")
    if isinstance(config, dict) and sftp is not None:
        try:
            timeout = max(0.1, min(float(kwargs.get("timeout", 5.0)), 5.0))
            raw = base._sftp_read(sftp, str(base.REMOTE_RUN / "rl-config-receipt.json"), timeout=timeout)
            receipt = base._json_bytes(raw)
            if isinstance(receipt, dict):
                config.update({
                    "updates": receipt.get("updates"),
                    "requested_updates": receipt.get("requested_updates", receipt.get("updates")),
                    "checkpoint_steps": receipt.get("checkpoint_steps"),
                })
        except Exception:
            # The shared summary has already captured the bounded observation;
            # validation below fails closed when these structured fields are
            # unavailable.
            pass
    preparation = value.get("preparation", {})
    if preparation.get("schema") == "Diverse-flow-rl-preparation-v1":
        value["phase_ready"] = True
    if EVALUATION_ONLY:
        sftp = args[0] if args else kwargs.get("sftp")
        timeout = max(0.1, min(float(kwargs.get("timeout", 5.0)), 1.0))
        recovery = {}
        terminal = {}
        if sftp is not None:
            try:
                raw = base._sftp_read(sftp, str(base.REMOTE_RUN / "evaluation-recovery-receipt.json"), timeout=timeout)
                recovery = base._json_bytes(raw) or {}
            except Exception:
                recovery = {}
            try:
                raw = base._sftp_read(sftp, str(base.REMOTE_RUN / "terminal-receipt.json"), timeout=timeout)
                terminal = base._json_bytes(raw) or {}
            except Exception:
                terminal = {}
        archived_preparation = value.get("preparation", {})
        archived_deployment = value.get("config", {}).get("deployment")
        value["evaluation_recovery"] = {
            "schema": recovery.get("schema"),
            "status": recovery.get("status"),
            "mode": recovery.get("mode"),
            "topology": recovery.get("topology"),
            "manifest_sha256": recovery.get("evaluation", {}).get("manifest_sha256") if isinstance(recovery.get("evaluation"), dict) else None,
        }
        value["preparation"] = {
            "schema": "Diverse-flow-evaluation-recovery-preparation-v1",
            "mode": "evaluation_only",
            "training_ready": False,
            "evaluation_ready": recovery.get("mode") == "evaluation_only",
            "archived_training_preparation": archived_preparation,
        }
        value["config"]["archived_training_deployment"] = archived_deployment
        value["config"]["deployment"] = {"type": "single_node", "gpus_per_node": 1, "num_train_gpus": 0, "num_infer_gpus": 1}
        value["phase_ready"] = recovery.get("mode") == "evaluation_only" and recovery.get("status") == "prepared_no_training"
        value["training_ready"] = False
        if isinstance(terminal, dict):
            value["terminal"].update({
                "recovery_status": terminal.get("status"),
                "exit_code": terminal.get("exit_code"),
                "checkpoint4_public_complete": terminal.get("checkpoint4_public_complete"),
                "archive_hash_verified": terminal.get("archive", {}).get("hash_verified") if isinstance(terminal.get("archive"), dict) else None,
                "public_archive_hash_verified": terminal.get("archive", {}).get("public_hash_verified") if isinstance(terminal.get("archive"), dict) else None,
            })
    return value


base._remote_summary = _remote_summary


def _validate_config_summary(summary):
    if EVALUATION_ONLY:
        recovery = summary.get("evaluation_recovery", {})
        topology = summary.get("config", {}).get("deployment")
        if recovery.get("schema") != "Diverse-flow-evaluation-recovery-v1" or recovery.get("mode") != "evaluation_only":
            raise base.OperationError("remote preparation is not evaluation-only ready")
        if topology != {"type": "single_node", "gpus_per_node": 1, "num_train_gpus": 0, "num_infer_gpus": 1}:
            raise base.OperationError("remote evaluation topology is not one inference GPU with no trainer")
        if summary.get("preparation", {}).get("training_ready") is not False:
            raise base.OperationError("evaluation receipt must not claim RL training readiness")
        return
    preparation = summary.get("preparation", {})
    if preparation.get("schema") != "Diverse-flow-rl-preparation-v1":
        raise base.OperationError("remote preparation is not Diverse-flow ready")
    config = summary.get("config", {})
    expected = (("updates", 20), ("checkpoint_steps", [4, 8, 12, 16, 20]), ("checkpoint_interval", 4), ("batch_size", 16), ("group_size", 4), ("initial_inflight", 32), ("max_inflight", 32))
    if any(config.get(key) != value for key, value in expected):
        raise base.OperationError("remote config shape does not match Diverse-flow pilot")
    profile = hardware_profile.get(HARDWARE_PROFILE)
    expected_deployment = {"type": "single_node", "gpus_per_node": profile["gpus_per_node"], "num_train_gpus": profile["num_train_gpus"], "num_infer_gpus": profile["num_infer_gpus"]}
    if (config.get("deployment") or {}) != expected_deployment:
        raise base.OperationError(f"remote config topology does not match hardware profile {HARDWARE_PROFILE}")
    if config.get("inference_parallelism") != {"tensor_parallel_size": profile["inference_tensor_parallel"], "data_parallel_size": 1}:
        raise base.OperationError(f"remote inference topology does not match hardware profile {HARDWARE_PROFILE}")


base._validate_config_summary = _validate_config_summary


def parser():
    value = base.argparse.ArgumentParser(description=__doc__)
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


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    window = base.make_window(pod_id=args.pod_id, rental_start_epoch=args.rental_start_epoch, price_usd_per_hour=args.price_usd_hour, cap_usd=args.cap_usd)
    source_bundle = base._regular_file(args.source_bundle, "source-bundle")
    token = base._regular_file(args.hf_token_file, "HF token file", owner_only=True)
    gateway = base._regular_file(args.gateway_env_file, "Gateway environment file", owner_only=True)
    if not base.COMMIT_RE.fullmatch(args.source_commit):
        raise SystemExit("source-commit must be an immutable full hash")
    if args.plan_only:
        import json
        topology = {"gpus_per_node": base.GPU_COUNT, "num_train_gpus": base.TRAIN_GPU_COUNT, "num_infer_gpus": base.INFER_GPU_COUNT}
        print(json.dumps({"schema": "Diverse-flow-single-operation-plan-v1", "run_name": base.RUN_NAME, "experiment_name": base.EXPERIMENT_NAME, "hardware_profile": HARDWARE_PROFILE, "gpu_cap_usd": window.cap_usd, "mode": "evaluation_only" if EVALUATION_ONLY else "training", "topology": topology, "max_updates": 0 if EVALUATION_ONLY else 20, "checkpoint_steps": [4] if EVALUATION_ONLY else [4, 8, 12, 16, 20], "judge": {"model": "xiaomi/mimo-v2.6-flash", "effort": "medium"}, "source_bundle_bytes": source_bundle.stat().st_size, "credentials_validated": True}, sort_keys=True))
        return 0
    return base.operate(window=window, source_bundle=source_bundle, source_commit=args.source_commit, hf_token_file=token, gateway_env_file=gateway, output_root=args.output_root)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (base.OperationError, OSError, ValueError) as exc:
        print(f"Diverse-flow operation: {exc}", file=sys.stderr)
        raise SystemExit(2)
