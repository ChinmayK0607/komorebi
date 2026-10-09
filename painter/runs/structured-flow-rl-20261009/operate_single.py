#!/usr/bin/env python3
"""Operate one bounded structured-flow run on an already rented pod.

Transport, private process ownership, bounded polling, and exact-pod release
come from the reviewed single-arm controller.  This adapter changes only the
run identity, remote command, and structured config checks.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shlex
import sys
import urllib.parse


RUN = Path(__file__).resolve().parent
SOURCE = RUN.parent / "throughput-recovery-20261003" / "operate_single.py"
spec = importlib.util.spec_from_file_location("structured_flow_operation_base", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"single operation controller unavailable: {SOURCE}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

base.RUN_NAME = "structured-flow-rl-20261009"
base.EXPERIMENT_NAME = "structured-flow-rl-20261009-flash"
base.REMOTE_RUN = base.REMOTE_REPO / "painter/runs/structured-flow-rl-20261009"
base.REMOTE_BUNDLE = base.REMOTE_ROOT / "source.bundle"
base.REMOTE_JOB_START = base.REMOTE_JOB_DIR / "structured-flow-remote-job-start.json"
base.REMOTE_JOB_PID = base.REMOTE_JOB_DIR / "structured-flow-remote-job.pid"
base.REMOTE_JOB_EXIT = base.REMOTE_JOB_DIR / "structured-flow-remote-job-exit.json"
base.REMOTE_JOB_STDOUT = base.REMOTE_JOB_DIR / "structured-flow-remote-job.stdout.log"
base.REMOTE_JOB_STDERR = base.REMOTE_JOB_DIR / "structured-flow-remote-job.stderr.log"
EVALUATION_ONLY = os.environ.get("PAINTER_EVALUATION_ONLY", "0")
if EVALUATION_ONLY not in {"0", "1"}:
    raise ValueError("PAINTER_EVALUATION_ONLY must be 0 or 1")
EVALUATION_ONLY = EVALUATION_ONLY == "1"
base.GPU_COUNT = 1 if EVALUATION_ONLY else 2
base.TRAIN_GPU_COUNT = 0 if EVALUATION_ONLY else 1
base.INFER_GPU_COUNT = 1


def _detached_job_command(*, source_commit: str, rental_start_epoch: float, price_usd_per_hour: float, cap_usd: float) -> str:
    values = {
        "PAINTER_SOURCE_COMMIT": source_commit,
        "PAINTER_NODE_ROOT": str(base.REMOTE_ROOT),
        "HF_TOKEN_PATH": str(base.REMOTE_TOKEN),
        "PAINTER_GATEWAY_ENV_FILE": str(base.REMOTE_GATEWAY),
        "PAINTER_RUN_ROOT": str(base.REMOTE_RUN),
        "PAINTER_RUN_NAME": base.RUN_NAME,
        "PAINTER_GPUS_PER_NODE": "1" if EVALUATION_ONLY else "2",
        "PAINTER_NUM_TRAIN_GPUS": "0" if EVALUATION_ONLY else "1",
        "PAINTER_NUM_INFER_GPUS": "1",
        "PAINTER_STRUCTURED_FLOW_POLICY": "1",
        "PAINTER_CODE_SUPERVISION_ENABLED": "1",
        "PAINTER_RUN_SUPERVISION": "1",
        # The evaluator owns a shorter archive/release tail; the training
        # branch retains its historical restart reserve.
        "PAINTER_RESERVE_SECONDS": "120" if EVALUATION_ONLY else "2700",
        "HF_CHECKPOINT_REPO": "CK0607/qwen3.8-27b-brush-painting",
        "HF_CHECKPOINT_RUN": "structured-flow-rl-20261009-sft-warmup",
    }
    if EVALUATION_ONLY:
        values.update({
            "PAINTER_EVALUATION_ONLY": "1",
            "PAINTER_EVAL_PREFERRED_STEP": "4",
            "PAINTER_EVAL_SCOPE": "all",
            "PAINTER_EVAL_EXPECTED_COUNT": "16",
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
            "PAINTER_RESTORE_SUPERVISION_EVIDENCE": "0",
            "PAINTER_CODE_SUPERVISION_ENABLED": "0",
            "PAINTER_STRUCTURED_FLOW_POLICY": "1",
            "PAINTER_RESTORE_ANCHOR_EVIDENCE": "1",
            "PAINTER_ANCHOR_EVIDENCE_SHA256": "9ec7942d8666f09ff794d635d306b6f1cfb6d440f55186fec32f6c1befb5ffdb",
            "PAINTER_ANCHOR_EVIDENCE_URL": "https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/resolve/cec8e7133738b03337ee4cae3f1c3fffdd91bc7f/runs/structured-flow-rl-20261009/evidence.tar.gz",
            "PAINTER_REUSE_MATCHED_BASELINE_EVIDENCE": "1",
            "PAINTER_MATCHED_BASELINE_EVIDENCE_SHA256": "ff4efa37835b1a7e3c5e90a628d685f1e94f39cf3d13760e47abab834c1a90c3",
            "PAINTER_MATCHED_BASELINE_EVIDENCE_URL": "https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/resolve/f72ee86cc24cb5fa3618c649aa1fe32ea22753ae/runs/structured-flow-rl-20261009/evidence.tar.gz",
            # Both the success and failure archive paths use this protected
            # remote token file; its contents never enter the command string.
            "PAINTER_RECOVERY_TOKEN_FILE": str(base.REMOTE_TOKEN),
        })
    # Warmup remains required by default.  The only supported bypass is an
    # explicit paired opt-in: both flags must be zero so prepare_node records
    # a pending supervision receipt instead of silently skipping SFT.
    run_warmup = os.environ.get("PAINTER_RUN_WARMUP", "0" if EVALUATION_ONLY else "1")
    warmup_required = os.environ.get("PAINTER_WARMUP_REQUIRED", "0" if EVALUATION_ONLY else "1")
    if run_warmup not in {"0", "1"} or warmup_required not in {"0", "1"}:
        raise ValueError("PAINTER_RUN_WARMUP and PAINTER_WARMUP_REQUIRED must be 0 or 1")
    if run_warmup != warmup_required:
        raise ValueError(
            "PAINTER_RUN_WARMUP and PAINTER_WARMUP_REQUIRED must be paired; "
            "use both 0 for the explicit RL-only fallback"
        )
    values.update({
        "PAINTER_RUN_WARMUP": run_warmup,
        "PAINTER_WARMUP_REQUIRED": warmup_required,
    })
    reuse_baseline = os.environ.get("PAINTER_REUSE_MATCHED_BASELINE_EVIDENCE", "1" if EVALUATION_ONLY else "0")
    if reuse_baseline not in {"0", "1"}:
        raise ValueError("PAINTER_REUSE_MATCHED_BASELINE_EVIDENCE must be 0 or 1")
    # matched_baseline.sh owns the pinned attempt16 URL/SHA defaults.  Forward
    # only the strict opt-in bit; arbitrary archive coordinates stay local to
    # that reviewed helper and are never accepted from the detached command.
    values["PAINTER_REUSE_MATCHED_BASELINE_EVIDENCE"] = reuse_baseline
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
    restore = os.environ.get("PAINTER_RESTORE_ANCHOR_EVIDENCE", "1" if EVALUATION_ONLY else "0")
    if restore not in {"0", "1"}:
        raise ValueError("PAINTER_RESTORE_ANCHOR_EVIDENCE must be 0 or 1")
    if restore == "1":
        restore_sha = os.environ.get(
            "PAINTER_ANCHOR_EVIDENCE_SHA256",
            "0c7fac0092ebdb94eb4f1e32d7d4df6a91e0d4908bf14ab7a0cc2ff476510266",
        )
        if len(restore_sha) != 64 or any(char not in "0123456789abcdef" for char in restore_sha):
            raise ValueError("PAINTER_ANCHOR_EVIDENCE_SHA256 must be lowercase hexadecimal")
        restore_url = os.environ.get(
            "PAINTER_ANCHOR_EVIDENCE_URL",
            "https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/resolve/"
            "cdf16bceda0212ba94e6f3fa5cc6f92b95f3c6d6/runs/structured-flow-rl-20261009/evidence.tar.gz",
        )
        parsed_url = urllib.parse.urlparse(restore_url)
        if parsed_url.scheme != "https" or parsed_url.netloc != "huggingface.co" or parsed_url.username or parsed_url.password or parsed_url.query or parsed_url.fragment:
            raise ValueError("PAINTER_ANCHOR_EVIDENCE_URL must be a credential-free pinned HTTPS Hugging Face URL")
        values.update({
            "PAINTER_RESTORE_ANCHOR_EVIDENCE": "1",
            "PAINTER_ANCHOR_EVIDENCE_SHA256": restore_sha,
            "PAINTER_ANCHOR_EVIDENCE_URL": restore_url,
        })
    supervision_restore = os.environ.get("PAINTER_RESTORE_SUPERVISION_EVIDENCE", "0")
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
        f"printf '{{\"schema\":\"structured-flow-remote-job-v1\",\"exit_code\":%s}}\\n' \"$rc\" >{exit_tmp}; "
        f"mv -f {exit_tmp} {shlex.quote(str(base.REMOTE_JOB_EXIT))}; exit \"$rc\""
    )
    return (
        "set -Eeuo pipefail; "
        f"rm -f {shlex.quote(str(base.REMOTE_JOB_START))} {shlex.quote(str(base.REMOTE_JOB_PID))} "
        f"{shlex.quote(str(base.REMOTE_JOB_EXIT))} {exit_tmp}; "
        f"nohup setsid bash -c {shlex.quote(inner)} </dev/null >/dev/null 2>&1 & pid=$!; "
        f"printf '%s\\n' \"$pid\" >{shlex.quote(str(base.REMOTE_JOB_PID))}; "
        f"printf '%s\\n' '{{\"schema\":\"structured-flow-remote-job-v1\",\"status\":\"started\"}}' >{shlex.quote(str(base.REMOTE_JOB_START))}"
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
    if preparation.get("schema") == "structured-flow-rl-preparation-v1":
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
            "schema": "structured-flow-evaluation-recovery-preparation-v1",
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
        if recovery.get("schema") != "structured-flow-evaluation-recovery-v1" or recovery.get("mode") != "evaluation_only":
            raise base.OperationError("remote preparation is not evaluation-only ready")
        if topology != {"type": "single_node", "gpus_per_node": 1, "num_train_gpus": 0, "num_infer_gpus": 1}:
            raise base.OperationError("remote evaluation topology is not one inference GPU with no trainer")
        if summary.get("preparation", {}).get("training_ready") is not False:
            raise base.OperationError("evaluation receipt must not claim RL training readiness")
        return
    preparation = summary.get("preparation", {})
    if preparation.get("schema") != "structured-flow-rl-preparation-v1":
        raise base.OperationError("remote preparation is not structured-flow ready")
    config = summary.get("config", {})
    expected = (("updates", 20), ("checkpoint_steps", [4, 8, 12, 16, 20]), ("checkpoint_interval", 4), ("batch_size", 16), ("group_size", 4), ("initial_inflight", 32), ("max_inflight", 32))
    if any(config.get(key) != value for key, value in expected):
        raise base.OperationError("remote config shape does not match structured-flow pilot")
    if (config.get("deployment") or {}) != {"type": "single_node", "gpus_per_node": 2, "num_train_gpus": 1, "num_infer_gpus": 1}:
        raise base.OperationError("remote config topology is not two GPUs with one trainer and one inference GPU")


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
        print(json.dumps({"schema": "structured-flow-single-operation-plan-v1", "run_name": base.RUN_NAME, "experiment_name": base.EXPERIMENT_NAME, "gpu_cap_usd": window.cap_usd, "mode": "evaluation_only" if EVALUATION_ONLY else "training", "topology": topology, "max_updates": 0 if EVALUATION_ONLY else 20, "checkpoint_steps": [4] if EVALUATION_ONLY else [4, 8, 12, 16, 20], "judge": {"model": "xiaomi/mimo-v2.6-flash", "effort": "medium"}, "source_bundle_bytes": source_bundle.stat().st_size, "credentials_validated": True}, sort_keys=True))
        return 0
    return base.operate(window=window, source_bundle=source_bundle, source_commit=args.source_commit, hf_token_file=token, gateway_env_file=gateway, output_root=args.output_root)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (base.OperationError, OSError, ValueError) as exc:
        print(f"structured-flow operation: {exc}", file=sys.stderr)
        raise SystemExit(2)
