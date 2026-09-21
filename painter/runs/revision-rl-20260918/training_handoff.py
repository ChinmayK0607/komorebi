"""Small, observational receipt for the native Prime batch boundary.

Prime already owns rollout grouping, score eligibility, policy freshness,
packing and optimizer admission.  This module does not reimplement those
decisions from reward sidecars.  At ``DataLoader.get_batch`` it retains the
actual ``rank_N.bin`` wire artifact and writes shape/provenance facts that can
be inspected after a run.  The JSON is an audit summary; the retained msgpack
file is the replayable batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any


SCHEMA = "painter.native-training-observation.v1"
PRIME_REVISION = "26b3131d2716a4f8210b165df584b83b4bc54f61"
STATE_SCHEMA = "painter.training-state.v2"
HANDOFF_NAME = "training-handoff.json"
STATE_NAME = "training-state.json"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _flat(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    elif hasattr(value, "reshape") and hasattr(value, "tolist"):
        value = value.reshape(-1).tolist()
    return list(value)


def _tensor_metadata(value: Any) -> dict[str, Any]:
    if hasattr(value, "detach"):
        # Keep this observer safe for bf16 tensors: the retained msgpack hash
        # covers bytes, while this JSON only needs shape/dtype facts.
        tensor = value.detach()
        return {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    return {"type": type(value).__name__}


def _copy_wire_artifact(source: Path, destination: Path) -> dict[str, Any]:
    """Copy one already-written native rank file without exposing a partial file."""
    if not source.is_file():
        return {"source": str(source), "status": "unavailable", "reason": "rank file not found"}
    source_hash = _sha(source)
    if destination.is_file():
        retained_hash = _sha(destination)
        if retained_hash != source_hash:
            raise ValueError(f"retained native rank file conflicts with source: {destination}")
        return {
            "source": str(source), "retained": str(destination), "status": "already_retained",
            "bytes": destination.stat().st_size, "sha256": retained_hash,
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)
    retained_hash = _sha(destination)
    if retained_hash != source_hash:
        raise ValueError(f"native rank file changed during retention: {source}")
    return {
        "source": str(source),
        "retained": str(destination),
        "status": "retained",
        "bytes": destination.stat().st_size,
        "sha256": retained_hash,
    }


def _score_observation(root: Path, trace_id: str) -> dict[str, Any]:
    """Join optional reward evidence; missing sidecars never block Prime."""
    path = root / "rl-rollouts" / trace_id / "reward.json"
    if not path.is_file():
        return {"status": "unavailable", "reason": "score sidecar not present", "trace_id": trace_id}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"status": "unavailable", "reason": "score sidecar unreadable", "trace_id": trace_id}
    evidence = value.get("score_evidence")
    if not isinstance(evidence, dict):
        return {"status": "unavailable", "reason": "score_evidence missing", "trace_id": trace_id}
    return {
        "status": evidence.get("status", "unknown"),
        "request_id": evidence.get("request_id"),
        "receipt_sha256": evidence.get("receipt_sha256"),
        "timeout": evidence.get("timeout"),
        "trace_id": trace_id,
    }


def record_microbatches(root: Path, micro_batches: list[Any], *, training_step: int,
                        max_off_policy_steps: int = 2,
                        native_rank_files: list[Path] | None = None) -> Path:
    """Retain and summarize the first decoded native batch before forward.

    Prime's decoded ``MicroBatch`` is observational input here.  Optional
    trace/reward metadata can be absent because Prime's TrainSink is the
    authority for those fields; absence is recorded rather than converted to
    a guessed policy revision or group.  This function therefore cannot turn
    an image or JSON reward into a trainer batch and cannot claim a CUDA step.
    """
    if not isinstance(training_step, int) or training_step < 0:
        raise ValueError("training_step must be a non-negative integer")
    summaries: list[dict[str, Any]] = []
    score_observations: list[dict[str, Any]] = []
    rank_files: list[dict[str, Any]] = []
    retention_dir = root / "training-handoffs" / f"step_{training_step}"
    # Retain the original wire file first.  Optional tensor introspection must
    # never make the only replayable artifact disappear (for example on bf16).
    for source in native_rank_files or []:
        rank_files.append(_copy_wire_artifact(Path(source), retention_dir / Path(source).name))
    for rank, micro in enumerate(micro_batches):
        ids = _flat(_field(micro, "input_ids"))
        mask = _flat(_field(micro, "loss_mask"))
        logprobs = _flat(_field(micro, "inference_logprobs"))
        advantages = _flat(_field(micro, "advantages"))
        temperatures = _flat(_field(micro, "temperatures"))
        lengths = _flat(_field(micro, "sequence_lengths"))
        trace_ids = _field(micro, "trace_ids")
        branch_indices = _field(micro, "branch_indices")
        trace_ids = list(trace_ids) if trace_ids is not None else []
        branch_indices = list(branch_indices) if branch_indices is not None else []
        summary = {
            "micro_batch_index": rank,
            "token_count": len(ids),
            "sequence_count": len(lengths),
            "sequence_lengths": lengths,
            "stream_lengths": {
                "input_ids": len(ids), "loss_mask": len(mask),
                "inference_logprobs": len(logprobs), "advantages": len(advantages),
                "temperatures": len(temperatures),
            },
            "trace_ids": [str(value) for value in trace_ids],
            "branch_indices": branch_indices,
            "native_fields_present": {
                name: _field(micro, name) is not None
                for name in ("input_ids", "loss_mask", "inference_logprobs", "advantages",
                             "temperatures", "mm_kwargs", "mm_token_type_ids",
                             "ref_logprobs", "rl_weights", "sampling_mask", "routed_experts")
            },
            "mm_kwargs": {key: _tensor_metadata(value)
                          for key, value in (_field(micro, "mm_kwargs") or {}).items()},
        }
        summaries.append(summary)
        for trace_id in trace_ids:
            score_observations.append(_score_observation(root, str(trace_id)))
    document = {
        "schema": SCHEMA,
        "status": "observed",
        "producer": "prime_rl.train_sink",
        "trainer_boundary": "prime_rl.trainer.rl.train.DataLoader.get_batch",
        "native_batch": True,
        "batch_id": f"step-{training_step}",
        "training_step": training_step,
        "max_off_policy_steps": max_off_policy_steps,
        "captured_before_forward": True,
        "optimizer_step_observed": False,
        "proves_cuda_backward": False,
        "native_transport": "filesystem-msgpack",
        "native_rank_files": rank_files,
        "micro_batches": summaries,
        "score_observations": score_observations,
        "metadata_authority": {
            "policy_revision": "Prime TrainSink; not reconstructed here",
            "groups_and_advantages": "Prime TrainSink/native MicroBatch",
        },
        "limitations": [
            "capture begins after rollout generation and score waits; it does not recover pending traces",
            "JSON is an audit summary; retained rank_N.bin is the replayable native payload",
            "an observation is not evidence of a CUDA forward, backward, or optimizer update",
        ],
        "captured_at": time.time(),
    }
    destination = retention_dir / "observation.json"
    _atomic_json(destination, document)
    _atomic_json(root / HANDOFF_NAME, document)
    return destination


def _pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def runtime_state(root: Path, *, launch_pid: int | None = None) -> dict[str, Any]:
    """Derive operational state from processes, native files, and audit evidence."""
    experiment = {}
    experiment_path = root / "EXPERIMENT.json"
    if experiment_path.is_file():
        try:
            experiment = json.loads(experiment_path.read_text())
        except json.JSONDecodeError:
            experiment = {}
    launch_exit = root / "async-launch.exit"
    setup_exit = root / "training-setup.exit"
    run_id = experiment.get("restart_run_name") or experiment.get("run_name")
    # RLConfig resolves output_dir/run.name; do not accidentally report a
    # stale rank file or audit from another attempt under async-output.
    run_root = root / "async-output" / run_id if isinstance(run_id, str) and run_id else None
    batch_roots = [run_root / "batches"] if run_root is not None else []
    batch_files = sorted({p for batch_root in batch_roots if batch_root.is_dir()
                          for p in batch_root.glob("**/rank_*.bin")})
    optimizer = ([run_root / "artifacts" / "optimizer-update-audit.jsonl"]
                 if run_root is not None else [])
    optimizer = [path for path in optimizer if path.is_file()]
    updates = 0
    for path in optimizer:
        try:
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    report = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(report.get("completed_updates"), int) and report["completed_updates"] > 0:
                    updates += 1
                elif report.get("method") == "exact pre/post native optimizer step CUDA tensor comparison":
                    updates += 1
        except OSError:
            pass
    pid = launch_pid
    pid_path = root / "training-launch.pid"
    if pid is None and pid_path.is_file():
        try:
            pid = int(pid_path.read_text().strip())
        except ValueError:
            pid = None
    alive = _pid_alive(pid)
    if launch_exit.is_file() and launch_exit.read_text().strip() not in {"", "0"}:
        status = "failed"
    elif updates:
        status = "optimizer_evidence_observed"
    elif batch_files:
        status = "first_native_batch_observed"
    elif alive:
        status = "launch_running_waiting_for_native_batch"
    elif launch_exit.is_file():
        status = "finished_without_native_batch"
    elif setup_exit.is_file() and setup_exit.read_text().strip() not in {"", "0"}:
        status = "setup_failed"
    elif (root / "training-ready.json").is_file():
        status = "setup_complete_waiting_for_launch"
    else:
        status = "not_ready"
    reasons = []
    if not batch_files:
        reasons.append("no native filesystem batch observed")
    if not updates:
        reasons.append("no optimizer update audit observed")
    return {
        "schema": STATE_SCHEMA,
        "status": status,
        "run_id": run_id,
        "source_prime_revision": PRIME_REVISION,
        "producer_ready": bool(batch_files),
        "launch_started": alive or bool(batch_files) or bool(updates),
        "trainer_started": bool(batch_files) or bool(updates),
        "first_native_batch": bool(batch_files),
        "optimizer_step_observed": bool(updates),
        "training_process": {"pid": pid, "alive": alive},
        "native_batch": {"observed": bool(batch_files), "files": [str(p) for p in batch_files]},
        "handoff": {"path": str(root / HANDOFF_NAME), "observed": (root / HANDOFF_NAME).is_file()},
        "optimizer": {"audit_files": [str(p) for p in optimizer], "updates_observed": updates},
        "reasons": reasons,
        "updated_at": time.time(),
        "proves_cuda_backward": False,
    }


def write_runtime_state(root: Path, *, launch_pid: int | None = None) -> Path:
    destination = root / STATE_NAME
    _atomic_json(destination, runtime_state(root, launch_pid=launch_pid))
    return destination


def static_preflight(root: Path) -> dict[str, Any]:
    """Verify only the pinned source identity and real capture boundary."""
    source = root / "prime-rl"
    observed_revision = None
    if (source / ".git").exists():
        result = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=False)
        if result.returncode == 0:
            observed_revision = result.stdout.strip()
    if observed_revision is None:
        raise ValueError(f"unable to verify pinned Prime source revision at {source}")
    if observed_revision != PRIME_REVISION:
        raise ValueError(f"Prime source revision {observed_revision} does not match pinned {PRIME_REVISION}")
    return {
        "schema": SCHEMA,
        "status": "source_verified_waiting_for_native_batch",
        "prime_revision": PRIME_REVISION,
        "observed_prime_revision": observed_revision,
        "capture_boundary": "FileSystemBatchReceiver.receive -> DataLoader.get_batch",
        "native_wire_artifact": "batches/step_<step>/rank_<rank>.bin (msgpack)",
        "optimizer_authority": "Prime native TrainSink/trainer; observer does not gate launch",
        "cuda_claim": False,
        "generated_at": time.time(),
    }


def write_static_preflight(root: Path) -> Path:
    destination = root / "training-preflight.json"
    _atomic_json(destination, static_preflight(root))
    return destination


def bootstrap_state(root: Path) -> Path:
    document = runtime_state(root)
    document.update({"status": "setup_complete_waiting_for_launch", "setup_complete": True})
    destination = root / "training-ready.json"
    _atomic_json(destination, document)
    write_runtime_state(root)
    return destination


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("bootstrap", "preflight", "state"):
        command = sub.add_parser(name)
        command.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "bootstrap":
        print(bootstrap_state(root))
    elif args.command == "preflight":
        print(write_static_preflight(root))
    else:
        print(write_runtime_state(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
