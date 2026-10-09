#!/usr/bin/env python3
"""Allocate one two-H100 offer for the finite structured-flow pilot."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


RUN = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name}: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


alloc = _load("structured_flow_allocator_base", RUN.parent / "throughput-recovery-20261003" / "allocate_single.py")
operation = _load("structured_flow_operation", RUN / "operate_single.py")

GPU_COUNT = 2
GPU_TYPE = "H100"
MAX_CAP_USD = 20.0
RUN_NAME = "structured-flow-rl-20261009"


def _remote_staging_root() -> Path:
    """Return the node-local staging root selected for this launch.

    The controller and node share the same absolute scratch convention.  The
    opt-in value is already forwarded by ``operate_single``; using it here
    keeps the large Prime/source bundles off the encrypted node-root mount
    before the node-side checkout begins.  The unset path preserves the
    generic controller's historical ``/root/painter`` behavior.
    """
    raw = os.environ.get("PAINTER_LOCAL_SCRATCH_ROOT", "").strip()
    if not raw:
        return Path(operation.base.REMOTE_ROOT)
    if not raw.startswith("/"):
        raise ValueError("PAINTER_LOCAL_SCRATCH_ROOT must be absolute")
    return Path(raw)


def _stage_prime_bundle(*, api, operation_module, window, prime_bundle: Path, prime_commit: str, prime_sha256: str, now=time.time) -> Path:
    """Run the reviewed staging sequence against the selected filesystem."""
    operation_base = operation_module.base
    staging_root = _remote_staging_root()
    previous_root = operation_base.REMOTE_ROOT
    operation_base.REMOTE_ROOT = staging_root
    try:
        alloc.stage_prime_bundle(
            api=api,
            operation=operation_base,
            window=window,
            prime_bundle=prime_bundle,
            prime_commit=prime_commit,
            prime_sha256=prime_sha256,
            now=now,
        )
    finally:
        operation_base.REMOTE_ROOT = previous_root
    return staging_root


def allocate_and_operate(*, api, max_price_usd_hour: float, cap_usd: float, source_bundle: Path, source_commit: str, prime_bundle: Path, prime_commit: str, prime_sha256: str, hf_token_file: Path, gateway_env_file: Path, output_root: Path, now=time.time) -> int:
    source_bundle = alloc._regular(source_bundle, "source-bundle")
    prime_bundle = alloc._regular(prime_bundle, "prime-bundle")
    token = alloc._regular(hf_token_file, "HF token file", owner_only=True)
    gateway = alloc._regular(gateway_env_file, "Gateway environment file", owner_only=True)
    if not operation.base.COMMIT_RE.fullmatch(source_commit) or not operation.base.COMMIT_RE.fullmatch(prime_commit):
        raise ValueError("source and Prime commits must be immutable full hashes")
    maximum, cap = alloc._positive("max-price-usd-hour", max_price_usd_hour), alloc._positive("cap-usd", cap_usd)
    if cap > MAX_CAP_USD:
        raise ValueError("cap-usd must be at most 20")
    source_sha = alloc.verify_source_bundle(source_bundle, source_commit)
    prime_sha = alloc.verify_prime_bundle(prime_bundle, prime_commit, prime_sha256)
    output_root = output_root.expanduser().resolve()
    if output_root.exists() and output_root.is_symlink():
        raise ValueError("output-root must not be a symlink")
    output_root.mkdir(parents=True, exist_ok=True)
    output_root.chmod(0o700)
    offers = operation.base._bounded_call(lambda: api.ls(gpu_type=GPU_TYPE, gpu_count=GPU_COUNT), alloc.OFFER_TIMEOUT_SECONDS)
    offer = alloc.select_offer(offers, max_price_usd_hour=maximum)
    price = alloc._offer_price(offer)
    started = float(now())
    if not math.isfinite(started) or started <= 0:
        raise ValueError("rental start time must be positive and finite")
    executor_id = alloc._offer_id(offer)
    allocation = operation.base._bounded_call(lambda: api.up(executor_id=executor_id, name=RUN_NAME, enable_volume_encryption=True), alloc.ALLOCATION_TIMEOUT_SECONDS)
    pod_id = alloc._pod_id(allocation)
    if not pod_id:
        raise RuntimeError("allocation returned no exact pod id")
    release_target = SimpleNamespace(id=pod_id)
    tracked = alloc._ReleaseTrackingAPI(api)
    receipt_path = output_root / "allocation-receipt.json"
    receipt = {"schema": "structured-flow-single-allocation-v1", "status": "allocated", "rental_start_epoch": started, "price_usd_per_hour": price, "cap_usd": cap, "source_commit": source_commit, "source_bundle_sha256": source_sha, "prime_commit": prime_commit, "prime_bundle_sha256": prime_sha, "topology": {"gpus_per_node": 2, "num_train_gpus": 1, "num_infer_gpus": 1}}
    try:
        alloc._write_receipt(receipt_path, receipt)
        ready = alloc.wait_ready_bounded(api=api, operation=operation.base, allocation=allocation)
        if not alloc._ready_pod(ready):
            raise RuntimeError("allocated pod did not become SSH-ready")
        window = operation.base.make_window(pod_id=pod_id, rental_start_epoch=started, price_usd_per_hour=price, cap_usd=cap)
        staging_root = _stage_prime_bundle(
            api=api,
            operation_module=operation,
            window=window,
            prime_bundle=prime_bundle,
            prime_commit=prime_commit,
            prime_sha256=prime_sha,
            now=now,
        )
        receipt["status"] = "prime_staged"
        alloc._write_receipt(receipt_path, receipt)
        previous_bundle = operation.base.REMOTE_BUNDLE
        operation.base.REMOTE_BUNDLE = staging_root / "source.bundle"
        try:
            return operation.base.operate(window=window, source_bundle=source_bundle, source_commit=source_commit, hf_token_file=token, gateway_env_file=gateway, output_root=output_root, api=tracked)
        finally:
            operation.base.REMOTE_BUNDLE = previous_bundle
    finally:
        if not tracked.released:
            try:
                operation.base._bounded_call(lambda: api.rm(release_target), alloc.RELEASE_TIMEOUT_SECONDS)
                tracked.released = True
            except BaseException:
                pass
        receipt["status"] = "released" if tracked.released else "release_attempted"
        try:
            alloc._write_receipt(receipt_path, receipt)
        except BaseException:
            pass


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--max-price-usd-hour", type=float, required=True)
    value.add_argument("--cap-usd", type=float, required=True)
    value.add_argument("--source-bundle", type=Path, required=True)
    value.add_argument("--source-commit", required=True)
    value.add_argument("--prime-bundle", type=Path, required=True)
    value.add_argument("--prime-commit", required=True)
    value.add_argument("--prime-sha256", required=True)
    value.add_argument("--hf-token-file", type=Path, default=alloc.Path.home() / ".cache/huggingface/token")
    value.add_argument("--gateway-env-file", type=Path, default=operation.base.default_gateway_env_file())
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--plan-only", action="store_true")
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.plan_only:
        source = alloc._regular(args.source_bundle, "source-bundle")
        prime = alloc._regular(args.prime_bundle, "prime-bundle")
        print(json.dumps({"schema": "structured-flow-single-allocation-plan-v1", "run_name": RUN_NAME, "gpu_type": GPU_TYPE, "gpu_count": GPU_COUNT, "cap_usd": args.cap_usd, "max_price_usd_hour": args.max_price_usd_hour, "source_bundle_bytes": source.stat().st_size, "prime_bundle_bytes": prime.stat().st_size, "credentials_validated": False, "provider_actions": "disabled in plan-only"}, sort_keys=True))
        return 0
    args.source_bundle = alloc._regular(args.source_bundle, "source-bundle")
    args.prime_bundle = alloc._regular(args.prime_bundle, "prime-bundle")
    token = alloc._regular(args.hf_token_file, "HF token file", owner_only=True)
    gateway = alloc._regular(args.gateway_env_file, "Gateway environment file", owner_only=True)
    api = alloc._load_lium()
    return allocate_and_operate(api=api, max_price_usd_hour=args.max_price_usd_hour, cap_usd=args.cap_usd, source_bundle=args.source_bundle, source_commit=args.source_commit, prime_bundle=args.prime_bundle, prime_commit=args.prime_commit, prime_sha256=args.prime_sha256, hf_token_file=token, gateway_env_file=gateway, output_root=args.output_root)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"structured-flow allocation: {exc}", file=sys.stderr)
        raise SystemExit(2)
