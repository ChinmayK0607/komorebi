#!/usr/bin/env python3
"""Allocate one H100 pair and run the finite matched campaign.

All source and Prime bundle checks happen before ``up``.  The exact returned
pod object is held through the outer ``finally`` and released after archive
attempts, including handoff and transport failures.  This module reuses the
reviewed allocator's bounded SDK/SSH/SFTP helpers without changing its
single-arm $20 defaults.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from typing import Any, Callable, Sequence


RUN = Path(__file__).resolve().parent


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


alloc = _load("matched_allocator_base", RUN.parent / "throughput-recovery-20261003" / "allocate_single.py")
operation = _load("matched_operation", RUN / "operate_campaign.py")

RUN_NAME = "rl-judge-matched-20261004"
GPU_COUNT = 2
GPU_TYPE = "H100"
MAX_CAP_USD = 58.5


def allocate_and_operate(*, api: Any, max_price_usd_hour: float, cap_usd: float,
                         source_bundle: Path, source_commit: str, prime_bundle: Path,
                         prime_commit: str, prime_sha256: str, hf_token_file: Path,
                         gateway_env_file: Path, output_root: Path,
                         now: Callable[[], float] = time.time) -> int:
    source_bundle = alloc._regular(source_bundle, "source-bundle")
    prime_bundle = alloc._regular(prime_bundle, "prime-bundle")
    hf_token_file = alloc._regular(hf_token_file, "HF token file", owner_only=True)
    gateway_env_file = alloc._regular(gateway_env_file, "Gateway environment file", owner_only=True)
    if not operation.base.COMMIT_RE.fullmatch(source_commit):
        raise ValueError("source-commit must be a full immutable Git commit")
    if not operation.base.COMMIT_RE.fullmatch(prime_commit):
        raise ValueError("prime-commit must be a full immutable Git commit")
    cap = alloc._positive("cap-usd", cap_usd)
    if cap > MAX_CAP_USD:
        raise ValueError("campaign cap must be at most 58.50")
    maximum = alloc._positive("max-price-usd-hour", max_price_usd_hour)
    source_sha = alloc.verify_source_bundle(source_bundle, source_commit)
    prime_sha = alloc.verify_prime_bundle(prime_bundle, prime_commit, prime_sha256)
    output_root = output_root.expanduser()
    if output_root.exists() and output_root.is_symlink():
        raise ValueError("output-root must not be a symlink")
    output_root.mkdir(parents=True, exist_ok=True)
    output_root.chmod(0o700)
    offers = operation.base._bounded_call(
        lambda: api.ls(gpu_type=GPU_TYPE, gpu_count=GPU_COUNT), alloc.OFFER_TIMEOUT_SECONDS
    )
    offer = alloc.select_offer(offers, max_price_usd_hour=maximum)
    price = alloc._offer_price(offer)
    started = float(now())
    if not math.isfinite(started) or started <= 0:
        raise ValueError("rental start time must be positive and finite")
    allocation = operation.base._bounded_call(
        lambda: api.up(executor_id=alloc._offer_id(offer), name=RUN_NAME, enable_volume_encryption=True),
        alloc.ALLOCATION_TIMEOUT_SECONDS,
    )
    pod_id = alloc._pod_id(allocation)
    if not pod_id:
        raise RuntimeError("provider allocation did not return an exact pod id")
    release_target: Any = SimpleNamespace(id=pod_id)
    tracked = alloc._ReleaseTrackingAPI(api)
    receipt_path = output_root / "allocation-receipt.json"
    receipt = {
        "schema": "matched-three-judge-allocation-v1", "status": "allocated",
        "rental_start_epoch": started, "price_usd_per_hour": price, "cap_usd": cap,
        "source_commit": source_commit, "source_bundle_sha256": source_sha,
        "prime_commit": prime_commit, "prime_bundle_sha256": prime_sha,
        "topology": {"gpus_per_node": 2, "num_train_gpus": 1, "num_infer_gpus": 1},
    }
    try:
        alloc._write_receipt(receipt_path, receipt)
        ready = alloc.wait_ready_bounded(api=api, operation=operation.base, allocation=allocation)
        if not alloc._ready_pod(ready):
            raise RuntimeError("allocated pod did not become SSH-ready")
        window = operation.make_campaign_window(pod_id=pod_id, rental_start_epoch=started,
                                                price_usd_per_hour=price, cap_usd=cap)
        alloc.stage_prime_bundle(api=api, operation=operation.base, window=window,
                                 prime_bundle=prime_bundle, prime_commit=prime_commit,
                                 prime_sha256=prime_sha, now=now)
        receipt["status"] = "prime_staged"
        alloc._write_receipt(receipt_path, receipt)
        return operation.base.operate(window=window, source_bundle=source_bundle,
                                      source_commit=source_commit, hf_token_file=hf_token_file,
                                      gateway_env_file=gateway_env_file, output_root=output_root,
                                      api=tracked)
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


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-price-usd-hour", type=float, default=3.8)
    parser.add_argument("--cap-usd", type=float, default=58.5)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--prime-bundle", type=Path, required=True)
    parser.add_argument("--prime-commit", default="26b3131d2716a4f8210b165df584b83b4bc54f61")
    parser.add_argument("--prime-sha256", default="64b035e176df1aa850280f9bcdad5cedef8108636b59a5c29f2af8b014cf3dbe")
    parser.add_argument("--hf-token-file", type=Path, default=operation.base.default_hf_token_file())
    parser.add_argument("--gateway-env-file", type=Path, default=operation.base.default_gateway_env_file())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    source = alloc._regular(args.source_bundle, "source-bundle")
    prime = alloc._regular(args.prime_bundle, "prime-bundle")
    token = alloc._regular(args.hf_token_file, "HF token file", owner_only=True)
    gateway = alloc._regular(args.gateway_env_file, "Gateway environment file", owner_only=True)
    if not operation.base.COMMIT_RE.fullmatch(args.source_commit) or not operation.base.COMMIT_RE.fullmatch(args.prime_commit):
        raise SystemExit("source and Prime commits must be immutable full hashes")
    if args.plan_only:
        print(json.dumps({"schema": "matched-three-judge-allocation-plan-v1", "run_name": RUN_NAME,
                          "gpu_type": GPU_TYPE, "gpu_count": GPU_COUNT, "max_price_usd_hour": args.max_price_usd_hour,
                          "campaign_cap_usd": args.cap_usd, "per_arm_cap_usd": args.cap_usd / 3,
                          "prime_commit": args.prime_commit, "source_bundle_bytes": source.stat().st_size,
                          "prime_bundle_bytes": prime.stat().st_size, "credential_paths_validated": True}, sort_keys=True))
        return 0
    api = alloc._load_lium()
    return allocate_and_operate(api=api, max_price_usd_hour=args.max_price_usd_hour,
                                cap_usd=args.cap_usd, source_bundle=source,
                                source_commit=args.source_commit, prime_bundle=prime,
                                prime_commit=args.prime_commit, prime_sha256=args.prime_sha256,
                                hf_token_file=token, gateway_env_file=gateway,
                                output_root=args.output_root)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (operation.base.OperationError, OSError, ValueError, RuntimeError) as exc:
        print(f"matched allocation: {exc}", file=sys.stderr)
        raise SystemExit(2)
