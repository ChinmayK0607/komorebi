#!/usr/bin/env python3
"""Allocate one reviewed hardware-profile offer for the finite pilot."""
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
if str(RUN) not in sys.path:
    sys.path.insert(0, str(RUN))
import hardware_profile


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name}: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


alloc = _load("diverse_flow_allocator_base", RUN.parent / "throughput-recovery-20261003" / "allocate_single.py")
operation = _load("diverse_flow_operation", RUN / "operate_single.py")

HARDWARE_PROFILE = hardware_profile.selected_name()
_PROFILE = hardware_profile.get(HARDWARE_PROFILE)
GPU_COUNT = _PROFILE["gpu_count"]
GPU_TYPE = _PROFILE["gpu_type"]
MIN_GPU_MEMORY_MIB = _PROFILE["min_gpu_memory_mib"]
# Lium prices are whole-node prices.  Keep each profile's boundary strict:
# the preferred L40S offer is $1.52/hr, while the available H100 offer is
# $3.70/hr and must remain below its reviewed $4.00/hr ceiling.
MAX_OFFER_USD_HOUR = _PROFILE["max_offer_usd_hour_exclusive"]
MAX_CAP_USD = 20.0
RUN_NAME = "diverse-flow-rl-20261009"


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


def _nested_values(offer, names: tuple[str, ...]):
    """Yield SDK fields from an offer and its small resource/spec mappings.

    Lium SDK releases have exposed the same offer data as either attributes,
    flat dictionaries, or a nested ``specs``/``resources`` mapping.  Keep the
    compatibility shim local to this run; the reviewed allocator's H100
    globals must not decide a Diverse topology.
    """
    pending = [offer]
    seen: set[int] = set()
    while pending:
        current = pending.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, (list, tuple)):
            for name in names:
                collected = []
                for item in current:
                    if isinstance(item, dict) and name in item:
                        collected.append(item[name])
                    elif not isinstance(item, dict):
                        value = getattr(item, name, None)
                        if value is not None:
                            collected.append(value)
                if collected:
                    # A GPU-details list is one offer's per-device capacity;
                    # callers reduce it to the weakest device.
                    yield collected
            pending.extend(item for item in current if isinstance(item, (dict, list, tuple)) or hasattr(item, "__dict__"))
            continue
        for name in names:
            if isinstance(current, dict):
                if name in current:
                    yield current[name]
            else:
                value = getattr(current, name, None)
                if value is not None:
                    yield value
        # ``specs.gpu.details`` is the current Lium SDK shape; retain the
        # older resource/configuration paths for archived SDK payloads.
        for key in ("specs", "resources", "gpu", "hardware", "configuration", "details"):
            if isinstance(current, dict):
                child = current.get(key)
            else:
                child = getattr(current, key, None)
            if isinstance(child, (dict, list, tuple)) or hasattr(child, "__dict__"):
                pending.append(child)


def _parse_memory_mib(value) -> float:
    """Normalize common Lium capacity forms to MiB."""
    if isinstance(value, bool):
        raise ValueError("boolean is not GPU capacity")
    if isinstance(value, (list, tuple)):
        values = [_parse_memory_mib(item) for item in value]
        if not values:
            raise ValueError("empty GPU capacity")
        return min(values)
    if isinstance(value, dict):
        for key in ("mib", "memory_mib", "capacity_mib", "bytes", "value"):
            if key in value:
                return _parse_memory_mib(value[key])
        raise ValueError("unrecognized GPU capacity mapping")
    if isinstance(value, (int, float)):
        number = float(value)
        if number <= 0:
            raise ValueError("GPU capacity must be positive")
        # SDK numeric byte capacities are tens of billions; small numbers are
        # conventionally already MiB.
        return number / (1024 * 1024) if number > 10_000_000 else number
    text = str(value).strip().replace(",", "")
    match = __import__("re").fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)?", text)
    if not match:
        raise ValueError(f"unrecognized GPU capacity: {value!r}")
    number = float(match.group(1))
    unit = (match.group(2) or "mib").lower()
    if unit in {"mib", "mi"}:
        return number
    if unit in {"gib", "gi"}:
        return number * 1024
    if unit in {"gb", "g"}:
        # Provider-readable "48 GB" denotes the 48-GiB class in the offer
        # table; treating it as binary preserves the hardware threshold.
        return number * 1024
    if unit in {"mb", "m"}:
        return number * (1000 / 1024)
    if unit in {"bytes", "byte", "b"}:
        return number / (1024 * 1024)
    raise ValueError(f"unrecognized GPU capacity unit: {unit!r}")


def _offer_gpu_memory_mib(offer) -> float:
    # Prefer GPU-specific fields before falling back to a nested ``memory``
    # label.  Some offers also report host RAM as top-level memory; consuming
    # that before a nested GPU capacity would admit an undersized node.
    specific = ("gpu_memory_mib", "memory_mib", "gpu_memory_mb", "memory_mb",
                "gpu_memory", "gpu_ram", "memory_gb")
    for fields in (specific, ("memory", "capacity")):
        for value in _nested_values(offer, fields):
            try:
                return _parse_memory_mib(value)
            except (TypeError, ValueError):
                continue
    raise ValueError("offer does not expose GPU memory capacity")


def _offer_gpu_count(offer) -> int:
    for value in _nested_values(offer, ("gpu_count", "num_gpus", "gpus", "count")):
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    # Current SDK payloads omit a top-level count and expose one detail row
    # per device below specs.gpu.details.  Count that row list only as a
    # fallback; scalar host ``count`` fields above retain precedence.
    for value in _nested_values(offer, ("name",)):
        if isinstance(value, (list, tuple)) and value:
            return len(value)
    return 0


def _offer_type(offer, gpu_type: str = GPU_TYPE) -> str:
    fallback = ""
    for value in _nested_values(offer, ("gpu_type", "gpu_name", "gpu_model", "accelerator", "name")):
        values = value if isinstance(value, (list, tuple)) else (value,)
        for item in values:
            text = str(item or "").strip().upper()
            if not text:
                continue
            if gpu_type in text:
                return text
            fallback = fallback or text
    if fallback:
        return fallback
    return ""


def _select_profile_offer(offers, *, profile_name: str, max_price_usd_hour: float):
    """Select an exact-card-count/type offer meeting the profile floor."""
    profile = hardware_profile.get(profile_name)
    max_offer = profile["max_offer_usd_hour_exclusive"]
    maximum = alloc._positive("max-price-usd-hour", max_price_usd_hour)
    if maximum >= max_offer:
        raise ValueError(f"max-price-usd-hour must be strictly below {max_offer:.2f}")
    candidates = []
    for offer in offers:
        if _offer_gpu_count(offer) != profile["gpu_count"] or profile["gpu_type"] not in _offer_type(offer, profile["gpu_type"]):
            continue
        try:
            memory = _offer_gpu_memory_mib(offer)
            price = alloc._offer_price(offer)
        except (TypeError, ValueError):
            continue
        if memory < profile["min_gpu_memory_mib"] or price >= max_offer or price > maximum:
            continue
        candidates.append((-memory, price, alloc._offer_id(offer), offer))
    if not candidates:
        raise RuntimeError(
            f"no {profile_name} offer with exactly {profile['gpu_count']} {profile['gpu_type']} card(s) "
            f"and at least {profile['min_gpu_memory_mib']} MiB per GPU is below the hourly price limit"
        )
    # Capacity first avoids accidentally choosing the cheaper 46-GiB offer
    # when a full 48-GiB-class node is available; price then breaks ties.
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][3]


def _select_l40s_offer(offers, *, max_price_usd_hour: float):
    """Compatibility spelling for the default four-L40S selector."""
    return _select_profile_offer(offers, profile_name="l40s_4", max_price_usd_hour=max_price_usd_hour)


def _select_h100_offer(offers, *, max_price_usd_hour: float):
    """Select exactly two H100 cards with the reviewed 80-GB floor."""
    return _select_profile_offer(offers, profile_name="h100_2", max_price_usd_hour=max_price_usd_hour)


# Public run-local spelling for offline callers and selector audits.  It is
# deliberately not imported from the shared two-H100 allocator.
select_offer = _select_l40s_offer if HARDWARE_PROFILE == "l40s_4" else _select_h100_offer


def allocate_and_operate(*, api, max_price_usd_hour: float, cap_usd: float, source_bundle: Path, source_commit: str, prime_bundle: Path, prime_commit: str, prime_sha256: str, hf_token_file: Path, gateway_env_file: Path, output_root: Path, now=time.time, profile_name: str | None = None) -> int:
    selected_profile = hardware_profile.selected_name(profile_name)
    if selected_profile != HARDWARE_PROFILE:
        raise ValueError("hardware profile must be selected before loading the allocator")
    profile = hardware_profile.get(selected_profile)
    hardware_profile.require_resume(selected_profile)
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
    max_offer = profile["max_offer_usd_hour_exclusive"]
    if maximum >= max_offer:
        raise ValueError(f"max-price-usd-hour must be strictly below {max_offer:.2f}")
    offers = operation.base._bounded_call(lambda: api.ls(gpu_type=profile["gpu_type"], gpu_count=profile["gpu_count"]), alloc.OFFER_TIMEOUT_SECONDS)
    offer = _select_profile_offer(offers, profile_name=selected_profile, max_price_usd_hour=maximum)
    price = alloc._offer_price(offer)
    offer_memory = _offer_gpu_memory_mib(offer)
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
    receipt = {"schema": "Diverse-flow-single-allocation-v1", "status": "allocated", "hardware_profile": selected_profile, "rental_start_epoch": started, "price_usd_per_hour": price, "cap_usd": cap, "source_commit": source_commit, "source_bundle_sha256": source_sha, "prime_commit": prime_commit, "prime_bundle_sha256": prime_sha, "offer_requirements": {"gpu_type": profile["gpu_type"], "gpu_count": profile["gpu_count"], "min_gpu_memory_mib": profile["min_gpu_memory_mib"], "max_price_usd_per_hour_exclusive": profile["max_offer_usd_hour_exclusive"], "selected_gpu_memory_mib": offer_memory}, "topology": {"gpus_per_node": profile["gpus_per_node"], "num_train_gpus": profile["num_train_gpus"], "num_infer_gpus": profile["num_infer_gpus"], "inference_tensor_parallel": profile["inference_tensor_parallel"]}}
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
    value.add_argument("--hardware-profile", choices=tuple(sorted(hardware_profile.PROFILES)))
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    selected_profile = hardware_profile.selected_name(args.hardware_profile)
    if selected_profile != HARDWARE_PROFILE:
        raise ValueError("--hardware-profile must match PAINTER_HARDWARE_PROFILE selected at process start")
    profile = hardware_profile.get(selected_profile)
    if args.plan_only:
        source = alloc._regular(args.source_bundle, "source-bundle")
        prime = alloc._regular(args.prime_bundle, "prime-bundle")
        if args.max_price_usd_hour >= profile["max_offer_usd_hour_exclusive"]:
            raise ValueError(f"max-price-usd-hour must be strictly below {profile['max_offer_usd_hour_exclusive']:.2f}")
        print(json.dumps({"schema": "Diverse-flow-single-allocation-plan-v1", "run_name": RUN_NAME, "hardware_profile": selected_profile, "gpu_type": profile["gpu_type"], "gpu_count": profile["gpu_count"], "min_gpu_memory_mib": profile["min_gpu_memory_mib"], "max_offer_usd_hour_exclusive": profile["max_offer_usd_hour_exclusive"], "cap_usd": args.cap_usd, "max_price_usd_hour": args.max_price_usd_hour, "source_bundle_bytes": source.stat().st_size, "prime_bundle_bytes": prime.stat().st_size, "credentials_validated": False, "provider_actions": "disabled in plan-only"}, sort_keys=True))
        return 0
    args.source_bundle = alloc._regular(args.source_bundle, "source-bundle")
    args.prime_bundle = alloc._regular(args.prime_bundle, "prime-bundle")
    token = alloc._regular(args.hf_token_file, "HF token file", owner_only=True)
    gateway = alloc._regular(args.gateway_env_file, "Gateway environment file", owner_only=True)
    api = alloc._load_lium()
    return allocate_and_operate(api=api, max_price_usd_hour=args.max_price_usd_hour, cap_usd=args.cap_usd, source_bundle=args.source_bundle, source_commit=args.source_commit, prime_bundle=args.prime_bundle, prime_commit=args.prime_commit, prime_sha256=args.prime_sha256, hf_token_file=token, gateway_env_file=gateway, output_root=args.output_root, profile_name=selected_profile)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Diverse-flow allocation: {exc}", file=sys.stderr)
        raise SystemExit(2)
