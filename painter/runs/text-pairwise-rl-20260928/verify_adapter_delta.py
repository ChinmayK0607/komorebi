#!/usr/bin/env python3
"""Measure a LoRA checkpoint change from the frozen starting adapter."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measure(start: Path, end: Path) -> dict:
    # Profiling must not seize the renderer/trainer node's CPU workers.
    torch.set_num_threads(1)
    start = start / "adapter_model.safetensors" if start.is_dir() else start
    end = end / "adapter_model.safetensors" if end.is_dir() else end
    if not start.is_file() or not end.is_file():
        raise FileNotFoundError("both adapter tensor files are required")
    start_sq = end_sq = delta_sq = 0.0
    changed = total = 0
    changed_elements = total_elements = 0
    absolute_delta_sum = 0.0
    dtypes: Counter[str] = Counter()
    max_delta = 0.0
    with safe_open(str(start), framework="pt", device="cpu") as before, safe_open(
            str(end), framework="pt", device="cpu") as after:
        if set(before.keys()) != set(after.keys()):
            raise ValueError("adapter tensor names changed")
        for key in before.keys():
            a = before.get_tensor(key)
            b = after.get_tensor(key)
            if a.shape != b.shape:
                raise ValueError(f"adapter tensor shape changed: {key}")
            if a.dtype != b.dtype:
                raise ValueError(f"adapter tensor dtype changed: {key}")
            dtypes[str(a.dtype)] += 1
            a = a.to(torch.float64)
            b = b.to(torch.float64)
            diff = b - a
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError(f"nonfinite adapter weights: {key}")
            start_sq += float(torch.sum(a * a))
            end_sq += float(torch.sum(b * b))
            delta_sq += float(torch.sum(diff * diff))
            max_delta = max(max_delta, float(torch.max(torch.abs(diff))))
            changed += int(bool(torch.any(diff != 0)))
            changed_elements += int(torch.count_nonzero(diff))
            total_elements += diff.numel()
            absolute_delta_sum += float(torch.sum(torch.abs(diff)))
            total += 1
    start_l2 = math.sqrt(start_sq)
    delta_l2 = math.sqrt(delta_sq)
    return {"schema": "painter.adapter-delta.v1",
            "start_sha256": sha(start), "end_sha256": sha(end),
            "tensor_count": total, "changed_tensor_count": changed,
            "tensor_dtypes": dict(dtypes),
            "total_elements": total_elements,
            "changed_elements": changed_elements,
            "changed_element_fraction": changed_elements / total_elements if total_elements else None,
            "mean_absolute_delta_over_changed_elements": (
                absolute_delta_sum / changed_elements if changed_elements else None),
            "start_l2": start_l2, "end_l2": math.sqrt(end_sq),
            "delta_l2": delta_l2,
            "relative_delta_l2": delta_l2 / start_l2 if start_l2 else None,
            "max_absolute_parameter_delta": max_delta,
            "interpretation": "Weight movement only; this does not establish visual improvement."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--end", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = measure(args.start, args.end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"changed_tensors": result["changed_tensor_count"],
                      "relative_delta_l2": result["relative_delta_l2"]}))


if __name__ == "__main__":
    main()
