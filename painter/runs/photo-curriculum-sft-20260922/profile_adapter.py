"""Adapter shape checks used by the DDP fresh-base hook."""

from __future__ import annotations

import torch


TENSORS = 992
PARAMETERS = 116727808


def audit_fresh_adapter(state: dict) -> dict:
    values = {name: value.full_tensor() if hasattr(value, "full_tensor") else value for name, value in state.items()}
    a_names = {name for name in values if ".lora_A." in name}
    b_names = {name for name in values if ".lora_B." in name}
    if len(values) != TENSORS or len(a_names) != TENSORS // 2 or len(b_names) != TENSORS // 2 or a_names | b_names != set(values):
        raise ValueError(f"fresh adapter tensor set differs from pinned profile: {len(values)}")
    if {name.replace(".lora_A.", ".lora_B.") for name in a_names} != b_names:
        raise ValueError("fresh adapter A/B names are not paired")
    if sum(value.numel() for value in values.values()) != PARAMETERS:
        raise ValueError("fresh adapter parameter count differs from pinned profile")
    for name, value in values.items():
        if value.ndim != 2 or value.shape[0 if name in a_names else 1] != 16:
            raise ValueError(f"wrong LoRA rank: {name} {tuple(value.shape)}")
        if not value.is_floating_point() or not torch.isfinite(value).all().item():
            raise ValueError(f"non-finite fresh adapter tensor: {name}")
        nonzero = bool(torch.count_nonzero(value).item())
        if name in b_names and nonzero:
            raise ValueError(f"fresh LoRA B is not zero: {name}")
        if name in a_names and not nonzero:
            raise ValueError(f"fresh LoRA A is zero: {name}")
    return {"tensor_count": TENSORS, "parameters": PARAMETERS, "all_finite": True,
            "a_tensors": TENSORS // 2, "b_tensors": TENSORS // 2,
            "zero_b_tensors": TENSORS // 2, "nonzero_a_tensors": TENSORS // 2,
            "initial_lora_contribution": "exactly_zero"}
