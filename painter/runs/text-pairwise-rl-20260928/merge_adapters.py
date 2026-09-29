"""Build an exact effective-weight interpolation of two compatible LoRAs.

Run on the Linux evaluation node. This does not merge the 27B base weights.
For ordinary LoRA, delta W = (alpha / rank) * B @ A. Concatenating the
factors at twice the rank and doubling alpha preserves that scaling exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def merge(sft: Path, rl: Path, output: Path, rl_fraction: float) -> dict:
    if not 0 < rl_fraction < 1:
        raise ValueError("rl_fraction must be strictly between 0 and 1; use original adapters for endpoints")
    if output.exists():
        raise FileExistsError(output)
    configs = [json.loads((folder / "adapter_config.json").read_text()) for folder in (sft, rl)]
    first, second = configs
    for field in ("r", "lora_alpha", "target_modules", "base_model_name_or_path", "peft_type", "use_rslora", "rank_pattern", "alpha_pattern", "fan_in_fan_out"):
        if first.get(field) != second.get(field):
            raise ValueError(f"incompatible adapter config: {field}")
    if first.get("peft_type") != "LORA" or first.get("use_rslora") or first.get("rank_pattern") or first.get("alpha_pattern"):
        raise ValueError("only uniform standard LoRA is supported")
    if first.get("bias") not in (None, "none") or first.get("modules_to_save"):
        raise ValueError("non-LoRA trainable tensors need a separate merge rule")
    rank = int(first["r"])
    if rank <= 0 or float(first["lora_alpha"]) <= 0:
        raise ValueError("invalid LoRA scale")
    weights = [folder / "adapter_model.safetensors" for folder in (sft, rl)]
    source = [load_file(str(path), device="cpu") for path in weights]
    if source[0].keys() != source[1].keys():
        raise ValueError("adapter tensor names differ")
    merged: dict[str, torch.Tensor] = {}
    a_suffix, b_suffix = ".lora_A.weight", ".lora_B.weight"
    for key in source[0]:
        left, right = source[0][key], source[1][key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"adapter tensor mismatch: {key}")
        if key.endswith(a_suffix):
            if left.ndim != 2 or left.shape[0] != rank:
                raise ValueError(f"unexpected A shape: {key}")
            merged[key] = torch.cat((left, right), dim=0).contiguous()
        elif key.endswith(b_suffix):
            if left.ndim != 2 or left.shape[1] != rank:
                raise ValueError(f"unexpected B shape: {key}")
            merged[key] = torch.cat((left * (1 - rl_fraction), right * rl_fraction), dim=1).contiguous()
        else:
            raise ValueError(f"unsupported trainable tensor: {key}")
    if not merged or sum(k.endswith(a_suffix) for k in merged) != sum(k.endswith(b_suffix) for k in merged):
        raise ValueError("missing or unpaired LoRA factors")
    for key in merged:
        if key.endswith(a_suffix) and key[:-len(a_suffix)] + b_suffix not in merged:
            raise ValueError(f"missing paired B tensor: {key}")
    config = dict(first)
    config["r"] = rank * 2
    config["lora_alpha"] = float(first["lora_alpha"]) * 2
    output.mkdir(parents=True)
    save_file(merged, str(output / "adapter_model.safetensors"))
    (output / "adapter_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema": "exact-lora-effective-delta-interpolation-v1",
        "rl_fraction": rl_fraction,
        "sft_adapter_sha256": sha256(weights[0]),
        "rl_adapter_sha256": sha256(weights[1]),
        "merged_adapter_sha256": sha256(output / "adapter_model.safetensors"),
        "rank": config["r"],
        "lora_alpha": config["lora_alpha"],
    }
    (output / "merge_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--rl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rl-fraction", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.sft, args.rl, args.output, args.rl_fraction), sort_keys=True))


if __name__ == "__main__":
    main()
