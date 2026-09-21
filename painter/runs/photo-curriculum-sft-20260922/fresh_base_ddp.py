"""Fresh-base LoRA initialization that is safe for multiple DDP ranks.

Prime calls the reset hook independently on every rank.  Each rank validates
its native adapter state; the rank-zero export hook writes the public step-0
receipt while its internal barriers keep all ranks synchronized.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path


def install(config, trainer, export_step_zero, profile_module) -> None:
    from prime_rl.trainer.lora import LoRAState
    from prime_rl.trainer.world import get_world

    if config.resume is not None or config.model.lora is None:
        raise ValueError("fresh-base initialization requires a fresh optimizer and LoRA")
    if not callable(export_step_zero):
        raise ValueError("fresh-base initialization requires a verified step-0 export")
    expected = profile_module.verify_snapshot(config.model.name)
    rank = get_world().rank
    world_size = get_world().world_size
    namespace = getattr(inspect.unwrap(trainer), "__globals__", {})
    original_setup = namespace.get("setup_model")
    if not callable(original_setup):
        raise ValueError("pinned Prime trainer does not expose setup_model")
    original_reset = LoRAState.reset_adapter_parameters
    model = None
    reset_started = False

    def setup(*args, **kwargs):
        nonlocal model
        if model is not None:
            raise RuntimeError("refusing a second model setup in one rank")
        model = original_setup(*args, **kwargs)
        return model

    def reset(self):
        nonlocal reset_started
        if reset_started or model is None:
            raise RuntimeError("fresh-base reset must happen exactly once after model setup")
        reset_started = True
        original_reset(self)
        report = profile_module.audit_fresh_adapter(self.adapter_state_dict())
        report.update({
            "initialization_mode": "fresh_base", "method": "native_prime_reset_adapter_parameters",
            "base_model": expected["model_id"], "base_revision": expected["revision"],
            "base_path": str(config.model.name), "source": None, "source_sha256": None,
            "rank": 16, "alpha": 32, "optimizer": "fresh", "native_resume": False,
            "completed_updates": 0, "ddp_rank": rank, "world_size": world_size,
            "before_optimizer_creation": True, "before_first_forward": True,
        })
        artifact = config.run_dir / "artifacts" / f"initial-adapter-audit.rank{rank}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps(report, indent=2) + "\n")
        receipt = export_step_zero(model, 0, initialization_mode="fresh_base")
        if get_world().is_master:
            if not receipt or not receipt.get("verified") or receipt.get("private") is not False:
                raise ValueError("fresh step-0 adapter was not verified public")
            report.update(step_zero_export="verified", step_zero_receipt=receipt)
            (config.run_dir / "artifacts" / "initial-adapter-audit.json").write_text(json.dumps(report, indent=2) + "\n")

    namespace["setup_model"] = setup
    LoRAState.reset_adapter_parameters = reset
