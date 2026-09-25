"""Strict portable-adapter initialization after PRIME resets, before optimization."""
import hashlib
import inspect
import json
from pathlib import Path

from model_profiles import profile
BASE_REVISION = profile()['revision']
BASE_MODEL = profile()['model_id']
TARGETS = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj',
           'in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj'}


def initialization_mode(adapter=None, fresh_base=False):
    if bool(adapter) == bool(fresh_base):
        raise ValueError('Select exactly one initializer: --init-adapter or --fresh-base')
    return 'fresh_base' if fresh_base else 'trained_adapter'


def audit_fresh_adapter(state):
    """Audit native tensors; never manufacture or repair adapter weights."""
    import torch
    values = {name: value.full_tensor() if hasattr(value, 'full_tensor') else value
              for name, value in state.items()}
    a_names = {name for name in values if '.lora_A.' in name}
    b_names = {name for name in values if '.lora_B.' in name}
    expected=profile();pairs=expected['tensors']//2
    if len(values) != expected['tensors'] or len(a_names) != pairs or len(b_names) != pairs or a_names | b_names != set(values):
        raise ValueError(f"Fresh adapter requires {expected['tensors']} tensors from pinned model profile")
    if {name.replace('.lora_A.', '.lora_B.') for name in a_names} != b_names:
        raise ValueError('Fresh adapter A/B keys are not paired')
    if sum(value.numel() for value in values.values()) != expected['parameters']:
        raise ValueError('Fresh adapter parameter count differs from pinned model profile')
    for name, value in values.items():
        if value.ndim != 2 or value.shape[0 if name in a_names else 1] != 16:
            raise ValueError(f'Fresh adapter must have rank 16: {name}')
        if not value.is_floating_point() or not torch.isfinite(value).all().item():
            raise ValueError(f'Non-finite or non-floating fresh adapter: {name}')
        nonzero = bool(torch.count_nonzero(value).item())
        if name in b_names and nonzero:
            raise ValueError(f'Fresh adapter B must be exactly zero: {name}')
        if name in a_names and not nonzero:
            raise ValueError(f'Fresh adapter A must be natively initialized, not all zero: {name}')
    return {'tensor_count': expected['tensors'], 'parameters': expected['parameters'], 'all_finite': True,
            'a_tensors': pairs, 'nonzero_a_tensors': pairs, 'b_tensors': pairs,
            'zero_b_tensors': pairs, 'nonzero_b_tensors': 0, 'initial_lora_contribution': 'exactly_zero'}


def install_fresh_base_hook(config, trainer, export_step_zero):
    """Observe native reset before optimizer creation, then export the real model."""
    from prime_rl.trainer.lora import LoRAState
    from prime_rl.trainer.world import get_world
    initialization_mode(None, True)
    lora = config.model.lora
    if config.resume is not None or lora is None:
        raise ValueError('Fresh base requires LoRA and a fresh optimizer; native resume is forbidden')
    if Path(config.model.name).resolve(strict=True).name != BASE_REVISION:
        raise ValueError('Fresh base requires the selected pinned model snapshot')
    if lora.rank != 16 or lora.alpha != 32 or set(lora.target_modules) != TARGETS:
        raise ValueError('Fresh base requires rank 16, alpha 32 and the complete pinned target modules')
    if not callable(export_step_zero):
        raise ValueError('Fresh base requires a verified portable step-0 export callback')
    namespace = getattr(inspect.unwrap(trainer), '__globals__', {})
    original_setup = namespace.get('setup_model')
    if not callable(original_setup):
        raise ValueError('Unwrapped trainer must bind callable setup_model')
    original_reset = LoRAState.reset_adapter_parameters
    model = None
    reset_started = False

    def setup(*args, **kwargs):
        nonlocal model
        if model is not None:
            raise RuntimeError('Refusing a second fresh-base model setup')
        model = original_setup(*args, **kwargs)
        return model

    def reset(self):
        nonlocal reset_started
        if reset_started or model is None:
            raise RuntimeError('Fresh base reset must occur exactly once after model setup')
        if get_world().world_size != 1:
            raise ValueError('Fresh base pilot requires exactly one training rank')
        reset_started = True
        original_reset(self)
        report = audit_fresh_adapter(self.adapter_state_dict())
        report.update(initialization_mode='fresh_base', method='native_prime_reset_adapter_parameters',
                      base_model=BASE_MODEL, base_revision=BASE_REVISION, base_path=str(config.model.name),
                      source=None, source_sha256=None, rank=16, alpha=32, optimizer='fresh',
                      native_resume=False, before_optimizer_creation=True, before_first_forward=True,
                      completed_updates=0, step_zero_export='pending')
        output = config.run_dir/'artifacts/initial-adapter-audit.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2)+'\n')
        # Export failure propagates before PRIME can construct the optimizer.
        receipt = export_step_zero(model, 0, initialization_mode='fresh_base')
        if not receipt or not receipt.get('verified') or receipt.get('private') is not False:
            raise ValueError('Fresh base requires verified public step-0 export before training')
        report.update(step_zero_export='verified', step_zero_receipt=receipt)
        output.write_text(json.dumps(report, indent=2)+'\n')

    namespace['setup_model'] = setup
    LoRAState.reset_adapter_parameters = reset


def copy_and_verify(source, destination, reread):
    import torch
    if set(source) != set(destination):
        raise ValueError(f'Adapter key mismatch: missing={sorted(set(destination)-set(source))}, unexpected={sorted(set(source)-set(destination))}')
    for name in source:
        if source[name].shape != destination[name].shape or source[name].dtype != destination[name].dtype:
            raise ValueError(f'Adapter shape/dtype mismatch: {name}')
        if not torch.isfinite(source[name]).all().item():
            raise ValueError(f'Non-finite source adapter: {name}')
    with torch.no_grad():
        for name, tensor in destination.items():
            local = tensor.to_local() if hasattr(tensor, 'to_local') else tensor
            if local.shape != source[name].shape:
                raise ValueError('Portable initialization supports unsharded one-GPU LoRA only')
            local.copy_(source[name].to(local.device))
    verified = reread()
    if set(verified) != set(source):
        raise ValueError('Post-load adapter key set changed')
    for name, tensor in verified.items():
        actual = tensor.full_tensor() if hasattr(tensor, 'full_tensor') else tensor
        if not torch.equal(actual.cpu(), source[name]):
            raise ValueError(f'Post-load adapter differs from source: {name}')
    b_names = [name for name in source if '.lora_B.' in name]
    nonzero = sum(bool(torch.count_nonzero(source[name]).item()) for name in b_names)
    expected_pairs=profile()['tensors']//2
    if len(b_names) != expected_pairs or nonzero != expected_pairs:
        raise ValueError(f'Expected all {expected_pairs} trained B matrices, found {len(b_names)} with {nonzero} nonzero')
    return {'tensor_count': len(source), 'exact_tensor_matches': len(verified),
            'b_tensors': len(b_names), 'nonzero_b_tensors': nonzero,
            'parameters': sum(value.numel() for value in source.values()), 'all_finite': True}


def install_init_hook(config, adapter):
    from prime_rl.trainer.lora import LoRAState
    from safetensors.torch import load_file
    from prime_rl.trainer.world import get_world
    if config.resume is not None or config.model.lora is None:
        raise ValueError('Portable init requires LoRA and fresh optimizer, not native resume')
    adapter = Path(adapter).resolve(strict=True)
    metadata = json.loads((adapter/'adapter_config.json').read_text())
    lora = config.model.lora
    if metadata.get('r') != lora.rank or metadata.get('lora_alpha') != lora.alpha or set(metadata['target_modules']) != set(lora.target_modules):
        raise ValueError('Source adapter rank/alpha/target modules differ from config')
    weights_path = adapter/'adapter_model.safetensors'
    source = load_file(str(weights_path), device='cpu')
    original_reset = LoRAState.reset_adapter_parameters
    loaded = False

    def reset(self):
        nonlocal loaded
        if loaded:
            raise RuntimeError('Refusing a second reset after portable adapter initialization')
        if get_world().world_size != 1:
            raise ValueError('Portable init pilot requires exactly one training rank')
        original_reset(self)
        report = copy_and_verify(source, self.adapter_state_dict(), self.adapter_state_dict)
        report.update(initialization_mode='trained_adapter', source=str(adapter), source_sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(),
                      optimizer='fresh', native_resume=False, before_first_forward=True)
        output = config.run_dir/'artifacts/initial-adapter-audit.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2)+'\n')
        loaded = True

    LoRAState.reset_adapter_parameters = reset
