"""Save native DCP state and export a separate portable LoRA checkpoint.

The native checkpoint remains local and resumable.  The public Hugging Face
artifact is deliberately limited to the adapter files accepted by
``upload_checkpoints.py``; native model or optimizer state never enters that
upload path.
"""
import hashlib
import json
import os
import sys
from pathlib import Path


CONTRACT_SCHEMA = 'staged-family-balanced-checkpoint-contract-v1'


def native_checkpoint_path(run_dir, step):
    return Path(run_dir) / 'checkpoints' / f'step_{step}' / 'trainer'


def adapter_checkpoint_path(run_dir, step):
    return Path(run_dir) / 'artifacts' / 'adapters' / f'step_{step}'


def _write_checkpoint_contract(run_dir, step, native, adapter, receipt):
    """Record the local/native and public/portable halves of one save."""

    folder = Path(adapter)
    native_path = Path(native)
    contract = {
        'schema': CONTRACT_SCHEMA,
        'step': step,
        'native_dcp': {
            'path': str(native_path),
            'resumable': native_path.is_dir() and (native_path / '.metadata').is_file(),
        },
        'portable_adapter': {
            'path': str(folder),
            'receipt_path': str(folder / 'hf-upload.json'),
            'repo_id': receipt.get('repo_id'),
            'remote_path': receipt.get('path'),
            'verified': receipt.get('verified') is True,
            'private': receipt.get('private'),
        },
    }
    temporary = folder / '.checkpoint-contract.json.tmp'
    temporary.write_text(json.dumps(contract, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(folder / 'checkpoint-contract.json')
    return contract


def validate_files(folder, step):
    import torch
    from safetensors.torch import load_file
    weights = load_file(str(folder / 'adapter_model.safetensors'))
    if not weights or not all(torch.isfinite(value).all().item() for value in weights.values()):
        raise ValueError('Exported adapter is empty or non-finite')
    metadata = json.loads((folder / 'adapter_config.json').read_text())
    if metadata['peft_type'] != 'LORA':
        raise ValueError('Expected portable LoRA metadata')
    from model_profiles import profile
    selected=profile()
    if selected['name']=='qwen38_27b' and (len(weights)!=selected['tensors'] or sum(v.numel() for v in weights.values())!=selected['parameters']):
        raise ValueError('Portable adapter does not match Qwen27 dimensions')
    metadata.update(base_model_name_or_path=selected['model_id'],revision=selected['revision'])
    (folder/'adapter_config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    report = {'step': step, 'parameters': sum(value.numel() for value in weights.values()),
              'tensor_count': len(weights), 'finite': True, 'base_model':selected['model_id'], 'base_revision':selected['revision'],
              'target_modules': metadata['target_modules'],
              'sha256': hashlib.sha256((folder/'adapter_model.safetensors').read_bytes()).hexdigest()}
    (folder / 'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def cloud_settings():
    repo = os.environ.get('HF_CHECKPOINT_REPO', '')
    run = os.environ.get('HF_CHECKPOINT_RUN', '')
    if not repo or not run:
        raise ValueError('Durable checkpoints required: set HF_CHECKPOINT_REPO and unique HF_CHECKPOINT_RUN; authenticate Hugging Face on the pod')
    scripts = Path(__file__).resolve().parents[3] / 'scripts'
    sys.path.insert(0, str(scripts))
    from upload_checkpoints import check_repository, upload_checkpoint
    check_repository(repo, run)
    return repo, run, upload_checkpoint


def install_export_hook(config):
    from prime_rl.configs.trainer import FileSystemWeightBroadcastConfig
    from prime_rl.trainer.ckpt import CheckpointManager
    from prime_rl.transports.weights.filesystem import FileSystemWeightSender
    from prime_rl.trainer.world import get_world
    import torch.distributed as dist
    if config.model.lora is None:
        raise ValueError('This pilot exports LoRA only')
    repo, run, upload = cloud_settings()
    original_save = CheckpointManager.save

    def export(model, step, *, initialization_mode=None):
        if initialization_mode is not None and (initialization_mode != 'fresh_base' or step != 0):
            raise ValueError('Only genuine fresh-base initialization may use the step-zero export mode')
        folder = adapter_checkpoint_path(config.run_dir, step)
        world = get_world()
        if world.is_master:
            if initialization_mode == 'fresh_base' and folder.exists():
                raise ValueError('Refusing to overwrite an existing step-zero adapter')
            folder.mkdir(parents=True, exist_ok=True)
            # A repeated save must never leave an old success marker on failure.
            (folder / '.finished').unlink(missing_ok=True)
            latest = folder.parent / 'latest.json'
            if latest.exists() and json.loads(latest.read_text()).get('step') == step:
                latest.unlink()
        dist.barrier()
        sender = FileSystemWeightSender(config.run_dir, FileSystemWeightBroadcastConfig(), config.model.lora)
        # Deliberately use only the pinned serializer, not broadcast(): the
        # public transport waits for an inference receiver's acknowledgement.
        sender._broadcast(model, step, folder)
        if world.is_master:
            report = validate_files(folder, step)
            if initialization_mode == 'fresh_base':
                from init_adapter import audit_fresh_adapter, BASE_MODEL, BASE_REVISION, TARGETS
                from safetensors.torch import load_file
                metadata = json.loads((folder/'adapter_config.json').read_text())
                if metadata.get('r') != 16 or metadata.get('lora_alpha') != 32 or set(metadata.get('target_modules', [])) != TARGETS:
                    raise ValueError('Fresh exported adapter metadata differs from pinned rank/alpha/targets')
                report.update(audit_fresh_adapter(load_file(str(folder/'adapter_model.safetensors'), device='cpu')))
                report.update(initialization_mode='fresh_base', completed_updates=0,
                              base_model=BASE_MODEL, base_revision=BASE_REVISION)
                (folder/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
            receipt = upload(folder, repo, run)
            if not receipt.get('verified'):
                raise ValueError('Cloud checkpoint verification did not succeed')
            if os.environ.get('ADAPTER_ONLY_CHECKPOINTS') == '1' and receipt.get('private') is not False:
                raise ValueError('Adapter-only checkpoint must be verified in a public repository')
            if initialization_mode == 'fresh_base' and receipt.get('private') is not False:
                raise ValueError('Fresh step-zero export must be verified public')
            _write_checkpoint_contract(
                config.run_dir, step, native_checkpoint_path(config.run_dir, step), folder, receipt,
            )
            (folder / '.finished').touch()
            (folder.parent / 'latest.json').write_text(json.dumps({'step': step, 'directory': folder.name,
                                                                'sha256': report['sha256'], 'huggingface': receipt['url']})+'\n')
        dist.barrier()
        return receipt if world.is_master else None

    def save(self, step, model, optimizers, scheduler, progress, dataloader=None):
        # Keep the native DCP for local recovery even when the public artifact
        # is adapter-only. ``upload_checkpoint`` receives only the portable
        # LoRA files; it never sees this native checkpoint or optimizer state.
        adapter_only = os.environ.get('ADAPTER_ONLY_CHECKPOINTS') == '1'
        original_save(self, step, model, optimizers, scheduler, progress, dataloader)
        if adapter_only:
            native = native_checkpoint_path(config.run_dir, step)
            if not native.is_dir() or not (native / '.metadata').is_file():
                raise ValueError('Adapter-only checkpoint contract requires the local native DCP checkpoint')
        export(model, step)

    CheckpointManager.save = save
    return export
