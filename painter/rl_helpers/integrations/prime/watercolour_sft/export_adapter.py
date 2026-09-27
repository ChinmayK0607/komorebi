"""Export portable LoRA files after native SFT checkpoints, without a receiver."""
import hashlib
import json
import os
import sys
from pathlib import Path


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
        folder = config.run_dir / 'artifacts/adapters' / f'step_{step}'
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
            if initialization_mode == 'fresh_base' and receipt.get('private') is not False:
                raise ValueError('Fresh step-zero export must be verified public')
            (folder / '.finished').touch()
            (folder.parent / 'latest.json').write_text(json.dumps({'step': step, 'directory': folder.name,
                                                                'sha256': report['sha256'], 'huggingface': receipt['url']})+'\n')
        dist.barrier()
        return receipt if world.is_master else None

    def save(self, step, model, optimizers, scheduler, progress, dataloader=None):
        # This short fresh-only diagnostic publishes adapters directly. The
        # immutable base is already on HF; native full-base copies are redundant.
        adapter_only = os.environ.get('ADAPTER_ONLY_CHECKPOINTS') == '1'
        if adapter_only:
            if config.resume is not None:
                raise ValueError('Adapter-only capacity test does not support native optimizer resume')
        else:
            original_save(self, step, model, optimizers, scheduler, progress, dataloader)
        export(model, step)

    CheckpointManager.save = save
    return export
