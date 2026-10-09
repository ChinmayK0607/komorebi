"""Observe native CUDA updates with rank-safe aggregation."""
import inspect
import json
import os
from pathlib import Path


OVERRIDE_SCHEMA = 'Diverse-flow-multirank-update-audit-v1'


def attach(optimizer, named_params, output, steps=8):
    import torch
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError('Update audit requires native AdamW without CPU offload')
    params = [(n, p) for n, p in named_params if p.requires_grad]
    local = lambda t: t.to_local() if hasattr(t, 'to_local') else t
    if not params or any(not local(p).is_cuda for _, p in params):
        raise ValueError('Update audit requires CUDA trainable parameters')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    import torch.distributed as dist
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if output.exists():
        raise ValueError('Refusing an existing update-audit log')
    state = {'step': 0, 'changed': 0}
    snapshots = []
    gradients = []
    handles = []

    def before(opt, args, kwargs):
        snapshots[:] = [local(p).detach().clone() for _, p in params]
        gradients[:] = []
        for _, p in params:
            g = local(p.grad) if p.grad is not None else None
            gradients.append(torch.stack((torch.count_nonzero(g), torch.isfinite(g).all().long())) if g is not None
                             else torch.tensor([0, 1], device=local(p).device, dtype=torch.long))

    def after(opt, args, kwargs):
        state['step'] += 1
        stats = torch.stack([torch.stack((torch.count_nonzero(local(p).detach() != old),
                                         torch.isfinite(local(p)).all().long(), grad[0], grad[1]))
                             for (_, p), old, grad in zip(params, snapshots, gradients)]).cpu().tolist()
        tensors = [dict(name=n, dtype=str(local(p).dtype), elements=p.numel(), changed=s[0], finite=bool(s[1]),
                        nonzero_gradient_elements=s[2], gradient_finite=bool(s[3]))
                   for (n, p), s in zip(params, stats)]
        changed = sum(r['changed'] for r in tensors)
        state['changed'] += changed
        report = dict(completed_updates=state['step'], lr=[g['lr'] for g in opt.param_groups],
                      changed_elements=changed, changed_tensors=sum(r['changed'] > 0 for r in tensors),
                      nonzero_gradient_elements=sum(r['nonzero_gradient_elements'] for r in tensors),
                      all_finite=all(r['finite'] and r['gradient_finite'] for r in tensors), tensors=tensors,
                      method='exact pre/post native optimizer step CUDA tensor comparison')
        reports = [report]
        if world_size > 1 and dist.is_available() and dist.is_initialized():
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, report)
            reports = gathered
        if rank == 0:
            aggregate = {
                'completed_updates': state['step'],
                'lr': report['lr'],
                'changed_elements': sum(int(item['changed_elements']) for item in reports),
                'changed_tensors': sum(int(item['changed_tensors']) for item in reports),
                'nonzero_gradient_elements': sum(int(item['nonzero_gradient_elements']) for item in reports),
                'all_finite': all(bool(item['all_finite']) for item in reports),
                'rank_reports': reports,
                'world_size': world_size,
                'schema': OVERRIDE_SCHEMA,
            }
            with output.open('a') as f:
                f.write(json.dumps(aggregate) + '\n')
        snapshots.clear(); gradients.clear()
        if not all(bool(item['all_finite']) for item in reports):
            raise RuntimeError('Nonfinite adapter/gradient; see update audit')
        if state['step'] == steps:
            for h in handles:
                h.remove()
            if state['changed'] == 0:
                raise RuntimeError('No representable parameter changes during audited optimizer steps')

    handles.extend((optimizer.register_step_pre_hook(before), optimizer.register_step_post_hook(after)))
    return handles


def install(config, trainer):
    if config.model.optim_cpu_offload or config.model.full_offload is not None:
        return
    namespace = inspect.unwrap(trainer).__globals__
    original = namespace['setup_optimizer']

    def setup(*args, **kwargs):
        optimizer, manager = original(*args, **kwargs)
        if manager is not None:
            raise ValueError('Unexpected asynchronous gradient manager')
        named_params = args[1] if len(args) > 1 else kwargs['named_params']
        attach(optimizer, named_params, config.run_dir / 'artifacts/optimizer-update-audit.jsonl', steps=config.max_steps)
        return optimizer, manager

    namespace['setup_optimizer'] = setup
