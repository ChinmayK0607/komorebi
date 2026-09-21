"""Build native Prime config from capacity, without provider/GPU-name assumptions."""
import copy
import json
import os
import sys
from pathlib import Path


def topology(info):
    gpus = info['gpus']
    # The frozen 27B bf16 base + LoRA optimizer/activations needs ~64 GiB
    # in our observed SFT run. RL still needs a measured startup/first-step check.
    if len(gpus) < 2:
        raise ValueError('Async RL requires distinct trainer and inference devices')
    if any(g['memory_gib'] < 76 or g['compute_capability'] < 8 for g in gpus[:2]):
        raise ValueError('This unsharded 27B profile needs two >=76 GiB bf16 CUDA devices; select a sharded profile for smaller devices')
    return dict(gpus_per_node=len(gpus), num_train_gpus=1, num_infer_gpus=1,
                type='single_node')


def training_limits(experiment):
    """Resolve and validate the bounded training horizon from a run config."""
    values = {
        'max_steps': experiment.get('max_steps', 8),
        'checkpoint_interval': experiment.get('checkpoint_interval', 2),
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    return values['max_steps'], values['checkpoint_interval']


def main():
    assert sys.platform == 'linux', 'Configuration validation and model setup run on node'
    import tomli_w
    from hardware import detect
    from prime_rl.configs.rl import RLConfig
    from training_handoff import write_static_preflight
    root = Path(__file__).resolve().parent
    experiment = json.loads((root/'EXPERIMENT.json').read_text())
    max_steps, checkpoint_interval = training_limits(experiment)
    run_id = os.environ.get('PAINTER_RUN_ID') or experiment.get('restart_run_name') or experiment.get('run_name')
    if not run_id or '/' in run_id or '\\' in run_id:
        raise ValueError('EXPERIMENT.json must provide a simple restart run id')
    model = (root/'model-path.txt').read_text().strip()
    info = detect()
    deployment = topology(info)
    config = dict(max_steps=max_steps, seq_len=16384, output_dir=str(root/'async-output'),
        run=dict(name=run_id), deployment=deployment,
        model=dict(name=model, vlm=dict(vision_encoder_attr='model.visual',
            language_model_attr='model.language_model', freeze_vision_encoder=True)),
        weight_broadcast=dict(type='filesystem', timeout=1200),
        ckpt=dict(interval=checkpoint_interval, keep_last=1), monitors=dict(file={}),
        trainer=dict(model=dict(impl='custom', compile='None', attn='auto',
            optimization_dtype='bfloat16', reduce_dtype='bfloat16',
            lora=dict(rank=16, alpha=32, target_modules=['q_proj','k_proj','v_proj','o_proj',
                'gate_proj','up_proj','down_proj','in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj']),
            ac=dict(freq=1), ac_offloading='None', optim_cpu_offload=False),
            optim=dict(type='adamw', lr=1e-6, weight_decay=.01, max_norm=1.),
            loss=dict(type='ipo', eps=.1, adv_tau=1., kl_tau=.001)),
        orchestrator=dict(batch_size=16, group_size=8, max_off_policy_steps=2,
            renderer=dict(name='qwen3.5', enable_thinking=False),
            concurrency=dict(min_inflight=32, initial_inflight=32, max_inflight=32),
            train=dict(sampling=dict(temperature=1., top_p=1., max_completion_tokens=8192),
                source=[dict(name='painter-revision',env=dict(
                    taskset=dict(id='painter-revision', root=str(root)),
                    agent=dict(harness=dict(id='null'),runtime=dict(type='subprocess'),
                        timeout=dict(rollout=1200,scoring=2100)),
                    timeout=dict(episode=3600)))])),
        inference=dict(vllm=dict(dtype='bfloat16', max_model_len=16384,
            gpu_memory_utilization=.90, max_num_seqs=16, max_num_batched_tokens=16384,
            limit_mm_per_prompt=dict(image=2,video=0),
            gdn_prefill_backend='triton', kernel_config=dict(enable_jit_warmup=False),
            mm_processor_kwargs=dict(min_pixels=3136,max_pixels=1048576))))
    resolved = RLConfig.model_validate(copy.deepcopy(config))
    (root/'rl.toml').write_text(tomli_w.dumps(config))
    (root/'async-config-receipt.json').write_text(json.dumps(dict(
        hardware=info, deployment=deployment, run_id=run_id, validated=True, run_dir=str(resolved.run_dir),
        optimizer_updates=0, group_size=8, batch_size=16, max_steps=max_steps,
        checkpoint_interval=checkpoint_interval,
        max_inflight=32, max_off_policy_steps=2,
        support='Single-node CUDA with separate unsharded 27B trainer/inference replicas. Smaller memory and distributed trainer profiles are not yet validated.',
        objective='Native GRPO mean-centered rewards and native token-normalized IPO loss; no length penalty'),indent=2)+'\n')
    # Static checks bind this run to the pinned Prime producer/consumer data
    # path.  It deliberately remains ``awaiting_native_batch``: setup is not a
    # training update and a fixture cannot stand in for a native payload.
    write_static_preflight(root)
    print('Validated native async configuration:', resolved.run_dir)

if __name__ == '__main__':
    main()
