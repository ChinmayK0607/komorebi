"""Describe capabilities and propose settings; no model-name-specific GPU checks."""
import json
import os
import subprocess


def detect():
    gpus=[]
    try:
        output=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,compute_cap','--format=csv,noheader,nounits'],text=True)
        for row in output.splitlines():
            name,memory,cc=[x.strip() for x in row.split(',')]
            gpus.append(dict(name=name,memory_gib=float(memory)/1024,compute_capability=float(cc)))
    except (OSError,subprocess.CalledProcessError):pass
    return dict(cpu_count=os.cpu_count(),gpus=gpus,backend='cuda' if gpus else 'cpu_or_external_endpoint')


def plan(info,model_parameters=27_780_000_000,global_batch=8):
    g=info['gpus'];weights_gib=model_parameters*2/1024**3
    # Conservative starting estimates, verified by a node smoke; not promises of fit.
    partitions=[n for n in (1,2,4,8) if n<=len(g) and weights_gib/n+12 <= min(x['memory_gib'] for x in g[:n])*.90]
    tp=partitions[0] if partitions else None
    world=max([n for n in range(1,len(g)+1) if global_batch%n==0] or [1])
    return dict(hardware=info,estimated_bf16_weights_gib=round(weights_gib,2),
      render_workers=max(1,min(4,(info['cpu_count'] or 2)//4)),
      serving=dict(tensor_parallel_size=tp,dtype='bfloat16' if g and all(x['compute_capability']>=8 for x in g) else 'float16',
        gpu_memory_utilization=.9,enforce_eager=True,max_num_seqs=4 if tp else None),
      training=dict(micro_batch_size=1,global_batch_size=global_batch,world_size=world,gradient_accumulation=global_batch//world,
        note='Preserve effective batch when changing GPU count. Training fit needs its own smoke.'),
      status='ready_for_serving_smoke' if tp else 'render_only_or_supply_external_endpoint',
      support='Provider-independent. Local GPU runtime currently pinned for Linux CUDA; other accelerators require their own tested runtime image.')


if __name__=='__main__':print(json.dumps(plan(detect()),indent=2))
