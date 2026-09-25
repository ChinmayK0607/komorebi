"""Add per-family validation and post-final validation to pinned PRIME in memory.

No upstream file edits. Fail if upstream source changes, and save exact effective
training function so this small instrumentation is reviewable with each run.
"""
import ast
import hashlib
import importlib
import inspect
from pathlib import Path
SOURCE_SHA='08c0ab3284000ee9b3fd0853a7007aedac21b221e66c675c621927c063b46ad0'

EXTRA='''        import json as _json
        _groups = sorted({r.get("mixture_group") or r.get("task_family") or "unspecified" for r in val_raw_dataset})
        _record = {"trainer_step": step, "completed_updates": step if phase == "after_optimizer" else max(0, step-1), "phase": phase, "aggregate_nll": mean_loss, "groups": {}}
        for _group in _groups:
            _raw = val_raw_dataset.filter(lambda r: (r.get("mixture_group") or r.get("task_family") or "unspecified") == _group)
            _ds = setup_dataset(tokenizer, config.val.data, config.model.cp, max_epochs=1, raw_dataset=_raw, renderer=renderer, multimodal=multimodal)
            _loss, _nans = run_eval_loop(setup_dataloader(_ds, config.val.data))
            _record["groups"][_group] = {"examples": len(_raw), "assistant_token_mean_nll": _loss, "nan_batches": _nans}
        _path = config.run_dir / "artifacts/group-validation.jsonl"
        _path.parent.mkdir(parents=True, exist_ok=True)
        with _path.open("a") as _file:
            _file.write(_json.dumps(_record) + "\\n")
        logger.info("Group validation: " + _json.dumps(_record))

'''

def transform(source):
    if hashlib.sha256(source.encode()).hexdigest()!=SOURCE_SHA:raise ValueError('Unrecognized PRIME SFT trainer; inspect before instrumentation')
    source=source.replace('def run_validation(step: int) -> None:', 'def run_validation(step: int, phase: str = "before_optimizer") -> None:')
    anchor='    gc_handler = GarbageCollection(config.gc.interval) if config.gc else None\n'
    if source.count(anchor)!=1:raise ValueError('Validation insertion anchor mismatch')
    source=source.replace(anchor,EXTRA+anchor)
    anchor='    # Write final checkpoint\n'
    if source.count(anchor)!=1:raise ValueError('Final validation anchor mismatch')
    source=source.replace(anchor,'    if config.val is not None:\n        run_validation(progress.step, phase="after_optimizer")\n\n'+anchor)
    tree=ast.parse(source);node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='train')
    return ast.unparse(node)+'\n'

def install(output, load_sft_dataset=None):
    module=importlib.import_module('prime_rl.trainer.sft.train')
    if load_sft_dataset is not None:
        module.load_sft_dataset = load_sft_dataset
    source=Path(module.__file__).read_text()
    effective=transform(source)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=output/'effective_prime_train.py';path.write_text(effective)
    exec(compile(effective,str(path),'exec'),module.__dict__)
    return module.train
