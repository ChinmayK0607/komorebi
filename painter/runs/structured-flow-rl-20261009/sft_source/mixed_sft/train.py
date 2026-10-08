"""Strict mixed-modality SFT: preserve every target token, one example per sequence."""
from pathlib import Path
import sys
from prime_rl.trainer.sft import data

class StrictSFTDataset(data.SFTDataset):
    def _process(self, example):
        limit=self.seq_len
        try:
            # Upstream truncates multimodal examples inside _process, so enlarge
            # its processing window before inspecting the actual full result.
            self.seq_len=2**62
            sample=super()._process(example)
        finally:
            self.seq_len=limit
        if sample is None: raise ValueError(f'{example.get("id")}: no trainable assistant tokens')
        if len(sample['input_ids'])>limit: raise ValueError(f'{example.get("id")}: {len(sample["input_ids"])} tokens exceeds {limit}; refusing truncation')
        import os
        if os.environ.get('CAPACITY_STATE_COMPLETE') == '1':
            from prime_rl.trainer.sft.data import _drop_null_fields
            from prime_rl.utils.chat_template import normalize_messages
            messages = _drop_null_fields(normalize_messages(example['messages'], default_role='assistant'))
            if [m['role'] for m in messages] != ['system', 'user', 'assistant']:
                raise ValueError('Capacity prefill masking requires one state-complete edit turn')
            prefix = list(self.renderer.render_ids(messages[:-1], add_generation_prompt=True))
            full = sample['input_ids'] + [sample['target_ids'][-1]]
            if full[:len(prefix)] != prefix:
                raise ValueError('Training prefix differs from generation prefix')
            # loss_mask[i] describes target token i+1. Every supplied prefix
            # token is context, including the disabled-thinking wrapper.
            sample['loss_mask'][:len(prefix)-1] = [False] * (len(prefix)-1)
        sample["example_id"]=example.get("id")
        for key in ('source_example_id','scheduled_step','scheduled_slot','schedule_group'):
            if example.get(key) is not None:
                sample[key] = example[key]
        return sample

class SingleExampleDataset(data.CatDataset):
    def __iter__(self):
        if self.pending_sample is not None: raise ValueError('Cannot resume packed data')
        for sample in self.dataset:
            if len(sample['input_ids'])>self.seq_len: raise ValueError('Oversize example')
            length=min(self.seq_len, ((len(sample['input_ids'])+255)//256)*256)
            import json, os
            trace=os.environ.get('MIXED_SFT_TRACE')
            if trace:
                record={'id':sample.get('example_id'),'raw_tokens':len(sample['input_ids']),'padded_tokens':length,
                        'supervised_tokens':sum(sample['loss_mask']),'has_image':bool(sample.get('mm_kwargs'))}
                record.update({key: sample[key] for key in ('source_example_id','scheduled_step','scheduled_slot','schedule_group') if key in sample})
                fd=os.open(trace,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
                try:os.write(fd,(json.dumps(record)+'\n').encode())
                finally:os.close(fd)
            yield self._finalize_pack(sample,length)

def install():
    data.SFTDataset=StrictSFTDataset
    data.CatDataset=SingleExampleDataset
    from scheduled_data import load_bound_config
    data.load_sft_dataset=load_bound_config

if __name__=='__main__':
    install()
    from prime_rl.configs.sft import SFTConfig
    from prime_rl.utils.config import cli
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'prime/watercolour_sft'))
    from export_adapter import install_export_hook
    from init_adapter import install_init_hook, install_fresh_base_hook, initialization_mode
    import os
    config=cli(SFTConfig)
    if config.data.micro_batch_size!=1 or (config.val and config.val.data.micro_batch_size!=1):
        raise ValueError('SingleExampleDataset requires micro_batch_size=1; changing this silently changes actual examples per optimizer update')
    from scheduled_data import guard_runtime
    schedule = guard_runtime(config, os.environ)
    if schedule:
        import json
        setup = Path(os.environ['BRUSH_ROOT'])/'runs'/(config.run.name+'-setup')
        setup.mkdir(parents=True, exist_ok=True)
        (setup/'schedule-guard.json').write_text(json.dumps({'status':'verified','version':schedule['version'],'train_sha256':schedule['train_sha256'],'steps':schedule['max_steps'],'shuffle':False,'world_size':1},indent=2)+'\n')
    from group_validation import install as install_group_validation
    train=install_group_validation(Path(os.environ['BRUSH_ROOT'])/'runs'/(config.run.name+'-setup'), load_sft_dataset=data.load_sft_dataset)
    fresh = os.environ.get('FRESH_BASE', '0')
    if fresh not in ('0', '1'):
        raise ValueError('FRESH_BASE must be 0 or 1')
    adapter = os.environ.get('WATERCOLOUR_INIT_ADAPTER')
    mode = initialization_mode(adapter, fresh == '1')
    if mode == 'fresh_base' and any(os.environ.get(key) for key in ('INIT_REPO', 'INIT_REVISION', 'INIT_PREFIX', 'INIT_WEIGHT_SHA')):
        raise ValueError('Fresh base cannot silently skip pretrained initializer settings')
    import json
    setup_report = json.loads((Path(os.environ['BRUSH_ROOT'])/'runs'/(config.run.name+'-setup')/'resolved.setup.json').read_text())
    if setup_report.get('initialization_mode') != mode or setup_report.get('init_adapter') != (str(Path(adapter).resolve(strict=True)) if adapter else None):
        raise ValueError('Runtime initializer differs from the prepared initialization report')
    if setup_report.get('step_zero_export_required') != (mode == 'fresh_base') or setup_report.get('optimizer') != 'fresh':
        raise ValueError('Prepared initialization/optimizer audit does not match runtime')
    export = install_export_hook(config)
    if mode == 'fresh_base':
        install_fresh_base_hook(config, train, export)
    else:
        install_init_hook(config, adapter)
    from update_audit import install as install_update_audit
    install_update_audit(config, train)
    train(config)
