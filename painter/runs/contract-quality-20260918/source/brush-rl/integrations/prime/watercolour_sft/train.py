"""PRIME trainer with isolated, padded image/code examples; no upstream edits."""
from prime_rl.trainer.sft import data


class SingleExampleDataset(data.CatDataset):
    def __iter__(self):
        if self.pending_sample is not None:
            raise ValueError('Cannot resume packed data into isolated-example pilot')
        for sample in self.dataset:
            if len(sample['input_ids']) > self.seq_len:
                raise ValueError('Oversize example; run audit_lengths.py first')
            yield self._finalize_pack(sample, self.seq_len)


data.CatDataset = SingleExampleDataset

if __name__ == '__main__':
    from prime_rl.configs.sft import SFTConfig
    from prime_rl.utils.config import cli
    from prime_rl.trainer.sft.train import train
    from export_adapter import install_export_hook
    config = cli(SFTConfig)
    install_export_hook(config)
    import os
    if os.environ.get('WATERCOLOUR_INIT_ADAPTER'):
        from init_adapter import install_init_hook
        install_init_hook(config, os.environ['WATERCOLOUR_INIT_ADAPTER'])
    train(config)
