"""Use PRIME's normal local coordinator with an explicit SFT-initialized trainer.

WATERCOLOUR_INIT_ADAPTER=/absolute/adapter uv run --no-sync python run_from_sft.py @ rl.toml
Only the pinned trainer child command is changed. No optimizer/data resume.
"""
from pathlib import Path
import os


def trainer_command(command, wrapper):
    result = list(command)
    target = ['-m', 'prime_rl.trainer.rl.train']
    matches = [i for i in range(len(result)-1) if result[i:i+2] == target]
    if not matches:
        return result
    if len(matches) != 1 or Path(result[0]).name != 'torchrun':
        raise ValueError('Unexpected PRIME trainer invocation')
    index = matches[0]
    return result[:index] + [str(wrapper)] + result[index+2:]


def main():
    import prime_rl.entrypoints.rl as entrypoint
    adapter = os.environ.get('WATERCOLOUR_INIT_ADAPTER')
    if not adapter or not (Path(adapter)/'adapter_model.safetensors').is_file():
        raise ValueError('An explicit portable SFT adapter is required')
    wrapper = Path(__file__).with_name('train_from_sft.py').resolve(strict=True)
    original = entrypoint.Popen
    trainer_started = False

    def launch(command, *args, **kwargs):
        nonlocal trainer_started
        modified = trainer_command(command, wrapper)
        if modified != list(command):
            if trainer_started:
                raise RuntimeError('Unexpected duplicate trainer start')
            trainer_started = True
            environment = dict(kwargs.get('env', os.environ))
            environment['WATERCOLOUR_INIT_ADAPTER'] = str(Path(adapter).resolve())
            kwargs['env'] = environment
        return original(modified, *args, **kwargs)

    entrypoint.Popen = launch
    entrypoint.main()


if __name__ == '__main__':
    main()
