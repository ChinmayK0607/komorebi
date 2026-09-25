#!/usr/bin/env python3
"""Retrieve and verify the immutable public bounded-JSON SFT adapter; never use RL weights."""
import argparse
import hashlib
import json
from pathlib import Path

REPO='CK0607/qwen3.5-9b-brush-painting'
COMMIT='fe34b542066b7cf532d4d6839d76115342ac0b94'
PREFIX='speedpainting-json-sft-20260909/step_40'
WEIGHT_SHA='4a7832a23d47c3ebae573eaaf45c10ec3b977b43297694baa4602492c4625b20'
WEIGHT_BYTES=86623432


def download(output, repo=REPO, revision=COMMIT, prefix=PREFIX, weight_sha=WEIGHT_SHA):
    import re
    if not re.fullmatch(r"[0-9a-f]{40}", revision): raise ValueError("Immutable HF commit required")
    if not re.fullmatch(r"[0-9a-f]{64}", weight_sha): raise ValueError("Pinned weight SHA256 required")
    if prefix.startswith("/") or ".." in Path(prefix).parts: raise ValueError("Invalid HF adapter path")
    from huggingface_hub import hf_hub_download
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path(hf_hub_download(repo,prefix+'/upload-manifest.json',revision=revision)).read_text())
    expected=manifest['files']['adapter_model.safetensors']
    if expected.get('sha256')!=weight_sha:raise ValueError('Immutable bounded SFT manifest mismatch')
    for name in ['adapter_config.json','adapter_model.safetensors','validation.json']:
        target=output/name;record=manifest['files'][name]
        if target.is_symlink():raise ValueError('Refusing symlinked adapter cache')
        raw=target.read_bytes() if target.exists() else b''
        if len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:
            raw=Path(hf_hub_download(repo,prefix+'/'+name,revision=revision)).read_bytes()
            if len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:raise ValueError('Downloaded adapter checksum mismatch')
            tmp=target.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(target)
    (output/'hf-source.json').write_text(json.dumps({'repo':repo,'commit':revision,'path':prefix,'sha256':weight_sha,'verified':True},indent=2)+'\n')
    return output.resolve()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repo',default=REPO);p.add_argument('--revision',default=COMMIT);p.add_argument('--prefix',default=PREFIX);p.add_argument('--weight-sha',default=WEIGHT_SHA)
    a=p.parse_args();print(download(a.output,a.repo,a.revision,a.prefix,a.weight_sha))
