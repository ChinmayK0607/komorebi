#!/usr/bin/env python3
"""Upload portable adapters to Hugging Face storage and verify each commit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

# Keep Xet scratch files writable without relocating existing HF credentials.
os.environ.setdefault('HF_XET_CACHE', str(Path(tempfile.gettempdir()) / 'brush-hf-xet-cache'))

FILES=('adapter_config.json','adapter_model.safetensors','validation.json')
CARD='''---
base_model: Qwen/Qwen3.5-9B
library_name: peft
license: apache-2.0
pipeline_tag: image-text-to-text
---
# Qwen3.5-9B brush painting checkpoints

Durable public archive of experimental rank-16 LoRA adapters for reference-image painting code.
The locked base revision is c202236235762e1c871ad0ccb60c8ee5ba337b9a.

Each run/step folder contains portable adapter weights, PEFT configuration, validation statistics,
and a SHA-256 upload manifest. These are adapter-only weight checkpoints, not optimizer-state resumes.
No dataset, reference images, generated images, credentials or optimizer/base weights are included.

watercolour-sft/step_29 is the initial 29-update SFT adapter. watercolour-sft2/step_N records N
additional updates from that adapter, with a fresh optimizer (total updates 29+N).
Results are experimental and narrowly evaluated; these checkpoints do not imply general artist-level fidelity.
'''


def sha(data):return hashlib.sha256(data).hexdigest()


def selected_model():
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'integrations/prime/watercolour_sft'))
    from model_profiles import profile
    return profile()


def payload(checkpoint,run):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}',run):raise ValueError('Invalid run namespace')
    checkpoint=Path(checkpoint)
    if checkpoint.is_symlink() or not re.fullmatch(r'step_[0-9]+',checkpoint.name):raise ValueError('Expected real step_N checkpoint directory')
    values={}
    for name in FILES:
        p=checkpoint/name
        if p.is_symlink() or not p.is_file():raise ValueError('Missing or symlinked adapter file: '+name)
        values[name]=p.read_bytes()
    config=json.loads(values['adapter_config.json']);validation=json.loads(values['validation.json'])
    if config.get('peft_type')!='LORA' or not validation.get('finite'):raise ValueError('Expected validated finite LoRA checkpoint')
    if validation.get('sha256')!=sha(values['adapter_model.safetensors']):raise ValueError('Adapter validation hash mismatch')
    model=selected_model()
    if model['name']!='qwen35_9b' and (validation.get('tensor_count')!=model['tensors'] or validation.get('parameters')!=model['parameters']):
        raise ValueError('Checkpoint dimensions differ from selected model')
    manifest={'run':run,'step':int(checkpoint.name[5:]),'base_model':model['model_id'],
              'base_revision':model['revision'],
              'files':{name:{'sha256':sha(data),'bytes':len(data)} for name,data in values.items()}}
    values['upload-manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    return run+'/'+checkpoint.name,values


def verify(api,repo,revision,prefix,values):
    from huggingface_hub import hf_hub_download
    paths=[prefix+'/'+name for name in values]
    found={v.path:v for v in api.get_paths_info(repo_id=repo,paths=paths,revision=revision,repo_type='model')}
    verified={}
    with tempfile.TemporaryDirectory(prefix='hf-checkpoint-verify-') as cache:
        for name,data in values.items():
            path=prefix+'/'+name;info=found.get(path)
            if info is None or info.size!=len(data):raise ValueError('Remote file size mismatch: '+path)
            lfs=getattr(info,'lfs',None)
            remote_sha=(lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)) if lfs else None
            if not remote_sha:
                local=hf_hub_download(repo_id=repo,filename=path,revision=revision,repo_type='model',cache_dir=cache,force_download=True)
                remote_sha=sha(Path(local).read_bytes())
            if remote_sha!=sha(data):raise ValueError('Remote SHA256 mismatch: '+path)
            verified[name]={'sha256':remote_sha,'bytes':info.size}
    return verified


def check_repository(repo, run):
    from huggingface_hub import HfApi
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_.-]*', repo):
        raise ValueError('Expected HF_CHECKPOINT_REPO namespace/repository')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}', run):
        raise ValueError('Invalid HF_CHECKPOINT_RUN namespace')
    api = HfApi()
    api.whoami()
    api.repo_info(repo, repo_type='model')


def upload_checkpoint(checkpoint,repo,run,create_private=False,create=False):
    from huggingface_hub import HfApi,CommitOperationAdd
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_.-]*',repo):raise ValueError('Expected namespace/repository')
    prefix,values=payload(checkpoint,run)
    api=HfApi()
    # Creation never changes visibility of an existing repository.
    if create or create_private:api.create_repo(repo_id=repo,repo_type='model',private=bool(create_private),exist_ok=True)
    info=api.repo_info(repo,repo_type='model')
    existing=api.list_repo_files(repo,repo_type='model')
    # Immutable prefixes: repeated identical uploads verify; differing files fail.
    present=[prefix+'/'+n for n in values if prefix+'/'+n in existing]
    if present:
        current={n:v for n,v in values.items() if prefix+'/'+n in existing}
        verify(api,repo,info.sha,prefix,current)
    if len(present)==len(values):
        commit=info.sha
    else:
        operations=[CommitOperationAdd(path_in_repo=prefix+'/'+n,path_or_fileobj=v) for n,v in values.items() if prefix+'/'+n not in existing]
        if 'README.md' not in existing:
            model=selected_model()
            card=CARD if model['name']=='qwen35_9b' else f"---\nbase_model: {model['model_id']}\nlibrary_name: peft\nlicense: apache-2.0\npipeline_tag: image-text-to-text\n---\n# {model['model_id']} reference painting adapters\n\nExperimental rank-16 LoRA checkpoints. Base revision: {model['revision']}.\nEach checkpoint has verified hashes and model identity. Adapter-only; optimizer state is not included.\nEvaluation is developmental and does not establish general artist-level quality.\n"
            operations.append(CommitOperationAdd(path_in_repo='README.md',path_or_fileobj=card.encode()))
        result=api.create_commit(repo_id=repo,repo_type='model',operations=operations,commit_message='Archive '+prefix,parent_commit=info.sha)
        commit=result.oid
    files=verify(api,repo,commit,prefix,values)
    receipt={'repo_id':repo,'private':bool(info.private),'commit':commit,'path':prefix,
             'url':f'https://huggingface.co/{repo}/tree/{commit}/{prefix}', 'verified':True,'files':files}
    p=Path(checkpoint)/'hf-upload.json';temporary=p.with_suffix('.tmp');temporary.write_text(json.dumps(receipt,indent=2)+'\n');temporary.replace(p)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True);p.add_argument('--run',required=True)
    p.add_argument('--checkpoint',type=Path,action='append',required=True)
    visibility=p.add_mutually_exclusive_group()
    visibility.add_argument('--create',action='store_true',help='Create public repository if absent; preserve existing visibility')
    visibility.add_argument('--create-private',action='store_true',help='Explicitly create private repository if absent; preserve existing visibility')
    a=p.parse_args()
    for checkpoint in a.checkpoint:
        print(json.dumps(upload_checkpoint(checkpoint,a.repo,a.run,a.create_private,a.create)))
