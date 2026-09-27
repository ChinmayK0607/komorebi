"""Pinned model identities and complete rank-16 adapter dimensions."""
import hashlib
import json
import os
from pathlib import Path

PROFILES={
 'qwen35_9b':dict(model_id='Qwen/Qwen3.5-9B',revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',tensors=496,parameters=43278336,config_sha256='d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05'),
 'qwen38_27b':dict(model_id='Qwen/Qwen3.8-27B',revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',tensors=992,parameters=116727808,config_sha256='191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab'),
}

def profile():
 key=os.environ.get('BRUSH_MODEL_PROFILE','qwen35_9b')
 if key not in PROFILES:raise ValueError('Unknown pinned brush model profile')
 return dict(name=key,**PROFILES[key])

def verify_snapshot(path):
 p=Path(path).resolve(strict=True);spec=profile()
 if p.name!=spec['revision']:raise ValueError('Model directory does not match selected immutable revision')
 if hashlib.sha256((p/'config.json').read_bytes()).hexdigest()!=spec['config_sha256']:raise ValueError('Pinned model config hash mismatch')
 return spec
