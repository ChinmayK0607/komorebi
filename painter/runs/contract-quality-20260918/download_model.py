import hashlib,json,os,sys
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'source/brush-rl/integrations/prime/watercolour_sft'))
from model_profiles import profile,verify_snapshot
m=profile();api=HfApi();info=api.model_info(m['model_id'],revision=m['revision'],files_metadata=True)
assert info.sha==m['revision']
path=Path(snapshot_download(m['model_id'],revision=m['revision'],allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','*.model']))
verify_snapshot(path)
files={}
for f in info.siblings:
 p=path/f.rfilename
 if not p.is_file():continue
 with p.open('rb') as stream:h=hashlib.file_digest(stream,'sha256').hexdigest()
 if f.lfs:assert h==f.lfs.sha256,f.rfilename
 files[f.rfilename]={'sha256':h,'bytes':p.stat().st_size}
index=json.loads((path/'model.safetensors.index.json').read_text())
assert set(index['weight_map'].values())<=files.keys()
(P/'model-path.txt').write_text(str(path)+'\n')
(P/'model-receipt.json').write_text(json.dumps(dict(model=m,files=files),indent=2)+'\n')

experiment=json.loads((P/'EXPERIMENT.json').read_text());init=experiment['initial_adapter']
adapter_snapshot=Path(snapshot_download(init['repo'],revision=init['revision'],allow_patterns=[init['prefix']+'/*']))
adapter=adapter_snapshot/init['prefix']
with (adapter/'adapter_model.safetensors').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==init['sha256']
(P/'initial-adapter-path.txt').write_text(str(adapter)+'\n')
(P/'initial-adapter-receipt.json').write_text(json.dumps(dict(init,verified=True),indent=2)+'\n')
