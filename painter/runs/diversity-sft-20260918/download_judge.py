import hashlib,json
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
P=Path(__file__).resolve().parent
spec=json.loads((P/'EXPERIMENT.json').read_text())['judge']
info=HfApi().model_info(spec['model'],revision=spec['revision'],files_metadata=True)
assert info.sha==spec['revision']
root=Path(snapshot_download(spec['model'],revision=spec['revision'],allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','*.model']))
files={}
for f in info.siblings:
 path=root/f.rfilename
 if not path.is_file():continue
 with path.open('rb') as stream:h=hashlib.file_digest(stream,'sha256').hexdigest()
 if f.lfs:assert h==f.lfs.sha256,f.rfilename
 files[f.rfilename]=dict(sha256=h,bytes=path.stat().st_size)
(P/'judge-path.txt').write_text(str(root)+'\n')
(P/'judge-receipt.json').write_text(json.dumps(dict(spec,files=files,verified=True),indent=2)+'\n')
