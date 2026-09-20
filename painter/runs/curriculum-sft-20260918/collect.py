"""Provider-independent evidence retrieval over a supplied SSH connection."""
import argparse,hashlib,json,shlex,subprocess,tarfile
from pathlib import Path,PurePosixPath


def collect(ssh_command,root,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    parts=shlex.split(ssh_command);assert parts[0]=='ssh';target=next(x for x in parts[1:] if '@' in x)
    opts=parts[1:];opts.remove(target);scp=['scp']
    for part in opts:scp.append('-P' if part=='-p' else part)
    def fetch(name,destination):subprocess.run(scp+[target+':'+root.rstrip('/')+'/'+name,str(destination)],check=True)
    fetch('export-receipt.json',output/'receipt.json');r=json.loads((output/'receipt.json').read_text())
    assert r['file']=='evidence.tar.gz' and not r['contains_weights'] and r['bytes']<768*1024**2
    archive=output/r['file'];fetch(r['file'],archive.with_suffix('.partial'));partial=archive.with_suffix('.partial')
    assert partial.stat().st_size==r['bytes'] and hashlib.sha256(partial.read_bytes()).hexdigest()==r['sha256'];partial.replace(archive)
    with tarfile.open(archive) as t:
        members=t.getmembers();assert len(members)==r['files'] and len({m.name for m in members})==len(members)
        assert all(m.isfile() and not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts for m in members)
        assert sum(m.size for m in members)<2*1024**3
        t.extractall(output/'evidence',filter='data')
    manifest=json.loads((output/'evidence/evidence-manifest.json').read_text())
    assert {m.name for m in members}==set(manifest)|{'evidence-manifest.json'}
    for name,meta in manifest.items():
        f=output/'evidence'/name;assert f.stat().st_size==meta['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==meta['sha256'],name
    result=dict(verified=True,files=r['files'],archive_sha256=r['sha256'],contains_weights=False)
    (output/'verified.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ssh-command',required=True);p.add_argument('--remote-root',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(collect(a.ssh_command,a.remote_root,a.output)))
