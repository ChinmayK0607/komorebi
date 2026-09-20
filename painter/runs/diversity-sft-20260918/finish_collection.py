"""Collect this already-running experiment promptly, without model work locally."""
import datetime
import json
import shlex
import subprocess
import time
from pathlib import Path

from collect import collect

P = Path(__file__).resolve().parent
POD = '3eb1baa6-d5b2-4692-829d-6dcc187852a3'


def connection():
    r = subprocess.run(['lium', 'ps', POD, '--format', 'json'],
                       capture_output=True, text=True, check=True)
    pods, _ = json.JSONDecoder().raw_decode(r.stdout.lstrip())
    assert len(pods) == 1 and pods[0]['id'] == POD
    pod = pods[0]
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'UserKnownHostsFile=' + str(P / 'known_hosts'),
            '-p', str(pod['ports']['22']), 'root@' + pod['ip']]


def remote(code):
    r = subprocess.run(connection() + ['python3 -'], input=code,
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-3000:], flush=True)
    r.check_returncode()
    return json.loads(r.stdout)


assert not (P / 'collected-final/verified.json').exists(), 'Already collected'
deadline = datetime.datetime(2026, 9, 18, 10, 32, tzinfo=datetime.timezone.utc)
while True:
    status = remote("""from pathlib import Path
import json
p=Path('/root/painter')
r={'phase':json.loads((p/'phase.json').read_text()),'evaluation':{}}
for name in ['parent','midpoint','trained']:
 f=p/'evaluation'/name/'progress.json'
 if f.exists():r['evaluation'][name]=json.loads(f.read_text())
print(json.dumps(r))
""")
    print(json.dumps(status), flush=True)
    if status['phase']['phase'] in ['complete', 'failed']:
        break
    if datetime.datetime.now(datetime.timezone.utc) >= deadline:
        print('Preserving partial results before pod TTL.', flush=True)
        break
    time.sleep(30)

receipt = remote("import runpy;runpy.run_path('/root/painter/export_runtime.py')")
print('Exported ' + json.dumps(receipt), flush=True)
result = collect(shlex.join(connection()), '/root/painter', P / 'collected-final')
result['phase_at_collection'] = status
(P / 'final-collection-result.json').write_text(json.dumps(result, indent=2) + '\n')
print('Verified ' + json.dumps(result), flush=True)
