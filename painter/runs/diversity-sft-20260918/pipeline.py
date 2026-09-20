"""Bounded, node-only baseline -> SFT -> matched evaluation. Each phase is logged."""
import os,sys,json,time,subprocess,signal,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;assert sys.platform=='linux';os.chdir(P)
env=dict(os.environ);env.update(HF_HOME=str(P/'cache/huggingface'),HF_TOKEN_PATH=str(P/'private/hf-token'),UV_CACHE_DIR=str(P/'cache/uv'),BRUSH_ROOT=str(P/'source/brush-rl'),PRIME_ROOT=str(P/'prime-rl'),BRUSH_MODEL_PROFILE='qwen38_27b',PLAYWRIGHT_BROWSERS_PATH=str(P/'browsers'),WANDB_MODE='disabled',TOKENIZERS_PARALLELISM='false')
server=None;telemetry=None

def phase(name,**extra):
 (P/'phase.json').write_text(json.dumps(dict(phase=name,at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra),indent=2)+'\n');print(name,flush=True)
def execute(args,log,timeout):
 with open(P/'logs'/log,'a') as f:
  proc=subprocess.Popen(args,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=proc.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   raise
 if code:raise RuntimeError(f'{log} exited {code}')
def start_server(adapters=()):
 global server
 with open(P/'logs'/('server-trained.log' if adapters else 'server-base.log'),'a') as f:server=subprocess.Popen(['bash','serve.sh',str(P),*adapters],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 execute([str(P/'runtime/bin/python'),'wait_server.py',str(server.pid),'base'],'server-ready.log',1200)
def stop_server():
 global server
 if server and server.poll() is None:
  os.killpg(server.pid,signal.SIGTERM)
  try:server.wait(timeout=45)
  except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
 server=None
try:
 phase('setup_and_rendering')
 deadline=time.monotonic()+6000
 while not (P/'training-ready.json').exists() or not (P/'teacher-review.json').exists():
  if time.monotonic()>deadline:raise TimeoutError('Environment setup timeout')
  if (P/'training-setup.exit').exists() and (P/'training-setup.exit').read_text().strip()!='0':raise RuntimeError('Environment setup failed')
  time.sleep(5)
 with open(P/'gpu-telemetry.csv','w') as f:telemetry=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw','--format=csv','-l','10'],stdout=f,stderr=subprocess.STDOUT)
 review=json.loads((P/'teacher-review.json').read_text())
 assert review['status']=='reviewed' and review['judge_model']=='gpt-5.6-luna'
 assert review['source_programs_sha256']==hashlib.sha256((P/'diversity/programs.jsonl').read_bytes()).hexdigest()
 if '--from-prepared' in sys.argv:
  receipt=json.loads((P/'dataset-receipt.json').read_text())
  for split in ['train','validation']:
   assert hashlib.sha256((P/'sft-data'/(split+'.jsonl')).read_bytes()).hexdigest()==receipt[split+'_sha256']
  assert json.loads((P/'preflight.json').read_text())['status']=='prepared'
  assert json.loads((P/'serving-parity.json').read_text())['status']=='passed'
  assert json.loads((P/'replay-id-repair.json').read_text())['actual_loader_passed']
 else:
  phase('preparing_training')
  execute([str(P/'prime-rl/.venv/bin/python'),'prepare_sft.py'],'preparation.log',3600)
  execute([str(P/'runtime/bin/python'),'serving_parity.py'],'serving-parity.log',300)
 phase('training');execute(['bash','train_only.sh',str(P)],'train-launch.log',11200)
 phase('matched_evaluation')
 root=P/'train-output/diversity-sft-20260918/artifacts/adapters'
 adapters={'parent':Path((P/'initial-adapter-path.txt').read_text().strip()),'midpoint':root/'step_192','trained':root/'step_384'}
 for name in ['midpoint','trained']:
  receipt=json.loads((adapters[name]/'hf-upload.json').read_text());assert receipt['verified'] and not receipt['private']
 start_server(['--lora-modules',*[name+'='+str(adapter) for name,adapter in adapters.items()]])
 for name in adapters:
  execute([str(P/'runtime/bin/python'),'evaluate_tasks.py','--model',name,'--samples','2'],name+'-evaluation.log',3600)
 stop_server();phase('complete',visual_judge='Luna agent after evidence collection',rl_optimizer_updates=0);(P/'pipeline.exit').write_text('0\n')
except BaseException as exc:
 phase('failed',error=f'{type(exc).__name__}: {exc}');(P/'pipeline.exit').write_text('1\n');raise
finally:
 stop_server()
 if telemetry and telemetry.poll() is None:telemetry.terminate();telemetry.wait(timeout=10)
