"""Single-node matched baseline -> short SFT -> matched checkpoint evaluation."""
import os,sys,json,time,subprocess,signal,fcntl
from pathlib import Path
from foundation_lock import check
P=Path(__file__).resolve().parent;assert sys.platform=='linux';os.chdir(P)
lock=(P/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
check()
env=dict(os.environ);env.update(HF_HOME=str(P/'cache/huggingface'),HF_TOKEN_PATH=str(P/'private/hf-token'),UV_CACHE_DIR=str(P/'cache/uv'),BRUSH_ROOT=str(P/'source/brush-rl'),PRIME_ROOT=str(P/'prime-rl'),BRUSH_MODEL_PROFILE='qwen38_27b',PLAYWRIGHT_BROWSERS_PATH=str(P/'browsers'),WANDB_MODE='disabled',TOKENIZERS_PARALLELISM='false')
server=None;telemetry=None

def phase(name,**extra):
 (P/'phase.json').write_text(json.dumps(dict(phase=name,at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra),indent=2)+'\n');print(name,flush=True)
def execute(args,log,timeout):
 with open(P/'logs'/log,'a') as f:proc=subprocess.Popen(args,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 try:code=proc.wait(timeout=timeout)
 except subprocess.TimeoutExpired:
  os.killpg(proc.pid,signal.SIGTERM)
  try:proc.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  raise
 if code:raise RuntimeError(f'{log} exited {code}')
def start_server(adapters):
 global server
 with open(P/'logs/server.log','a') as f:
  server=subprocess.Popen(['bash','serve.sh',str(P),'--lora-modules',*[name+'='+str(path) for name,path in adapters.items()]],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 execute([str(P/'runtime/bin/python'),'wait_server.py',str(server.pid),'base'],'server-ready.log',1200)
def stop_server():
 global server
 if server and server.poll() is None:
  os.killpg(server.pid,signal.SIGTERM)
  try:server.wait(timeout=45)
  except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
 server=None
try:
 (P/'logs').mkdir(exist_ok=True);phase('environment_setup')
 deadline=time.monotonic()+5400
 while not (P/'training-ready.json').exists():
  if time.monotonic()>deadline:raise TimeoutError('Environment setup timeout')
  if (P/'training-setup.exit').exists() and (P/'training-setup.exit').read_text().strip()!='0':raise RuntimeError('Environment setup failed')
  time.sleep(5)
 with open(P/'gpu-telemetry.csv','w') as f:telemetry=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw','--format=csv','-l','10'],stdout=f,stderr=subprocess.STDOUT)
 parent=Path((P/'initial-adapter-path.txt').read_text().strip())
 phase('matched_parent_baseline');start_server({'parent':parent})
 execute([str(P/'runtime/bin/python'),'evaluate_tasks.py','--model','parent','--samples','4'],'parent-evaluation.log',4200)
 phase('collecting_training_only_student_states')
 execute([str(P/'runtime/bin/python'),'evaluate_tasks.py','--model','parent','--samples','1','--tasks','training-state-tasks.json','--output','training-states'],'training-states.log',1200)
 stop_server()
 phase('preparing_sft')
 execute([str(P/'prime-rl/.venv/bin/python'),'prepare_sft.py'],'preparation.log',1800)
 execute([str(P/'runtime/bin/python'),'serving_parity.py'],'serving-parity.log',300)
 phase('training');execute(['bash','train_only.sh',str(P)],'train-launch.log',3000)
 phase('matched_checkpoint_evaluation')
 root=P/'train-output/contract-quality-20260918/artifacts/adapters'
 adapters={'parent':parent,'midpoint':root/'step_24','trained':root/'step_72'}
 for name in ['midpoint','trained']:
  receipt=json.loads((adapters[name]/'hf-upload.json').read_text());assert receipt['verified'] and not receipt['private']
 start_server(adapters)
 for name in ['midpoint','trained']:
  execute([str(P/'runtime/bin/python'),'evaluate_tasks.py','--model',name,'--samples','4'],name+'-evaluation.log',4200)
 stop_server();phase('exporting_evidence')
 execute(['python3','export_runtime.py'],'export.log',600)
 phase('complete',visual_judge='Luna blinded agent review after collection',rl_optimizer_updates=0)
 (P/'pipeline.exit').write_text('0\n')
except BaseException as exc:
 phase('failed',error=f'{type(exc).__name__}: {exc}');(P/'pipeline.exit').write_text('1\n');raise
finally:
 stop_server()
 if telemetry and telemetry.poll() is None:telemetry.terminate();telemetry.wait(timeout=10)
