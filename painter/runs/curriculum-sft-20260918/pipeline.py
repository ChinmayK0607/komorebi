"""Bounded, node-only baseline -> SFT -> matched evaluation. Each phase is logged."""
import os,sys,json,time,subprocess,signal
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
 phase('waiting_for_model_runtime')
 deadline=time.monotonic()+3600
 while not (P/'model-path.txt').exists() or not (P/'runtime/bin/vllm').exists():
  if time.monotonic()>deadline:raise TimeoutError('Model/runtime setup timeout')
  if (P/'training-setup.exit').exists() and (P/'training-setup.exit').read_text().strip()!='0':raise RuntimeError('Environment setup failed')
  time.sleep(5)
 with open(P/'gpu-telemetry.csv','w') as f:telemetry=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw','--format=csv','-l','10'],stdout=f,stderr=subprocess.STDOUT)
 phase('baseline_harness_check');start_server()
 # An unchanged base endpoint remains available after SFT, so the full pass@8
 # baseline can be collected alongside trained samples without blocking SFT.
 execute([str(P/'runtime/bin/python'),'evaluate_batch.py','--model','base','--samples','1','--wait-references'],'baseline-initial.log',3600)
 stop_server();phase('preparing_training')
 while not (P/'training-ready.json').exists():
  if time.monotonic()>deadline+1800:raise TimeoutError('Rendered data/setup timeout')
  if (P/'training-setup.exit').exists() and (P/'training-setup.exit').read_text().strip()!='0':raise RuntimeError('Environment setup failed')
  time.sleep(5)
 execute([str(P/'prime-rl/.venv/bin/python'),'prepare_sft.py'],'preparation.log',1800)
 execute([str(P/'runtime/bin/python'),'serving_parity.py'],'serving-parity.log',300)
 phase('training');execute(['bash','train_only.sh',str(P)],'train-launch.log',7500)
 phase('matched_evaluation')
 adapter=P/'train-output/curriculum-sft-20260918/artifacts/adapters/step_96'
 receipt=json.loads((adapter/'hf-upload.json').read_text());assert receipt['verified'] and not receipt['private']
 start_server(['--lora-modules','trained='+str(adapter)])
 execute([str(P/'runtime/bin/python'),'evaluate_batch.py','--model','base','--samples','8'],'baseline-full.log',3600)
 execute([str(P/'runtime/bin/python'),'evaluate_batch.py','--model','trained','--samples','8'],'trained-evaluation.log',3600)
 phase('complete',checkpoint=receipt['url']);(P/'pipeline.exit').write_text('0\n')
except BaseException as exc:
 phase('failed',error=f'{type(exc).__name__}: {exc}');(P/'pipeline.exit').write_text('1\n');raise
finally:
 stop_server()
 if telemetry and telemetry.poll() is None:telemetry.terminate();telemetry.wait(timeout=10)
