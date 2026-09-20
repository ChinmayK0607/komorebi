"""On-policy single-decision sampling with exact vLLM token IDs/logprobs."""
import json,sys,os,hashlib,re,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
P=Path(__file__).resolve().parent

def main():
 assert sys.platform=='linux','Inference and rendering run on the GPU node only'
 from PIL import Image
 from transformers import AutoProcessor
 from vllm import LLM,SamplingParams
 from vllm.lora.request import LoRARequest
 from contract import SYSTEM,NEXT_VERSION
 from render_demos import atomic
 e=json.loads((P/'EXPERIMENT.json').read_text());states=json.loads((P/'rl-states.json').read_text());model=(P/'model-path.txt').read_text().strip();adapter=(P/'initial-adapter-path.txt').read_text().strip()
 out=P/'rl-rollouts';out.mkdir(exist_ok=True)
 processor=AutoProcessor.from_pretrained(model,local_files_only=True)
 tasks=[];prompts=[]
 for state in states:
  program=(P/state['program']).read_text()
  messages=[{'role':'system','content':SYSTEM},{'role':'user','content':[{'type':'text','text':'REFERENCE'},{'type':'image'},{'type':'text','text':'CURRENT CANVAS'},{'type':'image'},{'type':'text','text':'Current program:\n'+program},{'type':'text','text':NEXT_VERSION}]}]
  text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
  photos=[Image.open(P/state[k]).convert('RGB') for k in ['reference','canvas']]
  for k in range(e['samples_per_state']):
   ident=f"{state['id']}-sample-{k:02d}";folder=out/ident;folder.mkdir(exist_ok=True)
   if (folder/'sample.json').exists():continue
   tasks.append((state,k,folder,text));prompts.append({'prompt':text,'multi_modal_data':{'image':photos}})
 if tasks:
  llm=LLM(model=model,dtype='bfloat16',max_model_len=16384,max_num_seqs=16,max_num_batched_tokens=16384,gpu_memory_utilization=.90,enable_lora=True,max_lora_rank=16,max_loras=1,limit_mm_per_prompt={'image':2,'video':0},gdn_prefill_backend='triton',kernel_config={'enable_jit_warmup':False},mm_processor_kwargs={'min_pixels':3136,'max_pixels':1048576})
  params=[SamplingParams(temperature=e['temperature'],top_p=1.0,top_k=-1,min_p=0.0,max_tokens=e['max_tokens'],logprobs=1,seed=918300+k) for state,k,folder,text in tasks]
  outputs=llm.generate(prompts,params,lora_request=LoRARequest('pilot',1,adapter),use_tqdm=True)
  for task,result in zip(tasks,outputs,strict=True):
   state,k,folder,text=task;choice=result.outputs[0]
   ids=list(choice.token_ids);lps=[float(row[token].logprob) for row,token in zip(choice.logprobs,ids,strict=True)]
   atomic(folder/'sample.json',dict(id=folder.name,state=state,sample=k,prompt=text,prompt_token_ids=list(result.prompt_token_ids),completion_token_ids=ids,completion_logprobs=lps,text=choice.text,finish_reason=choice.finish_reason,stop_reason=choice.stop_reason,temperature=e['temperature'],policy=e['initial_adapter'],source_sha256={key:hashlib.sha256((P/state[key]).read_bytes()).hexdigest() for key in ['reference','canvas','program']}))
  # Exit the inference process before rendering/training to release GPU memory.
 atomic(P/'sampling-complete.json',dict(samples=len(list(out.glob('*/sample.json'))),expected=len(states)*e['samples_per_state']))

if __name__=='__main__':main()
