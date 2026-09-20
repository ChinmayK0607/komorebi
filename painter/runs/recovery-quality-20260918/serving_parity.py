import base64,hashlib,io,json,sys
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
P=Path(__file__).resolve().parent;assert sys.platform=='linux'
processor=AutoProcessor.from_pretrained((P/'model-path.txt').read_text().strip(),local_files_only=True)
results=[]
for key,item in json.loads((P/'training-prefixes.json').read_text()).items():
 photos=[];hf=[]
 for message in item['row']['messages'][:-1]:
  parts=[]
  for part in message['content']:
   if part['type']=='image_url':photos.append(Image.open(io.BytesIO(base64.b64decode(part['image_url']['url'].split(',',1)[1]))).convert('RGB'));parts.append({'type':'image'})
   else:parts.append(part)
  hf.append(dict(role=message['role'],content=parts))
 text=processor.apply_chat_template(hf,tokenize=False,add_generation_prompt=True,enable_thinking=False)
 processed=processor(text=[text],images=photos or None,return_tensors='np',padding=False)
 checks=dict(prefix_equal=processed['input_ids'][0].tolist()==item['tokens'],image_grid_equal=processed.get('image_grid_thw',[]).tolist()==item['image_grid'] if photos else item['image_grid']==[],pixels_equal=hashlib.sha256(processed['pixel_values'].tobytes()).hexdigest()==item['pixel_sha256'] if photos else item['pixel_sha256'] is None)
 results.append(dict(images=int(key),**checks))
r=dict(status='passed' if all(all(v for k,v in row.items() if k!='images') for row in results) else 'mismatch',rows=results)
(P/'serving-parity.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if r['status']!='passed':raise SystemExit('Actual training/serving template or image preprocessing differs; inspect this diagnostic')
