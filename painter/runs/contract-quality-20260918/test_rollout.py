"""Regression for useful feedback after a failed browser render."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import rollout

class RolloutFeedbackTests(unittest.TestCase):
 def test_finish_keeps_existing_canvas_without_rendering(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);(p/'renderer.py').write_text('# fixture renderer');ref=p/'reference.png';ref.write_bytes(b'fixture-only');program=p/'program.js';program.write_text('function draw(){noLoop();}')
   response={'response':{'choices':[{'message':{'content':'Already complete.\nFINISHED'},'finish_reason':'stop'}],'usage':{'completion_tokens':5}},'seconds':1}
   cfg={'operational_limits':{'max_turns':2,'max_tokens_per_response':500,'episode_timeout_seconds':60,'render_timeout_seconds':10}}
   with patch.object(rollout.sys,'platform','linux'),patch.object(rollout,'completion',return_value=response),patch.object(rollout.subprocess,'run') as renderer:
    result=rollout.run(ref,p/'episode','http://unused','fixture',p/'renderer.py',p/'python',cfg,initial_canvas=ref,initial_program=program)
   renderer.assert_not_called();self.assertTrue(result['finished']);self.assertEqual(result['canvas'],str(ref.resolve()));self.assertEqual(result['turns'][0]['action'],'finish')

 def test_repeated_program_gets_no_change_feedback(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);(p/'renderer.py').write_text('# fixture renderer');ref=p/'reference.png';ref.write_bytes(b'fixture-only');program=p/'program.js';code='function draw(){noLoop();}\n';program.write_text(code);seen=[]
   def completion(*args,**kwargs):
    seen.append(args[2]);text='Plan.\n```javascript\n'+code+'```' if len(seen)==1 else 'Keep the current painting.\nFINISHED'
    return {'response':{'choices':[{'message':{'content':text},'finish_reason':'stop'}],'usage':{'completion_tokens':10}},'seconds':1}
   cfg={'operational_limits':{'max_turns':2,'max_tokens_per_response':500,'episode_timeout_seconds':60,'render_timeout_seconds':10}}
   with patch.object(rollout.sys,'platform','linux'),patch.object(rollout,'completion',completion),patch.object(rollout.subprocess,'run') as renderer:
    result=rollout.run(ref,p/'episode','http://unused','fixture',p/'renderer.py',p/'python',cfg,initial_canvas=ref,initial_program=program)
   renderer.assert_not_called();self.assertIn('made no change',seen[1][-1]['content'][-1]['text']);self.assertEqual(len(result['turns']),2);self.assertTrue(result['finished'])

 def test_finish_cannot_accept_blank_canvas(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);(p/'renderer.py').write_text('# fixture renderer');ref=p/'reference.png';ref.write_bytes(b'fixture-only')
   response={'response':{'choices':[{'message':{'content':'FINISHED'},'finish_reason':'stop'}],'usage':{'completion_tokens':2}},'seconds':1}
   cfg={'operational_limits':{'max_turns':1,'max_tokens_per_response':500,'episode_timeout_seconds':60,'render_timeout_seconds':10}}
   with patch.object(rollout.sys,'platform','linux'),patch.object(rollout,'completion',return_value=response),patch.object(rollout.subprocess,'run') as renderer:
    result=rollout.run(ref,p/'episode','http://unused','fixture',p/'renderer.py',p/'python',cfg)
   renderer.assert_not_called();self.assertFalse(result['finished']);self.assertFalse(result['valid_canvas']);self.assertIn('Cannot finish',result['turns'][0]['error'])

 def test_fence_whitespace_keeps_valid_program(self):
  code='function draw(){noLoop();}\n'
  for opening in ['```javascript\n','``` javascript\n','```\tjs\t\r\n','```\n']:
   with self.subTest(opening=opening):
    self.assertEqual(rollout.sketch('Plan.\n'+opening+code+'```\nFINISHED'),code)

 def test_finish_without_program_remains_explicit_failure(self):
  with self.assertRaisesRegex(ValueError,'No complete JavaScript code block'):
   rollout.sketch('FINISHED')

 def test_failed_render_returns_actual_js_error_and_can_recover(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);(p/'renderer.py').write_text('# fixture renderer');ref=p/'reference.png';ref.write_bytes(b'fixture-only')
   seen=[]
   def completion(endpoint,model,messages,*args,**kwargs):
    seen.append(messages)
    reply=('FINISHED' if len(seen)==3 else 'Plan.\n```javascript\nfunction draw(){noLoop();}\n```\nFINISHED')
    return {'response':{'choices':[{'message':{'content':reply},'finish_reason':'stop'}],'usage':{'completion_tokens':20}},'seconds':1}
   calls=[]
   def render(args,**kwargs):
    calls.append(args);source=Path(args[args.index('--output')-1]);dest=Path(args[args.index('--output')+1])
    result={'valid':len(calls)>1,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'errors':[] if len(calls)>1 else ['TypeError: brush.notAnAPI is not a function']}
    dest.with_suffix('.json').write_text(json.dumps(result))
    if result['valid']:
     dest.write_bytes(b'fixture-only');result['png_sha256']=hashlib.sha256(dest.read_bytes()).hexdigest()
     dest.with_suffix('.json').write_text(json.dumps(result))
   cfg={'operational_limits':{'max_turns':3,'max_tokens_per_response':500,'episode_timeout_seconds':60,'render_timeout_seconds':10}}
   with patch.object(rollout.sys,'platform','linux'),patch.object(rollout,'completion',completion),patch.object(rollout.subprocess,'run',render):
    result=rollout.run(ref,p/'episode','http://unused','fixture',p/'renderer.py',p/'python',cfg)
   self.assertIn('brush.notAnAPI',seen[1][-1]['content'][-1]['text'])
   self.assertTrue(result['finished']);self.assertTrue(result['valid_canvas']);self.assertEqual(len(result['turns']),3)
   self.assertEqual(result['generated_tokens'],60)

if __name__=='__main__':unittest.main()
