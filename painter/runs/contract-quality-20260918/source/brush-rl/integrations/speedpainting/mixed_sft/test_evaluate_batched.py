import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('batched_eval',HERE/'evaluate_bounded.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def manifest():
 return {'version':1,'examples':[{'id':str(i),'split':'validation','budget':64,'messages':[{'role':'system','content':[{'type':'text','text':'Paint using bounded-brush-v1 JSON only.'}]},{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,eA=='}}]}]} for i in range(4)]}

def response(finish='stop'):
 return {'choices':[{'message':{'content':json.dumps({'version':'bounded-brush-v1','paperRGB':[255,255,255],'actions':[{'type':'STOP'}]})},'finish_reason':finish}]}

class Tests(unittest.TestCase):
 def test_four_requests_overlap_and_match_ids(self):
  barrier=threading.Barrier(4)
  def fetch(request,timeout):
   self.assertEqual(timeout,1800);payload=json.loads(request.data);self.assertEqual(payload['seed'],42);self.assertEqual(payload['max_tokens'],24576)
   barrier.wait(timeout=5)
   return io.BytesIO(json.dumps(response()).encode())
  with tempfile.TemporaryDirectory() as d,patch.object(m,'urlopen',side_effect=fetch):
   out=Path(d)/'out';result=m.evaluate(manifest(),'http://localhost/v1','painting',out)
   self.assertEqual(result['valid'],4);self.assertEqual([r['id'] for r in result['results']],['0','1','2','3'])
   self.assertTrue((out/'0/response.raw').exists())
 def test_length_is_model_invalid_not_infrastructure(self):
  with tempfile.TemporaryDirectory() as d,patch.object(m,'urlopen',side_effect=lambda *a,**kw:io.BytesIO(json.dumps(response('length')).encode())):
   result=m.evaluate(manifest(),'http://localhost/v1','painting',Path(d)/'out')
   self.assertEqual(result['valid'],0);self.assertEqual(result['infrastructure_errors'],0)
   self.assertIn('truncated',result['results'][0]['validation_error'])
 def test_api_failure_is_infrastructure(self):
  with tempfile.TemporaryDirectory() as d,patch.object(m,'urlopen',side_effect=TimeoutError('deadline')):
   result=m.evaluate(manifest(),'http://localhost/v1','painting',Path(d)/'out')
   self.assertEqual(result['infrastructure_errors'],4);self.assertTrue(all('validation_error' not in r for r in result['results']))
 def test_v2_answer_is_invalid_for_legacy_v1_prompt(self):
  body=response();doc=json.loads(body['choices'][0]['message']['content']);doc['version']='bounded-brush-v2';body['choices'][0]['message']['content']=json.dumps(doc)
  with tempfile.TemporaryDirectory() as d,patch.object(m,'urlopen',side_effect=lambda *a,**kw:io.BytesIO(json.dumps(body).encode())):
   result=m.evaluate(manifest(),'http://localhost/v1','painting',Path(d)/'out')
   self.assertEqual(result['valid'],0);self.assertEqual(result['infrastructure_errors'],0)
   self.assertTrue(all('requested contract' in r['validation_error'] for r in result['results']))
 def test_v2_request_is_rejected_before_api(self):
  cases=manifest()
  for row in cases['examples']:
   row['action_contract']='bounded-brush-v2';row['messages'][0]['content'][0]['text']='Use bounded-brush-v2 JSON.'
  with tempfile.TemporaryDirectory() as d,patch.object(m,'urlopen') as fetch:
   with self.assertRaisesRegex(ValueError,'v1 only'):
    m.evaluate(cases,'http://localhost/v1','painting',Path(d)/'out')
   fetch.assert_not_called()

if __name__=='__main__':unittest.main()
