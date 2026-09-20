"""Pure metadata and mocked I/O tests; no model or painting renderer runs locally."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import contract
import demonstrations
import recovery_data
import render_demos
import rollout


class ContractTests(unittest.TestCase):
    def test_consumers_share_prompt_and_targets(self):
        self.assertEqual(rollout.SYSTEM, contract.SYSTEM)
        self.assertEqual(recovery_data.SYSTEM, contract.SYSTEM)
        self.assertEqual(render_demos.SYSTEM, contract.SYSTEM)
        _, programs=demonstrations.recipe('box',17,'train')
        for code in programs:
            answer=contract.paint_target('Match the reference.',code)
            self.assertNotIn('FINISHED', answer)
            self.assertEqual(rollout.sketch(answer).strip(),code.strip())
        self.assertNotIn('```',contract.finish_target())

    def test_legacy_transform_is_not_silently_mixed_back_into_sft(self):
        _, programs=demonstrations.recipe('box',17,'train')
        with self.assertRaisesRegex(ValueError,'Legacy affine'):
            contract.paint_target('Paint.', 'let frame=[1,0,0,1,0,0];\n'+programs[-1])
        for extra in ['translate(12,30);','scale(.5);','rotate(.3);']:
            with self.subTest(extra=extra),self.assertRaises(ValueError):
                contract.validate_teacher_program(programs[-1]+extra)

    def test_materialization_separates_paint_and_observed_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'ref.png').write_bytes(b'fixture');(root/'program.js').write_text(demonstrations.recipe('box',17,'train')[1][-1])
            row=dict(kind='paint',reference='ref.png',target_program='program.js',plan='Paint.')
            paint=recovery_data.materialize(root,row)
            self.assertEqual(paint['contract'],contract.identity())
            self.assertNotIn('FINISHED',paint['messages'][-1]['content'][0]['text'])
            with self.assertRaisesRegex(ValueError,'observed canvas'):
                recovery_data.materialize(root,dict(row,kind='finish'))
            finish=recovery_data.materialize(root,dict(row,kind='finish',canvas='ref.png',program='program.js'))
            self.assertIn('FINISHED',finish['messages'][-1]['content'][0]['text'])
            self.assertEqual(sum(x['type']=='image_url' for x in finish['messages'][1]['content']),2)


class PreviewTests(unittest.TestCase):
    def episode(self,replies,max_turns=None,tamper=False):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name)
        ref=root/'ref.png';ref.write_bytes(b'reference-fixture')
        renderer=root/'renderer.py';renderer.write_text('# fixture only')
        seen=[];rendered=[]
        def complete(endpoint,model,messages,*args,**kwargs):
            seen.append(messages)
            return dict(response=dict(choices=[dict(message=dict(content=replies[len(seen)-1]),finish_reason='stop')],usage=dict(completion_tokens=10)),seconds=0)
        def render(args,**kwargs):
            source=Path(args[args.index('--output')-1]);dest=Path(args[args.index('--output')+1])
            dest.write_bytes(b'new-fixture-'+str(len(rendered)).encode());rendered.append(dest)
            dest.with_suffix('.json').write_text(json.dumps(dict(valid=True,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                png_sha256='wrong' if tamper else hashlib.sha256(dest.read_bytes()).hexdigest())))
        config=dict(operational_limits=dict(max_turns=max_turns or len(replies),max_tokens_per_response=500,episode_timeout_seconds=60,render_timeout_seconds=10))
        with patch.object(rollout.sys,'platform','linux'),patch.object(rollout,'completion',complete),patch.object(rollout.subprocess,'run',render):
            result=rollout.run(ref,root/'out','http://unused','fixture',renderer,root/'python',config)
        return root,result,seen,rendered

    def test_code_plus_finish_observes_actual_canvas_before_finishing(self):
        code='```javascript\nfunction draw(){noLoop();}\n```\nFINISHED'
        root,result,seen,rendered=self.episode([code,'FINISHED'])
        self.assertEqual(len(rendered),1)
        self.assertFalse(result['turns'][0]['finished'])
        self.assertTrue(result['turns'][0]['finish_deferred'])
        self.assertTrue(result['finished'])
        self.assertEqual(result['turns'][1]['observed_canvas'],str(rendered[0]))
        self.assertIn(render_demos.image(rendered[0]),seen[1][1]['content'])
        self.assertEqual(json.loads((root/'out/binding.json').read_text())['contract'],contract.identity())

    def test_can_revise_after_a_successful_render(self):
        code=lambda n:'```js\nfunction draw(){/*'+str(n)+'*/noLoop();}\n```\nFINISHED'
        _,result,seen,rendered=self.episode([code(1),code(2),'FINISHED'])
        self.assertEqual(len(rendered),2)
        self.assertTrue(result['finished'])
        self.assertEqual(result['turns'][-1]['observed_canvas'],str(rendered[-1]))
        self.assertIn(render_demos.image(rendered[-1]),seen[-1][1]['content'])

    def test_turn_limit_does_not_count_uninspected_paint_as_completed(self):
        _,result,_,_=self.episode(['```js\nfunction draw(){noLoop();}\n```\nFINISHED'])
        self.assertTrue(result['valid_canvas'])
        self.assertFalse(result['finished'])
        self.assertTrue(result['operational_cutoff'])

    def test_png_tampering_does_not_become_a_valid_canvas(self):
        _,result,_,_=self.episode(['```js\nfunction draw(){noLoop();}\n```'],tamper=True)
        self.assertFalse(result['valid_canvas'])
        self.assertIn('PNG hash mismatch',result['turns'][0]['error'])

if __name__=='__main__':unittest.main()
