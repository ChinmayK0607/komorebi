"""Analyze preserved JSON/programs and display existing PNGs; no model/render work."""
import collections
import hashlib
import html
import json
import re
import statistics
from pathlib import Path

P = Path(__file__).resolve().parent
RUNS = P.parent
EVIDENCE = P / 'collected-final/evidence'


def read(path):
    return json.loads(path.read_text())


def lengths(rows):
    turns = [t for r in rows for t in r.get('turns', []) if t.get('tokens') is not None]
    values = [t['tokens'] for t in turns]
    clean = [t['tokens'] for t in turns if 'Operational token cutoff' not in t.get('error', '')]
    first = [r['turns'][0]['tokens'] for r in rows if r.get('turns')]
    return dict(episodes=len(rows), responses=len(values), tokens=sum(values),
                mean_per_turn=statistics.mean(values), median_per_turn=statistics.median(values),
                mean_excluding_token_cutoffs=statistics.mean(clean),
                token_cutoff_responses=len(values)-len(clean),
                mean_first_turn=statistics.mean(first), median_first_turn=statistics.median(first))


def main():
    out = P / 'diagnostic'
    out.mkdir(exist_ok=True)
    history, targets, controls = {}, {}, {}
    for name in ['curriculum-sft-20260918', 'recovery-quality-20260918', 'diversity-sft-20260918']:
        root = RUNS/name/'collected-final/evidence'
        history[name] = {}
        for f in sorted((root/'evaluation').glob('*/summary.json')):
            rows = read(f)['results']
            groups = {'all': lengths(rows)}
            for kind in ['paint', 'repair', 'finish']:
                selected = [r for r in rows if r.get('task', {}).get('kind') == kind]
                if selected:
                    groups[kind] = lengths(selected)
            history[name][f.parent.name] = groups
        buckets = collections.defaultdict(list)
        for row in read(root/'token-audit.json'):
            if not row['validation']:
                buckets[row['group']].append(row['supervised_tokens'])
        targets[name] = {k: dict(pool_examples=len(v), mean=statistics.mean(v),
                                median=statistics.median(v), minimum=min(v), maximum=max(v))
                         for k,v in buckets.items()}
        client = (root/'client.py').read_bytes()
        controls[name] = dict(client_sha256=hashlib.sha256(client).hexdigest(),
                              operational_limits=read(root/'campaign.json')['operational_limits'],
                              thinking_disabled=b"'enable_thinking':False" in client)

    decoded = [r for r in read(P/'luna-review/decoded-judgments.json')
               if r['comparison'] == 'parent_vs_trained']
    labels = {(r['task_id'], r['sample']): r for r in decoded}
    family, references, samples = (collections.defaultdict(collections.Counter) for _ in range(3))
    for row in decoded:
        for bucket, key in [(family,row['family']), (references,row['task_id']), (samples,row['sample'])]:
            bucket[key][row['outcome']] += 1
    coding, cases = {}, []
    for model in ['parent', 'midpoint', 'trained']:
        counts = collections.Counter()
        for row in read(EVIDENCE/'evaluation'/model/'summary.json')['results']:
            if row['task']['kind'] != 'paint':
                continue
            last = next(t for t in reversed(row['turns']) if t.get('valid'))
            folder = EVIDENCE/'evaluation'/model/row['task']['id']/f'sample-{row["sample"]:02d}'/f'turn-{last["turn"]}'
            code = (folder/'program.js').read_text()
            content = read(folder/'response.json')['response']['choices'][0]['message']['content']
            plan_words = len(content.split('```')[0].split())
            style = 'affine_helper' if re.search(r'function\s+mapped\(', code) else 'direct'
            counts[style] += 1
            result = labels[(row['task']['id'], row['sample'])]['outcome']
            if model == 'trained':
                counts[style+'_'+result] += 1
            cases.append(dict(model=model, task=row['task']['id'], sample=row['sample'], style=style,
                              final_turn_tokens=last['tokens'], episode_tokens=row['generated_tokens'],
                              attempts=len(row['turns']), code_path=str(folder/'program.js'),
                              plan_words=plan_words, code_characters=len(code),
                              helper_characters=len(code.split('function draw',1)[0]) if 'function draw' in code else None,
                              parent_comparison=result if model == 'trained' else None))
        coding[model] = dict(counts)

    # Verify the arithmetic against literals in this exact preserved program.
    cube = next(x for x in cases if x['model']=='trained' and x['task']=='v2-box-007-paint' and x['sample']==0)
    code = Path(cube['code_path']).read_text()
    matches = re.findall(r'frame\s*=\s*\[([^\]]+)\]', code)
    matrix = [float(x) for x in matches[-1].split(',')]
    xy = re.search(r'const x=([\d.]+),y=([\d.]+)', code)
    x, y = map(float, xy.groups())
    cube_math = dict(local_xy=[x,y], frame=matrix,
                     mapped_xy=[matrix[0]*x+matrix[2]*y+matrix[4], matrix[1]*x+matrix[3]*y+matrix[5]],
                     program=cube['code_path'], note='The preserved PNG is blank; mapped center is above the canvas.')
    result = dict(history=history, target_lengths=targets, serving_controls=controls,
                  family_outcomes=family, sample_outcomes=samples, reference_outcomes=references,
                  references_with_any_win=sum(v['win']>0 for v in references.values()),
                  references_with_two_wins=sum(v['win']==2 for v in references.values()),
                  coding_styles=coding, cube_arithmetic=cube_math, cases=cases,
                  caveats=['Historical runs use different task/reference sets; compare models within each run.',
                           'Style/outcome association is observational, not a randomized representation ablation.',
                           'Token counts are recorded completion usage; target counts also include EOS.',
                           'Two samples per reference do not establish reliable family-wide gains.'])
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')

    recent = history['diversity-sft-20260918']
    table = '<table><tr><th>Checkpoint</th><th>Painting tokens/turn</th><th>Excluding cutoff replies</th><th>Median</th></tr>'
    for model in ['parent','midpoint','trained']:
        v=recent[model]['paint']
        table+=f'<tr><td>{model}</td><td>{v["mean_per_turn"]:.0f}</td><td>{v["mean_excluding_token_cutoffs"]:.0f}</td><td>{v["median_per_turn"]:.0f}</td></tr>'
    table+='</table>'
    tasks={r['id']:r for r in read(EVIDENCE/'evaluation-tasks.json')}

    def figure(path, caption):
        relative=Path('..')/'collected-final/evidence'/str(path).removeprefix('/root/painter/')
        assert (out/relative).is_file()
        return f'<figure><a href="{html.escape(str(relative))}"><img src="{html.escape(str(relative))}"></a><figcaption>{html.escape(caption)}</figcaption></figure>'

    cards=[]
    descriptions={
        'v2-box-007-paint':'Confirmed coordinate error: the plan says (337,312), but the transform sends its center to (466,-122). More generated code did not prevent a blank canvas.',
        'v3-house-0011-paint':'Scale error: a local house roughly108 units wide receives another scale near0.23, producing a tiny mark.',
        'v3-flower-0005-paint':'Mixed capability: final sample0 changes the subject; sample1 is a recognizable flower and wins against its matched parent. Relative improvement is not an exact reconstruction.',
        'v2-bus-007-paint':'A genuine relative gain in sample1: a coherent bus after three execution repairs. Its geometry is still not an exact reference match.'}
    for ident, description in descriptions.items():
        columns=[figure(tasks[ident]['reference'],'Reference')]
        for model,k in [('parent',0),('trained',0),('parent',1),('trained',1)]:
            row=read(EVIDENCE/'evaluation'/model/ident/f'sample-{k:02d}/summary.json')
            columns.append(figure(row['canvas'],f'{model}, sample{k}: {row["turns"][-1]["tokens"]} tokens in final response; {row["generated_tokens"]} total'))
        cards.append(f'<article><h2>{html.escape(ident)}</h2><p>{html.escape(description)}</p><div class="images">'+''.join(columns)+'</div></article>')
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Regression and token audit</title><style>body{font:16px/1.5 system-ui;background:#f6f3ec;color:#292822;margin:24px;max-width:1600px}table{border-collapse:collapse}th,td{padding:9px 16px;border-bottom:1px solid #ccc;text-align:left}article{margin:28px 0;padding:20px;background:white}.images{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%}figcaption{font-size:13px}@media(max-width:800px){.images{grid-template-columns:repeat(2,minmax(0,1fr))}}</style><h1>What regressed, and what happened to tokens?</h1><p>The strongest concrete failure is mixing local-transform and canvas-coordinate conventions. The final checkpoint is not getting progressively shorter in this run. All examples below are preserved node renders; no new inference or painting was performed.</p>'''+table+'''<p>Earlier curriculum SFT did shorten replies:1,745→560 tokens/turn, toward teacher targets around500–600 tokens. Recovery's next apparent decrease primarily came from learning17-token finish-only replies. The current midpoint average is inflated by two16,384-token cutoffs.</p><p>All13 final-over-parent wins occurred on different references; no reference won both sampled comparisons. Flowers and sailboats each have2 wins,1 loss,1 tie across their four cases. Allfour box cases lose. This is mixed, inconsistent reference conditioning, not uniformly lost capability.</p>'''+''.join(cards)+'''<p>See <a href="REPORT.md">detailed diagnosis</a> and <a href="metrics.json">reproducible metrics</a>. Style correlations are not proof that a coordinate-only intervention fixes every failure.</p>'''
    (out/'index.html').write_text(page)
    print(json.dumps(dict(output=str(out), historical_runs=len(history), painting_cases=len(cases), references_with_any_win=result['references_with_any_win'], references_with_two_wins=result['references_with_two_wins'])))


if __name__=='__main__':
    main()
