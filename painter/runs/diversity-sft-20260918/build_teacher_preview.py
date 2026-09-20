"""Display existing teacher renders and their actual Luna labels; no rendering."""
import html
import json
from pathlib import Path

P = Path(__file__).resolve().parent / 'teacher-preview'
rows = json.loads((P / 'review-input.json').read_text())
judgments = {r['id']: r for r in json.loads((P / 'luna-judgments.json').read_text())}
sections = []
for family in sorted({r['primary_family'] for r in rows}):
    cards = []
    for row in (r for r in rows if r['primary_family'] == family):
        label = judgments[row['id']]
        image = row['id'] + '/reference.png'
        assert (P / image).is_file()
        cards.append(f'''<article><a href="{image}"><img loading="lazy" src="{image}" alt="{html.escape(row['id'])}"></a>
<h3>{html.escape(row['layout'])} · {row['object_count']} recipe object(s)</h3>
<span class="{label['grade']}">{label['grade']}{' · excluded from training' if label['grade'] != 'acceptable' else ''}</span>
<p>{html.escape(label['reasons'])}</p><small>{html.escape(row['id'])}</small></article>''')
    sections.append(f'<section><h2>{html.escape(family.replace("_", " "))}</h2><div class="grid">'+''.join(cards)+'</div></section>')
page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Diversity curriculum · teacher review</title><style>
body{margin:0;background:#171b22;color:#eef2f6;font:16px/1.5 system-ui,sans-serif}main{max-width:1250px;margin:auto;padding:32px}
h1{font-size:32px;line-height:1.2}h2{text-transform:capitalize;margin-top:40px}h3{font-size:17px;margin:12px 0 6px}
header p{max-width:950px;color:#bdc9d8}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}
article{background:#242b35;padding:12px;border-radius:12px}img{width:100%;display:block;border-radius:6px}article p{font-size:14px;color:#c7d0dc}
small{color:#9daaba}.acceptable{color:#8de0aa}.uncertain{color:#ffc46f}.unacceptable{color:#ff9696}a{color:#b3d0ff}
@media(max-width:800px){main{padding:16px}.grid{grid-template-columns:1fr}}
</style><main><header><h1>Diversity curriculum: teacher previews</h1>
<p>These are program-generated reference targets, not student outputs. Luna visually reviewed all 36 shown here: 34 acceptable, 2 uncertain. The two uncertain targets are excluded from training. This sampled review covers 36 of 2,304 references; the remaining images are not individually visually certified.</p>
<p>The new data varies geometry, scale, mirroring, placement and object combinations across 12 existing recipe families. This tests procedural composition and geometry, not photographs or unseen subject families. The run continues the recovery adapter with 50% painting, 20% repair, 10% finish, 10% previous teacher replay and 10% general replay by example count. No speed reward.</p>
<p><a href="review.md">Luna report</a> · <a href="luna-judgments.json">Raw judgments</a></p></header>'''+''.join(sections)+'</main></html>'
(P / 'index.html').write_text(page)
print(P / 'index.html')
