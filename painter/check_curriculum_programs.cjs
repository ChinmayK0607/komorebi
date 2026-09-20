// Syntax/provenance checks only. No painting program is executed or rendered.
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const crypto = require('crypto');
const assert = require('assert/strict');
const root = process.argv[2];
if (!root) throw Error('Usage: node check_curriculum_programs.cjs DATA_DIRECTORY');
const file = path.join(root, 'programs.jsonl');
const bytes = fs.readFileSync(file);
const rows = bytes.toString().trim().split('\n').map(JSON.parse);
const groups = new Set(), hashes = new Set();
let programs = 0;
for (const row of rows) {
  assert(!groups.has(row.source_group), 'Repeated source group: ' + row.source_group);
  groups.add(row.source_group);
  assert(['train', 'dev'].includes(row.split));
  assert.equal(row.object_count, row.objects.length);
  assert(row.programs.reference && row.programs.missing_parts);
  assert.equal(Object.keys(row.programs).length, 3);
  const hash = crypto.createHash('sha256').update(row.programs.reference).digest('hex');
  assert.equal(hash, row.target_sha256);
  assert(!hashes.has(hash), 'Duplicate reference program');
  hashes.add(hash);
  for (const object of row.objects) {
    assert.equal(object.frame.length, 6);
    assert(object.frame.every(Number.isFinite));
    const [a,b,c,d,x,y] = object.frame;
    for (const u of [-220,220]) for (const v of [-220,220]) {
      const px=a*u+c*v+x, py=b*u+d*v+y;
      assert(px>=24 && px<=576 && py>=24 && py<=576, 'Target bounds leave canvas');
    }
  }
  for (const [kind, source] of Object.entries(row.programs)) {
    new vm.Script(source, {filename: row.id + '/' + kind + '.js'});
    assert.equal((source.match(/translate\(-300,-300\)/g)||[]).length, 1);
    programs++;
  }
}
const result = {
  references: rows.length, programs, javascript_syntax_passed: true,
  source_sha256: crypto.createHash('sha256').update(bytes).digest('hex'),
  duplicate_reference_programs: 0, source_group_overlap: 0,
  conservative_target_bounds: 'inside 24..576 on both axes',
  rendered: false,
  method: 'vm.Script syntax compilation, source binding and geometric metadata checks; no painting execution'
};
fs.writeFileSync(path.join(root,'syntax-check.json'), JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result,null,2));
