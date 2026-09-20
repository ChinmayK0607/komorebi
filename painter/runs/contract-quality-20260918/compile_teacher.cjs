// Trusted procedural recipes ONLY. This is not a sandbox for model-written JS.
// Run on a Linux node: node compile_teacher.cjs old-programs.jsonl new-programs.jsonl
// Lower geometric transforms into explicit canvas-pixel brush calls, preserving
// their order and full numeric precision. Browser renders still need comparison.
const fs = require('fs');
const vm = require('vm');
const crypto = require('crypto');
const assert = require('assert/strict');
const sha = x => crypto.createHash('sha256').update(x).digest('hex');
const allowed = new Set(['seed','noStroke','noWash','fill','fillBleed','fillTexture',
                         'polygon','noFill','set','line','spline']);
function trace(source) {
  const phases = {setup:[], draw:[]};
  let phase = 'setup';
  function record(name, args) {
    const cloned = JSON.parse(JSON.stringify(args, (key,value) => {
      if (typeof value === 'number' && !Number.isFinite(value)) throw Error('Non-finite geometry');
      return value;
    }));
    phases[phase].push([name,cloned]);
  }
  const brush = new Proxy({}, {get(target,name) {
    assert(allowed.has(name), 'Unsupported brush call: '+String(name));
    return (...args) => record('brush.'+name,args);
  }});
  const sandbox = {brush, WEBGL:'WEBGL'};
  for (const name of ['createCanvas','pixelDensity','randomSeed','noiseSeed','background','translate','noLoop']) {
    sandbox[name] = (...args) => record(name,args);
  }
  const context = vm.createContext(sandbox, {codeGeneration:{strings:false,wasm:false}});
  new vm.Script(source+'\nsetup();').runInContext(context,{timeout:3000});
  phase='draw';
  new vm.Script('draw();').runInContext(context,{timeout:3000});
  assert.deepEqual(phases.setup.find(x=>x[0]==='createCanvas'), ['createCanvas',[600,600,'WEBGL']]);
  const translations = phases.draw.filter(x=>x[0]==='translate');
  assert.deepEqual(translations,[['translate',[-300,-300]]]);
  return phases;
}
function emit(phases) {
  const call = ([name,args]) => name+'('+args.map((x,i)=>name==='createCanvas' && i===2 ? 'WEBGL' : JSON.stringify(x)).join(',')+');';
  return ['setup','draw'].map(phase=>'function '+phase+'(){\n'+phases[phase].map(call).join('\n')+'\n}\n').join('');
}
function compile(source) {
  const before=trace(source), program=emit(before);
  assert.deepEqual(trace(program), before, 'Lowering changed the brush-call trace');
  return {program, receipt:{source_sha256:sha(source),target_sha256:sha(program),
    source_bytes:Buffer.byteLength(source),target_bytes:Buffer.byteLength(program),
    exact_brush_trace:true,render_parity_verified:false}};
}
if (require.main === module) {
  assert.equal(process.platform,'linux','Compile painting recipes on the node.');
  const [input,output] = process.argv.slice(2);
  assert(input && output,'Usage: node compile_teacher.cjs INPUT.jsonl OUTPUT.jsonl');
  const rows = fs.readFileSync(input,'utf8').trim().split('\n').map(JSON.parse);
  const result=rows.map(row=>{
    const compiled=Object.fromEntries(Object.entries(row.programs).map(([kind,source])=>[kind,compile(source)]));
    return {...row, programs:Object.fromEntries(Object.entries(compiled).map(([k,v])=>[k,v.program])),
      coordinate_representation:'canvas-pixels',
      compilation:Object.fromEntries(Object.entries(compiled).map(([k,v])=>[k,v.receipt])),
      target_sha256:compiled.reference.receipt.target_sha256,
      render_status:'pending_node_render',visual_review_status:'pending'};
  });
  fs.writeFileSync(output,result.map(x=>JSON.stringify(x)).join('\n')+'\n',{flag:'wx'});
  console.log(JSON.stringify({references:result.length,trace_parity:true,render_parity:false,
    note:'Measure target token lengths before SFT: explicit numeric programs may be much longer.'}));
}
module.exports={trace,emit,compile};
