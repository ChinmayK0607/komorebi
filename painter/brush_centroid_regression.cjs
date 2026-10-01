#!/usr/bin/env node
/*
 * Pure p5.brush centroid regression. No renderer, model, network, or API.
 * Usage from the repository root: node painter/brush_centroid_regression.cjs
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const sourcePath = process.argv[2] || path.join(
  __dirname, 'vendor/integrations/watercolour/assets/p5.brush.js'
);
const source = fs.readFileSync(sourcePath, 'utf8');
const prefix = 'function xn(t){';
const oldFallback = '{x:v[0].x,y:v[0].y}';
const newFallback = '{x:t[0].x,y:t[0].y}';
const start = source.indexOf(prefix);
assert.notEqual(start, -1, 'function xn(t) is missing');
let depth = 0;
let end = -1;
for (let i = source.indexOf('{', start); i < source.length; i += 1) {
  if (source[i] === '{') depth += 1;
  if (source[i] === '}') {
    depth -= 1;
    if (depth === 0) {
      end = i + 1;
      break;
    }
  }
}
assert.notEqual(end, -1, 'function xn(t) has unbalanced braces');
const originalFunction = source.slice(start, end);
assert.equal((source.match(new RegExp(oldFallback.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g')) || []).length, 1);
assert.equal(originalFunction.includes(newFallback), false);

function loadFunction(functionText, name) {
  const renamed = functionText.replace(prefix, `function ${name}(t){`);
  const context = {};
  vm.runInNewContext(`${renamed}; globalThis.result = ${name};`, context);
  // Normalize the cross-realm object returned by vm before strict assertions.
  return (...args) => JSON.parse(JSON.stringify(context.result(...args)));
}

const original = loadFunction(originalFunction, 'centroidOriginal');
const patchedSource = source.replace(oldFallback, newFallback);
assert.equal(patchedSource.includes(oldFallback), false);
assert.equal((patchedSource.match(new RegExp(newFallback.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g')) || []).length, 1);
const patchedFunction = patchedSource.slice(start, end);
const patched = loadFunction(patchedFunction, 'centroidPatched');

const zeroArea = [
  {x: 10, y: -3}, {x: 11, y: -3}, {x: 12, y: -3}, {x: 13, y: -3},
  {x: 14, y: -3}, {x: 15, y: -3}, {x: 16, y: -3}, {x: 17, y: -3}
];
let originalError;
try {
  original(zeroArea);
} catch (error) {
  originalError = error;
}
assert.equal(originalError && originalError.name, 'ReferenceError', 'original fallback must expose v ReferenceError');

const empty = patched([]);
const short = patched([{x: 1, y: 2}, {x: 3, y: 4}, {x: 5, y: 8}]);
const fixedZeroArea = patched(zeroArea);
const polygon = [
  {x: 0, y: 0}, {x: 4, y: 0}, {x: 4, y: 1}, {x: 4, y: 2},
  {x: 4, y: 3}, {x: 0, y: 3}, {x: 0, y: 2}, {x: 0, y: 1}
];
assert.deepEqual(empty, {x: 0, y: 0});
assert.deepEqual(short, {x: 3, y: 14 / 3});
assert.deepEqual(fixedZeroArea, zeroArea[0]);
assert.deepEqual(patched(polygon), {x: 2, y: 1.5});
assert.deepEqual(original(polygon), patched(polygon), 'nondegenerate centroid changed');
for (const value of [empty, short, fixedZeroArea, patched(polygon)]) {
  assert.equal(Number.isFinite(value.x), true);
  assert.equal(Number.isFinite(value.y), true);
}

process.stdout.write('p5.brush centroid regression passed\n');
