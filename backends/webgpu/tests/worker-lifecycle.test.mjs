import { test } from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
const code = readFileSync(new URL('../engine/bend-worker.js', import.meta.url), 'utf8');
function setup() {
  const messages = [];
  const context = vm.createContext({ URL, performance, Uint8Array, TextDecoder, postMessage: m => messages.push(m), importScripts() {}, location: { href: 'https://example.com/engine/bend-worker.js' }, BendWire: { Utf8Stream: class { push() { return 'x'; } end() { return ''; } } } });
  vm.runInContext(code, context);
  vm.runInContext(`encode=()=>[1,2,3]; piece=()=>new Uint8Array([120]); let resume,started; globalThis.started=new Promise(r=>started=r); session={reset(){},step(){started();return new Promise(r=>resume=()=>r({ms:1,top2:[{id:99}]}));},async free(){}};globalThis.resume=()=>resume();`, context);
  return { context, messages };
}
test('free settles active and queued chats exactly once', async () => {
  const { context, messages } = setup();
  vm.runInContext('chat({id:1,system:"",messages:[],maxNew:5})', context);
  await context.started;
  vm.runInContext('chat({id:2,system:"",messages:[],maxNew:5})', context);
  const free = vm.runInContext('free()', context);
  context.resume(); await free;
  for (const id of [1, 2]) {
    const terminal = messages.filter(m => m.id === id && ['done', 'error'].includes(m.t));
    assert.equal(terminal.length, 1); assert.equal(terminal[0].reason, 'aborted');
  }
});
test('open operations serialize even when an earlier open is delayed', async () => {
  const { context, messages } = setup();
  vm.runInContext(`let releaseOpen; globalThis.openStarted=new Promise(r=>globalThis.signalOpen=r); open=async m=>{ if(m.openId===1){signalOpen(); await new Promise(r=>releaseOpen=r)} post({t:'ready',openId:m.openId}) }; globalThis.releaseOpen=()=>releaseOpen();`,context);
  const a = vm.runInContext('onmessage({data:{t:"open",openId:1}})',context);
  await context.openStarted;
  const b = vm.runInContext('onmessage({data:{t:"open",openId:2}})',context);
  await Promise.resolve(); assert.equal(messages.length,0);
  context.releaseOpen(); await Promise.all([a,b]);
  assert.deepEqual(messages.map(m=>m.openId),[1,2]);
});
test('decision cancellation settles active and queued requests without later results', async () => {
  const { context, messages } = setup();
  vm.runInContext(`let releaseDecision; globalThis.decisionStarted=new Promise(r=>globalThis.signalDecision=r);
    decide=async (m,r)=>{signalDecision();await new Promise(resolve=>releaseDecision=resolve);if(!r.finished)post({t:'decided',id:m.id})};
    globalThis.releaseDecision=()=>releaseDecision();`, context);
  const active = vm.runInContext('onmessage({data:{t:"decide",id:10,body:"{}"}})', context);
  await context.decisionStarted;
  const queued = vm.runInContext('onmessage({data:{t:"decide",id:11,body:"{}"}})', context);
  await vm.runInContext('onmessage({data:{t:"cancel",id:11}})', context);
  await vm.runInContext('onmessage({data:{t:"cancel",id:10}})', context);
  context.releaseDecision(); await Promise.all([active, queued]);
  for (const id of [10,11]) {
    const terminal = messages.filter(m => m.id === id && ['done','error','decided'].includes(m.t));
    assert.equal(terminal.length, 1); assert.equal(terminal[0].reason, 'aborted');
  }
});
test('free cancels an active decision and waits for its underlying work', async () => {
  const { context, messages } = setup();
  vm.runInContext(`let releaseDecision;globalThis.decisionStarted=new Promise(r=>globalThis.signalDecision=r);
    decide=async (m,r)=>{signalDecision();await new Promise(resolve=>releaseDecision=resolve);if(!r.finished)post({t:'decided',id:m.id})};
    globalThis.releaseDecision=()=>releaseDecision();`, context);
  const active = vm.runInContext('onmessage({data:{t:"decide",id:12,body:"{}"}})', context);
  await context.decisionStarted;
  const freeing = vm.runInContext('onmessage({data:{t:"free"}})', context);
  assert.equal(messages.find(m=>m.id===12)?.reason, 'aborted');
  context.releaseDecision(); await Promise.all([active,freeing]);
  assert.equal(messages.filter(m=>m.id===12).length, 1);
});
test('Kev decision stops after the in-flight GPU step on cancellation', async () => {
  const { context, messages } = setup();
  vm.runInContext(`let resumeKev;let kevSteps=0;globalThis.kevStarted=new Promise(r=>globalThis.signalKev=r);
    encode=()=>[248050,248050,1];kev={boxEnd:248050};session={reset(){},step(){kevSteps++;signalKev();return new Promise(r=>resumeKev=()=>r({ms:1}));},async free(){}};
    globalThis.resumeKev=()=>resumeKev();globalThis.kevSteps=()=>kevSteps;`, context);
  const body = JSON.stringify({state:'Q',questions:{pick:{type:'choice',instructions:'pick',criteria:{A:'',B:''}}}});
  context.body = body;
  const running = vm.runInContext('onmessage({data:{t:"decide",id:13,body}})', context);
  await context.kevStarted;
  await vm.runInContext('onmessage({data:{t:"cancel",id:13}})', context);
  context.resumeKev(); await running;
  assert.equal(context.kevSteps(), 1);
  assert.equal(messages.filter(m=>m.id===13).length, 1);
  assert.equal(messages.find(m=>m.id===13).reason, 'aborted');
});

test('cancel during batched prefill settles once and prevents decode', async () => {
  const { context, messages } = setup();
  vm.runInContext(`let resumeBatch; globalThis.batchStarted=new Promise(r=>globalThis.signalBatch=r); let steps=0;
    session={numerics:'turbo', reset(){}, prefill(){signalBatch(); return new Promise(r=>resumeBatch=()=>r({ms:1,tokens:2}));},
      async step(){steps++;return {ms:1,top2:[{id:99}]};},async free(){}};
    globalThis.resumeBatch=()=>resumeBatch(); globalThis.steps=()=>steps;`, context);
  vm.runInContext('chat({id:3,system:"",messages:[],maxNew:5})', context);
  await context.batchStarted;
  await vm.runInContext('onmessage({data:{t:"cancel",id:3}})', context);
  context.resumeBatch();
  await vm.runInContext('tail', context);
  assert.equal(context.steps(), 0);
  assert.equal(messages.filter(m => m.id === 3 && m.t === 'done').length, 1);
  assert.equal(messages.find(m => m.id === 3 && m.t === 'done').reason, 'aborted');
});
