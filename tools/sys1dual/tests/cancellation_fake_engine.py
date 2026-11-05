#!/usr/bin/env python3
import json, os, sys, time
with open(os.environ['FAKE_ENGINE_STARTS'], 'a') as f: f.write(str(os.getpid())+'\n')
print('sys1: FP32 clef head + tokenizer ready', file=sys.stderr, flush=True)
for line in sys.stdin:
    req=json.loads(line)
    if 'decide_batch' in req:
        for i,_ in enumerate(req['decide_batch']):
            time.sleep(.4)
            if os.environ.get('LLB_BATCH_OOO'): print(json.dumps({'bx_item':i}),flush=True)
            print(json.dumps({'answers':{'q':{'choice':'yes'}},'usage':{'input_tokens':1}}),flush=True)
    elif req.get('mode') == 'generate':
        n=req.get('n',1)
        for _ in range(n): time.sleep(.08); print(json.dumps({'token':1,'text':'x'}),flush=True)
        print(json.dumps({'text':'x'*n,'ids':[1]*n,'prompt_ids':[1],'finish':'length'}),flush=True)
    else: print(json.dumps({'answers':{'q':{'choice':'yes'}},'usage':{'input_tokens':1}}),flush=True)
