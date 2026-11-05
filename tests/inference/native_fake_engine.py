#!/usr/bin/env python3
import json,sys,time,os
print('sys1: FP32 clef head + tokenizer ready',file=sys.stderr,flush=True)
for line in sys.stdin:
 try: r=json.loads(line)
 except Exception:
  print(json.dumps({'error':'malformed engine line'}),flush=True);continue
 if r.get('mode')=='generate':
  n=r.get('n',1)
  for i in range(n):
   time.sleep(.08)
   print(json.dumps({'token':10,'text':'x'}),flush=True)
  print(json.dumps({'text':'x'*n,'ids':[10]*n,'prompt_ids':[1],'finish':'length'}),flush=True)
 else: print(json.dumps({'answers':{'q':{'noul':1}},'usage':{'input_tokens':1}}),flush=True)
