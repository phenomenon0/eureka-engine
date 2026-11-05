#!/usr/bin/env python3
"""Real-worker cancellation smoke; caller must hold both house GPU locks."""
import json, os, signal, socket, subprocess, sys, time, urllib.request, urllib.error
server, engine, sidecar = sys.argv[1:]
s=socket.socket(); s.bind(('127.0.0.1',0)); port=s.getsockname()[1]; s.close()
p=subprocess.Popen([server,'--port',str(port),'--engine',engine,'--parallel','1','--drain-seconds','1','--','--gpu','12GB','--threads','16','--',sidecar,'0','--steps','512'],start_new_session=True)
def call(body):
    q=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    return urllib.request.urlopen(q, timeout=180)
def record(name, **values): print(json.dumps({'test':name,**values}),flush=True)
try:
    start=time.monotonic()
    for _ in range(3600):
        if p.poll() is not None: raise RuntimeError('server exited while loading')
        try: urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1).read(); break
        except Exception: time.sleep(.1)
    else: raise RuntimeError('load timed out')
    record('initial-ready',seconds=round(time.monotonic()-start,3))
    body={'messages':[{'role':'user','content':'Count from one to one thousand, slowly and in order.'}],'max_tokens':256,'stream':True}
    with call(body) as r:
        while True:
            line=r.readline()
            if not line: raise RuntimeError('stream ended before token')
            if b'"content"' in line: break
    record('client-cancelled-after-token')
    start=time.monotonic(); response=None
    while time.monotonic()-start<240:
        try:
            with call({'messages':[{'role':'user','content':'Say hello.'}],'max_tokens':8}) as r: response=json.load(r)
            break
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError): time.sleep(.2)
    if not response: raise RuntimeError('next request failed after cancellation')
    text=response['choices'][0]['message']['content']
    if not text: raise RuntimeError('empty follow-up output')
    record('following-completion',seconds=round(time.monotonic()-start,3),content=text,model=response.get('model'))
finally:
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try: p.wait(timeout=15)
        except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL); p.wait()
