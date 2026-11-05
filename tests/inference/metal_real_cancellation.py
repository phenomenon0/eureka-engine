#!/usr/bin/env python3
"""Real-model Metal server cancellation smoke. Run under the host heavy GPU lock."""
import json, os, pathlib, signal, socket, subprocess, sys, tempfile, time, urllib.request, urllib.error
server, engine, gguf, sidecar, logs = sys.argv[1:]
logs=pathlib.Path(logs); logs.mkdir(parents=True, exist_ok=True)
s=socket.socket(); s.bind(('127.0.0.1',0)); port=s.getsockname()[1]; s.close()
with tempfile.TemporaryDirectory(prefix='eureka-metal-smoke-') as tmp, (logs/'server.log').open('w') as out:
    env={**os.environ,'LLB_GGUF':gguf,'TMPDIR':tmp,'LLB_DUAL_FASTPF':'0','LLB_DECIDE_FASTPF':'0'}
    p=subprocess.Popen([server,'--port',str(port),'--api-key','test','--alias',pathlib.Path(gguf).name,'--reasoning','off','--parallel','1','--engine',engine,'--','--gpu','6GB','--threads','8','--',sidecar,'0','--steps','1024'],env=env,stdout=out,stderr=out,start_new_session=True)
    def call(body):
        q=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','X-Api-Key':'test'})
        return urllib.request.urlopen(q,timeout=180)
    def record(name,**values): print(json.dumps({'test':name,**values}),flush=True)
    try:
        start=time.monotonic()
        for _ in range(1800):
            if p.poll() is not None: raise RuntimeError('server exited while loading')
            try: urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1).read(); break
            except Exception: time.sleep(.1)
        else: raise RuntimeError('load timed out')
        record('initial-ready',seconds=round(time.monotonic()-start,3))
        with call({'messages':[{'role':'user','content':'Count from one to one thousand, in order.'}],'max_tokens':256,'stream':True}) as r:
            while True:
                line=r.readline()
                if not line: raise RuntimeError('stream ended before first token')
                if b'"content"' in line: break
        record('client-cancelled-after-token')
        start=time.monotonic(); response=None
        while time.monotonic()-start<180:
            try:
                with call({'messages':[{'role':'user','content':'Say hello.'}],'max_tokens':8}) as r: response=json.load(r)
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError): time.sleep(.2)
        if not response: raise RuntimeError('following completion failed')
        text=response['choices'][0]['message']['content']
        if not text: raise RuntimeError('empty following completion')
        record('following-completion',seconds=round(time.monotonic()-start,3),content=text,model=response.get('model'))
    finally:
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGTERM)
            try: p.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL); p.wait()
