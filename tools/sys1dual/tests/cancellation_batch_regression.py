#!/usr/bin/env python3
"""Exercise native batch and continuous-batch disconnect/restart; pass a built llb-serve."""
import json, os, pathlib, signal, socket, subprocess, sys, tempfile, time, urllib.request
here=pathlib.Path(__file__).resolve().parent
binary=sys.argv[1]
def request(port,path,body):
    req=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=8) as r: return json.loads(r.read())
for continuous in (False,True):
    with tempfile.TemporaryDirectory() as tmp:
        starts=pathlib.Path(tmp)/'starts'; sock=socket.socket(); sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]; sock.close()
        args=[binary,'--port',str(port),'--engine',str(here/'cancellation_fake_engine.py'),'--parallel','1','--native-batch','2','--drain-seconds','1']
        if continuous: args += ['--native-cont']
        args += ['--']
        p=subprocess.Popen(args,env={**os.environ,'FAKE_ENGINE_STARTS':str(starts),'LLB_GGUF':'fake'},stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        try:
            for _ in range(100):
                try: urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=.2).read(); break
                except Exception: time.sleep(.05)
            batch={'stream':True,'batch':[{'state':'a','questions':{'q':{'type':'noul'}}},{'state':'b','questions':{'q':{'type':'noul'}}}]}
            body=json.dumps(batch).encode(); c=socket.create_connection(('127.0.0.1',port)); c.settimeout(3)
            c.sendall(b'POST /v1/decide HTTP/1.1\r\nHost: localhost\r\nContent-Length: '+str(len(body)).encode()+b'\r\n\r\n'+body)
            assert b'200 OK' in c.recv(4096); time.sleep(.1); c.close()
            for _ in range(100):
                if starts.exists() and len(starts.read_text().splitlines()) >= 2: break
                time.sleep(.05)
            assert len(starts.read_text().splitlines()) >= 2, 'worker was not restarted'
            for _ in range(100):
                try:
                    response=request(port,'/v1/chat/completions',{'messages':[{'role':'user','content':'next'}],'max_tokens':1})
                    assert response['choices'][0]['message']['content']=='x'; break
                except Exception: time.sleep(.05)
            else: raise AssertionError('following request failed')
            print('native-cont' if continuous else 'native-batch', 'passed')
        finally:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try: p.wait(timeout=5)
                except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL); p.wait()
