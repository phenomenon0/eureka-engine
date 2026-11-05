#!/usr/bin/env python3
import json, os, pathlib, signal, socket, subprocess, sys, tempfile, time, urllib.request, urllib.error
here = pathlib.Path(__file__).resolve().parent
root = here.parent / 'src'
def request(port, path, body):
    data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=data, headers={'X-Api-Key':'test','Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=8) as r: return r.status, r.read().decode()
    except urllib.error.HTTPError as e: return e.code, e.read().decode()
def raw(port, tokens):
    s = socket.create_connection(('127.0.0.1',port))
    data = json.dumps({'messages':[{'role':'user','content':'hello'}], 'max_tokens':tokens, 'stream':True}).encode()
    s.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nX-Api-Key: test\r\nContent-Length: '+str(len(data)).encode()+b'\r\n\r\n'+data)
    return s
with tempfile.TemporaryDirectory() as tmp:
    binary = os.environ.get('EUREKA_TEST_SERVE') or str(pathlib.Path(tmp)/'serve')
    if not os.environ.get('EUREKA_TEST_SERVE'):
        subprocess.run(['clang','-O2','-pthread',*[str(root/x) for x in ('serve.c','json.c','speak_route.c')],'-o',binary],check=True)
    sock=socket.socket(); sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]; sock.close()
    args=[binary,'--port',str(port),'--engine',str(here/'native_fake_engine.py'),'--parallel',os.environ.get('EUREKA_TEST_SLOTS','1'),'--drain-seconds','1','--']
    if not os.environ.get('EUREKA_TEST_CUDA'): args[3:3] = ['--api-key','test']
    elif os.environ.get('EUREKA_TEST_BANKS'): args[3:3] = ['--banks',os.environ['EUREKA_TEST_BANKS']]
    proc=subprocess.Popen(args,env={**os.environ,'LLB_GGUF':'fake'},stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    try:
        for _ in range(100):
            try: urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=.2).read(); break
            except Exception: time.sleep(.05)
        pretty=json.dumps({'state':'hello','questions':{'q':{'type':'noul'}}},indent=2)
        assert request(port,'/v1/decide',pretty)[0] == 200
        assert request(port,'/v1/decide',pretty)[0] == 200
        assert request(port,'/v1/chat/completions','{"messages":[],"temperature":1e999}')[0] == 400
        assert request(port,'/v1/chat/completions',{'messages':[{'role':'user','content':'hi'}], 'max_tokens':1e300})[0] == 400
        if os.environ.get('EUREKA_TEST_SLOTS') == '2':
            active=raw(port,20); active.settimeout(3); active.recv(4096)
            other=raw(port,8); other.settimeout(3); data=other.recv(4096)
            active.close()
            while True:
                chunk=other.recv(4096)
                if not chunk: break
                data += chunk
            other.close()
            assert b'[DONE]' in data, data
            code,body=request(port,'/v1/chat/completions',{'messages':[{'role':'user','content':'next'}], 'max_tokens':1})
            assert code == 200 and json.loads(body)['choices'][0]['message']['content'] == 'x'
            print('native multislots regression passed')
        else:
            active=raw(port,20); active.recv(4096)
            queued=raw(port,1); time.sleep(.15); queued.close()
            active.close(); time.sleep(.15)
            start=time.monotonic()
            code,body=request(port,'/v1/chat/completions',{'messages':[{'role':'user','content':'next'}], 'max_tokens':1})
            assert code == 200 and json.loads(body)['choices'][0]['message']['content'] == 'x'
            assert time.monotonic()-start < 1.0
            print('native server regression passed')
    finally:
        try: os.killpg(proc.pid,signal.SIGTERM); proc.wait(timeout=5)
        except (ProcessLookupError,subprocess.TimeoutExpired):
            try: os.killpg(proc.pid,signal.SIGKILL); proc.wait()
            except ProcessLookupError: pass
