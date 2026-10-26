#!/usr/bin/env python3
"""serve.py — native macOS Kokoro-82M runner for POST /v1/speak.

Misaki English G2P (no espeak fallback, same config as the PC P3 frontend) ->
Kokoro-82M v1.0 ONNX (onnx-community) on onnxruntime -> 24 kHz mono PCM16 WAV.

  POST /v1/speak {text, voice?}  -> audio/wav, X-LLB-Speak-Ms (server-side, g2p+infer+encode)
  GET  /health                   -> json (model/bank sha, EP actually bound, ORT version)

env: TTS_ASSETS (dir with model*.onnx, af_heart.bin, tokenizer.json), TTS_MODEL (model.onnx),
     TTS_EP (cpu|coreml|coreml-ane), TTS_PORT (19022), TTS_THREADS (0 = ORT default).
"""
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _NoEspeakFallback:
    def find_spec(self, fullname, path=None, target=None):
        blocked = ('misaki.espeak', 'phonemizer', 'espeakng_loader', 'espeakng')
        if any(fullname == n or fullname.startswith(n + '.') for n in blocked):
            raise ImportError(f'GPL phonemizer backend is disabled: {fullname}')
        return None


sys.meta_path.insert(0, _NoEspeakFallback())

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from misaki import en  # noqa: E402

SR = 24000
MAX_TOKENS = 510
ASSETS = os.environ.get('TTS_ASSETS', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dl'))
MODEL = os.environ.get('TTS_MODEL', 'model.onnx')
EP = os.environ.get('TTS_EP', 'cpu')
PORT = int(os.environ.get('TTS_PORT', '19022'))
THREADS = int(os.environ.get('TTS_THREADS', '0'))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def providers_for(ep):
    if ep == 'cpu':
        return ['CPUExecutionProvider']
    units = {'coreml': 'ALL', 'coreml-ane': 'CPUAndNeuralEngine', 'coreml-gpu': 'CPUAndGPU'}[ep]
    opts = {'ModelFormat': 'MLProgram', 'MLComputeUnits': units}
    opts.update(json.loads(os.environ.get('TTS_COREML_OPTS') or '{}'))
    return [('CoreMLExecutionProvider', opts),
            'CPUExecutionProvider']


class Engine:
    def __init__(self):
        t0 = time.perf_counter()
        self.model_path = os.path.join(ASSETS, MODEL)
        vocab = json.load(open(os.path.join(ASSETS, 'tokenizer.json')))['model']['vocab']
        self.vocab = {k: int(v) for k, v in vocab.items()}
        self.voices = {}
        so = ort.SessionOptions()
        if THREADS:
            so.intra_op_num_threads = THREADS
        self.sess = ort.InferenceSession(self.model_path, sess_options=so, providers=providers_for(EP))
        self.inputs = [i.name for i in self.sess.get_inputs()]
        self.g2p = en.G2P(trf=False, british=False, fallback=None)
        self.lock = threading.Lock()
        self.info = {
            'runner': 'serve.py', 'ort_version': ort.__version__, 'ep_requested': EP,
            'ep_bound': self.sess.get_providers(), 'ep_options': providers_for(EP)[0], 'threads': THREADS, 'model': MODEL, 'model_sha256': sha256(self.model_path),
            'misaki': 'en.G2P(trf=False, british=False, fallback=None)', 'sample_rate': SR,
            'load_ms': round((time.perf_counter() - t0) * 1000, 1), 'pid': os.getpid(),
        }

    def voice(self, name):
        if not re.fullmatch(r'[a-z]{2}_[a-z]+', name or ''):
            raise ValueError(f'bad voice {name!r}')
        if name not in self.voices:
            p = os.path.join(ASSETS, f'{name}.bin')
            if not os.path.exists(p):
                raise ValueError(f'voice not installed: {name}')
            self.voices[name] = (np.fromfile(p, dtype=np.float32).reshape(-1, 1, 256), sha256(p))
        return self.voices[name]

    def chunks(self, phonemes):
        """Split phonemes on sentence/clause punctuation so each chunk fits the 510-token window."""
        out, cur = [], ''
        for piece in re.split(r'(?<=[.!?;:,])\s+', phonemes):
            if len(cur) + len(piece) + 1 > MAX_TOKENS and cur:
                out.append(cur)
                cur = piece
            else:
                cur = f'{cur} {piece}'.strip()
        if cur:
            out.append(cur)
        return [c[:MAX_TOKENS] for c in out]

    def speak(self, text, voice='af_heart'):
        bank, bank_sha = self.voice(voice)
        phonemes, _ = self.g2p(text)
        phonemes = re.sub(r'\s+', ' ', phonemes).strip()
        if not phonemes or '❓' in phonemes:
            raise ValueError(f'Misaki has no pronunciation for part of {text!r}: {phonemes!r}')
        audio = []
        with self.lock:
            for ph in self.chunks(phonemes):
                ids = [self.vocab[c] for c in ph if c in self.vocab]
                style = bank[min(len(ids), len(bank) - 1)]
                feed = {'input_ids': np.array([[0, *ids, 0]], dtype=np.int64),
                        'style': style.astype(np.float32), 'speed': np.ones(1, dtype=np.float32)}
                audio.append(self.sess.run(None, {k: feed[k] for k in self.inputs})[0].reshape(-1))
        wav = np.concatenate(audio)
        pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).astype('<i2')
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm.tobytes())
        return buf.getvalue(), phonemes, bank_sha


ENGINE = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):
        sys.stderr.write('%s %s\n' % (time.strftime('%H:%M:%S'), fmt % args))

    def send(self, code, body, ctype='application/json', headers=None):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ('/health', '/v1/health'):
            self.send(200, json.dumps({'ok': True, **ENGINE.info}).encode())
        else:
            self.send(404, b'{"error":"not found"}')

    def do_POST(self):
        if self.path != '/v1/speak':
            return self.send(404, b'{"error":"not found"}')
        try:
            n = int(self.headers.get('Content-Length', '0'))
            req = json.loads(self.rfile.read(n) or b'{}')
            text = str(req.get('text') or '').strip()
            voice = req.get('voice') or 'af_heart'
            if not text:
                return self.send(400, b'{"error":"text required"}')
            t0 = time.perf_counter()
            body, phonemes, bank_sha = ENGINE.speak(text, voice)
            ms = (time.perf_counter() - t0) * 1000
        except ValueError as e:
            return self.send(422, json.dumps({'error': str(e)}).encode())
        except Exception as e:  # noqa: BLE001
            return self.send(500, json.dumps({'error': repr(e)}).encode())
        self.send(200, body, 'audio/wav', {
            'X-LLB-Speak-Ms': f'{ms:.1f}', 'X-LLB-Sample-Rate': str(SR), 'X-LLB-Voice': voice,
            'X-LLB-G2P': 'misaki', 'X-LLB-EP': ','.join(ENGINE.info['ep_bound']),
            'X-LLB-Voice-Sha256': bank_sha,
            'X-LLB-Phonemes': phonemes.encode('ascii', 'backslashreplace').decode()})


def main():
    global ENGINE
    ENGINE = Engine()
    sys.stderr.write('serve: ' + json.dumps(ENGINE.info) + '\n')
    ENGINE.info['banner'] = (f'serve: tier NATIVE (ort {ort.__version__} {"+".join(ENGINE.info["ep_bound"])}, '
                              f'{MODEL}, intra_op={THREADS or "default"}, k={MAX_TOKENS} phoneme tokens, exact fp graph)')
    sys.stderr.write(ENGINE.info['banner'] + '\n')
    srv = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    sys.stderr.write(f'serve: listening 127.0.0.1:{PORT}\n')
    sys.stderr.flush()
    srv.serve_forever()


if __name__ == '__main__':
    main()
