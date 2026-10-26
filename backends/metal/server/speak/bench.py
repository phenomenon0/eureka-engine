#!/usr/bin/env python3
"""bench.py — 10 frozen P3 texts x 2 passes over POST /v1/speak (CUDA-final protocol: 4 warm-ups,
20 sequential measured requests, no outlier removal). RTF = X-LLB-Speak-Ms / WAV duration."""
import argparse
import hashlib
import json
import math
import os
import statistics
import struct
import time
import urllib.request

TEXTS = [
    'The rain stopped just before sunrise.',
    'Please close the window when you leave.',
    'I thought the package would arrive today.',
    'Our train leaves in about ten minutes.',
    'She found a quiet place to read.',
    'This is the first test of the serving path.',
    'We can meet after lunch tomorrow.',
    'The small dog ran across the garden.',
    'I heard a distant bell in the evening.',
    'They went home after the last song.',
]


def post(url, obj):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read(), {k.lower(): v for k, v in r.headers.items()}, (time.perf_counter() - t0) * 1000
    except urllib.error.HTTPError as e:
        return e.code, e.read(), {}, (time.perf_counter() - t0) * 1000


def wav_info(b):
    if len(b) < 44 or b[:4] != b'RIFF' or b[8:12] != b'WAVE':
        return None
    i, info, data = 12, {}, None
    while i + 8 <= len(b):
        cid, sz = b[i:i + 4], struct.unpack('<I', b[i + 4:i + 8])[0]
        body = b[i + 8:i + 8 + sz]
        if cid == b'fmt ':
            fmt, ch, sr, _, _, bits = struct.unpack('<HHIIHH', body[:16])
            info.update(format=fmt, channels=ch, sample_rate=sr, bits=bits)
        elif cid == b'data':
            data = body
            break
        i += 8 + sz + (sz & 1)
    n = len(data) // 2
    s = struct.unpack(f'<{n}h', data[:2 * n])
    rms = math.sqrt(sum(x * x for x in s) / n) / 32768 if n else 0
    info.update(samples=n, duration_s=n / info['sample_rate'],
                rms_dbfs=round(20 * math.log10(rms), 2) if rms else None,
                peak=round(max(abs(x) for x in s) / 32768, 4) if n else None,
                clipped=sum(1 for x in s if abs(x) >= 32767))
    return info


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, math.ceil(p / 100 * len(xs)) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='http://127.0.0.1:19022')
    ap.add_argument('--out', required=True)
    ap.add_argument('--voice', default='af_heart')
    ap.add_argument('--texts', help='json list of 10 texts (default: frozen P3 set)')
    a = ap.parse_args()
    global TEXTS
    if a.texts:
        TEXTS = json.loads(a.texts)
    os.makedirs(os.path.join(a.out, 'wavs'), exist_ok=True)
    url = a.base + '/v1/speak'
    health_before = json.loads(urllib.request.urlopen(a.base + '/health', timeout=10).read())
    warm = []
    for t in TEXTS[:4]:
        c, b, h, w = post(url, {'text': t, 'voice': a.voice})
        warm.append({'text': t, 'http': c, 'speak_ms': float(h.get('x-llb-speak-ms', 'nan')), 'wall_ms': round(w, 2)})
    rows = []
    for p in (1, 2):
        for i, t in enumerate(TEXTS):
            c, b, h, w = post(url, {'text': t, 'voice': a.voice})
            info = wav_info(b) if c == 200 else None
            sha = hashlib.sha256(b).hexdigest()
            if c == 200:
                open(os.path.join(a.out, 'wavs', f'p{p}-t{i}.wav'), 'wb').write(b)
            sm = float(h.get('x-llb-speak-ms', 'nan'))
            r = {'pass': p, 'i': i, 'text': t, 'http': c, 'bytes': len(b), 'sha256': sha,
                 'riff': info is not None, 'wav': info, 'speak_ms': sm, 'wall_ms': round(w, 2),
                 'ep': h.get('x-llb-ep'), 'voice_sha256': h.get('x-llb-voice-sha256'),
                 'g2p': h.get('x-llb-g2p'), 'phonemes': h.get('x-llb-phonemes')}
            if info:
                r['rtf'] = round(sm / 1000 / info['duration_s'], 6)
                r['rtf_wall'] = round(w / 1000 / info['duration_s'], 6)
                r['chars_per_s'] = round(len(t) / info['duration_s'], 2)
            rows.append(r)
            print(f"p{p} t{i} http={c} ms={sm:.1f} dur={info and info['duration_s']:.3f} rtf={r.get('rtf')} sha={sha[:12]}", flush=True)
    health_after = urllib.request.urlopen(a.base + '/health', timeout=10).status
    ok = [r for r in rows if r['http'] == 200 and r['riff']]
    rtf = [r['rtf'] for r in ok]
    pairs = sum(1 for i in range(len(TEXTS)) if rows[i]['sha256'] == rows[len(TEXTS) + i]['sha256'])
    fmt_ok = all(r['wav']['sample_rate'] == 24000 and r['wav']['channels'] == 1 and r['wav']['bits'] == 16
                 and r['wav']['format'] == 1 for r in ok)
    dur_pair_equal = sum(1 for i in range(len(TEXTS)) if rows[i]['wav'] and rows[len(TEXTS) + i]['wav']
                         and rows[i]['wav']['samples'] == rows[len(TEXTS) + i]['wav']['samples'])
    cps = [r['chars_per_s'] for r in ok]
    s = {
        'schema': 'macos-speak-native/v1', 'base': a.base,
        'protocol': '4 warm-ups, then 10 frozen P3 texts x 2 passes, 20 sequential requests; RTF = X-LLB-Speak-Ms / WAV duration; no outlier removal',
        'health_before': health_before, 'health_after_http': health_after,
        'requests': len(rows), 'http_200': sum(1 for r in rows if r['http'] == 200), 'valid_riff': len(ok),
        'format_24k_mono_pcm16': fmt_ok, 'unique_sha': len({r['sha256'] for r in ok}),
        'pairs_identical_sha': pairs, 'pairs_equal_length': dur_pair_equal,
        'rtf_median': round(statistics.median(rtf), 6), 'rtf_p95': round(pct(rtf, 95), 6),
        'rtf_min': round(min(rtf), 6), 'rtf_max': round(max(rtf), 6),
        'rtf_wall_median': round(statistics.median(r['rtf_wall'] for r in ok), 6),
        'speak_ms_median': round(statistics.median(r['speak_ms'] for r in ok), 1),
        'speak_ms_p95': round(pct([r['speak_ms'] for r in ok], 95), 1),
        'audio_s_median': round(statistics.median(r['wav']['duration_s'] for r in ok), 4),
        'chars_per_s': {'min': min(cps), 'median': statistics.median(cps), 'max': max(cps)},
        'rms_dbfs': {'min': min(r['wav']['rms_dbfs'] for r in ok), 'max': max(r['wav']['rms_dbfs'] for r in ok)},
        'peak_max': max(r['wav']['peak'] for r in ok), 'clipped_samples': sum(r['wav']['clipped'] for r in ok),
        'ep_seen': sorted({r['ep'] for r in ok}), 'voice_sha_seen': sorted({r['voice_sha256'] for r in ok}),
        'warmups': warm, 'rows': rows,
    }
    json.dump(s, open(os.path.join(a.out, 'speak.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in s.items() if k not in ('rows', 'warmups', 'health_before')}, indent=1))


if __name__ == '__main__':
    main()
