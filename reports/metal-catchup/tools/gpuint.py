# gpuint.py XML: parse xctrace metal-gpu-intervals; print the GPU timeline summary (per command buffer / encoder)
import sys, xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
ids = {}
for e in root.iter():
    if e.get('id'): ids[e.get('id')] = (e.text, e.get('fmt'))
def val(e):
    if e is None: return None
    r = e.get('ref')
    if r is not None: return ids[r]
    v = e.get('fmt') if e.text is None else e.text
    if e.get('id'): ids[e.get('id')] = (e.text, e.get('fmt'))
    return (e.text, e.get('fmt'))
rows = []
for row in root.iter('row'):
    cs = list(row)
    vals = [val(c) if c.tag != 'sentinel' else None for c in cs]
    # also register nested ids
    for c in cs:
        for sub in c.iter():
            if sub.get('id') and sub.get('id') not in ids: ids[sub.get('id')] = (sub.text, sub.get('fmt'))
    rows.append(vals)
# columns: start, duration, channel, frame, latency, depth, label, state, conn, color, process, gpu, subtitle, ios, bytes, cb, enc, sub
out = []
for v in rows:
    try:
        st = int(v[0][0]); du = int(v[1][0]); ch = v[2][1]; dp = v[5][1] if v[5] else '?'; lab = v[6][1] if v[6] else ''
        cb = v[15][1] if len(v) > 15 and v[15] else ''; en = v[16][1] if len(v) > 16 and v[16] else ''
    except Exception as ex:
        continue
    out.append((st, du, ch, dp, lab, cb, en))
out.sort()
import collections
print("rows", len(out), collections.Counter((o[2], o[3]) for o in out))
for o in out[:5]: print(o)
t0 = out[0][0]
if len(sys.argv) > 2:
    a, b = float(sys.argv[2]) * 1e9, float(sys.argv[3]) * 1e9
    for o in out:
        if a <= o[0] - t0 <= b: print(round((o[0] - t0) / 1e6, 3), round(o[1] / 1e3, 1), o[2], o[3], o[4][:60], o[5], o[6])
