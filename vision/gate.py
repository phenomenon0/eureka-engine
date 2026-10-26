#!/usr/bin/env python3
"""V3 gate: ours (vit_run dumps) vs the Metal golden, judged by the REGISTERED bounds in vision/TOLERANCES.tsv.
gate.py OURS_DIR GOLDEN_DIR IMAGE -> one row per bounded tensor (+ unbounded ones as INFO); exit 0 iff all PASS."""
import sys, os, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spread import shapes, metrics
ours, gold, img = sys.argv[1], sys.argv[2], sys.argv[3]
# (ENGINE-IPHONEVIS) TOL=<bounds tsv> (vision/TOLERANCES-4B.tsv for the 4B tower); default = R1's registered table
TOL = os.environ.get('TOL', os.path.join(os.path.dirname(__file__), '../TOLERANCES.tsv'))
tol = {r['tensor']: r for r in csv.DictReader(open(TOL), delimiter='\t') if r['image'] == img}
so, sg = shapes(ours), shapes(gold)
fails = 0; n_bounded = 0
order = lambda n: (0 if n in ('patch_bias', 'inp_pos_emb') else 1 if '-' in n and n.split('-')[-1].isdigit() else 2,
                   int(n.split('-')[-1]) if n.split('-')[-1].isdigit() else 0, n)
print(f"{'tensor':<18} {'mean|d|':>10} {'bound':>10} {'max|d|':>10} {'bound':>10} {'minrowcos':>12} {'bound':>12}  verdict")
for n in sorted((n for n in sg if n in so and n not in ('pre', 'embd_instr')), key=order):
    a = np.fromfile(f"{ours}/{n}.f32", '<f4'); b = np.fromfile(f"{gold}/{n}.f32", '<f4')
    if a.size != b.size: print(f"{n:<18} SIZE {a.size} vs {b.size}  FAIL"); fails += 1; continue
    ne0 = sg[n][0] * (sg[n][1] if n.startswith(('Qcur', 'Kcur')) else 1)
    mx, mn, rms, amax, cos = metrics(a, b, ne0)
    t = tol.get(n)
    if t:
        n_bounded += 1
        ok = mn <= float(t['mean_abs_max']) and mx <= float(t['max_abs_max']) and cos >= float(t['min_row_cos_min'])
        fails += not ok
        print(f"{n:<18} {mn:10.4g} {float(t['mean_abs_max']):10.4g} {mx:10.4g} {float(t['max_abs_max']):10.4g} {cos:12.9f} {float(t['min_row_cos_min']):12.9f}  {'PASS' if ok else 'FAIL'}")
    else:
        print(f"{n:<18} {mn:10.4g} {'-':>10} {mx:10.4g} {'-':>10} {cos:12.9f} {'-':>12}  INFO")
print(f"== {img}: {n_bounded - fails}/{n_bounded} bounded tensors PASS" + ("" if n_bounded else " (no registered bounds: INFO only)"))
sys.exit(1 if fails else 0)
