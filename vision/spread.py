#!/usr/bin/env python3
"""spread.py A_DIR B_DIR [names...]: per-tensor metrics of A vs B (B = reference) over every .f32 both dirs share
(or the listed names), using manifest.tsv shapes. TSV: image tensor n max_abs mean_abs rms_ref absmax_ref min_row_cos"""
import sys, os, numpy as np
def shapes(d):
    s = {}
    for line in open(os.path.join(d, 'manifest.tsv')).read().splitlines()[1:]:
        f = line.split('\t'); s[f[0]] = [int(x) for x in f[1:]]
    return s
def metrics(a, b, ne0):
    a = a.astype(np.float64); b = b.astype(np.float64); d = np.abs(a - b)
    A = a.reshape(-1, ne0); B = b.reshape(-1, ne0)
    cos = np.sum(A*B, 1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-30)
    return d.max(), d.mean(), float(np.sqrt((b*b).mean())), float(np.abs(b).max()), float(cos.min())
def main():
    A, B = sys.argv[1], sys.argv[2]; sa, sb = shapes(A), shapes(B)
    names = sys.argv[3:] or [n for n in sb if n in sa]
    img = os.path.basename(B.rstrip('/'))
    for n in names:
        if n not in sa or n not in sb: print(f"{img}\t{n}\tMISSING"); continue
        a = np.fromfile(f"{A}/{n}.f32", '<f4'); b = np.fromfile(f"{B}/{n}.f32", '<f4')
        if a.size != b.size: print(f"{img}\t{n}\tSIZE {a.size} vs {b.size}"); continue
        ne0 = sb[n][0] * (sb[n][1] if n.startswith(('Qcur','Kcur')) else 1)
        mx, mn, rms, amax, cos = metrics(a, b, ne0)
        print(f"{img}\t{n}\t{a.size}\t{mx:.6g}\t{mn:.6g}\t{rms:.6g}\t{amax:.6g}\t{cos:.9f}")
if __name__ == '__main__': main()
