"""Honest 2-half search over blend tweaks on cross-fitted member predictions.

Usage: python3 tweak.py MEMBER_DIR KIT_PUBLIC [DEFAULT_SIZE_MEMBER]
Fit each tweak on half A (even episodes), score on half B, and vice versa.
Points on the 20-point scale.
"""
import csv, itertools, sys
from pathlib import Path
import numpy as np

mdir, kit = Path(sys.argv[1]), Path(sys.argv[2]) / "validation"
def load(p):
    with open(p) as f:
        return {r["query_id"]: np.array([float(r[c]) for c in ("p_0", "p_1", "p_2", "p_5", "p_20")]) for r in csv.DictReader(f)}
tgt, ref = load(kit / "targets.csv"), load(kit / "reference.csv")
cond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(kit / "conditions.csv"))}
ep = {r["query_id"]: r["episode_id"] for r in csv.DictReader(open(kit / "targets.csv"))}
ids = sorted(tgt)
eps = sorted(set(ep.values()))
half = {e: i % 2 for i, e in enumerate(eps)}
members = {p.stem: load(p) for p in sorted(mdir.glob("*.csv")) if p.stem != "blend_val"}
print("members:", list(members))
T = np.array([tgt[q] for q in ids]); R = np.array([ref[q] for q in ids])
C = np.array([cond[q] for q in ids]); E = np.array([ep[q] for q in ids]); H = np.array([half[ep[q]] for q in ids])

def points(P, mask):
    ratios = []
    for c in sorted(set(C)):
        m = mask & (C == c)
        def err(X):
            per = [((X[m & (E == e)] - T[m & (E == e)]) ** 2).sum(1).mean() for e in set(E[m])]
            return np.mean(per)
        ratios.append(err(P) / max(err(R), 1e-9))
    return 40 / (1 + np.mean(ratios))

def L(p):
    p = np.clip(p, 1e-5, 1 - 1e-5); return np.log(p / (1 - p))
names = list(members)
W = {n: L(1 - np.array([members[n][q][0] for q in ids])) for n in names}
S = {}
for n in names:
    s = np.array([members[n][q][1:] for q in ids]); t = s.sum(1, keepdims=True)
    S[n] = np.where(t > 0, s / np.where(t > 0, t, 1), 0.25)

def build(w, temp, size):
    z = sum(w[i] * W[n] for i, n in enumerate(names)) * temp
    pw = 1 / (1 + np.exp(-z))
    P = np.column_stack([1 - pw, pw[:, None] * S[size]])
    P = np.maximum(P, 1e-4); return P / P.sum(1, keepdims=True)

grid = [np.array(w) / 10 for w in itertools.product(range(11), repeat=len(names)) if sum(w) == 10]
eq = np.ones(len(names)) / len(names)
temps = (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2)
default_size = sys.argv[3] if len(sys.argv) > 3 else names[0]
base = build(eq, 1.0, default_size)
print(f"default (equal, T=1, size={default_size}): all {points(base, H >= 0):.4f}  A {points(base, H == 0):.4f}  B {points(base, H == 1):.4f}")
for h in (0, 1):
    best = max(((points(build(w, t, s), H == h), (tuple(w), t, s)) for w in grid for t in temps for s in names), key=lambda x: x[0])[1]
    o = 1 - h
    print(f"fit on half {'AB'[h]}: best {best} -> held-out {'AB'[o]}: {points(build(np.array(best[0]), best[1], best[2]), H == o):.4f} vs default {points(base, H == o):.4f}")
for t in temps:
    P = build(eq, t, default_size); print(f"T={t}: A {points(P, H == 0):.4f} B {points(P, H == 1):.4f}")
for s in names:
    P = build(eq, 1.0, s); print(f"size={s}: A {points(P, H == 0):.4f} B {points(P, H == 1):.4f}")
for i, n in enumerate(names):
    w = eq.copy(); w[i] = 0; w /= w.sum(); P = build(w, 1.0, default_size); print(f"drop {n}: A {points(P, H == 0):.4f} B {points(P, H == 1):.4f}")
