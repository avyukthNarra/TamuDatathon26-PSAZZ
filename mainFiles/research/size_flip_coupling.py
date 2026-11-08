"""Task 4: are wrong size modes concentrated where v7c's WIN flip swap fires? (validation truth)"""
import sys
from collections import defaultdict
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import flip_transfer as FT
    import size_flip as F
E = FT.E
def pb_err(r):
    A, B = F.MODES[r["type"]]
    t = [x / (1 - r["truth"][0]) for x in r["truth"][1:]]
    tb = 1.0 if sum((a - b) ** 2 for a, b in zip(t, B)) < sum((a - b) ** 2 for a, b in zip(t, A)) else 0.0
    p = [x / (1 - r["v7c"][0]) for x in r["v7c"][1:]]
    pb = min(max(sum((x - a) * (b - a) for x, a, b in zip(p, A, B)) / sum((b - a) ** 2 for a, b in zip(A, B)), 0), 1)
    return abs(pb - tb)
for cond in ("shared_change", "drift", "stable", "short_history"):
    g = defaultdict(list)
    for r in E.P.VAL:
        if r["condition"] != cond or 1 - r["truth"][0] <= 0.05 or r["type"] not in F.MODES: continue
        g[abs(r["flip_d"]) > 1e-9].append(pb_err(r) > 0.5)
    on, off = g[True], g[False]
    print(f"{cond:14s} win-flip active: {sum(on)}/{len(on)} size-wrong ({(sum(on)/len(on) if on else 0):.1%}) | not active: {sum(off)}/{len(off)} ({(sum(off)/len(off) if off else 0):.1%})")
