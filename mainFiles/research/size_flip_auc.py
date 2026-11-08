"""Can the confidently-wrong size modes be told apart from their recent wins? (validation truth)"""
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import size_flip as F
E = F.E
def wrong(r):
    A, B = F.MODES[r["type"]]
    t = [x / (1 - r["truth"][0]) for x in r["truth"][1:]]
    tb = 1.0 if sum((a - b) ** 2 for a, b in zip(t, B)) < sum((a - b) ** 2 for a, b in zip(t, A)) else 0.0
    p = [x / (1 - r["v7c"][0]) for x in r["v7c"][1:]]
    pb = min(max(sum((x - a) * (b - a) for x, a, b in zip(p, A, B)) / sum((b - a) ** 2 for a, b in zip(A, B)), 0), 1)
    return abs(pb - tb) > 0.5
for K in (4, 6, 10, 15):
    F.prep(K)
    for cond in ("shared_change", "drift", "stable"):
        rows = [r for r in E.P.VAL if r["condition"] == cond and 1 - r["truth"][0] > 0.05 and r["type"] in F.MODES and r[f"llr{K}"] is not None]
        pos = [r[f"llr{K}"] for r in rows if wrong(r)]; neg = [r[f"llr{K}"] for r in rows if not wrong(r)]
        if not pos: print(f"K {K} {cond}: no wrong machines with enough wins"); continue
        auc = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
        print(f"K {K:2d} {cond:14s}: wrong {len(pos):2d} / {len(rows)}, AUC of 'inverted rule fits recent wins better' = {auc:.2f}, mean LLR wrong {sum(pos)/len(pos):+.2f} vs right {sum(neg)/len(neg):+.2f}")
