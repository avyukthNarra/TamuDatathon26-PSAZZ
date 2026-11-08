"""Round-2 residual hunt on the peer-corrected model: win AND size (run from mainFiles/)."""
import math, sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
sys.argv = sys.argv[:1] + ["--methods", "none"]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import peer_correction as P
VAL, TRAIN = P.VAL, P.TRAIN
def lg(p): p = min(max(p, 1e-4), 1 - 1e-4); return math.log(p / (1 - p))
for rows, truth in ((VAL, True), (TRAIN, False)):
    for r in rows:
        sh = r["x"]["peers 1-5"]
        f = r["f"] if sh is None else 1 / (1 + math.exp(-(lg(r["f"]) + (sh - 0.4187))))
        r["f2"] = f
        r["res"] = ((1 - r["truth"][0]) if truth else (1.0 if r["outcome"] else 0.0)) - f
        m = r["mixer"]; pbig = (m[3] + m[4]) / (1 - m[0])                 # forecast P(5x or 20x | win)
        if truth:
            t = r["truth"]; r["size_res"] = (t[3] + t[4]) / (1 - t[0]) - pbig if 1 - t[0] > 0.05 else None
        else:
            r["size_res"] = ((1.0 if r["outcome"] >= 3 else 0.0) - pbig) if r["outcome"] > 0 else None
def lag(r, k): return r["seq"].get(r["cutoff"] - k)
def peers_at(r, k, test):
    v = [test(s[r["cutoff"] - k]) for s in r["peers"] if r["cutoff"] - k in s]
    return None if not v else sum(v) / len(v)
SIG = {}
for k in range(1, 9):
    SIG[f"own lag {k} won"] = lambda r, k=k: None if lag(r, k) is None else float(lag(r, k) > 0)
    SIG[f"own lag {k} 5x+"] = lambda r, k=k: None if lag(r, k) is None else float(lag(r, k) >= 5)
    SIG[f"peers lag {k} 5x+ share"] = lambda r, k=k: peers_at(r, k, lambda v: v >= 5)
    SIG[f"peers lag {k} win share"] = lambda r, k=k: peers_at(r, k, lambda v: v > 0)
SIG["recent10 - recent40 own win rate"] = lambda r: (lambda ts: None if len(ts) < 40 else sum(r["seq"][t] > 0 for t in ts[-10:]) / 10 - sum(r["seq"][t] > 0 for t in ts[-40:]) / 40)(sorted(t for t in r["seq"] if t < r["cutoff"]))
SIG["own big-win share last 40 wins"] = lambda r: (lambda w: None if len(w) < 10 else sum(v >= 5 for v in w) / len(w))([r["seq"][t] for t in sorted(t for t in r["seq"] if t < r["cutoff"]) if r["seq"][t] > 0][-40:])
SIG["signal"] = lambda r: float(r["signal"])
SIG["cutoff"] = lambda r: float(r["cutoff"])
def slope(rows, fn, key):
    xs, ys = [], []
    for r in rows:
        y = r[key]
        if y is None: continue
        x = fn(r)
        if x is None: continue
        xs.append(x); ys.append(y)
    n = len(xs)
    if n < 50: return None
    mx, my = sum(xs) / n, sum(ys) / n; sxx = sum((x - mx) ** 2 for x in xs)
    if not sxx: return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    se = math.sqrt(sum((y - my - b * (x - mx)) ** 2 for x, y in zip(xs, ys)) / (n - 2) / sxx)
    return b, b / se
for key, label in (("res", "WIN chance"), ("size_res", "SIZE P(5x+ | win)")):
    print(f"\n=== residual of {label}: signals with |z| > 2 on BOTH splits, same sign ===")
    for name, fn in SIG.items():
        v, t = slope(VAL, fn, key), slope(TRAIN, fn, key)
        if v and t and v[0] * t[0] > 0 and abs(v[1]) > 2 and abs(t[1]) > 2:
            print(f"  {name:36s} validation {v[0]:+.3f} (z {v[1]:+.1f}) | train {t[0]:+.3f} (z {t[1]:+.1f})")
