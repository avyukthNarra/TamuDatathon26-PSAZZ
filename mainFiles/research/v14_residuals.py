"""Residual hunt on v14 (win and size), both splits (run from mainFiles/)."""
import math, sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import v13_eval as E
V14 = E.make(ww=0.4, extra_peer=0.5)
P = E.P
for split, rows in (("validation", P.VAL), ("train", P.TRAIN)):
    for r in rows:
        p = V14(r, None); r["p14"] = p
        w = 1 - p[0]
        pbig = (p[3] + p[4]) / w
        if split == "validation":
            t = r["truth"]; r["wres"] = (1 - t[0]) - w
            r["sres"] = (t[3] + t[4]) / (1 - t[0]) - pbig if 1 - t[0] > 0.05 else None
        else:
            r["wres"] = (1.0 if r["outcome"] else 0.0) - w
            r["sres"] = ((1.0 if r["outcome"] >= 3 else 0.0) - pbig) if r["outcome"] > 0 else None
def lag(r, k): return r["seq"].get(r["cutoff"] - k)
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
def peers(r, k, test):
    v = [test(s[r["cutoff"] - k]) for s in r["peers"] if r["cutoff"] - k in s]
    return sum(v) / len(v) if v else None
SIG = {}
for k in range(1, 9):
    SIG[f"own lag {k} won"] = lambda r, k=k: None if lag(r, k) is None else float(lag(r, k) > 0)
    SIG[f"own lag {k} >=2x"] = lambda r, k=k: None if lag(r, k) is None else float(lag(r, k) >= 2)
    SIG[f"own lag {k} >=5x"] = lambda r, k=k: None if lag(r, k) is None else float(lag(r, k) >= 5)
    SIG[f"peers lag {k} win"] = lambda r, k=k: peers(r, k, lambda v: v > 0)
    SIG[f"peers lag {k} >=2x"] = lambda r, k=k: peers(r, k, lambda v: v >= 2)
SIG["signal"] = lambda r: float(r["signal"])
SIG["n observed"] = lambda r: float(sum(1 for t in r["seq"] if t < r["cutoff"]))
SIG["forecast win"] = lambda r: 1 - r["p14"][0]
SIG["own win share last 10"] = lambda r: (lambda ts: None if len(ts) < 10 else sum(r["seq"][t] > 0 for t in ts) / len(ts))(sorted(t for t in r["seq"] if t < r["cutoff"])[-10:])
SIG["v7c win - our win"] = lambda r: (1 - r["v7c"][0]) - E.v11(r)
for key, label in (("wres", "WIN"), ("sres", "SIZE P(5x+|win)")):
    print(f"=== {label}: |z| > 2 on both splits, same sign")
    for name, fn in SIG.items():
        v, t = slope(P.VAL, fn, key), slope(P.TRAIN, fn, key)
        if v and t and v[0] * t[0] > 0 and abs(v[1]) > 2 and abs(t[1]) > 2:
            print(f"  {name:26s} validation {v[0]:+.4f} (z {v[1]:+.1f}) | train {t[0]:+.4f} (z {t[1]:+.1f})")
