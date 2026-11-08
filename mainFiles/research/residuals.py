"""Residual hunt: what does the 56.54 model's win chance miss? (run from mainFiles/)

Residual = true p_win - forecast (validation, exact) and outcome - forecast (train, noisy but
8448 rows). For each candidate signal we report the mean residual when it is on vs off, with a
standard error, on both splits. A real missed effect shows the same sign on both.
"""
import json, math, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slot_forecasting.io import load_inputs

pub = Path("public"); cache = Path("runs/research-cache")
VAL = json.load(open(cache / "validation.json")); TRAIN = json.load(open(cache / "train.json"))

def logit(p): p = min(max(p, 1e-4), 1 - 1e-4); return math.log(p / (1 - p))
def pool(r): return 1 / (1 + math.exp(-(0.7 * logit(1 - r["mixer"][0]) + 0.3 * logit(r["hmm"]))))

HIST = {}
for split in ("validation", "train"):
    h = defaultdict(dict)
    for p in load_inputs(pub / split)[1]:
        h[(p.episode_id, p.machine_id)][p.tick] = p.payout
    HIST[split] = h
for split, rows in (("validation", VAL), ("train", TRAIN)):
    for r in rows:
        f = pool(r)
        y = (1 - r["truth"][0]) if split == "validation" else (1.0 if r["outcome"] else 0.0)
        r["res"], r["f"] = y - f, f
        r["seq"] = HIST[split][(r["episode"], r["machine"])]
        peers = [k for k in HIST[split] if k[0] == r["episode"] and k[1] != r["machine"]]
        r["peers"] = [HIST[split][k] for k in peers]

def lag(r, k):
    t = r["cutoff"] - k
    return r["seq"].get(t)

SIGNALS = {}
for k in range(1, 9):
    SIGNALS[f"own lag {k} won"] = lambda r, k=k: None if lag(r, k) is None else lag(r, k) > 0
    SIGNALS[f"own lag {k} paid 5x+"] = lambda r, k=k: None if lag(r, k) is None else lag(r, k) >= 5
for k in (1, 2, 3):
    SIGNALS[f"peers' lag {k} win share > 0.5"] = lambda r, k=k: (lambda v: None if not v else sum(v) / len(v) > 0.5)(
        [s[r["cutoff"] - k] > 0 for s in r["peers"] if r["cutoff"] - k in s])
SIGNALS["three-lag parity (1,3,5) even"] = lambda r: None if any(lag(r, k) is None for k in (1, 3, 5)) else sum(lag(r, k) > 0 for k in (1, 3, 5)) % 2 == 0
SIGNALS["forecast > 0.5"] = lambda r: r["f"] > 0.5
SIGNALS["recent 10 win share > older"] = lambda r: (lambda ts: None if len(ts) < 30 else sum(r["seq"][t] > 0 for t in ts[-10:]) / 10 > sum(r["seq"][t] > 0 for t in ts[:-10]) / len(ts[:-10]))(sorted(t for t in r["seq"] if t < r["cutoff"]))
SIGNALS["cutoff in last third of episode (>= 235)"] = lambda r: r["cutoff"] >= 235

def effect(rows, fn, mask=lambda r: True):
    on, off = [], []
    for r in rows:
        if not mask(r): continue
        v = fn(r)
        if v is None: continue
        (on if v else off).append(r["res"])
    if len(on) < 20 or len(off) < 20: return None
    m = lambda x: sum(x) / len(x)
    var = lambda x: sum((a - m(x)) ** 2 for a in x) / (len(x) - 1)
    d = m(on) - m(off); se = math.sqrt(var(on) / len(on) + var(off) / len(off))
    return d, se

def _report():
    print(f"{"signal":42s} {'validation (truth) diff  z':>28s} {'train (outcomes) diff  z':>28s}")
    for name, fn in SIGNALS.items():
        v, t = effect(VAL, fn), effect(TRAIN, fn)
        if v and t:
            flag = "  <-- same sign, |z|>2 on both" if (v[0] * t[0] > 0 and abs(v[0] / v[1]) > 2 and abs(t[0] / t[1]) > 2) else ""
            print(f"{name:42s} {v[0]:+.4f} {v[0]/v[1]:+6.1f}            {t[0]:+.4f} {t[0]/t[1]:+6.1f}{flag}")
    print("\nmean residual: validation %+.4f  train %+.4f" % (sum(r['res'] for r in VAL) / len(VAL), sum(r['res'] for r in TRAIN) / len(TRAIN)))
    
    
    # ---- peer effect map: slope of residual on the peers' win share at each lag (centred) ----
    def slope(rows, fn, mask=lambda r: True):
        xs, ys = [], []
        for r in rows:
            if not mask(r): continue
            x = fn(r)
            if x is None: continue
            xs.append(x); ys.append(r["res"])
        n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs); b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        resid = [y - my - b * (x - mx) for x, y in zip(xs, ys)]
        se = math.sqrt(sum(e * e for e in resid) / (n - 2) / sxx)
        return b, se
    
    def peer_share(r, k):
        v = [s[r["cutoff"] - k] > 0 for s in r["peers"] if r["cutoff"] - k in s]
        return sum(v) / len(v) if v else None
    
    print("\nslope of residual on peers' win share (0..1)  [validation | train]")
    for k in range(1, 9):
        v, t = slope(VAL, lambda r: peer_share(r, k)), slope(TRAIN, lambda r: peer_share(r, k))
        print(f"  peers' lag {k}: {v[0]:+.3f} (z {v[0]/v[1]:+.1f}) | {t[0]:+.3f} (z {t[0]/t[1]:+.1f})")
    for w in (3, 5, 10):
        fn = lambda r, w=w: (lambda v: sum(v) / len(v) if v else None)([x for k in range(1, w + 1) for x in ([peer_share(r, k)] if peer_share(r, k) is not None else [])])
        v, t = slope(VAL, fn), slope(TRAIN, fn)
        print(f"  peers' mean win share, lags 1-{w}: {v[0]:+.3f} (z {v[0]/v[1]:+.1f}) | {t[0]:+.3f} (z {t[0]/t[1]:+.1f})")
    

if __name__ == "__main__":
    _report()
