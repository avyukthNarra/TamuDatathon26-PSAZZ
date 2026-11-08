"""Round-3 residual scan on v10 with new signal families (run from mainFiles/)."""
import math, sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
sys.argv = sys.argv[:1] + ["--methods", "none"]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import residuals2 as R2
VAL, TRAIN = R2.VAL, R2.TRAIN
TYPE = {}
for r in VAL + TRAIN: TYPE[(r["episode"], r["machine"])] = r["type"]
def peer_items(r):
    for s_key, s in R2.P.R.HIST["validation" if r["episode"].startswith("main-validation") else "train"].items():
        if s_key[0] == r["episode"] and s_key[1] != r["machine"]:
            yield s_key, s
def peers_stat(r, lags, fn, same=None):
    vals = []
    for key, s in peer_items(r):
        if same is not None and (TYPE.get(key) == r["type"]) != same: continue
        for k in lags:
            t = r["cutoff"] - k
            if t in s: vals.append(fn(s[t]))
    return sum(vals) / len(vals) if vals else None
def own_ticks(r): return sorted(t for t in r["seq"] if t < r["cutoff"])
def streak(r, want_win):
    n = 0
    for t in reversed(own_ticks(r)):
        if (r["seq"][t] > 0) == want_win: n += 1
        else: break
    return float(min(n, 20))
def since(r, test):
    ts = own_ticks(r)
    for i, t in enumerate(reversed(ts)):
        if test(r["seq"][t]): return float(min(i, 60))
    return 60.0
SIG = {
    "peers mean payout lags 1-5": lambda r: peers_stat(r, range(1, 6), float),
    "peers 5x+ share lags 1-5": lambda r: peers_stat(r, range(1, 6), lambda v: float(v >= 5)),
    "peers 20x share lags 1-10": lambda r: peers_stat(r, range(1, 11), lambda v: float(v >= 20)),
    "SAME-type peers win share 1-5": lambda r: peers_stat(r, range(1, 6), lambda v: float(v > 0), same=True),
    "OTHER-type peers win share 1-5": lambda r: peers_stat(r, range(1, 6), lambda v: float(v > 0), same=False),
    "peers win share lags 9-20": lambda r: peers_stat(r, range(9, 21), lambda v: float(v > 0)),
    "peers win share lags 21-60": lambda r: peers_stat(r, range(21, 61), lambda v: float(v > 0)),
    "own loss streak": lambda r: streak(r, False),
    "own win streak": lambda r: streak(r, True),
    "spins since own last win": lambda r: since(r, lambda v: v > 0),
    "spins since own last 5x+": lambda r: since(r, lambda v: v >= 5),
    "own win share lags 9-20": lambda r: (lambda ts: None if not ts else sum(r["seq"][t] > 0 for t in ts) / len(ts))([t for t in own_ticks(r) if r["cutoff"] - 20 <= t <= r["cutoff"] - 9]),
    "own mean payout lags 1-8": lambda r: (lambda v: sum(v) / len(v) if v else None)([r["seq"][r["cutoff"] - k] for k in range(1, 9) if r["cutoff"] - k in r["seq"]]),
}
print(f"{'signal':34s} {'validation slope (z)':>22s} {'train slope (z)':>20s}")
for name, fn in SIG.items():
    v, t = R2.slope(VAL, fn, "res"), R2.slope(TRAIN, fn, "res")
    if not v or not t: continue
    flag = "  <-- BOTH" if v[0] * t[0] > 0 and abs(v[1]) > 2 and abs(t[1]) > 2 else ""
    print(f"{name:34s} {v[0]:+.4f} ({v[1]:+5.1f})        {t[0]:+.4f} ({t[1]:+5.1f}){flag}", flush=True)
