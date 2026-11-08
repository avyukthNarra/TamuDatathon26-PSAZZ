"""(A) sub-cell correction: our win model pools 'both won' with 'both lost' (and the two 'differ'
patterns) into parity cells. Per machine: pick its pair by parity likelihood on its own history,
then logit += alpha * (logit r4 - logit r_par), r4 = 4-cell rate of the cutoff's cell shrunk toward
the parity rate with strength K. (B) HMM off where v7c's flip swap is active. Both checks vs v16."""
import math, sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import flip_transfer as FT
E, lg = FT.E, FT.lg
PAIRS = ((1, 3), (2, 5), (4, 7))
def cells(r):
    seq = {t: v > 0 for t, v in r["seq"].items() if t < r["cutoff"]}
    best = None
    for a, b in PAIRS:
        par = {True: [0, 0], False: [0, 0]}; four = {}
        for t, w in seq.items():
            if t - a in seq and t - b in seq:
                k = (seq[t - a], seq[t - b]); par[k[0] == k[1]][0] += w; par[k[0] == k[1]][1] += 1
                c = four.setdefault(k, [0, 0]); c[0] += w; c[1] += 1
        ll = sum(w * math.log((pw + .5) / (pn + 1)) + (pn - pw) * 0 for pw, pn in par.values() for w in [0])
        ll = sum(pw * math.log((pw + .5) / (pn + 1)) + (pn - pw) * math.log((pn - pw + .5) / (pn + 1)) for pw, pn in par.values() if pn)
        if best is None or ll > best[0]: best = (ll, (a, b), par, four)
    _, (a, b), par, four = best
    t = r["cutoff"]
    if t - a not in seq or t - b not in seq: return None
    k = (seq[t - a], seq[t - b]); pw, pn = par[k[0] == k[1]]; fw, fn = four.get(k, [0, 0])
    return pw, pn, fw, fn
for rows in (E.P.VAL, E.P.TRAIN):
    for r in rows: r["cells"] = cells(r)
def corr(r, K):
    c = r["cells"]
    if c is None: return 0.0
    pw, pn, fw, fn = c
    rp = (pw + 1) / (pn + 2)
    r4 = (fw + K * rp) / (fn + K)
    return lg(r4) - lg(rp)
def v(alpha=0.0, K=16, types=None, hmm_off_flip=False):
    def f(r, so):
        a = E.v11(r)
        if hmm_off_flip and abs(r["flip_d"]) > 1e-9:
            a = 1 - r["mixer"][0]   # mixer only (peer terms are added below anyway)
        b = 1 - r["v7c"][0]
        za = lg(a) + r["flip_d"]
        if alpha and (types is None or r["type"] in types):
            za += alpha * corr(r, K)
        z = 0.7 * za + 0.3 * lg(b) + 0.5 * E.flat(r) + 0.05 * r["signal"]
        win = 1 / (1 + math.exp(-z))
        sb = [x / b for x in r["v7c"][1:]]
        raw = [max(x, 1e-4) for x in [1 - win] + [win * s for s in sb]]; t = sum(raw)
        return [x / t for x in raw]
    return f
E.BASE = v()
E.evaluate("v16", v())
for K in (8, 16, 32):
    E.evaluate(f"(A) sub-cell all types, K {K}", v(1.0, K))
E.evaluate("(A) sub-cell jackpot only, K 16", v(1.0, 16, {"jackpot"}))
E.evaluate("(A) sub-cell all, K 16, alpha 0.5", v(0.5, 16))
E.evaluate("(B) HMM off where flip swap active", v(hmm_off_flip=True))
