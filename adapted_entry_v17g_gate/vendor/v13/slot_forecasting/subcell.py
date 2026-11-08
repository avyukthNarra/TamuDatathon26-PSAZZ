"""Sub-cell correction for jackpot machines (v17).

Our win experts use PARITY cells: "the two lags agree" pools "both won" with "both lost" (and
"differ" pools won/lost with lost/won). Those sub-cells do not share a win rate, and for jackpot
machines that pooling is most of the remaining win error on stable and short histories (found
independently by Sai's sai/master-test jackpot patch on another base). Per machine: pick its lag
pair by parity likelihood on its own history (pairs (1,3), (2,5), (4,7)), then
    logit p += alpha * (logit r4 - logit rp)
rp = parity rate of the cutoff's cell, (w + 1) / (n + 2); r4 = its 4-cell rate shrunk toward rp,
(w4 + K rp) / (n4 + K). Research (research/subcell.py, subcell_plateau.py): vs v16 validation
57.57 (57.62 / 57.52) -> 57.69 (57.71 / 57.68), held-out train window dR -0.0011; K 8-32 and
alpha 0.75-1.25 all give 57.68-57.70. Applied to all types it raised validation but not train,
so it is jackpot-only. Only ticks before the cutoff are read.
"""
import math

PAIRS = ((1, 3), (2, 5), (4, 7))
SUBCELL_K = 16.0
SUBCELL_ALPHA = 1.0
SUBCELL_TYPES = ("jackpot",)


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def subcell_shift(seq, cutoff, k=SUBCELL_K):
    """Log-odds shift (logit r4 - logit rp) for one machine, or 0 if the cutoff's cell is unknown."""
    won = {t: v > 0 for t, v in seq.items() if t < cutoff}
    best = None
    for a, b in PAIRS:
        par = {True: [0, 0], False: [0, 0]}
        four = {}
        for t, w in won.items():
            if t - a in won and t - b in won:
                key = (won[t - a], won[t - b])
                par[key[0] == key[1]][0] += w
                par[key[0] == key[1]][1] += 1
                cell = four.setdefault(key, [0, 0])
                cell[0] += w
                cell[1] += 1
        ll = sum(pw * math.log((pw + .5) / (pn + 1)) + (pn - pw) * math.log((pn - pw + .5) / (pn + 1))
                 for pw, pn in par.values() if pn)
        if best is None or ll > best[0]:
            best = (ll, (a, b), par, four)
    _, (a, b), par, four = best
    if cutoff - a not in won or cutoff - b not in won:
        return 0.0
    key = (won[cutoff - a], won[cutoff - b])
    pw, pn = par[key[0] == key[1]]
    fw, fn = four.get(key, (0, 0))
    rp = (pw + 1) / (pn + 2)
    return _logit((fw + k * rp) / (fn + k)) - _logit(rp)


def subcell_shifts(queries, history, types=SUBCELL_TYPES):
    from .schema import index_history
    indexed = index_history(queries, history)
    out = {}
    for q in queries:
        if q.machine_type in types:
            seq = {p.tick: p.payout for p in indexed.get((q.episode_id, q.machine_id), [])}
            out[q.query_id] = SUBCELL_ALPHA * subcell_shift(seq, q.cutoff)
    return out
