"""Per-machine residual hunt: does each machine have structure the model misses? (run from mainFiles/)

For every train machine with a long history, walk the pair mixer (settings as adapt() picks them)
and take the residual won_t - p_t at every observed tick t >= 60. Against each candidate pattern
(agree/disagree of two lags a<b<=8, or single lag k won) compute the residual's mean difference
z-score. Record each machine's max |z| over patterns. Null: the same with the machine's own
win sequence shifted by 23 ticks to build the patterns (same marginals, no timing link).
If real max |z| clearly exceeds the null, machines carry extra rules the model misses.
"""
import itertools, math, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slot_forecasting.io import load_inputs, read_labels, read_targets
from slot_forecasting.pair_mixer import CLASS_INDEX, adapt, index_history, size_modes_from_targets, win_prior_by_type_from_targets

tq, th = load_inputs("public/train"); vq, _ = load_inputs("public/validation"); vt = read_targets("public/validation/targets.csv")
lab = read_labels("public/train/labels.csv", tq)
mixer = adapt(tq, th, lab, log=lambda *_: None, size_modes=size_modes_from_targets(vt, vq),
              win_prior=win_prior_by_type_from_targets(vt, vq), hmm_weight=0)
PATTERNS = [("pair", a, b) for a, b in itertools.combinations(range(1, 9), 2)] + [("lag", k, None) for k in range(1, 9)]
indexed = index_history(tq, th)
real, null, best_pat = [], [], Counter()
import csv
cond = {r['query_id']: r['condition'] for r in csv.DictReader(open('public/train/conditions.csv'))}
by_cond = []
for q in tq:
    plays = indexed.get((q.episode_id, q.machine_id), [])
    if len(plays) < 150: continue
    own = [None] * (q.cutoff + 1)
    for p in plays: own[p.tick] = CLASS_INDEX[p.payout]
    emitted = {}
    mixer.walk(own, q.machine_type, q.cutoff - 1, emitted.__setitem__, q.signal)
    won = {t: own[t] > 0 for t in range(q.cutoff) if own[t] is not None}
    res = {t: (1.0 if won[t] else 0.0) - (1 - emitted[t][0]) for t in won if t >= 60}
    def maxz(shift):
        src = {t + shift: v for t, v in won.items()} if shift else won
        top, which = 0.0, None
        for kind, a, b in PATTERNS:
            on, off = [], []
            for t, e in res.items():
                if kind == "pair":
                    if t - a in src and t - b in src: (on if src[t - a] == src[t - b] else off).append(e)
                elif t - a in src: (on if src[t - a] else off).append(e)
            if len(on) < 15 or len(off) < 15: continue
            m1, m2 = sum(on) / len(on), sum(off) / len(off)
            v = (sum((x - m1) ** 2 for x in on) / len(on) / len(on) + sum((x - m2) ** 2 for x in off) / len(off) / len(off))
            z = abs(m1 - m2) / math.sqrt(v) if v > 0 else 0.0
            if z > top: top, which = z, (kind, a, b)
        return top, which
    z, which = maxz(0); real.append(z); best_pat[which] += 1
    lv = [e for t, e in res.items() if 1 - emitted[t][0] > 0.5]; dv = [e for t, e in res.items() if 1 - emitted[t][0] <= 0.5]
    by_cond.append((cond[q.query_id], z, sum(lv) / max(len(lv), 1), sum(dv) / max(len(dv), 1)))
    null.append(maxz(23)[0])
real.sort(); null.sort()
q = lambda xs, f: xs[int(f * (len(xs) - 1))]
print(f"machines {len(real)} | max|z| quantiles 50/75/90/99 real {[round(q(real, f), 2) for f in (.5, .75, .9, .99)]} | null {[round(q(null, f), 2) for f in (.5, .75, .9, .99)]}")
print("share of machines with max|z| above the null's 95th percentile:", round(sum(z > q(null, .95) for z in real) / len(real), 3), "(0.05 = nothing)")
print("most common strongest patterns:", best_pat.most_common(8))

thr = q(null, .95)
for c in ("stable", "short_history", "drift", "shared_change"):
    g = [x for x in by_cond if x[0] == c]
    if not g: continue
    print(f"{c:14s} machines {len(g):3d}: above null-95: {sum(x[1] > thr for x in g) / len(g):.3f} | mean residual where forecast>0.5 {sum(x[2] for x in g) / len(g):+.4f}, where <=0.5 {sum(x[3] for x in g) / len(g):+.4f}")
