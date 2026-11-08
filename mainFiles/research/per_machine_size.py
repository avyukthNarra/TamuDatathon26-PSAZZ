"""Per-machine SIZE hunt: at winning ticks, residual = (won 5x+) - forecast P(5x+ | win), against
"own lag k paid 5x+" / "own lag k won" (k = 1..8) and peers' 5x+ at lags 1..3; max |z| per machine
vs a 23-tick-shifted null (run from mainFiles/)."""
import math, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slot_forecasting.io import load_inputs, read_labels, read_targets
from slot_forecasting.pair_mixer import CLASS_INDEX, adapt, index_history, size_modes_from_targets, win_prior_by_type_from_targets
tq, th = load_inputs("public/train"); vq, _ = load_inputs("public/validation"); vt = read_targets("public/validation/targets.csv")
lab = read_labels("public/train/labels.csv", tq)
mixer = adapt(tq, th, lab, log=lambda *_: None, size_modes=size_modes_from_targets(vt, vq), win_prior=win_prior_by_type_from_targets(vt, vq), hmm_weight=0)
indexed = index_history(tq, th)
PATS = [("big", k) for k in range(1, 9)] + [("won", k) for k in range(1, 9)]
real, null, best = [], [], Counter()
pooled = {p: [[], []] for p in PATS}
for q in tq:
    plays = indexed.get((q.episode_id, q.machine_id), [])
    if len(plays) < 150: continue
    own = [None] * (q.cutoff + 1)
    for p in plays: own[p.tick] = CLASS_INDEX[p.payout]
    em = {}
    mixer.walk(own, q.machine_type, q.cutoff - 1, em.__setitem__, q.signal)
    res = {}
    for t in range(60, q.cutoff):
        if own[t] and own[t] > 0:
            p = em[t]; res[t] = (1.0 if own[t] >= 3 else 0.0) - (p[3] + p[4]) / (1 - p[0])
    def maxz(shift):
        top, which = 0.0, None
        for kind, k in PATS:
            on, off = [], []
            for t, e in res.items():
                s = t - k - shift
                if s < 0 or own[s] is None: continue
                hit = own[s] >= 3 if kind == "big" else own[s] > 0
                (on if hit else off).append(e)
                if not shift: pooled[(kind, k)][0 if hit else 1].append(e)
            if len(on) < 8 or len(off) < 8: continue
            m1, m2 = sum(on) / len(on), sum(off) / len(off)
            v = sum((x - m1) ** 2 for x in on) / len(on) ** 2 + sum((x - m2) ** 2 for x in off) / len(off) ** 2
            z = abs(m1 - m2) / math.sqrt(v) if v > 0 else 0
            if z > top: top, which = z, (kind, k)
        return top, which
    z, w = maxz(0); real.append(z); best[w] += 1; null.append(maxz(23)[0])
real.sort(); null.sort(); qq = lambda x, f: x[int(f * (len(x) - 1))]
print(f"machines {len(real)} | size max|z| 50/90/99 real {[round(qq(real, f), 2) for f in (.5, .9, .99)]} null {[round(qq(null, f), 2) for f in (.5, .9, .99)]}")
print("share above null 95th pct:", round(sum(z > qq(null, .95) for z in real) / len(real), 3), "| strongest patterns:", best.most_common(6))
print("pooled residual difference (on - off) per pattern:")
for p, (on, off) in pooled.items():
    if on and off:
        m1, m2 = sum(on) / len(on), sum(off) / len(off)
        print(f"  {p}: {m1 - m2:+.4f} (n on {len(on)})")
