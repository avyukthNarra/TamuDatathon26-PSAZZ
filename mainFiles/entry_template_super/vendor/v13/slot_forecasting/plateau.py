"""v11: average (in log-odds) of three near-equivalent win-chance models, for variance reduction.

All three share the pair mixer's payout sizes and differ only in how the win chance is formed:
  A  0.7 x logit(mixer) + 0.3 x logit(HMM) + 1.0 x (peers' lag 1-5 win share - centre)   (= v10)
  B  0.6 x logit(mixer) + 0.4 x logit(HMM) + 1.0 x (same peer term)
  C  0.7 x logit(mixer) + 0.3 x logit(HMM) + sum_k w_k (peers' lag-k win share - centre_k)
     w = LAG_WEIGHTS: per-lag weights fitted on main-round train outcomes (out-of-fold base
     forecasts), scaled by 0.55; lags 2-4 dominate.
Each variant ties v10 but wins on a different split / half; their average improved both checks:
validation 56.55 -> 56.58 in the research cache (halves 56.52 / 56.60 -> 56.55 / 56.63) and
held-out train window dR -0.0006 [-0.0009, -0.0002] with the per-lag weights refit without each
fold (research/plateau_oof.py). Centres are recomputed from each round's train at fit time.
"""
import math

from .peers import _by_machine, peer_share, train_center
from .schema import probabilities

LAG_WEIGHTS = (-0.0196, 0.3143, 0.3479, 0.3302, -0.073, -0.0535, 0.1373, 0.0555)
MIN_PROBABILITY = 1e-4


def _logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def lag_centers(queries, history):
    by_machine = _by_machine(queries, history)
    centers = []
    for k in range(1, 9):
        vals = [s for q in queries if (s := peer_share((q.episode_id, q.machine_id), q.cutoff, by_machine, (k,))) is not None]
        centers.append(sum(vals) / len(vals) if vals else 0.5)
    return centers


class PlateauAverage:
    def __init__(self, mixer, hmm, center, centers):
        self.mixer, self.hmm, self.center, self.centers = mixer, hmm, center, centers
        self.pairs, self.cell_mode = mixer.pairs, mixer.cell_mode

    def predict(self, queries, history):
        base = self.mixer.predict(queries, history)
        hmm = self.hmm.win_chances(queries, history)
        by_machine = _by_machine(queries, history)
        out = {}
        for q in queries:
            probs = base[q.query_id]
            key = (q.episode_id, q.machine_id)
            lm, lh = _logit(1.0 - probs[0]), _logit(hmm[q.query_id])
            share = peer_share(key, q.cutoff, by_machine)
            flat = 0.0 if share is None else share - self.center
            lagged = 0.0
            for k, (w, c) in enumerate(zip(LAG_WEIGHTS, self.centers), start=1):
                s = peer_share(key, q.cutoff, by_machine, (k,))
                lagged += 0.0 if s is None else w * (s - c)
            z = ((0.7 * lm + 0.3 * lh + flat) + (0.6 * lm + 0.4 * lh + flat) + (0.7 * lm + 0.3 * lh + lagged)) / 3.0
            win = 1.0 / (1.0 + math.exp(-z))
            old = 1.0 - probs[0]
            raw = [max(p, MIN_PROBABILITY) for p in [1.0 - win] + [win * s / old for s in probs[1:]]]
            total = math.fsum(raw)
            out[q.query_id] = probabilities([p / total for p in raw], q.query_id)
        return out


def build_plateau(mixer, hmm, queries, history, log=print):
    center, centers = train_center(queries, history), lag_centers(queries, history)
    log(f"v11 plateau average: centre {center:.4f}, per-lag centres {[round(c, 3) for c in centers]}")
    return PlateauAverage(mixer, hmm, center, centers)
