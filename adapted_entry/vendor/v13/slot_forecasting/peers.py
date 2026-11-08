"""Peer correction: machines win more when the other machines in their episode just won.

Residual analysis of the 56.54 model (research/residuals.py) found its win chance too low when
the episode's OTHER machines won in the last few ticks, and too high when they lost, on
validation truth (z +4.0) and train outcomes (z +6.0), at every lag 1-8 (strongest 3-4) and in
every condition and type. The per-machine response spreads no wider than noise, so one shared
coefficient is used:

    logit p = logit p_model + COEF * (share of peers' observed spins at lags 1-5 that won - CENTER)

CENTER is the average of that share over train queries (learned at fit time). COEF = 1.0: train
fitted 1.8, validation prefers 0.75; on both checks 0.5-1.0 pass (validation 56.51 -> 56.56,
both halves up; train window dR -0.0037 [-0.0057, -0.0019]). Only ticks before the cutoff are read.
"""
import math

from .schema import index_history, probabilities

PEER_COEF = 1.0
PEER_LAGS = range(1, 6)
MIN_PROBABILITY = 1e-4


def peer_share(own_key, cutoff, by_machine, lags=PEER_LAGS):
    episode = own_key[0]
    wins = total = 0
    for key, ticks in by_machine.items():
        if key[0] != episode or key == own_key:
            continue
        for k in lags:
            v = ticks.get(cutoff - k)
            if v is not None:
                wins += v > 0
                total += 1
    return wins / total if total else None


def _by_machine(queries, history):
    out = {}
    for key, plays in index_history(queries, history).items():
        out[key] = {p.tick: p.payout for p in plays}
    return out


def train_center(queries, history):
    by_machine = _by_machine(queries, history)
    shares = [s for q in queries if (s := peer_share((q.episode_id, q.machine_id), q.cutoff, by_machine)) is not None]
    return sum(shares) / len(shares) if shares else 0.5


class PeerAdjusted:
    def __init__(self, model, center, coef=PEER_COEF):
        self.model, self.center, self.coef = model, center, coef

    def predict(self, queries, history):
        base = self.model.predict(queries, history)
        by_machine = _by_machine(queries, history)
        out = {}
        for q in queries:
            probs = base[q.query_id]
            share = peer_share((q.episode_id, q.machine_id), q.cutoff, by_machine)
            if share is None:
                out[q.query_id] = probs
                continue
            win = min(max(1.0 - probs[0], 1e-4), 1 - 1e-4)
            z = math.log(win / (1 - win)) + self.coef * (share - self.center)
            new = 1.0 / (1.0 + math.exp(-z))
            raw = [max(p, MIN_PROBABILITY) for p in [1.0 - new] + [new * s / win for s in probs[1:]]]
            total = math.fsum(raw)
            out[q.query_id] = probabilities([p / total for p in raw], q.query_id)
        return out
