"""Three-model win-chance ensemble: pair mixer + hidden-rule HMM + change-point regime model.

Each model sees the same per-machine history but explains it differently:
  pair mixer   decaying lag-pair tallies weighted by recent accuracy (slot_forecasting/pair_mixer.py)
  rule HMM     a hidden rule that can jump, fixed live / dead levels (slot_forecasting/rule_hmm.py)
  regime model Bayesian change points with each cell's level learned since the regime began
               (slot_forecasting/changepoint.py)
The final win chance is their equal-weight average; payout sizes (given a win) come from the
pair mixer. Equal weights are used on purpose: the validation optimum (0.35 / 0.3 / 0.35) is a
flat plateau and 1/3 each needs no tuning in a new round.
"""
import math

from .changepoint import RegimeModel
from .pair_mixer import adapt
from .rule_hmm import HiddenRuleHMM, levels_from_prior
from .schema import probabilities

MIN_PROBABILITY = 1e-4


def _logit(p):
    p = min(max(p, 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


class Ensemble:
    def __init__(self, mixer, members, weights, pool="linear"):
        # pool: "linear" averages win probabilities; "logodds" averages their log-odds (a confident
        #   model pulls harder). Research on pranav/stacked-model: log-odds pooling of the same
        #   0.7 mixer / 0.3 HMM blend improved train cutoff dR -0.0013 [-0.0021, -0.0004] and both
        #   validation halves; the regime model did not pass the train check, so it is left out.
        self.mixer, self.members, self.weights, self.pool = mixer, members, weights, pool

    def predict(self, queries, history):
        base = self.mixer.predict(queries, history)
        wins = [member.win_chances(queries, history) for member in self.members]
        out = {}
        for query_id, probs in base.items():
            win = 1.0 - probs[0]
            values = [win] + [w[query_id] for w in wins]
            if self.pool == "logodds":
                z = math.fsum(weight * _logit(v) for weight, v in zip(self.weights, values))
                blended = 1.0 / (1.0 + math.exp(-z))
            else:
                blended = math.fsum(weight * v for weight, v in zip(self.weights, values))
            raw = [max(p, MIN_PROBABILITY) for p in [1.0 - blended] + [blended * s / win for s in probs[1:]]]
            total = math.fsum(raw)
            out[query_id] = probabilities([p / total for p in raw], query_id)
        return out


def build_ensemble(queries, history, labels, size_modes, win_prior, log=print, variant="peers"):
    """Fit the pair mixer on train (as adapt() does, without its own HMM blend) and add the other
    members, which need only the lag pairs it found and the win prior.

    variant "strongest": v12 = v11 with payout sizes from the size-rule HMM (strongest.py).
    variant "plateau": v11, the log-odds average of v10, a 0.6 / 0.4 version and per-lag peers.
    variant "peers" (default): the "logodds" model plus the peer correction (peers.py).
    variant "logodds": 0.7 mixer + 0.3 hidden-rule HMM, pooled in log-odds.
    variant "three-way": 1/3 mixer + 1/3 HMM + 1/3 regime model, linear (better on validation,
    not confirmed on train).
    """
    mixer = adapt(queries, history, labels, log=log, size_modes=size_modes, win_prior=win_prior, hmm_weight=0)
    if not isinstance(win_prior, dict):
        log("no per-type win prior: using the pair mixer alone")
        return mixer
    hmm = HiddenRuleHMM(mixer.pairs, levels_from_prior(win_prior))
    if variant == "three-way":
        log("ensemble: 1/3 pair mixer + 1/3 hidden-rule HMM + 1/3 change-point regime model (win chance)")
        return Ensemble(mixer, [hmm, RegimeModel(mixer.pairs, win_prior, temper=0.5)], (1 / 3, 1 / 3, 1 / 3))
    log("ensemble: 0.7 pair mixer + 0.3 hidden-rule HMM, pooled in log-odds (win chance)")
    if variant in ("plateau", "strongest"):
        from .plateau import build_plateau
        model = build_plateau(mixer, hmm, queries, history, log)
        if variant == "strongest":
            from .strongest import with_size_rules
            model = with_size_rules(model, queries, history, size_modes, log)
        return model
    model = Ensemble(mixer, [hmm], (0.7, 0.3), pool="logodds")
    if variant == "logodds":
        return model
    from .peers import PEER_COEF, PeerAdjusted, train_center
    center = train_center(queries, history)
    log(f"peer correction: logit += {PEER_COEF} x (peers' lag 1-5 win share - {center:.4f})")
    return PeerAdjusted(model, center)
