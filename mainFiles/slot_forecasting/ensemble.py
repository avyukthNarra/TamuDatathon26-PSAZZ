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


class Ensemble:
    def __init__(self, mixer, members, weights):
        self.mixer, self.members, self.weights = mixer, members, weights

    def predict(self, queries, history):
        base = self.mixer.predict(queries, history)
        wins = [member.win_chances(queries, history) for member in self.members]
        out = {}
        for query_id, probs in base.items():
            win = 1.0 - probs[0]
            values = [win] + [w[query_id] for w in wins]
            blended = math.fsum(weight * v for weight, v in zip(self.weights, values))
            raw = [max(p, MIN_PROBABILITY) for p in [1.0 - blended] + [blended * s / win for s in probs[1:]]]
            total = math.fsum(raw)
            out[query_id] = probabilities([p / total for p in raw], query_id)
        return out


def build_ensemble(queries, history, labels, size_modes, win_prior, log=print):
    """Fit the pair mixer on train (as adapt() does, without its own HMM blend) and add the
    other two members, which need only the lag pairs it found and the win prior."""
    mixer = adapt(queries, history, labels, log=log, size_modes=size_modes, win_prior=win_prior, hmm_weight=0)
    if not isinstance(win_prior, dict):
        log("no per-type win prior: using the pair mixer alone")
        return mixer
    members = [HiddenRuleHMM(mixer.pairs, levels_from_prior(win_prior)),
               RegimeModel(mixer.pairs, win_prior, temper=0.5)]
    log("ensemble: 1/3 pair mixer + 1/3 hidden-rule HMM + 1/3 change-point regime model (win chance)")
    return Ensemble(mixer, members, (1 / 3, 1 / 3, 1 / 3))
