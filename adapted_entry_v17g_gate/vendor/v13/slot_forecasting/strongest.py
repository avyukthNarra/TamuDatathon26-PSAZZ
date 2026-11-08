"""v12: v11's win chance (plateau.py) with payout sizes from the size-rule HMM (size_rule_hmm.py).

Size given a win is one of two exact mixes per machine type, and which one applies is set per
machine by ONE earlier spin: whether the payout k spins back was >= 2x (k in {1, 2, 4}, either
direction) - found by the zhill-56.28-hmm / latest-update-push-zhill branch. The size HMM infers
that rule per machine from its own past wins; the core rules are re-found on each round's train.
Types without exactly two clear size modes keep the pair mixer's sizes.

Research (research/size_hmm_eval.py): cross-fitted validation 56.58 -> 57.33 (halves 57.47 /
57.18), held-out train window dR -0.0162 [-0.0236, -0.0090]; every condition improves.
"""
import math

from .schema import probabilities
from .size_rule_hmm import SizeRuleHMM, core_from_train, default_rules

MIN_PROBABILITY = 1e-4


def two_modes(size_modes):
    """{type: (small, big)} from the mixer's size modes when a type has exactly two."""
    out = {}
    for kind, spec in (size_modes or {}).items():
        modes = spec["modes"] if isinstance(spec, dict) else spec
        if len(modes) == 2:
            a, b = sorted((tuple(m) for m in modes), key=lambda m: m[2] + m[3])
            out[kind] = (a, b)
    return out


class WithSizeRules:
    def __init__(self, model, size_hmm):
        self.model, self.size_hmm = model, size_hmm
        self.pairs, self.cell_mode = model.pairs, model.cell_mode

    def predict(self, queries, history):
        base = self.model.predict(queries, history)
        splits = self.size_hmm.splits(queries, history)
        out = {}
        for qid, probs in base.items():
            if qid not in splits:
                out[qid] = probs
                continue
            win = 1.0 - probs[0]
            raw = [max(p, MIN_PROBABILITY) for p in [1.0 - win] + [win * s for s in splits[qid]]]
            total = math.fsum(raw)
            out[qid] = probabilities([p / total for p in raw], qid)
        return out


def with_size_rules(model, queries, history, size_modes, log=print):
    modes = two_modes(size_modes)
    if not modes:
        log("size-rule HMM skipped: no type has exactly two size modes")
        return model
    core = core_from_train(queries, history, modes)
    log(f"size-rule HMM for {sorted(modes)}; core rules from train: {core}")
    return WithSizeRules(model, SizeRuleHMM(modes, prior=default_rules(core_weight=0.97, core=core), switch=0.0))
