"""(Adapted verbatim from the latest-update-push-zhill branch, commit 00d0d9d.)

Size-rule HMM: which of a few known rules sets a machine's payout-size mix.

Payout size given a win is exactly one of two mixes per machine type ("small" A, "big" B, read
off validation targets). Which mix applies at a spin is set by ONE earlier spin: whether the
payout `k` spins back was >= 2x, with k in {1, 2, 4} and either direction (found by maximum
likelihood on long train AND validation histories: these 6 rules win for ~2/3 of machines; the
rest are mostly the same rules lost to noise). Other lags / thresholds get a small prior so the
filter can still find them.

Hidden state = (threshold, lag, inverted) or None (50/50 mix). Forward filtering over the sizes
of the machine's own past wins, with a small per-tick switch chance (drift), gives P(state), and
the forecast size split at the cutoff is the posterior-weighted mix. Only ticks below the cutoff
are read. Standard library only.
"""
import math

from .schema import index_history

CLASS = {1: 0, 2: 1, 5: 2, 20: 3}
CORE = [(2, k, d) for k in (1, 2, 4) for d in (False, True)]


ALL_RULES = [(t, k, d) for t in (2, 5) for k in range(1, 9) for d in (False, True)]


def default_rules(core_weight=0.9, none_weight=0.02, core=CORE):
    core = list(core)
    others = [r for r in ALL_RULES if r not in core]
    prior = {r: core_weight / len(core) for r in core}
    prior.update({r: (1 - core_weight - none_weight) / len(others) for r in others})
    prior[None] = none_weight
    return prior


class SizeRuleHMM:
    def __init__(self, modes, prior=None, switch=0.001):
        self.modes = modes  # {machine_type: (A split, B split)}, each 4 probs for 1/2/5/20x
        self.prior = prior or default_rules()
        self.switch = switch

    def big_probability(self, seq, cutoff, A, B):
        states = list(self.prior)
        prior = [self.prior[s] for s in states]
        post = list(prior)
        last = None
        for t in sorted(seq):
            v = seq[t]
            if v == 0 or v not in CLASS:
                continue
            if last is not None:
                stay = (1 - self.switch) ** (t - last)
                post = [stay * p + (1 - stay) * q for p, q in zip(post, prior)]
            last = t
            c = CLASS[v]
            like = []
            for s in states:
                big = self._big(seq, t, s)
                like.append(0.5 * (A[c] + B[c]) if big is None else (B[c] if big else A[c]))
            post = [p * l for p, l in zip(post, like)]
            total = math.fsum(post)
            post = [p / total for p in post]
        if last is not None:
            stay = (1 - self.switch) ** (cutoff - last)
            post = [stay * p + (1 - stay) * q for p, q in zip(post, prior)]
        pb = 0.0
        for s, p in zip(states, post):
            big = self._big(seq, cutoff, s)
            pb += p * (0.5 if big is None else float(big))
        return pb

    @staticmethod
    def _big(seq, t, state):
        if state is None:
            return None
        threshold, lag, inverted = state
        x = seq.get(t - lag)
        if x is None:
            return None
        return (x >= threshold) != inverted

    def splits(self, queries, history):
        """{query_id: 4-prob size split given a win} (only for machine types with known modes)."""
        indexed = index_history(queries, history)
        out = {}
        for q in queries:
            if q.machine_type not in self.modes:
                continue
            A, B = self.modes[q.machine_type]
            seq = {p.tick: p.payout for p in indexed.get((q.episode_id, q.machine_id), [])}
            pb = self.big_probability(seq, q.cutoff, A, B)
            out[q.query_id] = tuple((1 - pb) * a + pb * b for a, b in zip(A, B))
        return out


def modes_from_targets(targets, queries):
    """The two distinct size splits per machine type in validation targets (small-mix first)."""
    kind = {q.query_id: q.machine_type for q in queries}
    found = {}
    for t in targets:
        k = kind.get(t.query_id)
        w = 1 - t.probabilities[0]
        if k is None or w < 1e-6:
            continue
        split = tuple(round(x / w, 6) for x in t.probabilities[1:])
        found.setdefault(k, {}).setdefault(split, 0)
        found[k][split] += 1
    out = {}
    for k, counts in found.items():
        top = sorted(counts, key=counts.get, reverse=True)[:2]
        # only when two exact mixes explain (almost) every answer for this type; else no size HMM
        if len(top) == 2 and counts[top[1]] >= 10 and counts[top[0]] + counts[top[1]] >= 0.9 * sum(counts.values()):
            a, b = sorted(top, key=lambda s: s[2] + s[3])  # small mix = less 5x/20x mass
            out[k] = (a, b)
    return out


def core_from_train(queries, history, modes, min_spins=150, min_share=0.05):
    """Rediscover the common size rules on this round's train: for each long-history machine, the
    rule with the highest likelihood of its observed win sizes; rules that win for >= min_share of
    machines become the core. Falls back to CORE when nothing qualifies (e.g. no long histories)."""
    indexed = index_history(queries, history)
    wins, n = {}, 0
    for q in queries:
        if q.machine_type not in modes:
            continue
        seq = {p.tick: p.payout for p in indexed.get((q.episode_id, q.machine_id), [])}
        if len(seq) < min_spins:
            continue
        A, B = modes[q.machine_type]
        best, best_ll = None, None
        for rule in [None] + ALL_RULES:
            ll = 0.0
            for t, v in seq.items():
                if v not in CLASS or v == 0:
                    continue
                c = CLASS[v]
                big = SizeRuleHMM._big(seq, t, rule)
                ll += math.log(max(0.5 * (A[c] + B[c]) if big is None else (B[c] if big else A[c]), 1e-9))
            if best_ll is None or ll > best_ll:
                best, best_ll = rule, ll
        n += 1
        wins[best] = wins.get(best, 0) + 1
    core = sorted((r for r, c in wins.items() if r is not None and c >= min_share * n), key=ALL_RULES.index)
    return core or list(CORE)
