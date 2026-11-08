"""Population priors (empirical Bayes) for the pair mixer's expert weights.

The pair mixer starts every machine with uniform weights over its win experts (lag pair x memory)
and its payout-size experts (which earlier spin's payout keys the size mode), and must learn them
from that machine's own spins. On a short history (~40 spins, ~15 wins) that is too little. Train
has ~900 machines: here each machine type's AVERAGE final expert log-weights are learned on train
(walking every train machine to its cutoff) and used, scaled by `strength`, as the starting
weights of every machine of that type. Strength 0 reproduces the pair mixer exactly.

PopulationMixer subclasses PairMixer without editing it: walk() is rebuilt from PairMixer.walk's
source with two hooks (start weights from the type's population prior; record the final weights).
"""
import inspect
import math
import textwrap

from .pair_mixer import PairMixer
from .schema import index_history

_source = textwrap.dedent(inspect.getsource(PairMixer.walk))
for old, new in (
    ("def walk(self, own, machine_type, stop, emit, signal=None):",
     "def walk(self, own, machine_type, stop, emit, signal=None):\n    _pop = self.population.get(machine_type, {})"),
    ("    logw = list(prior)\n    size_logw = [0.0] * len(self.size_keys)",
     "    logw = [p + self.win_strength * q for p, q in zip(prior, _pop.get('win', [0.0] * len(prior)))]\n"
     "    size_logw = [self.size_strength * q for q in _pop.get('size', [0.0] * len(self.size_keys))]"),
    ("        if t == stop:\n            break",
     "        if t == stop:\n            self._final = (list(logw), list(size_logw))\n            break"),
):
    assert old in _source, old[:40]
    _source = _source.replace(old, new)
_namespace = dict(vars(__import__("slot_forecasting.pair_mixer", fromlist=["*"])))
exec(compile(_source, "<population walk>", "exec"), _namespace)


class PopulationMixer(PairMixer):
    walk = _namespace["walk"]

    def __init__(self, *args, population=None, win_strength=0.0, size_strength=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.population = population or {}
        self.win_strength = win_strength
        self.size_strength = size_strength


def _centred(values):
    top = max(values)
    logs = [v - top for v in values]
    mean = sum(logs) / len(logs)
    return [v - mean for v in logs]


def learn_population(mixer, queries, history):
    """Average (centred, capped) final expert log-weights per machine type over train machines."""
    from .pair_mixer import CLASS_INDEX
    indexed = index_history(queries, history)
    sums = {}
    for q in queries:
        own = [None] * (q.cutoff + 1)
        for p in indexed.get((q.episode_id, q.machine_id), []):
            own[p.tick] = CLASS_INDEX[p.payout]
        mixer.walk(own, q.machine_type, q.cutoff, lambda t, p: None, q.signal)
        win, size = mixer._final
        row = sums.setdefault(q.machine_type, {"win": [0.0] * len(win), "size": [0.0] * len(size), "n": 0})
        for key, values in (("win", win), ("size", size)):
            for i, v in enumerate(_centred([max(x, max(values) - 20.0) for x in values])):
                row[key][i] += v
        row["n"] += 1
    return {kind: {key: [v / row["n"] for v in row[key]] for key in ("win", "size")} for kind, row in sums.items()}


def population_mixer(base, queries, history, win_strength, size_strength):
    """A PopulationMixer with `base`'s settings and population priors learned on (queries, history)."""
    settings = {k: v for k, v in vars(base).items() if k in inspect.signature(PairMixer.__init__).parameters}
    probe = PopulationMixer(**settings)
    probe.type_size_priors = base.type_size_priors
    probe.size_modes, probe.win_prior, probe.win_priors = base.size_modes, base.win_prior, base.win_priors
    population = learn_population(probe, queries, history)
    model = PopulationMixer(**settings, population=population, win_strength=win_strength, size_strength=size_strength)
    for attr in ("type_size_priors", "size_modes", "win_prior", "win_priors", "pairs", "pair_prior", "half_lives",
                 "cell_mode", "size_keys", "size_cell"):
        setattr(model, attr, getattr(base, attr))
    return model
