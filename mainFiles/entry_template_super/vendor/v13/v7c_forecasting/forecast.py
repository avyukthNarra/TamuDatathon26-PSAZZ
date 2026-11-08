"""Shared predict path for the runner and the packaged entry.

Size replacement keeps the pair mixer's p_0. The flip, when enabled, relabels
parity counts inside that walk before the size rule runs. Both are optional so
the ablation runner can turn them off without a second copy of the walk.
"""

from .episode_flip import FlipGate
from .size_gate import fit_size_gate
from .size_rule import SizeRule


def fit_size(queries, history, enabled, labels=None, gate_enabled=True):
    if not enabled:
        return None
    rule = SizeRule.learn(queries, history)
    gate = fit_size_gate(queries, history, labels) if gate_enabled else None
    rule.enabled = gate is None or bool(gate["use_candidate"])
    rule.gate_diagnostics = None if gate is None else gate["diagnostics"]
    rule.gate_choice = "size_rule" if rule.enabled else "v6_size_head"
    return rule


def fit_flip(model, queries, history, enabled):
    if not enabled:
        return None
    return FlipGate.fit(model.pairs, queries, history)


def predict_split(model, queries, history, size_rule, gate, swaps=None):
    if swaps is None and gate is not None:
        swaps, _detect = gate.swaps(queries, history, model.pairs)
    raw = model.predict(queries, history, swap_at=swaps)
    if size_rule is None or not getattr(size_rule, "enabled", True):
        return raw
    return size_rule.apply(raw, queries, history)
