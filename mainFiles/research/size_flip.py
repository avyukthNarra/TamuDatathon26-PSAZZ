"""Episode-wide payout-size flip detection on top of v15 (run from mainFiles/).

Per machine: its MAP size rule (size_rule_hmm, fitted on wins BEFORE its last K wins), then the
log-likelihood ratio of its last K wins under the INVERTED rule vs the rule. An episode is flagged
when at least M machines have LLR > T; then every machine in it gets P(big mode) -> 1 - P(big mode)
on v15's (v7c) sizes. Both checks, vs v15.
"""
import math, sys
from collections import defaultdict
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import v13_eval as E
from slot_forecasting.io import load_inputs, read_targets
from slot_forecasting.size_rule_hmm import CLASS, SizeRuleHMM, default_rules, modes_from_targets, core_from_train
lg = E.lg
tq, th = load_inputs("public/train"); vq, _ = load_inputs("public/validation"); vt = read_targets("public/validation/targets.csv")
MODES = modes_from_targets(vt, vq); CORE = core_from_train(tq, th, MODES)
HMM = SizeRuleHMM(MODES, prior=default_rules(core_weight=0.97, core=CORE), switch=0.0)
def flip_llr(r, K):
    if r["type"] not in MODES: return None
    A, B = MODES[r["type"]]
    seq = {t: v for t, v in r["seq"].items() if t < r["cutoff"]}
    wins = [t for t in sorted(seq) if seq[t] > 0]
    if len(wins) < K + 10: return None
    early = {t: v for t, v in seq.items() if t < wins[-K]}
    states = list(HMM.prior); post = []
    for s in states:  # likelihood of early wins under each rule
        ll = math.log(HMM.prior[s])
        for t, v in early.items():
            if v <= 0 or v not in CLASS: continue
            big = HMM._big(early, t, s); c = CLASS[v]
            ll += math.log(0.5 * (A[c] + B[c]) if big is None else (B[c] if big else A[c]))
        post.append(ll)
    best = states[max(range(len(states)), key=lambda i: post[i])]
    if best is None: return None
    inv = (best[0], best[1], not best[2]); llr = 0.0
    for t in wins[-K:]:
        c = CLASS[seq[t]]; b1, b2 = HMM._big(seq, t, best), HMM._big(seq, t, inv)
        if b1 is None: continue
        llr += math.log((B[c] if b2 else A[c]) / (B[c] if b1 else A[c]))
    return llr
def prep(K):
    for rows in (E.P.VAL, E.P.TRAIN):
        for r in rows: r[f"llr{K}"] = flip_llr(r, K)
def flagged(rows, K, T, M):
    by = defaultdict(int)
    for r in rows:
        if r[f"llr{K}"] is not None and r[f"llr{K}"] > T: by[(r["episode"], r["cutoff"])] += 1
    return {k for k, n in by.items() if n >= M}
def v15(r, so, flip=False):
    a, b = E.v11(r), 1 - r["v7c"][0]
    win = 1 / (1 + math.exp(-(0.6 * lg(a) + 0.4 * lg(b) + 0.5 * E.flat(r) + 0.05 * r["signal"])))
    sb = [x / b for x in r["v7c"][1:]]
    if flip and r["type"] in MODES:
        A, B = MODES[r["type"]]
        pb = min(max(sum((x - p) * (q - p) for x, p, q in zip(sb, A, B)) / sum((q - p) ** 2 for p, q in zip(A, B)), 0), 1)
        sb = [pb * p + (1 - pb) * q for p, q in zip(A, B)]   # swap: P(big) -> 1 - P(big)
    raw = [max(x, 1e-4) for x in [1 - win] + [win * s for s in sb]]; t = sum(raw)
    return [x / t for x in raw]
E.BASE = lambda r, so: v15(r, so)
for K in (6, 10):
    prep(K)
    for T, M in ((2.0, 2), (3.0, 2), (2.0, 3)):
        fv, ft = flagged(E.P.VAL, K, T, M), flagged(E.P.TRAIN, K, T, M)
        E.evaluate(f"K {K} T {T} M {M} (flag {len(fv)} val / {len(ft)} train)", lambda r, so, fv=fv, ft=ft: v15(r, so, (r["episode"], r["cutoff"]) in (fv if r["episode"].startswith("main-validation") else ft)))
