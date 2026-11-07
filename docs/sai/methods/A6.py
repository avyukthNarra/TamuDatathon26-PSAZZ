"""A6: prior strength scaled by effective sample size: alpha_eff = a0 * (W/W_med)^g, W = recency-weighted observed
tick count (hl 64), W_med = median over the split's queries (data-derived). g<0: less data -> stronger prior.

Pre-declared grid: g in {-1,-0.5,0,0.5,1} x a0 in {4,8,16,32} x prior in {type, cell}, ridge and freq modes (80 cells).
"""
import itertools
import _freq as f

BASE = [dict(mode="ridge", prior="self", alpha=8, hl=64), dict(mode="ridge", prior="type", alpha=8, hl=64), dict(mode="freq", prior="type", alpha=8, hl=64)]
cells = list(BASE)
for md, pr, g, a in itertools.product(("ridge", "freq"), ("type", "cell"), (-1, -0.5, 0.0, 0.5, 1), (4, 8, 16, 32)):
    cells.append(dict(mode=md, prior=pr, alpha=a, hl=64, g=g))

if __name__ == "__main__":
    f.run_ids("A6", cells, "prior strength scaled by effective sample size")
