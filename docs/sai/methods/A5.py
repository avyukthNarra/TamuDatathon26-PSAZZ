"""A5: inverse-density weights for short-history older ticks. Short history is detected from tick gaps only
(observed fraction < 0.6 of cutoff and a dense observed tail); older sampled ticks each get weight (gap_factor)^e5,
gap_factor = (#ticks before the dense tail) / (#observed older ticks), ~7 measured from the data.

Pre-declared grid: e5 in {0,0.5,1} x e5r (also weight ridge rows of the win model: 0/1) x alpha in {8,24}
x hl in {64,128} x prior in {type, cell}, ridge mode; e5r only matters for ridge. Freq mode: e5 in {0,0.5,1} x
hl in {64,128} x prior in {type,cell}.
"""
import itertools
import _freq as f

BASE = [dict(mode="ridge", prior="self", alpha=8, hl=64), dict(mode="ridge", prior="type", alpha=8, hl=64), dict(mode="freq", prior="type", alpha=8, hl=64)]
cells = list(BASE)
for e, er, a, h, pr in itertools.product((0, 0.5, 1), (0, 1), (8, 24), (64, 128), ("type", "cell")):
    cells.append(dict(mode="ridge", prior=pr, alpha=a, hl=h, e5=e, e5r=er))
for e, h, pr in itertools.product((0, 0.5, 1), (64, 128), ("type", "cell")):
    cells.append(dict(mode="freq", prior=pr, alpha=8, hl=h, hlw=h, e5=e))

if __name__ == "__main__":
    f.run_ids("A5", cells, "inverse-density weights for short-history older ticks")
