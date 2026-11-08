"""A1: type-level Dirichlet smoothing strength alpha for the size split.

Pre-declared grid: prior in {self (B3), type (train), cell (train, signal cross)} x alpha in {2,8,24,64},
hl 64, in ridge mode and freq mode (24 cells) + baselines.
Extension (added because alpha=64 was the grid edge and still best): alpha in {128,256,512} for prior in {type,cell}, ridge mode (6 cells).
"""
import itertools
import _freq as f

BASE = [dict(mode="ridge", prior="self", alpha=8, hl=64), dict(mode="ridge", prior="type", alpha=8, hl=64), dict(mode="freq", prior="type", alpha=8, hl=64)]
cells = list(BASE)
for md, pr, a in itertools.product(("ridge", "freq"), ("self", "type", "cell"), (2, 8, 24, 64)):
    cells.append(dict(mode=md, prior=pr, alpha=a, hl=64))
for pr, a in itertools.product(("type", "cell"), (128, 256, 512)):
    cells.append(dict(mode="ridge", prior=pr, alpha=a, hl=64))

if __name__ == "__main__":
    f.run_ids("A1", cells, "alpha grid for type Dirichlet smoothing")
