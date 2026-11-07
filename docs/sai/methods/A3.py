"""A3: half-life grid for the size split (and, in freq mode, also for the win frequency: hlw = hl).

Pre-declared grid: hl in {4,8,16,24,32,48,64,128,inf} x alpha in {8,24} x norm in {False (B3 convention, shorter hl
= relatively stronger prior), True (rescaled to common effective sample)} x mode {ridge, freq}, prior type (72 cells).
"""
import itertools
import _freq as f

BASE = [dict(mode="ridge", prior="self", alpha=8, hl=64), dict(mode="ridge", prior="type", alpha=8, hl=64), dict(mode="freq", prior="type", alpha=8, hl=64)]
cells = list(BASE)
for md, h, a, nm in itertools.product(("ridge", "freq"), (4, 8, 16, 24, 32, 48, 64, 128, 1e9), (8, 24), (False, True)):
    cells.append(dict(mode=md, prior="type", alpha=a, hl=h, hlw=h, norm=nm))

if __name__ == "__main__":
    f.run_ids("A3", cells, "size-split half-life grid")
