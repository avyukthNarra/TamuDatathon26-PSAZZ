"""A2: 9-cell (machine_type x signal) priors vs type-only, plus signal-conditioned win-rate shrinkage.

Pre-declared grid:
  cell0 B3 repro, cell1 train type prior, cell2 freq baseline
  prior in {type, cell} x alpha in {2,8,24} x wk (win-rate prior pseudo-ticks, ridge pw shrunk to the type/cell
  train win rate) in {0,50,200,800}  (ridge mode, 24 cells)
  freq mode: prior in {type, cell} x alpha in {8,24} x aw in {20,60,200}  (12 cells)
  The type-vs-cell contrast is the plan's with/without-signal cross.
"""
import itertools
import _freq as f

BASE = [dict(mode="ridge", prior="self", alpha=8, hl=64), dict(mode="ridge", prior="type", alpha=8, hl=64), dict(mode="freq", prior="type", alpha=8, hl=64)]
cells = list(BASE)
for pr, a, wk in itertools.product(("type", "cell"), (2, 8, 24), (0, 50, 200, 800)):
    cells.append(dict(mode="ridge", prior=pr, alpha=a, hl=64, wk=wk))
for pr, a, aw in itertools.product(("type", "cell"), (8, 24), (20, 60, 200)):
    cells.append(dict(mode="freq", prior=pr, alpha=a, hl=64, aw=aw))

if __name__ == "__main__":
    f.run_ids("A2", cells, "9-cell type x signal priors")
