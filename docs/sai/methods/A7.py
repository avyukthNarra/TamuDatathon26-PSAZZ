"""A7: factorized (win-rate x conditional severity) vs direct 5-class frequencies, and their blend.

Pre-declared grid (all cells scored on dev):
  cell0  B3 repro (prior self, alpha 8, hl 64, ridge pw)   [must equal 45.82]
  cell1  same with train-estimated type prior
  mode ridge (B3 pair-ridge pw x severity) | freq (recency-weighted win freq x severity, hlw 64, aw 20), prior type
  direct blend weight w in {0.25,0.5,1.0} x direct half-life dhl in {32,64,128,inf} x pseudo-ticks da in {10,30,100}
  (w=1 in mode ridge is pure direct 5-class, independent of pw; reported once under mode "freq")
  plus prior cell (9-cell) variants of cell1 and of the best-direct setting.
"""
import itertools
import _freq as f

cells = [dict(mode="ridge", prior="self", alpha=8, hl=64), dict(mode="ridge", prior="type", alpha=8, hl=64),
         dict(mode="freq", prior="type", alpha=8, hl=64)]
for w, dhl, da in itertools.product((0.25, 0.5), (32, 64, 128, 1e9), (10, 30, 100)):
    cells.append(dict(mode="ridge", prior="type", alpha=8, hl=64, a7=dict(w=w, dhl=dhl, da=da)))
for dhl, da in itertools.product((32, 64, 128, 1e9), (10, 30, 100)):
    cells.append(dict(mode="freq", prior="type", alpha=8, hl=64, a7=dict(w=1.0, dhl=dhl, da=da)))
cells += [dict(mode="ridge", prior="cell", alpha=8, hl=64),
          dict(mode="ridge", prior="cell", alpha=8, hl=64, a7=dict(w=0.5, dhl=64, da=30)),
          dict(mode="freq", prior="cell", alpha=8, hl=64, a7=dict(w=1.0, dhl=64, da=30))]
if __name__ == "__main__":
    f.run_ids("A7", cells, "factorized vs direct 5-class frequency, blended")
