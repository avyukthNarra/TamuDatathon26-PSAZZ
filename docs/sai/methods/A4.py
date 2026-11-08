"""A4: class-specific half-life and alpha for the size split (frequent classes p1,p2 vs rare p5,p20).

Pre-declared grid (norm=True: class sums rescaled by W(64)/W(hl_k) so classes share one effective sample scale):
  cell0 B3 repro, cell1 train type prior, cell2 freq-mode baseline
  hl_frequent (p1,p2) in {16,32,64} x hl_rare (p5,p20) in {64,128,inf} x alpha_rare in {8,24,64} x alpha_frequent in {4,8}
  (54 cells, ridge pw), prior type; plus freq-mode cells for the 9 (hl_f,hl_r) pairs at alpha 8/8.
  Also the same 9 pairs with norm=False (B3 convention) at alpha 8/8, ridge.
"""
import itertools
import _freq as f

BASE = [dict(mode="ridge", prior="self", alpha=8, hl=64), dict(mode="ridge", prior="type", alpha=8, hl=64), dict(mode="freq", prior="type", alpha=8, hl=64)]
cells = list(BASE)
for hf, hr, ar, af in itertools.product((16, 32, 64), (64, 128, 1e9), (8, 24, 64), (4, 8)):
    cells.append(dict(mode="ridge", prior="type", hl=[hf, hf, hr, hr], alpha=[af, af, ar, ar], norm=True))
for hf, hr in itertools.product((16, 32, 64), (64, 128, 1e9)):
    cells.append(dict(mode="freq", prior="type", hl=[hf, hf, hr, hr], alpha=8, norm=True))
    cells.append(dict(mode="ridge", prior="type", hl=[hf, hf, hr, hr], alpha=8, norm=False))

if __name__ == "__main__":
    f.run_ids("A4", cells, "class-specific half-life and alpha")
