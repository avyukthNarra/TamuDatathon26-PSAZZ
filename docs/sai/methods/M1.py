"""M1: holdout-selected half-life (B3 mechanism) with varied holdout length, half-life grid, and weight floor.

Pre-declared grid (full cross, 5 x 6 x 3 = 90 cells):
  holdout in {20, 30, 40, 60, 80}
  grid in {B3 (inf,128,64,32,16); +8; +(12,8,4); (inf,256,128,64,32); fine (inf,192,128,96,64,48,32,24,16);
           short-heavy (128,64,32,16,8,4)}
  floor (minimum row weight after exponential decay) in {0, 0.05, 0.15}
Selection rule, ridge, size split are exactly B3's. Cell (40, B3, 0) must reproduce B3 (45.82).
Run: python3 methods/M1.py [--smoke]
"""
import sys, itertools
sys.path.insert(0, __import__("os").path.dirname(__file__))
import _mem as m

GRIDS = {"B3": m.B3_GRID, "B3+8": m.B3_GRID + (8,), "B3+12,8,4": m.B3_GRID + (12, 8, 4),
         "long": (m.INF, 256, 128, 64, 32), "fine": (m.INF, 192, 128, 96, 64, 48, 32, 24, 16),
         "short": (128, 64, 32, 16, 8, 4)}


def make(p):
    def predict(split):
        qs = m.prep(split)
        g = GRIDS[p["grid"]]
        return m.assemble(split, [m.pw_hl(q, m.b3_hl(q, g, p["holdout"], p["floor"]), p["floor"]) for q in qs])
    return predict


cells = [{"holdout": h, "grid": g, "floor": f} for h, g, f in itertools.product((40, 20, 30, 60, 80), GRIDS, (0.0, 0.05, 0.15))]
if __name__ == "__main__":
    m.main_runner("M1", __doc__, cells, make, "B3 holdout half-life selection; vary holdout/grid/floor")
