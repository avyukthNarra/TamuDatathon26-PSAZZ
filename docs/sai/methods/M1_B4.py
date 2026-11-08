"""M1_B4: holdout half-life selection on the B4 base (lags (7,), pairs (1,3),(2,5),(4,7), near-OLS ridge).

Base cell (holdout 40, grid B3, floor 0, lam 0.05) must reproduce B4 (50.89).
Pre-declared grids (all scope dev):
  Round A (lam 0.05): holdout {20,30,40,60,80} x grid {B3 (inf,128,64,32,16); +8; +(12,8,4); long (inf,256,128,64,32);
     fine (inf,192,128,96,64,48,32,24,16); short (128,64,32,16,8,4)} x floor {0,.05,.15} = 90 cells
  Round B (ridge lam x memory): lam {0.5,2} x holdout {20,40,80} x grid {B3, +(12,8,4), long, fine} x floor {0,.05} = 48 cells
  Round C (extension, declared because best may sit on long edge; lam 0.05): grid xlong (inf,512,256,128,64,32,16)
     and xfine (inf,384,256,192,128,96,64,48,32,24,16) x holdout {40,60,80} x floor {0,.05} = 12 cells
Selection by total dev points. Run: python3 methods/M1_B4.py [--smoke]
"""
import sys, itertools
sys.path.insert(0, __import__("os").path.dirname(__file__))
import _mem4 as m

G = {"B3": m.B3_GRID, "B3+8": m.B3_GRID + (8,), "B3+12,8,4": m.B3_GRID + (12, 8, 4),
     "long": (m.INF, 256, 128, 64, 32), "fine": (m.INF, 192, 128, 96, 64, 48, 32, 24, 16),
     "short": (128, 64, 32, 16, 8, 4), "xlong": (m.INF, 512, 256, 128, 64, 32, 16),
     "xfine": (m.INF, 384, 256, 192, 128, 96, 64, 48, 32, 24, 16)}


def make(p):
    def predict(split):
        m.set_lam(p["lam"])
        qs = m.prep(split)
        g = G[p["grid"]]
        return m.assemble(split, [m.pw_hl(q, m.b3_hl(q, g, p["holdout"], p["floor"]), p["floor"]) for q in qs])
    return predict


cells = [{"lam": 0.05, "holdout": h, "grid": g, "floor": f} for h, g, f in
         itertools.product((40, 20, 30, 60, 80), list(G)[:6], (0.0, 0.05, 0.15))]
cells += [{"lam": l, "holdout": h, "grid": g, "floor": f} for l, h, g, f in
          itertools.product((0.5, 2.0), (20, 40, 80), ("B3", "B3+12,8,4", "long", "fine"), (0.0, 0.05))]
cells += [{"lam": 0.05, "holdout": h, "grid": g, "floor": f} for g, h, f in
          itertools.product(("xlong", "xfine"), (40, 60, 80), (0.0, 0.05))]
if __name__ == "__main__":
    m.main_runner("M1_B4", __doc__, cells, make, "B4 base holdout half-life selection; vary holdout/grid/floor/lam")
