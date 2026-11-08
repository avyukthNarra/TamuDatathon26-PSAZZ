"""L3: all pairs among lags 1-8 with null controls (pairs among lags 9-16), then triples.
Base = L2-structure: strict lag(s) learned from train + top3 learned pairs. Variants (pair sets / triple sets):
 pairs: top3 | top6 | scr | all8 (28 pairs of lags 1-8) | null16 (28 pairs of lags 9-16, REPLACING top3) |
        top3+null16 (null features added) | all8+null16
 triples (on top of base lags + top3 pairs): core (all triples among lags in the screened sets: lags 1-5,7 union learned) |
        all8 (56 triples of lags 1-8) | null16 (56 triples of lags 9-16)
PRE-DECLARED GRID: 10 structures x lam {0.1,1,3,10,30,100} = 60 cells. Payout split = B3's.
Rationale for null controls: the simulator interacts only up to lag 8, so null features show the pure overfit cost.
Run: python3 methods/L3.py [--smoke]
"""
import itertools
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _core as K
from _run import run_grid
from harness import load_split

LN = K.learned(load_split("train"))
L0, P3 = LN["lag_strict"], LN["pair_top3"]
null16 = tuple(itertools.combinations(range(9, 17), 2))
core_lags = tuple(sorted(set(LN["lag_scr"]) | {x for p in LN["pair_scr"] for x in p}))
STRUCT = {
    "pairs_top3": dict(pairs=P3),
    "pairs_top6": dict(pairs=LN["pair_top6"]),
    "pairs_scr": dict(pairs=LN["pair_scr"]),
    "pairs_all8": dict(pairs=LN["pair_all"]),
    "pairs_null16": dict(pairs=null16),
    "pairs_top3+null16": dict(pairs=P3 + null16),
    "pairs_all8+null16": dict(pairs=LN["pair_all"] + null16),
    "tri_core": dict(pairs=P3, triples=tuple(itertools.combinations(core_lags, 3))),
    "tri_all8": dict(pairs=P3, triples=tuple(itertools.combinations(range(1, 9), 3))),
    "tri_null16": dict(pairs=P3, triples=tuple(itertools.combinations(range(9, 17), 3))),
}

def predict(split, p):
    return K.predict(split, K.C(**p))

cells = [(f"{n}_lam{lam}", dict(lags=L0, lam=float(lam), **kw)) for n, kw in STRUCT.items() for lam in (0.1, 1, 3, 10, 30, 100)]
if __name__ == "__main__":
    print("core_lags", core_lags)
    run_grid("L3", "all pairs 1-8 vs null-control pairs 9-16, then triples (ridge win model)", cells, predict)
