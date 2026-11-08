"""L2: B3 per-machine ridge on lags + pairs; vary lambda, lag set, pair set.
PRE-DECLARED GRID (full cross, 11x4x5 = 220 cells; lag set '7' is replaced by data-learned 'strict' lags, which is (7,)) + extras:
  lam in {0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80}   (lower values added in two rounds because the best cell sat on the lower edge each time)
  lag sets: B3 (1,2,3,4,5,7) | screened (from train lag_screen, see _core.screened_sets) | 1-8 | (7,)
  pair sets: B3 ((1,3),(2,5),(4,7)) | screened (data-chosen from train) | strict (screened pairs > 3x null max) | top3 (3 largest pooled pair excess in train) | none
  extras: hl_grid with 8 added; holdout 60; holdout 25.
Payout-size split is B3's. Lags/pairs for 'screened'/'strict' are learned from train histories, not hard-coded.
Section-5 cross: pairs on/off is covered by the 'none' pair set; signal cannot enter a per-machine fit (constant per machine), see L9.
Run: python3 methods/L2.py [--smoke]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _core as K
from _run import run_grid
from harness import load_split

t = load_split("train")
SL, SP = K.screened_sets(t)
s = K.screen(t)
# strict pairs: excess above 3x the null max in at least one type
strict = []
for (ty, a, b), val in s["pair"].items():
    if b <= 8:
        nm = max(v for (t2, a2, b2), v in s["pair"].items() if t2 == ty and a2 >= 9)
        if val > 3 * max(nm, 1.0) and (a, b) not in strict:
            strict.append((a, b))
STRICT = tuple(sorted(strict))
lstrict = []
for (ty, l), val in s["single"].items():
    if l <= 8 and val > 3 * max(max(v for (t2, l2), v in s["single"].items() if t2 == ty and l2 >= 9), 1.0) and l not in lstrict:
        lstrict.append(l)
LSTRICT = tuple(sorted(lstrict))
agg = {}
for (ty, a, b), val in s["pair"].items():
    if b <= 8:
        agg[(a, b)] = agg.get((a, b), 0) + val
TOP3 = tuple(sorted(sorted(agg, key=lambda k: -agg[k])[:3]))
LAGSETS = {"B3": K.B3["lags"], "scr": SL, "1-8": tuple(range(1, 9)), "strict": LSTRICT}
PAIRSETS = {"B3": K.B3["pairs"], "scr": SP, "strict": STRICT, "top3": TOP3, "none": ()}

def predict(split, p):
    return K.predict(split, K.C(**p))

cells = []
for lam in (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80):
    for ln, l in LAGSETS.items():
        for pn, pr in PAIRSETS.items():
            cells.append((f"lam{lam}_lags{ln}_pairs{pn}", dict(lam=float(lam), lags=l, pairs=pr)))
extras = [("hl+8", dict(hl_grid=(1e9, 128, 64, 32, 16, 8))), ("holdout60", dict(holdout=60)), ("holdout25", dict(holdout=25))]
if __name__ == "__main__":
    print("lstrict", LSTRICT, "top3", TOP3, "screened lags", SL, "pairs", SP, "strict", STRICT)
    run_grid("L2", "B3 ridge win model; varied lambda x lag set x pair set (sets learned from train screen)", cells, predict, extras)
