"""L7: per-machine sparse feature selection from a wide candidate set, then ridge on the selected columns.
Selection per machine on its own pre-cutoff rows: univariate |corr| z-score of each candidate column with the win flag,
keep columns with z >= thr, or the top-k columns by z (rule 'k'). Columns: lag flags and pair products.
Candidate sets: A = lags 1-8 + all 28 pairs of 1-8 | B = A + null controls (lags 9-16 + 28 pairs of 9-16) |
 C = screened lags/pairs learned from train. Half-life chosen with the same holdout rule as B3 (selection is made on all
 pre-cutoff rows, once, before the half-life holdout; a mild within-machine optimism in the half-life choice only).
Payout split = B3's.
PRE-DECLARED GRID: candidate {A,B,C} x rule {z>=0 (all),0.5,1,1.5,2,3,4,5,6; top-k 2,3,4,6} x lam {0.1,1,5} = 117 cells (z 0-1.5 added after the first 81-cell run peaked at the z>=2 edge).
Run: python3 methods/L7.py [--smoke]
"""
import itertools
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _core as K
from _run import run_grid
from harness import load_split

LN = K.learned(load_split("train"))
l8, p8 = LN["lag_all"], LN["pair_all"]
CAND = {"A": (l8, p8), "B": (l8 + tuple(range(9, 17)), p8 + tuple(itertools.combinations(range(9, 17), 2))),
        "C": (LN["lag_scr"], LN["pair_scr"])}

def predict(split, p):
    return K.predict(split, K.C(**p))

cells = []
for cn, (cl, cp) in CAND.items():
    for rule in (("thr", 0), ("thr", 0.5), ("thr", 1), ("thr", 1.5), ("thr", 2), ("thr", 3), ("thr", 4), ("thr", 5), ("thr", 6), ("k", 2), ("k", 3), ("k", 4), ("k", 6)):
        sel = dict(cand_lags=cl, cand_pairs=cp, **({"thr": rule[1]} if rule[0] == "thr" else {"k": rule[1]}))
        for lam in (0.1, 1, 5):
            cells.append((f"cand{cn}_{rule[0]}{rule[1]}_lam{lam}", dict(lags=(), pairs=(), sel=sel, lam=float(lam))))
if __name__ == "__main__":
    run_grid("L7", "per-machine univariate-z selection from a wide candidate set + ridge", cells, predict)
