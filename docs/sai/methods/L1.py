"""L1: one-hot lags 1-8 + missing flag, regularised logistic (and linear) win model, with/without pair products.
Binary win/loss target (not a 5-class multinomial: payout split stays B3's, so only the win model changes, as in L2).
Columns per lag: win indicator and missing indicator (loss = baseline). Half-life by the B3 holdout rule.
PRE-DECLARED GRID: link {lin, logit} x lag set {1-8, 1-16 (lags 9-16 are null controls), strict, B3}
 x pairs {none, learned top3} x lam {0.1, 0.3, 1, 3, 10, 30} = 96 cells (0.1, 0.3 added after the first 64-cell run peaked at lam=1, the lower edge). coding = onehot throughout.
Run: python3 methods/L1.py [--smoke]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _core as K
from _run import run_grid
from harness import load_split

LN = K.learned(load_split("train"))
LAGS = {"1-8": LN["lag_all"], "1-16": tuple(range(1, 17)), "strict": LN["lag_strict"], "B3": K.B3["lags"]}
PAIRS = {"none": (), "top3": LN["pair_top3"]}

def predict(split, p):
    return K.predict(split, K.C(**p))

cells = [(f"{lk}_lags{ln}_pairs{pn}_lam{lam}", dict(link=lk, coding="onehot", lags=l, pairs=pr, lam=float(lam)))
         for lk in ("lin", "logit") for ln, l in LAGS.items() for pn, pr in PAIRS.items() for lam in (0.1, 0.3, 1, 3, 10, 30)]
if __name__ == "__main__":
    run_grid("L1", "one-hot lags + missing flags, regularised logistic/linear win model (+/- pairs)", cells, predict)
