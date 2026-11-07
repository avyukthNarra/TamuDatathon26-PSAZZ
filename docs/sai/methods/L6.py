"""L6: partial pooling of per-machine ridge coefficients toward the machine-type mean (instead of toward 0).
Two passes over the split's own queries (only ticks < cutoff): pass 1 = per-machine fit (lam1), type mean of the
coefficient vector; pass 2 = per-machine ridge with penalty pool_lam*||beta - mu_type||^2 (intercept unpenalised).
Payout-size split = B3's. Feature sets are learned from train histories (_core.learned).
PRE-DECLARED GRID: feature sets {base = strict lags + top3 pairs (L2 best structure), B3 (hard-coded B3 sets, reference),
 scr (screened lags/pairs), all8 (lags 1-8 + all 28 pairs)} x pool_lam {1,3,10,30,100,300} x lam1 {0.5,5}  = 48 cells
 extras (control, no pooling, plain ridge lam = same grid values) for each set: 24 cells.
Run: python3 methods/L6.py [--smoke]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _core as K
from _run import run_grid
from harness import load_split

LN = K.learned(load_split("train"))
SETS = {"base": (LN["lag_strict"], LN["pair_top3"]), "B3": (K.B3["lags"], K.B3["pairs"]),
        "scr": (LN["lag_scr"], LN["pair_scr"]), "all8": (LN["lag_all"], LN["pair_all"])}

def predict(split, p):
    return K.predict(split, K.C(**p))

cells, extras = [], []
for sn, (l, pr) in SETS.items():
    for pl in (1, 3, 10, 30, 100, 300):
        for l1 in (0.5, 5):
            cells.append((f"{sn}_pool{pl}_lam1_{l1}", dict(lags=l, pairs=pr, lam=float(l1), pool="type", pool_lam=float(pl))))
        extras.append((f"{sn}_nopool_lam{pl}", dict(lags=l, pairs=pr, lam=float(pl))))
if __name__ == "__main__":
    print({k: v for k, v in LN.items() if k.startswith(("lag_s", "pair_t", "pair_s"))})
    run_grid("L6", "per-machine ridge with coefficients partially pooled to the type mean (2-pass)", cells, predict, extras)
