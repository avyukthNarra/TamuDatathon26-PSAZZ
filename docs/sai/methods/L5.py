"""L5: 3-state (loss / small win / big win) and log(1+y) level coding vs the win/loss flag, for single lags and pairs.
Coding of single-lag columns: flag (B3) | onehot (win + missing indicators) | flag_big (flag + centred big-win indicator,
big = payout >= big_thr) | flag_log (flag + centred log1p payout) | log (centred log1p payout only).
pair_coding: flag (flag x flag) | flag+big (adds big x big product per pair) | flag+log (adds log x log product per pair).
PRE-DECLARED GRID: coding {flag, onehot, flag_big thr2, flag_big thr5, flag_log, log} x pair_coding {flag, flag+big, flag+log}
 (big_thr = coding's thr, 5 otherwise) x lag set {strict (learned from train), B3 lags} x lam {0.1, 1, 3, 10} = 144 cells.
Pairs = learned top3 throughout. Payout split = B3's. Win flag target throughout (only the features change).
Run: python3 methods/L5.py [--smoke]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _core as K
from _run import run_grid
from harness import load_split

LN = K.learned(load_split("train"))
P3 = LN["pair_top3"]
LAGSETS = {"strict": LN["lag_strict"], "B3": K.B3["lags"]}
CODINGS = [("flag", "flag", 5), ("onehot", "onehot", 5), ("big2", "flag_big", 2), ("big5", "flag_big", 5), ("flog", "flag_log", 5), ("log", "log", 5)]

def predict(split, p):
    return K.predict(split, K.C(**p))

cells = []
for cn, cod, thr in CODINGS:
    for pc in ("flag", "flag+big", "flag+log"):
        for ln, ls in LAGSETS.items():
            for lam in (0.1, 1, 3, 10):
                cells.append((f"{cn}_{pc}_lags{ln}_lam{lam}", dict(lags=ls, pairs=P3, coding=cod, big_thr=thr, pair_coding=pc, lam=float(lam))))
if __name__ == "__main__":
    run_grid("L5", "3-state / log-level coding of lag features vs win-loss flag (ridge win model)", cells, predict)
