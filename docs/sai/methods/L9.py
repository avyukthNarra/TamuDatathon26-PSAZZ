"""L9: signal x lag and type x lag interactions. Per-machine models cannot hold a signal term (constant within a
machine), so the interaction lives in the coefficient prior: pass-1 per-machine betas are averaged within a cell
and used as ridge prior mean in pass 2 (as L6). Cell definitions: global, type, signal, type x signal (9 cells),
type_signal_lin (per type: beta ~ a + b*signal regression across machines). Payout split = B3's.
Also a train-screen diagnostic: lag/pair excess z^2 by signal level (printed, saved in results extra via report).
PRE-DECLARED GRID: cell kind {global,type,signal,type_signal,type_signal_lin} x pool_lam {1,3,10,30}
 x feature set {base (strict lag + top3 pairs), B3, scr}, lam1 = 1.0 (pass 1) = 60 cells.
 extras: no-pooling control with lam=pool_lam for each set (12 cells).
Run: python3 methods/L9.py [--smoke]
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _core as K
from _run import run_grid
from harness import load_split

t = load_split("train")
LN = K.learned(t)
SETS = {"base": (LN["lag_strict"], LN["pair_top3"]), "B3": (K.B3["lags"], K.B3["pairs"]), "scr": (LN["lag_scr"], LN["pair_scr"])}

def predict(split, p):
    return K.predict(split, K.C(**p))

cells, extras = [], []
for sn, (l, pr) in SETS.items():
    for pl in (1, 3, 10, 30):
        for kind in ("global", "type", "signal", "type_signal", "type_signal_lin"):
            cells.append((f"{sn}_{kind}_pool{pl}", dict(lags=l, pairs=pr, lam=1.0, pool=kind, pool_lam=float(pl))))
        extras.append((f"{sn}_nopool_lam{pl}", dict(lags=l, pairs=pr, lam=float(pl))))

def diagnostic():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ls", Path(K.HERE.parent) / "scripts" / "lag_screen.py")
    ls = importlib.util.module_from_spec(spec); spec.loader.exec_module(ls)
    out = {}
    for sg in (-1, 0, 1):
        idx = [i for i in range(t.n) if t.signal[i] == sg]
        W = [K._flags(t.grid[i], int(t.cutoff[i])) for i in idx]
        z7 = np.array([ls.z_single(w, 7) for w in W]); z7 = z7[~np.isnan(z7)]
        zp = np.array([ls.z_pair(w, 1, 3) for w in W]); zp = zp[~np.isnan(zp)]
        out[sg] = dict(n=len(idx), lag7_excess=float(np.mean(z7 ** 2) - 1), pair13_excess=float(np.mean(zp ** 2) - 1))
    return out

if __name__ == "__main__":
    print("signal diagnostic (train):", diagnostic())
    run_grid("L9", "pooled-prior interactions: coefficient prior mean by type x signal cells vs global/type/signal", cells, predict, extras)
