"""Shared volatility features + episode-OOF machinery for V1/V2/V3 (grid-only, ticks < cutoff, NaN never zero).
Per-machine features are computed on OBSERVED ticks only (windows count observed ticks, not calendar ticks)."""
import sys, hashlib
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from harness import load_split, read_pred_csv, evaluate, diff_ci, cached
from harness.scoring import scope_episodes
from harness.stack_w5 import row_weights, parts
FOLDS = 6
NAMES = []


def _ewvar(x, hl):
    a = 1 - 0.5 ** (1.0 / hl)
    m = x[0]; s2 = 0.0
    for z in x[1:]:
        d = z - m
        m += a * d
        s2 = (1 - a) * (s2 + a * d * d)
    return s2


def machine_feats(g, cutoff):
    obs = np.flatnonzero(~np.isnan(g))
    p = g[obs]
    w = (p > 0).astype(float)
    y = np.log1p(p); y = y - y.mean()
    n = len(obs)
    f = {}
    f["n_obs"] = n
    f["obs_frac"] = n / max(cutoff, 1)
    for k in (8, 16, 40):
        f[f"vw{k}"] = w[-k:].var()
        f[f"vy{k}"] = y[-k:].var()
    for hl in (8, 16, 32):
        f[f"ewv_w{hl}"] = _ewvar(w, hl)
        f[f"ewv_y{hl}"] = _ewvar(y, hl)
    # vol-of-vol: std of rolling var(16) of the win flag over the last 40 observed points
    rv = [w[i - 16:i].var() for i in range(max(16, n - 40), n + 1)] if n >= 17 else [0.0]
    f["vol_of_vol"] = float(np.std(rv))
    lw, sw = w.mean(), w[-8:].mean()
    f["gap_z"] = (sw - lw) / np.sqrt(max(lw * (1 - lw), 1e-3) / 8)
    f["abs_gap_z"] = abs(f["gap_z"])
    # GARCH-like: conditional variance of the win residual (EWMA mean residual, EWMA squared residual)
    a = 0.1; m = w[0]; h = lw * (1 - lw)
    for z in w[1:]:
        e = z - m
        h = 0.05 * lw * (1 - lw) + 0.1 * e * e + 0.85 * h
        m += a * e
    f["garch_h"] = h
    # time since last large shift: last observed point where |mean(prev 8) - mean(next 8)| z > 2.5 (tick distance to cutoff)
    last = None
    for i in range(8, n - 7):
        d = w[i:i + 8].mean() - w[i - 8:i].mean()
        if abs(d) > 0.5: last = i
    f["since_shift"] = float(cutoff - obs[last]) / max(cutoff, 1) if last is not None else 1.0
    f["long_rate"] = lw
    return f


def build(split):
    key = f"volfeat_{split.name}_{split.n}"
    def b():
        rows = [machine_feats(split.grid[i], int(split.cutoff[i])) for i in range(split.n)]
        names = list(rows[0].keys())
        M = np.array([[r[k] for k in names] for r in rows])
        # episode level
        E = np.zeros((split.n, 6 * 4))
        sel = [names.index(k) for k in ("ewv_w16", "vw16", "gap_z", "garch_h", "vol_of_vol", "vy16")]
        for i in range(split.n):
            pe = split.peers(i)
            allr = M[[i] + pe][:, sel]
            pr = M[pe][:, sel] if pe else allr
            E[i] = np.r_[pr.mean(0), pr.max(0), allr.std(0), np.abs(pr.mean(0) - M[i, sel])]
        en = [f"{p}_{s}" for p in ("peer_mean", "peer_max", "agree_sd", "peer_gap") for s in
              ("ewv_w16", "vw16", "gap_z", "garch_h", "vol_of_vol", "vy16")]
        return names + en, np.column_stack([M, E])
    return cached(key, b)


def folds(v, dev):
    order = sorted(dev, key=lambda e: hashlib.sha256(f"w5fold|{e}".encode()).hexdigest())
    return {e: i % FOLDS for i, e in enumerate(order)}


def setup():
    v = load_split("validation")
    names, F = build(v)
    dev = scope_episodes(v, "dev")
    aud = [e for e in v.episodes if e not in set(dev)]
    rows = np.concatenate([v.episode_rows(e) for e in dev])
    arows = np.concatenate([v.episode_rows(e) for e in aud])
    w = row_weights(v, set(rows.tolist()))
    fold = folds(v, dev)
    return v, names, F, dev, rows, arows, w, fold


def oof_run(v, dev, fold, rows, arows, fitfn, predfn):
    """fitfn(train_rows)->model ; predfn(model, rows)->(len,5). Returns full (n,5) with dev=OOF, audit=fit on all dev."""
    out = np.tile(np.array([0.5, 0.2, 0.15, 0.1, 0.05]), (v.n, 1))
    for f in range(FOLDS):
        tr = np.concatenate([v.episode_rows(e) for e in dev if fold[e] != f])
        te = np.concatenate([v.episode_rows(e) for e in dev if fold[e] == f])
        out[te] = predfn(fitfn(tr), te)
    m = fitfn(rows)
    out[arows] = predfn(m, arows)
    return out, m


def report(P, v, base, tag=""):
    ev = evaluate(P, v, "dev", bootstrap=1000)
    d = diff_ci(P, base, v, "dev")
    r = {k[:5]: round(x, 3) for k, x in ev["condition_ratios"].items()}
    print(f"{tag} dev {ev['points']:.2f} {[round(x,2) for x in ev['ci95']]} vs TMC {d['diff']:+.2f} {[round(x,2) for x in d['ci95']]} {r}")
    return ev, d
