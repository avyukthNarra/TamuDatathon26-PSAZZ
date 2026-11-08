"""Shared plumbing for G1-G3 (blending across all methods). Episode-OOF on dev (6 folds, same hash as stack_w5),
audit rows = same method fit on all dev rows. Never touches audit truth."""
import hashlib, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["TEAM"] = "1"
from harness import load_split, read_pred_csv, evaluate, diff_ci, to_csv
from harness.scoring import scope_episodes
from harness.stack_w5 import row_weights, fit as w5fit, combine as w5combine, parts, load, PW, SH

HERE = Path(__file__).resolve().parents[1]
FOLDS = 6
ALL = ["TM_COMBINED", "TM_MIXER", "TM_RULE", "TM_E6", "TM_E7", "TM_E67", "C2", "C1", "A9", "M9b_B4", "M2_B4", "M3_B4",
       "B4", "L5", "L7", "A1_B4"]


def setup():
    v = load_split("validation")
    dev = scope_episodes(v, "dev")
    aud = scope_episodes(v, "audit")
    rows = np.concatenate([v.episode_rows(e) for e in dev])
    arows = np.concatenate([v.episode_rows(e) for e in aud])
    w = row_weights(v, set(rows.tolist()))
    order = sorted(dev, key=lambda e: hashlib.sha256(f"w5fold|{e}".encode()).hexdigest())
    fold = {e: i % FOLDS for i, e in enumerate(order)}
    folds = []
    for f in range(FOLDS):
        folds.append((np.concatenate([v.episode_rows(e) for e in dev if fold[e] != f]),
                      np.concatenate([v.episode_rows(e) for e in dev if fold[e] == f])))
    return v, rows, arows, w, folds


def obs_features(v):
    """Observable-only features: n observed ticks, frac missing, cutoff, machine type onehot, signal."""
    nobs = (~np.isnan(v.grid)).sum(1).astype(float)
    cut = v.cutoff.astype(float)
    fm = 1 - nobs / np.maximum(cut, 1)
    types = sorted(set(v.mtype.tolist()))
    oh = np.stack([(v.mtype == t).astype(float) for t in types], 1)
    try:
        sig = v.signal.astype(float)
    except Exception:
        s = sorted(set(v.signal.tolist())); sig = np.array([s.index(x) for x in v.signal], float)
    return dict(nobs=nobs, cut=cut, fm=fm, oh=oh, types=types, sig=sig)


def run_cv(v, rows, arows, folds, fitfn, predfn):
    """fitfn(train_rows)->state ; predfn(state, rows)->(len(rows),5). Returns full (n,5) pred and train-error."""
    P = np.tile(np.array([0.5, 0.2, 0.15, 0.1, 0.05]), (v.n, 1))
    for tr, te in folds:
        P[te] = predfn(fitfn(tr), te)
    st = fitfn(rows)
    P[arows] = predfn(st, arows)
    return P, st


def team_stack_oof(v, rows, w, folds):
    PWm = [parts(load(v, n))[0] for n in PW]; SHm = [parts(load(v, n))[1] for n in SH]
    oof = np.tile(np.array([0.5, 0.2, 0.15, 0.1, 0.05]), (v.n, 1))
    for tr, te in folds:
        a, b = w5fit(PWm, SHm, v, tr, w)
        oof[te] = w5combine([m[te] for m in PWm], [m[te] for m in SHm], a, b)
    return oof


def report(P, v, TMC, TS, name):
    ev = evaluate(P, v, "dev")
    d1 = diff_ci(P, TMC, v, "dev"); d2 = diff_ci(P, TS, v, "dev")
    return dict(points=ev["points"], ci95=ev["ci95"], ratios=ev["condition_ratios"], vs_TMC=d1, vs_TEAM=d2)


def fmt(r):
    return (f"{r['points']:.2f} [{r['ci95'][0]:.1f},{r['ci95'][1]:.1f}] vsTMC {r['vs_TMC']['diff']:+.2f} "
            f"[{r['vs_TMC']['ci95'][0]:+.2f},{r['vs_TMC']['ci95'][1]:+.2f}] vsTEAM {r['vs_TEAM']['diff']:+.2f} "
            f"[{r['vs_TEAM']['ci95'][0]:+.2f},{r['vs_TEAM']['ci95'][1]:+.2f}] "
            + " ".join(f"{k[:5]}={x:.3f}" for k, x in r["ratios"].items()))
