"""G3 = tiny meta-learner on top of the 7-member global linear pool (same members as G1/G2).
base = global softmax pool of the members (6 params, refit inside each fold). The meta-learner predicts the residual
q - base (5 outputs) from [observable features (nobs, frac missing, cutoff, signal, type one-hot), member pw minus mean pw (7),
base (5)], output P = clip(base + resid) renormalised. Two learners, fixed seeds (random_state=0):
  gbt : GradientBoostingRegressor per class, depth 2, n_rounds in {10,30}, lr 0.05, subsample 0.7, sample_weight = R weights
  mlp : MLPRegressor(8 hidden, tanh, alpha in {1,10}) on standardised inputs, early_stopping off, max_iter 300
Episode-OOF on dev (6 folds), audit rows = fit on all dev. Train error is measured on in-fold training rows (R-weighted MSE)
and compared with the OOF error to show overfit."""
import json, sys, time, warnings
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from G_common import *
from G_gate import fit_gate, pred_gate
from G1 import MEMBERS
warnings.filterwarnings("ignore")


def build(v, Pm, base):
    f = obs_features(v)
    pw = 1 - Pm[:, :, 0]
    X = np.column_stack([f["nobs"], f["fm"], f["cut"], f["sig"], f["oh"], (pw - pw.mean(0, keepdims=True)).T, base])
    return X


def fit_meta(learner, hp, v, Pm, tr, w):
    B = fit_gate(Pm[:, tr], np.zeros((len(tr), 0)), v.truth[tr], w[tr], 0.0)
    base_all = pred_gate(B, Pm, np.zeros((v.n, 0)))
    X = build(v, Pm, base_all); R = v.truth - base_all
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xs = (X - mu) / sd
    sw = w[tr] / w[tr].mean()
    if learner == "gbt":
        ms = [GradientBoostingRegressor(n_estimators=hp["rounds"], max_depth=2, learning_rate=0.05, subsample=0.7,
                                        random_state=0).fit(Xs[tr], R[tr, c], sample_weight=sw) for c in range(5)]
        pr = lambda rows: np.column_stack([m.predict(Xs[rows]) for m in ms])
    else:
        # weighted fit via row resampling with fixed seed (MLP has no sample_weight): replicate rows proportional to weight
        rng = np.random.RandomState(0)
        idx = rng.choice(tr, size=len(tr) * 2, p=w[tr] / w[tr].sum())
        m = MLPRegressor((8,), activation="tanh", alpha=hp["alpha"], max_iter=300, random_state=0, tol=1e-6).fit(Xs[idx], R[idx])
        pr = lambda rows: m.predict(Xs[rows])
    return dict(B=B, base=base_all, pr=pr)


def apply(st, rows):
    P = np.clip(st["base"][rows] + st["pr"](rows), 1e-4, None)
    return P / P.sum(1, keepdims=True)


def wmse(P, v, rows, w):
    return float((w[rows] / w[rows].sum() * ((P[rows] - v.truth[rows]) ** 2).sum(1)).sum())


def main():
    v, rows, arows, w, folds = setup()
    Pm = np.stack([load(v, n) for n in MEMBERS])
    TMC = load(v, "TM_COMBINED"); TS = team_stack_oof(v, rows, w, folds)
    grid = []; best = None
    cells = [("gbt", {"rounds": 10}), ("gbt", {"rounds": 30}), ("mlp", {"alpha": 1.0}), ("mlp", {"alpha": 10.0})]
    for learner, hp in cells:
        P = np.tile(np.array([0.5, 0.2, 0.15, 0.1, 0.05]), (v.n, 1)); trerr = []; basee = []
        for tr, te in folds:
            st = fit_meta(learner, hp, v, Pm, tr, w); P[te] = apply(st, te)
            trerr.append(wmse(apply(st, tr), v, np.arange(v.n), w) if False else
                         float((w[tr] / w[tr].sum() * ((apply(st, tr) - v.truth[tr]) ** 2).sum(1)).sum()))
            basee.append(float((w[tr] / w[tr].sum() * ((st["base"][tr] - v.truth[tr]) ** 2).sum(1)).sum()))
        st = fit_meta(learner, hp, v, Pm, rows, w); P[arows] = apply(st, arows)
        oof = wmse(P, v, rows, w)
        r = report(P, v, TMC, TS, learner)
        extra = {"train_wmse": float(np.mean(trerr)), "base_train_wmse": float(np.mean(basee)), "oof_wmse": oof}
        print(learner, hp, fmt(r), f"| train wmse {np.mean(trerr):.5f} (base {np.mean(basee):.5f}) OOF wmse {oof:.5f}", flush=True)
        grid.append({"params": {"learner": learner, **hp}, "points": r["points"], "ci95": r["ci95"], "condition_ratios": r["ratios"],
                     "vs_TMC": r["vs_TMC"], "vs_TEAM": r["vs_TEAM"], **extra})
        if best is None or r["points"] > best[0]:
            best = (r["points"], learner, hp, P, r, extra)
    return v, grid, best


if __name__ == "__main__":
    t0 = time.perf_counter()
    v, grid, (pts, learner, hp, P, r, extra) = main()
    rt = time.perf_counter() - t0
    print("BEST", learner, hp, fmt(r))
    # determinism check
    v2, g2, b2 = None, None, None
    to_csv(HERE / "preds/G3_validation.csv", v, P)
    (HERE / "grids/G3.json").write_text(json.dumps(grid, indent=1))
    from harness import write_result
    write_result("G3", "claude", f"tiny meta-learner ({learner} {hp}) on 7-member pool residual", {"learner": learner, **hp,
                 "members": MEMBERS}, v, P, rt, extra={"vs_TM_COMBINED": r["vs_TMC"], "vs_TEAM_stack": r["vs_TEAM"], **extra})
