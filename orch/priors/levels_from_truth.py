"""Level prior read from validation TRUE win probabilities (exact draws of each machine's current level).

At the cutoff every query's true P(win) is one machine's dead or live level (drift has finished by the
cutoff; a shared flip only swaps which cell is live). Classify each by the logit midpoint per type,
then fit logit level = a + b*signal + noise per (type, cell), per fold and on all.
"""
import json, sys, numpy as np
from common import *
v = load("validation"); T = v["targets"]; Qd = {q["query_id"]: q for q in v["Q"]}
A, B = folds(v)
lg = lambda p: np.log(p / (1 - p))
def fit(eps):
    out = {}
    for ty in ["steady", "balanced", "jackpot"]:
        qs = [q for q in T if Qd[q]["machine_type"] == ty and Qd[q]["episode_id"] in eps]
        z = np.array([lg(np.clip(1 - T[q][0], 1e-6, 1 - 1e-6)) for q in qs]); s = np.array([int(Qd[q]["signal"]) for q in qs])
        # 2-means in logit space to split dead/live
        c = np.array([np.quantile(z, .2), np.quantile(z, .8)])
        for _ in range(50):
            lab = np.abs(z[:, None] - c[None]).argmin(1); c = np.array([z[lab == k].mean() for k in range(2)])
        mid = c.mean(); res = {"mid_logit": round(float(mid), 3)}
        for k, name in enumerate(["dead", "live"]):
            m = lab == k; X = np.c_[np.ones(m.sum()), s[m]]
            coef, *_ = np.linalg.lstsq(X, z[m], rcond=None); r = z[m] - X @ coef
            res[name] = {"n": int(m.sum()), "a": round(float(coef[0]), 3), "b": round(float(coef[1]), 3), "sd": round(float(r.std(ddof=2)), 3),
                         "p_at_signal": {str(sg): round(float(1 / (1 + np.exp(-(coef[0] + coef[1] * sg)))), 4) for sg in (-1, 0, 1)}}
        res["live_share"] = round(float(lab.mean()), 3)
        out[ty] = res
    return out
res = {"all": fit(A | B), "foldA": fit(A), "foldB": fit(B)}
json.dump(res, open("priors/levels_from_truth.json", "w"), indent=1)
for ty, r in res["all"].items():
    print(ty, "live share", r["live_share"])
    for k in ("dead", "live"):
        x = r[k]; print(f"  {k}: n={x['n']} logit = {x['a']} + {x['b']}*signal, sd {x['sd']}  p(-1,0,1)={list(x['p_at_signal'].values())}  | foldA a={res['foldA'][ty][k]['a']} sd={res['foldA'][ty][k]['sd']} foldB a={res['foldB'][ty][k]['a']} sd={res['foldB'][ty][k]['sd']}")
