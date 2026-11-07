"""Q1: per-machine Gaussian random walk on log-odds (GBM on odds) grid filter; win flag observed at observed ticks, propagated over gaps.
Sigma grid {0,.03,.06,.12,.25,.5}/tick. Variants: 'machine' (argmax own prequential LL), 'pooled' (argmax summed TRAIN LL),
'bma' (own-LL-weighted average over sigma). Prior N(logit train win rate, 1.5). Size shares from BEST (TM_COMBINED3). No validation truth used,
so dev and audit rows are the same parameter-free method. Run: python3 methods/Q1.py [--smoke]"""
import sys, json, time
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
import _gbm as G
from harness import load_split, read_pred_csv, evaluate, diff_ci, to_csv, write_result, smoke_split
from harness.stack_w5 import parts


def compute(v, t, rows=None):
    def b():
        mu = G.global_mu(t)
        LLt, _ = G.multi_sigma(t, mu)
        LL, PW = G.multi_sigma(v, mu, rows)
        return mu, LLt, LL, PW
    return b() if rows is not None else G.cached(f"Q1_filters_{v.n}_{t.n}", b)


def variants(mu, LLt, LL, PW):
    kp = int(np.argmax(LLt.sum(0)))
    w = np.exp(LL - LL.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    return {"machine": PW[np.arange(len(PW)), LL.argmax(1)], "pooled": PW[:, kp], "bma": (w * PW).sum(1)}, kp


def assemble(pw, base):
    _, sh = parts(base)
    pw = np.clip(pw, 1e-3, 1 - 1e-3)
    return np.column_stack([1 - pw, pw[:, None] * sh])


if __name__ == "__main__":
    v = load_split("validation"); t = load_split("train")
    base = read_pred_csv(G.HERE / "preds" / "TM_COMBINED3_validation.csv", v)
    if "--smoke" in sys.argv:
        rows = smoke_split(v); rows = np.arange(v.n)[:0] if rows is None else rows
        rows = list(rows.episode_rows(0)) if False else list(range(0, 768, 8))
        mu, LLt, LL, PW = compute(v, t, rows)
        V, kp = variants(mu, LLt, LL, PW)
        for k, pw in V.items():
            print(k, "pw mean", pw[rows].mean(), "corr w BEST", np.corrcoef(pw[rows], 1 - base[rows, 0])[0, 1])
        sys.exit()
    t0 = time.time()
    mu, LLt, LL, PW = compute(v, t)
    V, kp = variants(mu, LLt, LL, PW)
    print("pooled sigma", G.SIG[kp], "train LL by sigma", LLt.sum(0).round(0), "runtime", time.time() - t0)
    grid = []; best = None
    for k, pw in V.items():
        P = assemble(pw, base)
        ev = evaluate(P, v, "dev", bootstrap=1000); d = diff_ci(P, base, v, "dev")
        print(k, round(ev["points"], 2), [round(x, 2) for x in ev["ci95"]], "vs BEST", round(d["diff"], 2), {a[:5]: round(b, 3) for a, b in ev["condition_ratios"].items()})
        grid.append({"params": {"variant": k, "pooled_sigma": float(G.SIG[kp])}, "points": ev["points"], "condition_ratios": ev["condition_ratios"], "diff_vs_BEST": d["diff"]})
        if best is None or ev["points"] > best[0]: best = (ev["points"], k, P)
    _, k, P = best
    to_csv(G.HERE / "preds" / "Q1_validation.csv", v, P)
    np.save(G.HERE / "cache" / "Q1_pw.npy", V[k])
    write_result("Q1", "sai-worker", "per-machine logit random-walk grid filter win prob; shares from TM_COMBINED3", {"variant": k, "pooled_sigma": float(G.SIG[kp]), "sigma_grid": G.SIG.tolist(), "prior_mu": mu}, v, P, time.time() - t0)
    json.dump(grid, open(G.HERE / "grids" / "Q1.json", "w"), indent=1)
    for kk, pw in V.items():
        np.save(G.HERE / "cache" / f"Q1_pw_{kk}.npy", pw)
