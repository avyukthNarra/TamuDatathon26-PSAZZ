"""Q2: grid (histogram) log-odds filter for each machine's win probability, sigma linked to volatility.

Model per machine: hidden log-odds x follows a Gaussian random walk, step sd sigma*sqrt(gap); P(win at t) = sigmoid(x_t).
Grid: 121 points on [-6, 6]. Prior N(logit(type win-rate), 2^2) on the state before the first observed tick.
Predict: S' = S @ K(sigma*sqrt(gap)), K a row-stochastic Gaussian kernel. Update: multiply by sigmoid(x) (win) or 1-sigmoid(x).
Only observed ticks (< cutoff) are processed. gap = ticks since the last update of that state, so missing ticks only add diffusion.
Cutoff prediction = sum(S propagated over (cutoff - last) * sigmoid(x)).

Q2a: sigma_i = exp(c0 + c1*z_i), z_i = standardised EWMA(hl 16) variance of the win flag over observed ticks.
  Fit on TRAIN only: each train row uses pseudo-cutoff s = cutoff//2; z uses ticks < s (mu, sd of z from train rows);
  log-likelihood sums predictions at ticks >= s (each predicted from earlier ticks). Grid c0 x c1 chosen by max.
Q2b: one state per cell of a lag pair (a, b): cell = (win[t-a] == win[t-b]); a tick is assigned only if both lags and
  the tick itself are observed. Each cell state diffuses with its own gap. The cutoff uses the cell of (cutoff-a, cutoff-b);
  if either lag is unobserved it falls back to the Q2a prediction (fallback share reported).
Member: pw = (1-w)*pw_BEST + w*pw_Q, shares from BEST. w fitted episode-OOF on dev (6 folds, R-weighted squared error,
  grid 0..1 step 0.05). Audit rows use w fitted on all dev rows and are never scored. Candidate choice among Q2a/Q2b pairs
  uses dev member OOF points (reported).
Grids (planned before running): c0 in {-4,-3,-2.5,-2,-1.5}, c1 in {-0.5,0,0.5}; pairs (1,3), (2,5), (4,7).
  OMP_NUM_THREADS=1 ... nice -n 10 timeout 600 python3 methods/Q2.py smoke   (10 dev episodes, crash check)
  OMP_NUM_THREADS=1 ... nice -n 10 timeout 600 python3 methods/Q2.py          (full)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "methods"))
from harness import load_split, read_pred_csv, evaluate, diff_ci, to_csv, write_result
from harness.scoring import scope_episodes
from harness.stack_w5 import row_weights
from volfeat import _ewvar, folds

G = 121
XS = np.linspace(-6, 6, G)
D2 = (XS[:, None] - XS[None, :]) ** 2
SIG = 1 / (1 + np.exp(-XS))
C0S = [-4, -3, -2.5, -2, -1.5]
C1S = [-0.5, 0.0, 0.5]
PAIRS = [(1, 3), (2, 5), (4, 7)]
WGRID = np.round(np.arange(0, 1.0001, 0.05), 2)


def kernels(sd):
    s2 = np.maximum(sd, 1e-6) ** 2
    K = np.exp(-D2[None] / (2 * s2[:, None, None]))
    K /= K.sum(2, keepdims=True)
    return K


def run(grid, sigma, mu, C, ncell, cut, cut_cell):
    """Filter all machines at once. C (M,T): cell id per tick (-1 = not assigned). cut_cell (M,): cell for the cutoff (-1 none).
    Returns cutoff predictions (M,) (NaN where cut_cell < 0) and per-tick predictions (M,T) (NaN where not assigned)."""
    M, T = grid.shape
    win = (np.nan_to_num(grid) > 0).astype(float)
    prior = np.exp(-(XS[None, :] - mu[:, None]) ** 2 / 8.0)
    prior /= prior.sum(1, keepdims=True)
    S = np.repeat(prior[:, None, :], ncell, 1)
    last = -np.ones((M, ncell))
    pred = np.full((M, T), np.nan)
    for t in range(T):
        idx = np.flatnonzero(C[:, t] >= 0)
        if idx.size == 0:
            continue
        c = C[idx, t]
        gap = t - last[idx, c]
        K = kernels(sigma[idx] * np.sqrt(gap))
        Sp = np.einsum("mk,mkj->mj", S[idx, c], K)
        pred[idx, t] = Sp @ SIG
        y = win[idx, t][:, None]
        Sn = Sp * (y * SIG + (1 - y) * (1 - SIG))
        S[idx, c] = Sn / Sn.sum(1, keepdims=True)
        last[idx, c] = t
    pcut = np.full(M, np.nan)
    idx = np.flatnonzero(cut_cell >= 0)
    c = cut_cell[idx]
    K = kernels(sigma[idx] * np.sqrt(cut[idx] - last[idx, c]))
    pcut[idx] = np.einsum("mk,mkj->mj", S[idx, c], K) @ SIG
    return pcut, pred


def single_cells(grid):
    return np.where(~np.isnan(grid), 0, -1)


def pair_cells(grid, a, b):
    obs = ~np.isnan(grid)
    win = (np.nan_to_num(grid) > 0)
    M, T = grid.shape
    C = np.full((M, T), -1)
    for t in range(max(a, b), T):
        ok = obs[:, t] & obs[:, t - a] & obs[:, t - b]
        C[:, t] = np.where(ok, (win[:, t - a] == win[:, t - b]).astype(int), -1)
    return C


def pair_cut_cells(grid, cut, a, b):
    r = np.arange(grid.shape[0])
    ga, gb = grid[r, cut - a], grid[r, cut - b]
    ok = ~np.isnan(ga) & ~np.isnan(gb)
    return np.where(ok, ((ga > 0) == (gb > 0)).astype(int), -1)


def ewv_at(grid, upto):
    out = np.zeros(grid.shape[0])
    for i in range(grid.shape[0]):
        g = grid[i, :upto[i]]
        w = (g[~np.isnan(g)] > 0).astype(float)
        out[i] = _ewvar(w, 16) if len(w) > 1 else 0.25
    return out


def type_logit_prior(train, mtypes):
    obs = ~np.isnan(train.grid)
    win = (np.nan_to_num(train.grid) > 0)
    overall = win[obs].mean()
    rate = {}
    for m in np.unique(train.mtype):
        sel = train.mtype == m
        rate[m] = win[sel][obs[sel]].mean()
    r = np.array([rate.get(m, overall) for m in mtypes])
    r = np.clip(r, 0.02, 0.98)
    return np.log(r / (1 - r))


def prequential_ll(train, sig, mu):
    C = single_cells(train.grid)
    _, pred = run(train.grid, sig, mu, C, 1, train.cutoff, np.full(train.n, -1))
    s = train.cutoff // 2
    t_idx = np.arange(train.grid.shape[1])[None, :]
    mask = (~np.isnan(train.grid)) & (t_idx >= s[:, None]) & ~np.isnan(pred)
    p = np.clip(pred, 1e-9, 1 - 1e-9)
    y = (np.nan_to_num(train.grid) > 0).astype(float)
    return float(np.where(mask, y * np.log(p) + (1 - y) * np.log(1 - p), 0).sum())


def main(mode):
    t0 = time.time()
    v_full = load_split("validation")
    t_full = load_split("train")
    best_full = read_pred_csv(HERE / "preds" / "TM_COMBINED3_validation.csv", v_full)
    if mode == "smoke":
        dev_full = scope_episodes(v_full, "dev")[:10]
        rows = np.concatenate([v_full.episode_rows(e) for e in dev_full])
        v = v_full.subset(rows)
        BEST = best_full[rows]
        tr_eps = t_full.episodes[:10]
        t = t_full.subset(np.concatenate([t_full.episode_rows(e) for e in tr_eps]))
    else:
        v, t, BEST = v_full, t_full, best_full

    # ---- train-only fit of c0, c1 (Q2a) ----
    z_raw_tr = ewv_at(t.grid, t.cutoff // 2)
    mu_z, sd_z = z_raw_tr.mean(), z_raw_tr.std()
    z_tr = (z_raw_tr - mu_z) / sd_z
    mu_tr = type_logit_prior(t, t.mtype)
    train_grid = []
    best_ll, best_c = -np.inf, None
    for c0 in C0S:
        for c1 in C1S:
            ll = prequential_ll(t, np.exp(c0 + c1 * z_tr), mu_tr)
            train_grid.append({"params": {"c0": c0, "c1": c1}, "train_prequential_ll": ll})
            if ll > best_ll:
                best_ll, best_c = ll, (c0, c1)
    c0, c1 = best_c
    print(f"[train] best c0={c0} c1={c1} ll={best_ll:.1f} ({time.time()-t0:.0f}s)", flush=True)

    # ---- validation candidates (z from full observed history < cutoff) ----
    z_v = (ewv_at(v.grid, v.cutoff) - mu_z) / sd_z
    sig_v = np.exp(c0 + c1 * z_v)
    mu_v = type_logit_prior(t, v.mtype)
    pq_a, _ = run(v.grid, sig_v, mu_v, single_cells(v.grid), 1, v.cutoff, np.zeros(v.n, int))
    cands = {"Q2a": (pq_a, {"c0": c0, "c1": c1, "pair": None, "fallback_share": 0.0})}
    for a, b in PAIRS:
        C = pair_cells(v.grid, a, b)
        cc = pair_cut_cells(v.grid, v.cutoff, a, b)
        pc, _ = run(v.grid, sig_v, mu_v, C, 2, v.cutoff, cc)
        fb = np.isnan(pc)
        pc = np.where(fb, pq_a, pc)
        cands[f"Q2b_{a}_{b}"] = (pc, {"c0": c0, "c1": c1, "pair": [a, b], "fallback_share": float(fb.mean())})
    print(f"[val] candidates done ({time.time()-t0:.0f}s)", flush=True)

    # ---- composition with BEST, episode-OOF w on dev ----
    shB = BEST[:, 1:] / BEST[:, 1:].sum(1, keepdims=True)
    pwB = 1 - BEST[:, 0]
    dev = scope_episodes(v, "dev")
    aud = [e for e in v.episodes if e not in set(dev)]
    rows_dev = np.concatenate([v.episode_rows(e) for e in dev])
    rows_aud = np.concatenate([v.episode_rows(e) for e in aud]) if aud else np.zeros(0, int)
    wts = row_weights(v, set(rows_dev.tolist()))
    fold = folds(v, dev)
    q = v.truth

    def fit_w(pwQ, fit_rows):
        losses = []
        for w in WGRID:
            pwc = (1 - w) * pwB[fit_rows] + w * pwQ[fit_rows]
            P = np.column_stack([1 - pwc, pwc[:, None] * shB[fit_rows]])
            losses.append(float((wts[fit_rows] * ((P - q[fit_rows]) ** 2).sum(1)).sum()))
        return float(WGRID[int(np.argmin(losses))])

    def member(pwQ):
        pwc = np.zeros(v.n)
        wf = []
        for f in range(6):
            tr_r = np.concatenate([v.episode_rows(e) for e in dev if fold[e] != f])
            te_r = np.concatenate([v.episode_rows(e) for e in dev if fold[e] == f])
            w = fit_w(pwQ, tr_r)
            wf.append(w)
            pwc[te_r] = (1 - w) * pwB[te_r] + w * pwQ[te_r]
        w_all = fit_w(pwQ, rows_dev)
        if len(rows_aud):
            pwc[rows_aud] = (1 - w_all) * pwB[rows_aud] + w_all * pwQ[rows_aud]
        P = np.column_stack([1 - pwc, pwc[:, None] * shB])
        return P, wf, w_all

    def standalone(pq):
        return np.column_stack([1 - pq, pq[:, None] * shB])

    results = {}
    for name, (pq, info) in cands.items():
        P_m, wf, w_all = member(np.nan_to_num(pq, nan=0.5))
        ev_m = evaluate(P_m, v, "dev", bootstrap=500)
        ev_s = evaluate(standalone(np.nan_to_num(pq, nan=0.5)), v, "dev", bootstrap=500)
        d = diff_ci(P_m, BEST, v, "dev", bootstrap=500)
        results[name] = {"info": info, "P_member": P_m, "w_folds": wf, "w_all": w_all, "member": ev_m,
                         "standalone": ev_s, "member_vs_best": d}
        print(f"[{name}] standalone {ev_s['points']:.2f} | member OOF {ev_m['points']:.2f} "
              f"{[round(x, 2) for x in ev_m['ci95']]} vs BEST {d['diff']:+.2f} {[round(x, 2) for x in d['ci95']]} "
              f"w_all={w_all} ({time.time()-t0:.0f}s)", flush=True)

    best_name = max(results, key=lambda k: results[k]["member"]["points"])
    print("chosen member composition:", best_name, flush=True)
    if mode == "smoke":
        ev_b = evaluate(BEST, v, "dev", bootstrap=0)
        print("smoke BEST points", round(ev_b["points"], 2) if ev_b["points"] == ev_b["points"] else "nan")
        print("smoke finite:", {k: bool(np.isfinite(r["P_member"]).all()) for k, r in results.items()})
        return

    # ---- determinism of the chosen candidate ----
    if best_name == "Q2a":
        pq_chk, _ = run(v.grid, sig_v, mu_v, single_cells(v.grid), 1, v.cutoff, np.zeros(v.n, int))
    else:
        a, b = cands[best_name][1]["pair"]
        pq_chk, _ = run(v.grid, sig_v, mu_v, pair_cells(v.grid, a, b), 2, v.cutoff, pair_cut_cells(v.grid, v.cutoff, a, b))
    pq_chk = np.where(np.isnan(pq_chk), pq_a, pq_chk)
    det = float(np.nanmax(np.abs(pq_chk - cands[best_name][0])))

    best = results[best_name]
    P_best = best["P_member"]
    to_csv(HERE / "preds" / "Q2_validation.csv", v, P_best)
    grids = {"train_fit": train_grid, "candidates": [
        {"name": k, "params": r["info"], "standalone_dev": {"points": r["standalone"]["points"],
                                                             "ci95": r["standalone"]["ci95"],
                                                             "condition_ratios": r["standalone"]["condition_ratios"]},
         "member_oof_dev": {"points": r["member"]["points"], "ci95": r["member"]["ci95"],
                            "condition_ratios": r["member"]["condition_ratios"]},
         "member_vs_best": r["member_vs_best"], "w_folds": r["w_folds"], "w_all": r["w_all"]}
        for k, r in results.items()]}
    (HERE / "grids" / "Q2.json").write_text(json.dumps(grids, indent=1))
    runtime = time.time() - t0
    params = {"model": "grid log-odds filter, 121 pts on [-6,6], prior N(type logit,2^2), sigma=exp(c0+c1*z)",
              "c0": best["info"]["c0"], "c1": best["info"]["c1"], "pair": best["info"]["pair"],
              "z": "standardised EWMA(hl16) var of win flag", "w_step": 0.05, "member_with": "TM_COMBINED3"}
    write_result("Q2", "claude", f"Q2 volatility-linked grid filter ({best_name}) as member of TM_COMBINED3",
                 params, v, P_best, runtime, determinism=det, bootstrap=500)
    print("RESULT", best_name, round(best["member"]["points"], 3), "standalone", round(best["standalone"]["points"], 3),
          "vs BEST", round(best["member_vs_best"]["diff"], 3), [round(x, 3) for x in best["member_vs_best"]["ci95"]],
          "det", det, f"runtime {runtime:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
