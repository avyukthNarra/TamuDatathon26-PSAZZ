"""Jackpot-only win probability, patched onto v13's cross-fitted forecasts.

The size split is left alone: positive masses stay proportional to v13's
p[1:] / (1 - p[0]). Only queries with machine_type == jackpot get a new
P(win).

Model (structure from the one-pair parity kernel; every coefficient is fit
on train histories, not typed in):

  one lag pair, one direction, two logit-normal levels given type and signal
  stable / short_history: levels fixed for the whole episode
  drift: linear swap of the two levels, anchored at the cutoff
  shared_change: one episode-wide direction flip; the tick is located from
  the other machines in the episode, then the jackpot's own spins update it

A win is cheap under the live level and very expensive under the dead level,
so the posterior over pair and direction concentrates on the cell that holds
the wins. The reported win probability is the posterior predictive at the
cutoff (levels integrated by Gauss-Hermite on the logit-normal prior).

Variants compared:
  post      posterior mean, pair prior = train frequencies with add-0.5
  map       same prior, single best (pair, direction), levels still integrated
  flatpair  posterior mean with a uniform prior on all 28 pairs
The headline cross-fit picks among these three on fold A and scores fold B,
then swaps. Train-only is `post` with no validation choice.

Usage (from the worktree root):
  python jackpot/run_jackpot.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "mainFiles" / "public"
V13_PREDS = Path(
    "/Users/saisreed/Desktop/Coding/TamuDatathon26-PSAZZ/worktrees/_tm/v13"
    "/mainFiles/runs/v13/crossfit_validation_predictions.csv"
)
SCORER = Path(
    "/Users/saisreed/Desktop/Coding/TamuDatathon26-PSAZZ/worktrees/fresh-orch/orch/common.py"
)
OUT_DIR = ROOT / "jackpot"
PAIRS = tuple((a, b) for a in range(1, 8) for b in range(a + 1, 9))
PAIR_INDEX = {pair: i for i, pair in enumerate(PAIRS)}
CONDITIONS = ("stable", "short_history", "drift", "shared_change")
PCOLS = ["p_0", "p_1", "p_2", "p_5", "p_20"]
# map is the parity posterior. atomK uses the four lag-patterns of the MAP pair
# and shrinks that pattern's rate toward its parity pool with strength K.
# atomK: four lag-patterns of the MAP pair, shrunk toward that pattern's parity
# pool with strength K. The cross-fit picks K. Drift is not special-cased;
# a "leave drift on v13" option was tried and a fold rejected it.
MENU = ("map", "atom4", "atom8", "atom16")
SEED = 0


def _load_scorer():
    spec = importlib.util.spec_from_file_location("orch_common", SCORER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def logsumexp(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -np.inf
    top = finite.max()
    return float(top + np.log(np.exp(values - top).sum()))


def quadrature(n=7):
    """Nodes and weights for a pair of independent standard normals."""
    x, w = hermgauss(n)
    z = np.sqrt(2.0) * x
    wt = w / np.sqrt(np.pi)
    z1, z2 = np.meshgrid(z, z, indexing="ij")
    weight = np.outer(wt, wt).ravel()
    weight = weight / weight.sum()
    return z1.ravel(), z2.ravel(), weight


def load_split(split):
    folder = PUB / split
    queries = list(csv.DictReader(open(folder / "queries.csv")))
    for query in queries:
        query["signal"] = int(query["signal"])
        query["cutoff"] = int(query["cutoff"])
    history = {}
    with open(folder / "history.csv") as handle:
        for row in csv.DictReader(handle):
            key = (row["episode_id"], row["machine_id"])
            history.setdefault(key, {})[int(row["tick"])] = int(row["payout"])
    out = {"Q": queries, "H": history}
    cond_path = folder / "conditions.csv"
    if cond_path.exists():
        out["cond"] = {
            row["query_id"]: row["condition"]
            for row in csv.DictReader(open(cond_path))
        }
    return out


def series_from_history(payouts, cutoff):
    obs = np.zeros(cutoff, dtype=bool)
    win = np.zeros(cutoff, dtype=bool)
    for tick, payout in payouts.items():
        if 0 <= tick < cutoff:
            obs[tick] = True
            win[tick] = payout > 0
    return obs, win


def cell_counts(obs, win, cutoff, a, b, t0, t1):
    """Wins and visits in the agree cell and the differ cell on ticks [t0, t1)."""
    if t1 <= b:
        return (0, 0, 0, 0)
    idx = np.arange(max(b, t0), min(cutoff, t1))
    if idx.size == 0:
        return (0, 0, 0, 0)
    valid = obs[idx] & obs[idx - a] & obs[idx - b]
    if not valid.any():
        return (0, 0, 0, 0)
    agree = win[idx - a] == win[idx - b]
    y = win[idx]
    use_agree = valid & agree
    use_differ = valid & ~agree
    return (
        int(y[use_agree].sum()),
        int(use_agree.sum()),
        int(y[use_differ].sum()),
        int(use_differ.sum()),
    )


def binomial_gain(k0, n0, k1, n1):
    def ll(k, n):
        if n <= 0:
            return 0.0
        p = min(1.0 - 1e-6, max(1e-6, k / n))
        return k * np.log(p) + (n - k) * np.log(1.0 - p)

    return ll(k0, n0) + ll(k1, n1) - ll(k0 + k1, n0 + n1)


def best_pair(obs, win, cutoff, t0, t1, min_n=10):
    """Pair with the largest two-cell binomial gain. None if both cells are thin."""
    best, best_gain, best_counts = None, 0.0, None
    for pair in PAIRS:
        counts = cell_counts(obs, win, cutoff, pair[0], pair[1], t0, t1)
        k0, n0, k1, n1 = counts
        if n0 < min_n or n1 < min_n:
            continue
        gain = binomial_gain(k0, n0, k1, n1)
        if gain > best_gain:
            best, best_gain, best_counts = pair, gain, counts
    return best, best_gain, best_counts


def fit_pair_prior(train):
    """How often each of the 28 pairs wins the two-cell score on stable machines."""
    cond = train["cond"]
    counts = np.zeros(len(PAIRS))
    n_scored = 0
    agree_live = 0
    for query in train["Q"]:
        if cond[query["query_id"]] != "stable":
            continue
        obs, win = series_from_history(
            train["H"][(query["episode_id"], query["machine_id"])], query["cutoff"]
        )
        pair, gain, counts_ = best_pair(obs, win, query["cutoff"], 0, query["cutoff"])
        if pair is None:
            continue
        counts[PAIR_INDEX[pair]] += 1
        n_scored += 1
        k_agree, n_agree, k_differ, n_differ = counts_
        if n_agree and n_differ and (k_agree / n_agree) >= (k_differ / n_differ):
            agree_live += 1
    smoothed = (counts + 0.5) / (counts.sum() + 0.5 * len(PAIRS))
    return {
        "counts": counts,
        "n_scored": n_scored,
        "smoothed": smoothed,
        "flat": np.full(len(PAIRS), 1.0 / len(PAIRS)),
        "p_agree_live": (agree_live / n_scored) if n_scored else 0.5,
    }


def fit_jackpot_levels(train):
    """Logit-normal levels on jackpot machines, using a window that is still unswapped.

    Stable uses the whole history. Drift uses ticks more than 110 before the
    cutoff (before a ramp that ends ~30 ticks out). Shared change uses the
    first 45% (before the typical flip). The live cell is the one with the
    higher rate; that orientation is per machine and is not a coefficient.
    """
    cond = train["cond"]
    rows = []
    for query in train["Q"]:
        if query["machine_type"] != "jackpot":
            continue
        kind = cond[query["query_id"]]
        cutoff = query["cutoff"]
        if kind == "stable":
            t0, t1 = 0, cutoff
        elif kind == "drift":
            t0, t1 = 0, cutoff - 110
        elif kind == "shared_change":
            t0, t1 = 0, int(0.45 * cutoff)
        else:
            # Short histories hide ticks; the kernel matches stable, but the
            # sample is too thin for a level prior. Skip.
            continue
        obs, win = series_from_history(
            train["H"][(query["episode_id"], query["machine_id"])], cutoff
        )
        pair, gain, counts = best_pair(obs, win, cutoff, t0, t1, min_n=8)
        if pair is None:
            continue
        k0, n0, k1, n1 = counts
        r0, r1 = k0 / n0, k1 / n1
        if r0 >= r1:
            live, dead, n_live, n_dead = r0, r1, n0, n1
        else:
            live, dead, n_live, n_dead = r1, r0, n1, n0
        rows.append((query["signal"], live, dead, n_live, n_dead))

    if len(rows) < 8:
        raise RuntimeError(f"only {len(rows)} jackpot machines for the level prior")

    # Two rows per machine: dead (live indicator 0) and live (1).
    signal, y, weight, live_flag, resid_pair = [], [], [], [], []
    for sig, live, dead, n_live, n_dead in rows:
        for rate, n, flag in ((dead, n_dead, 0.0), (live, n_live, 1.0)):
            p = min(1.0 - 0.5 / n, max(0.5 / n, rate))
            signal.append(sig)
            y.append(np.log(p / (1.0 - p)))
            weight.append(n * p * (1.0 - p))
            live_flag.append(flag)
        resid_pair.append((sig, live, dead, n_live, n_dead))

    signal = np.asarray(signal, float)
    y = np.asarray(y, float)
    weight = np.asarray(weight, float)
    live_flag = np.asarray(live_flag, float)
    design = np.column_stack([np.ones(len(y)), signal, live_flag])
    w_design = design * np.sqrt(weight)[:, None]
    beta, *_ = np.linalg.lstsq(w_design, y * np.sqrt(weight), rcond=None)
    fitted = design @ beta
    # Machine-level residuals after removing a + b*signal + c*live, so the
    # correlation is the shared shock, not the binomial noise of one cell.
    machine_resid = []
    sampling_var = []
    for sig, live, dead, n_live, n_dead in resid_pair:
        def _logit_resid(rate, n, flag, sig=sig):
            p = min(1.0 - 0.5 / n, max(0.5 / n, rate))
            logit = np.log(p / (1.0 - p))
            center = beta[0] + beta[1] * sig + beta[2] * flag
            var = 1.0 / max(n * p * (1.0 - p), 1e-6)
            return logit - center, var

        e_dead, v_dead = _logit_resid(dead, n_dead, 0.0)
        e_live, v_live = _logit_resid(live, n_live, 1.0)
        machine_resid.append((e_dead, e_live))
        sampling_var.append((v_dead, v_live))
    machine_resid = np.asarray(machine_resid)
    sampling_var = np.asarray(sampling_var)
    # Subtract average binomial logit-variance. Floor so a small sample cannot
    # collapse the prior to a point mass.
    var_dead = max(0.05, float(machine_resid[:, 0].var()) - float(sampling_var[:, 0].mean()))
    var_live = max(0.05, float(machine_resid[:, 1].var()) - float(sampling_var[:, 1].mean()))
    sigma = float(np.sqrt(0.5 * (var_dead + var_live)))
    if machine_resid.shape[0] >= 5:
        rho = float(np.corrcoef(machine_resid[:, 0], machine_resid[:, 1])[0, 1])
    else:
        rho = 0.0
    if not np.isfinite(rho):
        rho = 0.0
    rho = float(np.clip(rho, -0.85, 0.85))
    return {
        "a": float(beta[0]),
        "b": float(beta[1]),
        "c": float(beta[2]),
        "sigma": sigma,
        "rho": rho,
        "n_machines": len(rows),
        "var_dead": var_dead,
        "var_live": var_live,
        "by_signal": _level_table(rows),
    }


def _level_table(rows):
    table = {}
    for sig in (-1, 0, 1):
        live = [r[1] for r in rows if r[0] == sig]
        dead = [r[2] for r in rows if r[0] == sig]
        table[str(sig)] = {
            "n": len(live),
            "live_mean": float(np.mean(live)) if live else None,
            "dead_mean": float(np.mean(dead)) if dead else None,
        }
    return table


def _plugin_orientation(obs, win, cutoff, t0, t1, min_n=12, min_gap=0.2):
    """Early-window pair, whether agree is the live cell, and the two rates."""
    best = None
    for pair in PAIRS:
        k0, n0, k1, n1 = cell_counts(obs, win, cutoff, pair[0], pair[1], t0, t1)
        if n0 < min_n or n1 < min_n:
            continue
        gap = abs(k0 / n0 - k1 / n1)
        if best is None or gap > best[0]:
            best = (gap, pair, k0, n0, k1, n1)
    if best is None or best[0] < min_gap:
        return None
    _, pair, k0, n0, k1, n1 = best
    r0, r1 = k0 / n0, k1 / n1
    agree_is_live = r0 >= r1
    live, dead = (r0, r1) if agree_is_live else (r1, r0)
    live = min(1.0 - 1e-3, max(1e-3, live))
    dead = min(live - 1e-3, max(1e-3, dead))
    return pair, agree_is_live, live, dead


def _flip_ll_curve(obs, win, cutoff, pair, agree_is_live, live, dead):
    """Profile log-likelihood of a single flip at c, levels fixed at the early rates."""
    a, b = pair
    idx = np.arange(b, cutoff)
    valid = obs[idx] & obs[idx - a] & obs[idx - b]
    agree = win[idx - a] == win[idx - b]
    early = agree if agree_is_live else ~agree
    y = win[idx]
    # p_pre / p_post along idx
    p_pre = np.where(early, live, dead)
    p_post = np.where(early, dead, live)
    def _tick_ll(p):
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.where(valid, np.where(y, np.log(p), np.log(1.0 - p)), 0.0)

    ll_pre = _tick_ll(p_pre)
    ll_post = _tick_ll(p_post)
    # Map back to absolute tick. idx[j] = b + j.
    curve = np.full(cutoff + 1, 0.0)
    csum_pre = np.cumsum(ll_pre)
    csum_post = np.cumsum(ll_post)
    total_post = csum_post[-1] if csum_post.size else 0.0
    # c is the first flipped tick. Ticks in idx with tick < c use pre.
    for j, tick in enumerate(idx):
        # sum of ll_pre over idx ticks < tick, + ll_post over idx ticks >= tick
        pre = csum_pre[j - 1] if j else 0.0
        post_before = csum_post[j - 1] if j else 0.0
        curve[tick] = pre + (total_post - post_before)
    if idx.size:
        curve[idx[-1] + 1:] = csum_pre[-1]
        curve[:idx[0]] = total_post
    return curve


def fit_flip_prior(train):
    """Mean and sd of the shared flip tick, from train shared_change episodes.

    The search is flat on [0.35, 0.90] of the episode. Machines without a
    clear early pair are left out, which drops most jackpots and keeps the
    tick estimate on the machines whose two cells actually separate.
    """
    by_episode = {}
    for query in train["Q"]:
        if train["cond"][query["query_id"]] != "shared_change":
            continue
        by_episode.setdefault(query["episode_id"], []).append(query)
    fracs = []
    residuals_ready = []
    details = []
    for episode, queries in sorted(by_episode.items()):
        cutoff = queries[0]["cutoff"]
        total = np.zeros(cutoff + 1)
        used = 0
        lo, hi = int(0.35 * cutoff), int(0.90 * cutoff)
        for query in queries:
            obs, win = series_from_history(
                train["H"][(episode, query["machine_id"])], cutoff
            )
            found = _plugin_orientation(obs, win, cutoff, 0, int(0.45 * cutoff))
            if found is None:
                continue
            curve = _flip_ll_curve(obs, win, cutoff, *found)
            total += curve
            used += 1
        if used == 0:
            continue
        window = np.full(cutoff + 1, -np.inf)
        window[lo:hi + 1] = total[lo:hi + 1]
        c_hat = int(np.argmax(window))
        fracs.append(c_hat / cutoff)
        details.append({"episode": episode, "c": c_hat, "cutoff": cutoff, "machines": used})
    fracs = np.asarray(fracs)
    frac = float(fracs.mean())
    # Residual tick sd around frac * cutoff, not the sd of the fraction.
    resid = []
    for row in details:
        resid.append(row["c"] - frac * row["cutoff"])
    sd = float(np.std(resid, ddof=1)) if len(resid) > 2 else 15.0
    sd = float(np.clip(sd, 3.0, 40.0))
    return {"frac": frac, "sd": sd, "n_episodes": len(details), "fracs": fracs.tolist(), "details": details}


def fit_drift(train):
    """Grid-search the linear swap u = clip((t - (cutoff - anchor)) / duration, 0, 1).

    Early levels and the early-live cell come from ticks more than 110 before
    the cutoff, and only from non-jackpot machines with a clear gap. Jackpot
    dead rates are too close to zero to time a ramp.
    """
    anchors = np.arange(60, 141, 2)
    durations = np.arange(24, 101, 2)
    total = np.zeros((anchors.size, durations.size))
    used = 0
    for query in train["Q"]:
        if train["cond"][query["query_id"]] != "drift":
            continue
        if query["machine_type"] == "jackpot":
            continue
        cutoff = query["cutoff"]
        obs, win = series_from_history(
            train["H"][(query["episode_id"], query["machine_id"])], cutoff
        )
        found = _plugin_orientation(obs, win, cutoff, 0, max(8, cutoff - 110), min_n=15, min_gap=0.25)
        if found is None:
            continue
        pair, agree_is_live, live, dead = found
        a, b = pair
        idx = np.arange(b, cutoff)
        valid = obs[idx] & obs[idx - a] & obs[idx - b]
        if valid.sum() < 40:
            continue
        agree = win[idx - a] == win[idx - b]
        early = agree if agree_is_live else ~agree
        y = win[idx]
        ticks = idx.astype(float)
        # u[anchor, duration, tick]
        u = np.clip(
            (ticks[None, None, :] - (cutoff - anchors[:, None, None])) / durations[None, :, None],
            0.0,
            1.0,
        )
        p = np.where(
            early[None, None, :],
            (1.0 - u) * live + u * dead,
            (1.0 - u) * dead + u * live,
        )
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        ll = np.where(y, np.log(p), np.log(1.0 - p))
        ll = np.where(valid, ll, 0.0).sum(axis=-1)
        total += ll
        used += 1
    if used < 10:
        raise RuntimeError(f"drift fit used only {used} machines")
    ia, id_ = np.unravel_index(int(total.argmax()), total.shape)
    anchor = int(anchors[ia])
    duration = int(durations[id_])
    # Binned flip fraction under this fit, for the report. Orient each machine
    # the same way and average (L - r) / (L - D) by ticks-before-cutoff.
    return {
        "anchor": anchor,
        "duration": duration,
        "u_at_cutoff": float(np.clip(anchor / duration, 0.0, 1.0)),
        "n_machines": used,
        "ll": float(total[ia, id_]),
        "grid_margin": float(total[ia, id_] - np.partition(total.ravel(), -2)[-2]),
    }


def level_nodes(signal, params):
    center = params["a"] + params["b"] * signal
    e1 = params["sigma"] * params["z1"]
    scale = np.sqrt(max(0.0, 1.0 - params["rho"] ** 2))
    e2 = params["sigma"] * (params["rho"] * params["z1"] + scale * params["z2"])
    p_dead = np.clip(sigmoid(center + e1), 1e-4, 1.0 - 1e-4)
    p_live = np.clip(sigmoid(center + params["c"] + e2), 1e-4, 1.0 - 1e-4)
    return p_dead, p_live


def _tick_loglik(p, win, valid):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.where(valid[:, None], np.where(win[:, None], np.log(p), np.log(1.0 - p)), 0.0)


def hypothesis_table(obs, win, cutoff, p_dead, p_live, node_w, regime, anchor, duration, c_logprior):
    """Integrated-level evidence and predictive P(win) for every pair and direction.

    Direction 1 means the agree cell is live at the start of the episode.
    """
    log_node = np.log(np.clip(node_w, 1e-300, None))
    ticks = np.arange(cutoff, dtype=float)
    logevid = np.empty((len(PAIRS), 2))
    pred = np.empty((len(PAIRS), 2))
    lags_ok = np.zeros(len(PAIRS), dtype=bool)
    agree_now = np.zeros(len(PAIRS), dtype=bool)
    for index, (a, b) in enumerate(PAIRS):
        valid = np.zeros(cutoff, dtype=bool)
        agree = np.zeros(cutoff, dtype=bool)
        if cutoff > b:
            idx = np.arange(b, cutoff)
            ok = obs[idx] & obs[idx - a] & obs[idx - b]
            valid[idx] = ok
            agree[idx] = win[idx - a] == win[idx - b]
        lag_a = cutoff - a
        lag_b = cutoff - b
        if lag_a >= 0 and lag_b >= 0 and obs[lag_a] and obs[lag_b]:
            lags_ok[index] = True
            agree_now[index] = bool(win[lag_a] == win[lag_b])
        for direction, early in ((1, agree), (0, ~agree)):
            if regime == "drift":
                u = np.clip((ticks - (cutoff - anchor)) / duration, 0.0, 1.0)
                p = np.where(
                    early[:, None],
                    (1.0 - u)[:, None] * p_live + u[:, None] * p_dead,
                    (1.0 - u)[:, None] * p_dead + u[:, None] * p_live,
                )
                ll = _tick_loglik(p, win, valid).sum(axis=0)
                log_post = ll + log_node
                logevid[index, direction] = logsumexp(log_post)
                post = np.exp(log_post - log_post.max())
                post /= post.sum()
                u_now = float(np.clip(anchor / duration, 0.0, 1.0))
                if not lags_ok[index]:
                    p_g = 0.5 * (p_live + p_dead)
                else:
                    early_now = agree_now[index] if direction == 1 else not agree_now[index]
                    if early_now:
                        p_g = (1.0 - u_now) * p_live + u_now * p_dead
                    else:
                        p_g = (1.0 - u_now) * p_dead + u_now * p_live
                pred[index, direction] = float(post @ p_g)
            elif regime == "shared":
                p_pre = np.where(early[:, None], p_live, p_dead)
                p_post = np.where(early[:, None], p_dead, p_live)
                ll_pre = _tick_loglik(p_pre, win, valid)
                ll_post = _tick_loglik(p_post, win, valid)
                # cs[c] = sum of per-tick contributions with tick < c
                cs_pre = np.vstack([np.zeros((1, p_live.size)), np.cumsum(ll_pre, axis=0)])
                cs_post = np.vstack([np.zeros((1, p_live.size)), np.cumsum(ll_post, axis=0)])
                total_post = cs_post[-1]
                ll_c = cs_pre + total_post - cs_post  # (cutoff+1, G); c from 0..cutoff
                joint = ll_c + log_node[None, :] + c_logprior[:, None]
                logevid[index, direction] = logsumexp(joint)
                post_g = np.exp(joint - np.nanmax(joint)).sum(axis=0)
                post_g /= post_g.sum()
                # The flip tick is always before the cutoff, so the query sees the swapped rule.
                if not lags_ok[index]:
                    p_g = 0.5 * (p_live + p_dead)
                else:
                    early_now = agree_now[index] if direction == 1 else not agree_now[index]
                    p_g = p_dead if early_now else p_live
                pred[index, direction] = float(post_g @ p_g)
            else:
                p = np.where(early[:, None], p_live, p_dead)
                ll = _tick_loglik(p, win, valid).sum(axis=0)
                log_post = ll + log_node
                logevid[index, direction] = logsumexp(log_post)
                post = np.exp(log_post - log_post.max())
                post /= post.sum()
                if not lags_ok[index]:
                    p_g = 0.5 * (p_live + p_dead)
                else:
                    early_now = agree_now[index] if direction == 1 else not agree_now[index]
                    p_g = p_live if early_now else p_dead
                pred[index, direction] = float(post @ p_g)
    return logevid, pred


def mix_hypotheses(logevid, pred, pair_prior):
    log_h = logevid + np.log(np.clip(pair_prior, 1e-300, None))[:, None]
    # Direction prior is 1/2 and cancels in the weights; kept explicit so a
    # fitted direction prior can drop in without touching the evidence.
    log_h = log_h + np.log(0.5)
    flat = log_h - log_h.max()
    weight = np.exp(flat)
    weight /= weight.sum()
    p_post = float((weight * pred).sum())
    pick = np.unravel_index(int(log_h.argmax()), log_h.shape)
    p_map = float(pred[pick])
    return p_post, p_map, {
        "pair": list(PAIRS[pick[0]]),
        "direction_agree_early_live": int(pick[1] == 1),
        "map_weight": float(weight[pick]),
    }


def global_c_logprior(cutoff, frac, sd):
    c = np.arange(cutoff + 1)
    logp = -0.5 * ((c - frac * cutoff) / sd) ** 2
    # Keep the search off the first lags and off the cutoff itself.
    logp[: max(8, int(0.35 * cutoff))] = -np.inf
    logp[int(0.90 * cutoff) + 1:] = -np.inf
    finite = np.isfinite(logp)
    logp[finite] -= logsumexp(logp[finite])
    return logp


def episode_c_logpriors(queries, history, cond, frac, sd):
    """For each shared_change machine, log prior of the flip tick from the OTHER machines.

    Other machines contribute a flip profile with levels plugged in from their
    own early window. Added to the train Gaussian prior and normalised.
    """
    by_episode = {}
    for query in queries:
        if cond[query["query_id"]] != "shared_change":
            continue
        by_episode.setdefault(query["episode_id"], []).append(query)
    out = {}
    for episode, group in by_episode.items():
        cutoff = group[0]["cutoff"]
        curves = {}
        for query in group:
            obs, win = series_from_history(history[(episode, query["machine_id"])], cutoff)
            found = _plugin_orientation(obs, win, cutoff, 0, int(0.45 * cutoff))
            if found is None:
                curves[query["machine_id"]] = np.zeros(cutoff + 1)
            else:
                curves[query["machine_id"]] = _flip_ll_curve(obs, win, cutoff, *found)
        base = global_c_logprior(cutoff, frac, sd)
        for query in group:
            others = np.zeros(cutoff + 1)
            for machine_id, curve in curves.items():
                if machine_id != query["machine_id"]:
                    others += curve
            logp = base.copy()
            finite = np.isfinite(logp)
            logp[finite] = logp[finite] + others[finite]
            logp[finite] -= logsumexp(logp[finite])
            out[query["query_id"]] = logp
    return out


def predict_jackpot(obs, win, cutoff, signal, regime, params, c_logprior):
    p_dead, p_live = level_nodes(signal, params)
    if regime == "shared" and c_logprior is None:
        c_logprior = global_c_logprior(cutoff, params["flip_frac"], params["flip_sd"])
    logevid, pred = hypothesis_table(
        obs, win, cutoff, p_dead, p_live, params["node_w"],
        regime, params["anchor"], params["duration"], c_logprior,
    )
    post, map_, info = mix_hypotheses(logevid, pred, params["pair_prior"])
    post_flat, _, _ = mix_hypotheses(logevid, pred, params["pair_prior_flat"])
    return {"post": post, "map": map_, "flatpair": post_flat, "info": info}


def four_cell_counts(obs, win, cutoff, a, b, t0):
    """Win counts for the four (lag a won, lag b won) patterns on ticks [t0, cutoff)."""
    cells = {(0, 0): [0, 0], (0, 1): [0, 0], (1, 0): [0, 0], (1, 1): [0, 0]}
    start = max(int(t0), b)
    for t in range(start, cutoff):
        if not (obs[t] and obs[t - a] and obs[t - b]):
            continue
        key = (int(win[t - a]), int(win[t - b]))
        cells[key][0] += int(win[t])
        cells[key][1] += 1
    return cells


def atomic_win_probability(obs, win, cutoff, pair, t0, strength, p_live, p_dead, parity_strength=8.0):
    """P(win) from the exact lag pattern, shrunk toward that pattern's parity pool.

    Jackpot's two "agree" patterns (both lags won, both lost) are not the same
    rate, and the same is true of the two "differ" patterns. Pooling them, which
    is what a parity model does, pulls a hot pattern down and a cold one up.
    The atomic count is often small, so it is shrunk toward the parity pool,
    and the pool is shrunk toward the train live or dead level.
    """
    a, b = pair
    if min(cutoff - a, cutoff - b) < 0 or not (obs[cutoff - a] and obs[cutoff - b]):
        return None
    key = (int(win[cutoff - a]), int(win[cutoff - b]))
    cells = four_cell_counts(obs, win, cutoff, a, b, t0)
    same = key[0] == key[1]
    group = ((0, 0), (1, 1)) if same else ((0, 1), (1, 0))
    other = ((0, 1), (1, 0)) if same else ((0, 0), (1, 1))
    w_g = sum(cells[k][0] for k in group)
    n_g = sum(cells[k][1] for k in group)
    w_o = sum(cells[k][0] for k in other)
    n_o = sum(cells[k][1] for k in other)
    rate_g = w_g / n_g if n_g else None
    rate_o = w_o / n_o if n_o else None
    group_is_live = rate_o is None or (rate_g is not None and rate_g >= rate_o)
    prior = p_live if group_is_live else p_dead
    if n_g == 0:
        p_par = prior
    else:
        p_par = (w_g + parity_strength * prior) / (n_g + parity_strength)
    w_a, n_a = cells[key]
    return (w_a + strength * p_par) / (n_a + strength)


def clean_window_start(regime, cutoff, params, c_logprior):
    """First tick of the regime that is actually in force at the cutoff.

    Drift has finished its swap by `anchor - duration` ticks before the cutoff.
    Shared change flips at the episode tick. Stable and short history use everything.
    """
    if regime == "drift":
        return max(0, cutoff - int(params["anchor"]) + int(params["duration"]))
    if regime == "shared":
        if c_logprior is None:
            return int(round(params["flip_frac"] * cutoff))
        return int(np.argmax(c_logprior))
    return 0


def regime_of(condition):
    if condition == "drift":
        return "drift"
    if condition == "shared_change":
        return "shared"
    return "stable"


def self_check(params):
    """A long stable series with one planted pair must put most mass on that pair."""
    rng = np.random.default_rng(1)
    cutoff = 210
    # A rare pair, so this fails if the train pair prior drowns the likelihood.
    a, b = 6, 8
    win = np.zeros(cutoff, dtype=bool)
    # Latent wins at every tick, then the parity rule. Tick 0..b-1 are fair coins.
    for t in range(cutoff):
        if t < b:
            win[t] = bool(rng.random() < 0.3)
            continue
        agree = win[t - a] == win[t - b]
        p = 0.62 if agree else 0.02
        win[t] = bool(rng.random() < p)
    obs = np.ones(cutoff, dtype=bool)
    # Tight levels around the planted rates so this checks pair recovery, not the prior.
    saved = {k: params[k] for k in ("a", "b", "c", "sigma", "rho")}
    params["a"], params["b"], params["c"] = float(np.log(0.02 / 0.98)), 0.0, float(
        np.log(0.62 / 0.38) - np.log(0.02 / 0.98)
    )
    params["sigma"], params["rho"] = 0.35, 0.0
    result = predict_jackpot(obs, win, cutoff, 0, "stable", params, None)
    params.update(saved)
    # The query's lags are the last two relevant ticks; rebuild what the rule says at `cutoff`
    # by extending one step. Recovery is about the MAP pair on the history, which predict
    # reports. Planted pair is (2, 5), direction agree = early live.
    ok = (
        result["info"]["pair"] == [6, 8]
        and result["info"]["direction_agree_early_live"] == 1
        and result["info"]["map_weight"] > 0.5
    )
    return {
        "pass": bool(ok),
        "map_pair": result["info"]["pair"],
        "map_weight": result["info"]["map_weight"],
        "p_post": result["post"],
        "direction_agree_early_live": result["info"]["direction_agree_early_live"],
    }


def patch_vector(base, p_win):
    pos = np.asarray(base[1:], dtype=float)
    total = pos.sum()
    size = pos / total if total > 1e-12 else np.full(4, 0.25)
    p_win = float(np.clip(p_win, 1e-6, 1.0 - 1e-6))
    out = np.empty(5)
    out[0] = 1.0 - p_win
    out[1:] = p_win * size
    out /= out.sum()
    return out


def bootstrap_points(pred, data, scorer, repeats=1000, seed=314159):
    """95% interval of contest points, resampling episodes inside each condition."""
    ep = {q["query_id"]: q["episode_id"] for q in data["Q"]}
    by_condition = {c: {} for c in CONDITIONS}
    for query in data["Q"]:
        qid = query["query_id"]
        cond = data["cond"][qid]
        episode = ep[qid]
        se_hat = float(((pred[qid] - data["targets"][qid]) ** 2).sum())
        se_ref = float(((data["reference"][qid] - data["targets"][qid]) ** 2).sum())
        by_condition[cond].setdefault(episode, []).append((se_hat, se_ref))
    episode_mean = {
        cond: {e: (float(np.mean([a for a, _ in rows])), float(np.mean([b for _, b in rows])))
               for e, rows in episodes.items()}
        for cond, episodes in by_condition.items()
    }
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repeats):
        ratios = []
        for cond in CONDITIONS:
            episodes = list(episode_mean[cond])
            draw = rng.choice(episodes, size=len(episodes), replace=True)
            num = np.mean([episode_mean[cond][e][0] for e in draw])
            den = np.mean([episode_mean[cond][e][1] for e in draw])
            ratios.append(num / max(den, 1e-9))
        scores.append(60.0 / (1.0 + float(np.mean(ratios))))
    scores.sort()
    lo = scores[int(0.025 * (len(scores) - 1))]
    hi = scores[int(0.975 * (len(scores) - 1))]
    return [lo, hi]


def bootstrap_delta(pred, base, data, repeats=1000, seed=314159):
    """Paired 95% interval of (pred points - base points), same episode draws."""
    ep = {q["query_id"]: q["episode_id"] for q in data["Q"]}
    means = {c: {} for c in CONDITIONS}
    for query in data["Q"]:
        qid = query["query_id"]
        cond = data["cond"][qid]
        episode = ep[qid]
        means[cond].setdefault(episode, []).append((
            float(((pred[qid] - data["targets"][qid]) ** 2).sum()),
            float(((base[qid] - data["targets"][qid]) ** 2).sum()),
            float(((data["reference"][qid] - data["targets"][qid]) ** 2).sum()),
        ))
    episode_mean = {
        cond: {e: tuple(np.mean(rows, axis=0)) for e, rows in episodes.items()}
        for cond, episodes in means.items()
    }
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repeats):
        ratios_p, ratios_b = [], []
        for cond in CONDITIONS:
            episodes = list(episode_mean[cond])
            draw = rng.choice(episodes, size=len(episodes), replace=True)
            num_p = np.mean([episode_mean[cond][e][0] for e in draw])
            num_b = np.mean([episode_mean[cond][e][1] for e in draw])
            den = np.mean([episode_mean[cond][e][2] for e in draw])
            den = max(float(den), 1e-9)
            ratios_p.append(num_p / den)
            ratios_b.append(num_b / den)
        deltas.append(60.0 / (1.0 + float(np.mean(ratios_p))) - 60.0 / (1.0 + float(np.mean(ratios_b))))
    deltas.sort()
    return [
        deltas[int(0.025 * (len(deltas) - 1))],
        deltas[int(0.975 * (len(deltas) - 1))],
    ]


def score_pack(pred, data, scorer, qids=None):
    pts, ratios = scorer.points(pred, data, qids=qids, max_points=60)
    return {"points": float(pts), "ratios": {k: float(v) for k, v in ratios.items()}}


def jackpot_win_rmse(p_win, data, qids):
    err = []
    for qid in qids:
        err.append((p_win[qid] - (1.0 - data["targets"][qid][0])) ** 2)
    return float(np.sqrt(np.mean(err))) if err else None


def write_preds(path, pred):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["query_id", *PCOLS])
        for qid in sorted(pred):
            writer.writerow([qid, *[f"{v:.17g}" for v in pred[qid]]])


def main():
    scorer = _load_scorer()
    train = load_split("train")
    val = load_split("validation")
    # Official targets and the frozen reference, for scoring only.
    scored = scorer.load("validation", pub=PUB)
    v13 = {
        row["query_id"]: np.array([float(row[c]) for c in PCOLS])
        for row in csv.DictReader(open(V13_PREDS))
    }
    if set(v13) != set(scored["targets"]):
        raise RuntimeError("v13 predictions do not cover the validation queries")

    print("fitting pair prior, jackpot levels, drift, shared-change tick...", flush=True)
    pair_fit = fit_pair_prior(train)
    levels = fit_jackpot_levels(train)
    drift = fit_drift(train)
    flip = fit_flip_prior(train)
    z1, z2, node_w = quadrature(7)
    params = {
        **levels,
        "pair_prior": pair_fit["smoothed"],
        "pair_prior_flat": pair_fit["flat"],
        "anchor": drift["anchor"],
        "duration": drift["duration"],
        "flip_frac": flip["frac"],
        "flip_sd": flip["sd"],
        "z1": z1,
        "z2": z2,
        "node_w": node_w,
    }
    check = self_check(params)
    print("self_check", check, flush=True)
    if not check["pass"]:
        raise RuntimeError(f"planted-pair self-check failed: {check}")

    c_prior = episode_c_logpriors(val["Q"], val["H"], val["cond"], flip["frac"], flip["sd"])
    jackpot_queries = [q for q in val["Q"] if q["machine_type"] == "jackpot"]
    print(f"scoring {len(jackpot_queries)} jackpot validation queries", flush=True)
    variant_p = {name: {} for name in (*MENU, "post", "flatpair", "nocond", "atom16_keepdrift")}
    map_info = {}
    for i, query in enumerate(jackpot_queries):
        obs, win = series_from_history(
            val["H"][(query["episode_id"], query["machine_id"])], query["cutoff"]
        )
        kind = val["cond"][query["query_id"]]
        regime = regime_of(kind)
        c_log = c_prior.get(query["query_id"])
        hat = predict_jackpot(obs, win, query["cutoff"], query["signal"], regime, params, c_log)
        variant_p["map"][query["query_id"]] = hat["map"]
        variant_p["post"][query["query_id"]] = hat["post"]
        variant_p["flatpair"][query["query_id"]] = hat["flatpair"]
        map_info[query["query_id"]] = hat["info"]
        p_dead, p_live = level_nodes(query["signal"], params)
        # Prior means of the two levels (quadrature weights, sigmoid is nonlinear).
        t0 = clean_window_start(regime, query["cutoff"], params, c_log)
        for strength, name in ((4.0, "atom4"), (8.0, "atom8"), (16.0, "atom16")):
            atomic = atomic_win_probability(
                obs, win, query["cutoff"], tuple(hat["info"]["pair"]), t0, strength,
                float(params["node_w"] @ p_live), float(params["node_w"] @ p_dead),
            )
            variant_p[name][query["query_id"]] = hat["map"] if atomic is None else float(atomic)
        if kind == "drift":
            variant_p["atom16_keepdrift"][query["query_id"]] = 1.0 - float(v13[query["query_id"]][0])
        else:
            variant_p["atom16_keepdrift"][query["query_id"]] = variant_p["atom16"][query["query_id"]]
        # Condition-blind ablation: stable kernel 1/2 (stable + short_history),
        # drift 1/4, shared_change 1/4. Shared tick uses only the train prior.
        parts = []
        for regime_name, weight in (("stable", 0.5), ("drift", 0.25), ("shared", 0.25)):
            blind = predict_jackpot(
                obs, win, query["cutoff"], query["signal"], regime_name, params,
                None if regime_name == "shared" else None,
            )
            parts.append(weight * blind["post"])
        variant_p["nocond"][query["query_id"]] = float(sum(parts))
        if i and i % 40 == 0:
            print(f"  {i}/{len(jackpot_queries)}", flush=True)

    def apply(p_win_map, qids):
        pred = {qid: v13[qid].copy() for qid in v13}
        for qid in qids:
            if qid in p_win_map:
                pred[qid] = patch_vector(v13[qid], p_win_map[qid])
        return pred

    jackpot_ids = [q["query_id"] for q in jackpot_queries]
    all_ids = list(v13)
    # Oracle: true jackpot win probability, v13 size split. Ceiling for this patch.
    oracle_p = {qid: 1.0 - scored["targets"][qid][0] for qid in jackpot_ids}
    base_pred = {qid: v13[qid].copy() for qid in v13}
    oracle_pred = apply(oracle_p, jackpot_ids)
    ref_pred = {qid: scored["reference"][qid].copy() for qid in all_ids}

    fold_a, fold_b = scorer.folds(scored)
    qids_by_fold = {"A": [], "B": []}
    for query in val["Q"]:
        qids_by_fold["A" if query["episode_id"] in fold_a else "B"].append(query["query_id"])

    # Cross-fit the discrete variant. Parameters stay train-only.
    chosen = {}
    cross_p = {}
    fold_choice = {}
    for tune, score_fold, tune_name, score_name in (
        ("A", "B", "A", "B"),
        ("B", "A", "B", "A"),
    ):
        best_name, best_pts = None, -1.0
        tune_scores = {}
        for name in MENU:
            pred = apply(variant_p[name], qids_by_fold[tune])
            pack = score_pack(pred, scored, scorer, qids=qids_by_fold[tune])
            tune_scores[name] = pack["points"]
            if pack["points"] > best_pts:
                best_name, best_pts = name, pack["points"]
        fold_choice[f"tune_{tune_name}_score_{score_name}"] = {
            "chosen": best_name,
            "tune_points": tune_scores,
        }
        for qid in qids_by_fold[score_fold]:
            if qid in variant_p[best_name]:
                cross_p[qid] = variant_p[best_name][qid]
                chosen[qid] = best_name

    cross_pred = apply(cross_p, jackpot_ids)
    # Train-only strength: cutoff Brier against train labels, stable and short
    # jackpot only, pair chosen by the parity score. No validation target is read.
    train_labels = {
        row["query_id"]: int(row["outcome"]) > 0
        for row in csv.DictReader(open(PUB / "train" / "labels.csv"))
    }
    brier = {name: [] for name in MENU}
    for query in train["Q"]:
        if query["machine_type"] != "jackpot":
            continue
        if train["cond"][query["query_id"]] not in ("stable", "short_history"):
            continue
        obs, win = series_from_history(
            train["H"][(query["episode_id"], query["machine_id"])], query["cutoff"]
        )
        pair, gain, _counts = best_pair(obs, win, query["cutoff"], 0, query["cutoff"], min_n=8)
        if pair is None:
            continue
        y = train_labels[query["query_id"]]
        p_dead_n, p_live_n = level_nodes(query["signal"], params)
        p_live_m = float(params["node_w"] @ p_live_n)
        p_dead_m = float(params["node_w"] @ p_dead_n)
        for strength, name in ((0.0, "map"), (4.0, "atom4"), (8.0, "atom8"), (16.0, "atom16")):
            if name == "map":
                # Parity pool of the active pattern, same prior shrink as the atomic parent.
                k0, n0, k1, n1 = cell_counts(obs, win, query["cutoff"], pair[0], pair[1], 0, query["cutoff"])
                if query["cutoff"] <= pair[1] or not (obs[query["cutoff"] - pair[0]] and obs[query["cutoff"] - pair[1]]):
                    continue
                agree = bool(win[query["cutoff"] - pair[0]] == win[query["cutoff"] - pair[1]])
                w, n = (k0, n0) if agree else (k1, n1)
                other_w, other_n = (k1, n1) if agree else (k0, n0)
                rate = w / n if n else 0.0
                other = other_w / other_n if other_n else 0.0
                prior = p_live_m if rate >= other else p_dead_m
                p_hat = (w + 8.0 * prior) / (n + 8.0) if n else prior
            else:
                p_hat = atomic_win_probability(
                    obs, win, query["cutoff"], pair, 0, strength, p_live_m, p_dead_m,
                )
                if p_hat is None:
                    continue
            brier[name].append((p_hat - float(y)) ** 2)
    train_brier = {name: float(np.mean(v)) if v else None for name, v in brier.items()}
    train_choice = min((name for name, v in train_brier.items() if v is not None), key=lambda name: train_brier[name])
    train_only_pred = apply(variant_p[train_choice], jackpot_ids)
    full_variant = {
        name: score_pack(apply(variant_p[name], jackpot_ids), scored, scorer)
        for name in (*MENU, "post", "flatpair", "nocond", "atom16_keepdrift")
    }
    # Descriptive full-validation numbers. The selection itself is the cross-fit above.
    cross_pack = score_pack(cross_pred, scored, scorer)
    train_pack = score_pack(train_only_pred, scored, scorer)
    base_pack = score_pack(base_pred, scored, scorer)
    oracle_pack = score_pack(oracle_pred, scored, scorer)
    anchor_pack = score_pack(ref_pred, scored, scorer)

    def ci(pred):
        return bootstrap_points(pred, scored, scorer)

    # Win-probability error on jackpot queries, by condition.
    by_cond_rmse = {}
    for cond in CONDITIONS:
        ids = [q["query_id"] for q in jackpot_queries if val["cond"][q["query_id"]] == cond]
        v13_p = {qid: 1.0 - v13[qid][0] for qid in ids}
        by_cond_rmse[cond] = {
            "n": len(ids),
            "v13": jackpot_win_rmse(v13_p, scored, ids),
            "post": jackpot_win_rmse(variant_p["post"], scored, ids),
            "cross": jackpot_win_rmse(cross_p, scored, ids),
            "oracle": 0.0,
        }

    top_pairs = sorted(
        ((list(PAIRS[i]), int(pair_fit["counts"][i])) for i in range(len(PAIRS))),
        key=lambda row: -row[1],
    )[:8]
    results = {
        "self_check": check,
        "pair_prior_top": top_pairs,
        "pair_n_scored": pair_fit["n_scored"],
        "p_agree_live": pair_fit["p_agree_live"],
        "levels": {k: levels[k] for k in ("a", "b", "c", "sigma", "rho", "n_machines", "var_dead", "var_live", "by_signal")},
        "drift": drift,
        "flip": {k: flip[k] for k in ("frac", "sd", "n_episodes")},
        "n_jackpot_val": len(jackpot_queries),
        "v13": base_pack,
        "v13_ci95": ci(base_pred),
        "train_only_choice": train_choice,
        "train_label_brier": train_brier,
        "train_only_post": train_pack,
        "train_only_ci95": ci(train_only_pred),
        "crossfit": cross_pack,
        "crossfit_ci95": ci(cross_pred),
        "crossfit_choice": fold_choice,
        "oracle_jackpot_win": oracle_pack,
        "pooled_anchor": anchor_pack,
        "full_validation_variants_descriptive": full_variant,
        "jackpot_win_rmse": by_cond_rmse,
        "delta_points_cross_minus_v13": cross_pack["points"] - base_pack["points"],
        "delta_cross_ci95": bootstrap_delta(cross_pred, base_pred, scored),
        "delta_train_only_ci95": bootstrap_delta(train_only_pred, base_pred, scored),
        "delta_points_oracle_minus_v13": oracle_pack["points"] - base_pack["points"],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    write_preds(OUT_DIR / "validation_preds_crossfit.csv", cross_pred)
    write_preds(OUT_DIR / "validation_preds_trainonly.csv", train_only_pred)
    print(json.dumps({k: results[k] for k in (
        "self_check", "levels", "drift", "flip", "v13", "train_only_post", "crossfit",
        "crossfit_choice", "oracle_jackpot_win", "pooled_anchor",
        "full_validation_variants_descriptive", "jackpot_win_rmse",
        "delta_points_cross_minus_v13", "delta_points_oracle_minus_v13", "pair_prior_top",
    )}, indent=2))


if __name__ == "__main__":
    main()
