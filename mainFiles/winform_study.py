"""Functional form of P(win) at the cutoff, and the win-only ceiling of each piece.

Validation targets are fit outcomes on one fold and scores on the other. They are
never features of the query they score. Pair, direction, and level updates use
only ticks strictly before the cutoff. Condition labels choose the regime
hypothesis under test; they are not numeric features.

The headline generator is a logit-normal level per parity cell, with a separate
signal slope in each cell, marginalizing the three pairs and two directions.
Drift uses a pure pre-ramp window and a pure post-ramp window. Shared change
uses one episode flip tick. Sizes in the scored vectors are the target's own
conditional size distribution, so the points isolate win-probability error.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from slot_forecasting.contest import contest_score  # noqa: E402
from slot_forecasting.io import load_inputs, read_rows, read_submission, read_targets  # noqa: E402
from slot_forecasting.schema import CONDITIONS  # noqa: E402

PAIRS = ((1, 3), (2, 5), (4, 7))
TYPES = ("steady", "balanced", "jackpot")
CELLS = ("dead", "live")
# Nominal drift ramp from the generative description. Measured, not assumed, below.
RAMP_START_BACK = 96
RAMP_END_BACK = 34
GRID = 161


def clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def u_at_cutoff(anchor, length):
    """Flip fraction at the query under u = clip((t - (cutoff - anchor)) / length, 0, 1)."""
    if length <= 0:
        raise ValueError("length must be positive")
    return clip01(anchor / length)


def dist_from_p(p_win, sizes):
    """Five payout probabilities with the true conditional size distribution."""
    p = float(p_win)
    if not math.isfinite(p):
        p = 0.5
    p = min(max(p, 0.0), 1.0)
    sizes = np.asarray(sizes, dtype=np.float64)
    sizes = np.clip(sizes, 0.0, None)
    total = float(sizes.sum())
    if total <= 0:
        sizes = np.full(4, 0.25)
    else:
        sizes = sizes / total
    out = np.empty(5, dtype=np.float64)
    out[0] = 1.0 - p
    out[1:] = p * sizes
    # Reconstructing p * (q / p) can miss 1 by a couple of ulps.
    error = float(out.sum() - 1.0)
    if abs(error) > 0:
        out[0] -= error
    return out


def true_sizes(probability):
    probability = np.asarray(probability, dtype=np.float64)
    p_win = 1.0 - probability[0]
    if p_win < 1e-12:
        return np.full(4, 0.25)
    sizes = np.clip(probability[1:], 0.0, None) / p_win
    sizes = np.clip(sizes, 0.0, None)
    return sizes / sizes.sum()


def level_table(mu, sd, ngrid=GRID):
    """Quadrature nodes for a normal prior on the logit."""
    sd = max(float(sd), 1e-3)
    theta = np.linspace(mu - 8.0 * sd, mu + 8.0 * sd, ngrid)
    dtheta = float(theta[1] - theta[0])
    log_prior = -0.5 * ((theta - mu) / sd) ** 2 - np.log(sd * math.sqrt(2.0 * math.pi))
    p = expit(theta)
    return theta, p, log_prior, np.log(np.clip(p, 1e-15, 1.0)), np.log(np.clip(1.0 - p, 1e-15, 1.0)), dtheta


def level_stats(k, n, mu, sd, ngrid=GRID):
    """Log evidence, posterior mean of p, posterior mean of the logit, posterior sd."""
    k = float(k)
    n = float(n)
    if sd <= 1e-8:
        p = float(expit(mu))
        p = min(max(p, 1e-8), 1.0 - 1e-8)
        logz = 0.0 if n <= 0 else k * math.log(p) + (n - k) * math.log(1.0 - p)
        return logz, p, float(mu), 0.0
    theta, p, log_prior, log_p, log_q, dtheta = level_table(mu, sd, ngrid)
    log_joint = log_prior + k * log_p + (n - k) * log_q
    m = float(log_joint.max())
    w = np.exp(log_joint - m) * dtheta
    z = float(w.sum())
    logz = math.log(z) + m
    w /= z
    mean_theta = float(w @ theta)
    var = float(w @ (theta - mean_theta) ** 2)
    return logz, float(w @ p), mean_theta, math.sqrt(max(var, 0.0))


def softmax_rows(logev):
    m = np.max(logev)
    w = np.exp(logev - m)
    w /= w.sum()
    return w


def build_sequences(queries, history):
    seqs = {(q.episode_id, q.machine_id): np.full(q.cutoff + 1, -1, dtype=np.int16) for q in queries}
    for play in history:
        seq = seqs[(play.episode_id, play.machine_id)]
        if play.tick < len(seq) - 1:
            seq[play.tick] = play.payout
    return seqs


def pair_alpha(seq, cutoff):
    """Agree indicators on 0..cutoff inclusive. Index `cutoff` is the query cell; no outcome is stored there."""
    alpha = np.full((len(PAIRS), cutoff + 1), -1.0)
    ticks = np.arange(cutoff + 1)
    for j, (a, b) in enumerate(PAIRS):
        usable = ticks >= b
        tt = ticks[usable]
        left = seq[tt - a]
        right = seq[tt - b]
        ok = (left >= 0) & (right >= 0)
        alpha[j, tt[ok]] = ((left[ok] > 0) == (right[ok] > 0)).astype(float)
    return alpha


def _cell_counts(seq, alpha_row, direction, use, flip_at):
    """Binomial counts for the dead and live levels.

    `direction` 1 means agreement is live before the flip. Ticks >= flip_at are swapped.
    """
    outcome = seq[: len(use)]
    observed = (outcome >= 0) & use & (alpha_row[: len(use)] >= 0)
    agree = alpha_row[: len(use)] == 1.0
    early_live = agree if direction == 1 else ~agree
    if flip_at is None:
        is_live = early_live
    else:
        flipped = np.arange(len(use)) >= flip_at
        is_live = np.where(flipped, ~early_live, early_live)
    win = outcome > 0
    live = observed & is_live
    dead = observed & ~is_live
    return np.array([dead.sum(), live.sum()], dtype=np.float64), np.array(
        [(dead & win).sum(), (live & win).sum()], dtype=np.float64
    )


def cutoff_cell(alpha_row, cutoff, direction, flipped):
    agree = alpha_row[cutoff]
    if agree < 0:
        return None
    early_live = (agree == 1.0) if direction == 1 else (agree == 0.0)
    is_live = (not early_live) if flipped else early_live
    return "live" if is_live else "dead"


def map_pair(seq, alpha, cutoff, window):
    """Pair by two-cell likelihood. Direction is which parity has the higher rate. Direction does not change the likelihood."""
    outcome = seq[:cutoff]
    observed = outcome >= 0
    win = outcome > 0
    gains = []
    rates = []
    for j in range(len(PAIRS)):
        known = observed & window & (alpha[j, :cutoff] >= 0)
        agree = alpha[j, :cutoff] == 1.0
        ns = np.array([np.sum(known & ~agree), np.sum(known & agree)], dtype=np.float64)
        ks = np.array([np.sum(known & ~agree & win), np.sum(known & agree & win)], dtype=np.float64)
        gains.append(_two_cell_gain(ks, ns))
        rate = np.divide(ks, ns, out=np.full(2, np.nan), where=ns > 0)
        rates.append(rate)
    order = np.argsort(gains)
    best = int(order[-1])
    second = float(gains[order[-2]]) if len(order) > 1 else float("-inf")
    rate = rates[best]
    if np.isnan(rate).any():
        direction = 1
        gap_rate = 0.0
    else:
        direction = 1 if rate[1] >= rate[0] else 0
        gap_rate = float(abs(rate[1] - rate[0]))
    return {
        "pair": best,
        "direction": direction,
        "gain": float(gains[best]),
        "gap": float(gains[best] - second),
        "rate_gap": gap_rate,
        "uncertain": bool(gains[best] - second < 3.0 or gap_rate < 0.15),
    }


def _two_cell_gain(ks, ns):
    def one(k, n):
        if n <= 0:
            return 0.0
        p = min(max(k / n, 1e-6), 1.0 - 1e-6)
        return float(k * np.log(p) + (n - k) * np.log(1.0 - p))

    pooled = one(ks.sum(), ns.sum())
    separate = one(ks[0], ns[0]) + one(ks[1], ns[1])
    return separate - pooled


def fit_line(signals, logits):
    x = np.column_stack([np.ones(len(signals)), np.asarray(signals, dtype=float)])
    y = np.asarray(logits, dtype=float)
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    dof = max(len(y) - x.shape[1], 1)
    sd = float(np.sqrt(np.sum(resid ** 2) / dof))
    return float(beta[0]), float(beta[1]), sd, resid


def fit_prior(rows):
    """rows: (type, cell, signal, logit). Separate intercept and slope per cell."""
    prior = {}
    for typ in TYPES:
        for cell in CELLS:
            subset = [r for r in rows if r[0] == typ and r[1] == cell]
            if len(subset) < 4:
                prior[(typ, cell)] = {"a": 0.0, "b": 0.0, "sd": 0.6, "n": len(subset)}
                continue
            a, b, sd, _ = fit_line([r[2] for r in subset], [r[3] for r in subset])
            prior[(typ, cell)] = {"a": a, "b": b, "sd": float(np.clip(sd, 0.05, 2.0)), "n": len(subset)}
    return prior


def fit_shared_slope(rows):
    """One signal slope per type, separate dead and live intercepts. Residual sd stays per cell."""
    out = {}
    for typ in TYPES:
        subset = [r for r in rows if r[0] == typ]
        if len(subset) < 6:
            continue
        live = np.array([1.0 if r[1] == "live" else 0.0 for r in subset])
        signal = np.array([r[2] for r in subset], dtype=float)
        y = np.array([r[3] for r in subset], dtype=float)
        x = np.column_stack([np.ones(len(subset)), live, signal])
        beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        for cell, flag in (("dead", 0.0), ("live", 1.0)):
            mask = live == flag
            dof = max(int(mask.sum()) - 2, 1)
            sd = float(np.sqrt(np.sum(resid[mask] ** 2) / dof)) if mask.any() else 0.6
            intercept = float(beta[0] + flag * beta[1])
            out[(typ, cell)] = {"a": intercept, "b": float(beta[2]), "sd": float(np.clip(sd, 0.05, 2.0)), "n": int(mask.sum())}
    return out


def fit_no_signal(rows):
    out = {}
    for typ in TYPES:
        for cell in CELLS:
            subset = [r[3] for r in rows if r[0] == typ and r[1] == cell]
            if len(subset) < 2:
                out[(typ, cell)] = {"a": 0.0, "b": 0.0, "sd": 0.6, "n": len(subset)}
                continue
            a = float(np.mean(subset))
            sd = float(np.std(subset, ddof=1)) if len(subset) > 2 else 0.6
            out[(typ, cell)] = {"a": a, "b": 0.0, "sd": float(np.clip(sd, 0.05, 2.0)), "n": len(subset)}
    return out


def mu_sd(prior, typ, cell, signal, shift=0.0):
    par = prior[(typ, cell)]
    return par["a"] + par["b"] * signal + shift, par["sd"]


def predict_machine(seq, alpha, cutoff, prior, signal, typ, regime, mode, flip_at=None, ramp=(RAMP_START_BACK, RAMP_END_BACK), u_cut=1.0, shift=0.0):
    """Marginal P(win) over three pairs and two directions."""
    use = np.ones(cutoff, dtype=bool)
    flipped = False
    if mode == "unflipped" or regime in ("stable", "short_history"):
        flip_at = None
        flipped = False
    elif regime == "drift":
        start, end = ramp
        ticks = np.arange(cutoff)
        use = (ticks < cutoff - start) | (ticks >= cutoff - end)
        flip_at = cutoff - end
        flipped = True
    elif regime == "shared_change":
        flipped = True
        if flip_at is None:
            flip_at = int(0.66 * cutoff)
    logev = np.zeros(len(PAIRS) * 2)
    pred = np.zeros(len(PAIRS) * 2)
    eth = np.zeros(len(PAIRS) * 2)
    hyp_cell = []
    idx = 0
    shrink = mode != "point"
    for j in range(len(PAIRS)):
        for direction in (0, 1):
            n, k = _cell_counts(seq, alpha[j], direction, use, flip_at if flipped else None)
            cell = cutoff_cell(alpha[j], cutoff, direction, flipped)
            evidence = 0.0
            post_p = {}
            post_theta = {}
            for c_i, cell_name in enumerate(CELLS):
                mu, sd = mu_sd(prior, typ, cell_name, signal, shift)
                if shrink:
                    logz, ep, et, _ = level_stats(k[c_i], n[c_i], mu, sd)
                else:
                    ep = float(expit(mu))
                    ep_c = min(max(ep, 1e-8), 1.0 - 1e-8)
                    logz = 0.0 if n[c_i] <= 0 else float(k[c_i] * np.log(ep_c) + (n[c_i] - k[c_i]) * np.log(1.0 - ep_c))
                    et = mu
                evidence += logz
                post_p[cell_name] = ep
                post_theta[cell_name] = et
            if cell is None:
                p = 0.5 * (post_p["dead"] + post_p["live"])
                e_theta = 0.5 * (post_theta["dead"] + post_theta["live"])
            else:
                p = post_p[cell]
                e_theta = post_theta[cell]
                if mode == "blend" and regime == "drift" and cell is not None and u_cut < 0.999:
                    other = "dead" if cell == "live" else "live"
                    p = u_cut * post_p[cell] + (1.0 - u_cut) * post_p[other]
            logev[idx] = evidence
            pred[idx] = p
            eth[idx] = e_theta
            hyp_cell.append(cell if cell is not None else "unknown")
            idx += 1
    weights = softmax_rows(logev)
    p_hat = float(weights @ pred)
    # MAP hypothesis, for diagnostics only.
    m = int(np.argmax(weights))
    return {
        "p": p_hat,
        "entropy": float(-np.sum(weights * np.log(weights + 1e-300))),
        "map_p": float(pred[m]),
        "map_theta": float(eth[m]),
        "map_cell": hyp_cell[m],
        "map_weight": float(weights[m]),
    }


def window_for(regime, cutoff):
    ticks = np.arange(cutoff)
    if regime in ("drift", "shared_change"):
        # Before either the nominal ramp or the shared-flip region.
        return ticks < int(0.40 * cutoff)
    return np.ones(cutoff, dtype=bool)


def label_rows(machines, regimes, include_conditions):
    rows = []
    kept = 0
    dropped = 0
    for m in machines:
        if m["condition"] not in include_conditions:
            continue
        if m["rule"]["uncertain"]:
            dropped += 1
            continue
        cell = cutoff_cell(m["alpha"][m["rule"]["pair"]], m["cutoff"], m["rule"]["direction"], regimes[m["condition"]])
        if cell is None:
            dropped += 1
            continue
        rows.append((m["type"], cell, m["signal"], m["logit"]))
        kept += 1
    return rows, {"kept": kept, "dropped_uncertain": dropped}


def train_level_prior(machines):
    """Debiased logit-normal prior from stable training histories. No validation targets."""
    buckets = {(typ, cell): [] for typ in TYPES for cell in CELLS}
    cov_rows = {typ: [] for typ in TYPES}
    for m in machines:
        if m["condition"] != "stable":
            continue
        rule = m["rule"]
        if rule["uncertain"]:
            continue
        use = np.ones(m["cutoff"], dtype=bool)
        n, k = _cell_counts(m["seq"], m["alpha"][rule["pair"]], rule["direction"], use, None)
        mles = []
        sampling = []
        ok = True
        for c_i in range(2):
            if n[c_i] < 25 or k[c_i] <= 0 or k[c_i] >= n[c_i]:
                ok = False
                break
            mle = math.log((k[c_i] + 0.5) / (n[c_i] - k[c_i] + 0.5))
            var = 1.0 / (k[c_i] + 0.5) + 1.0 / (n[c_i] - k[c_i] + 0.5)
            buckets[(m["type"], CELLS[c_i])].append((m["signal"], mle, var, n[c_i]))
            mles.append(mle)
            sampling.append(var)
        if ok:
            cov_rows[m["type"]].append((m["signal"], mles[0], mles[1], sampling[0], sampling[1]))
    prior = {}
    for key, values in buckets.items():
        if len(values) < 4:
            prior[key] = {"a": 0.0, "b": 0.0, "sd": 0.6, "n": len(values)}
            continue
        signal = np.array([v[0] for v in values], dtype=float)
        mle = np.array([v[1] for v in values], dtype=float)
        var = np.array([v[2] for v in values], dtype=float)
        weight = np.array([v[3] for v in values], dtype=float)
        x = np.column_stack([np.ones(len(values)), signal])
        beta = np.linalg.lstsq(x * weight[:, None], mle * weight, rcond=None)[0]
        resid = mle - x @ beta
        excess = float(np.average(resid ** 2 - var, weights=weight))
        prior[key] = {
            "a": float(beta[0]),
            "b": float(beta[1]),
            "sd": float(np.sqrt(max(excess, 0.02))),
            "n": len(values),
        }
    corr = {}
    for typ, values in cov_rows.items():
        if len(values) < 8:
            corr[typ] = None
            continue
        signal = np.array([v[0] for v in values], dtype=float)
        dead = np.array([v[1] for v in values])
        live = np.array([v[2] for v in values])
        xd = np.column_stack([np.ones(len(values)), signal])
        rd = dead - xd @ np.linalg.lstsq(xd, dead, rcond=None)[0]
        rl = live - xd @ np.linalg.lstsq(xd, live, rcond=None)[0]
        cov = np.cov(np.vstack([rd, rl]), ddof=1)
        sampling = np.mean([v[3] for v in values]), np.mean([v[4] for v in values])
        deb = cov.copy()
        deb[0, 0] = max(cov[0, 0] - sampling[0], 0.01)
        deb[1, 1] = max(cov[1, 1] - sampling[1], 0.01)
        raw = float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))
        denom = math.sqrt(deb[0, 0] * deb[1, 1])
        debiased = float(cov[0, 1] / denom) if denom > 0 else None
        # Sampling-variance subtraction often drives the latent correlation outside [-1, 1]
        # because the delta-method variance is too large near 0 and 1. Keep the raw number.
        corr[typ] = {
            "n": len(values),
            "raw_corr": raw,
            "raw_sd_dead": float(np.sqrt(cov[0, 0])),
            "raw_sd_live": float(np.sqrt(cov[1, 1])),
            "mean_sampling_var_dead": float(sampling[0]),
            "mean_sampling_var_live": float(sampling[1]),
            "excess_var_dead": float(cov[0, 0] - sampling[0]),
            "excess_var_live": float(cov[1, 1] - sampling[1]),
            "debiased_corr": None if debiased is None or abs(debiased) > 1 else debiased,
            "cov": float(cov[0, 1]),
        }
    return prior, corr


def _drift_views(group):
    """Early-window rates and the post-35% spins used to score a ramp. One row per usable machine."""
    cutoff = group[0]["cutoff"]
    views = []
    for m in group:
        rule = m["rule"]
        if rule["gap"] < 1.0:
            continue
        early = np.arange(cutoff) < int(0.35 * cutoff)
        n, k = _cell_counts(m["seq"], m["alpha"][rule["pair"]], rule["direction"], early, None)
        if n.min() < 8:
            continue
        rates = np.clip((k + 0.5) / (n + 1.0), 1e-3, 1 - 1e-3)
        alpha = m["alpha"][rule["pair"], :cutoff]
        seq = m["seq"][:cutoff]
        known = (seq >= 0) & (alpha >= 0)
        early_live = (alpha == 1.0) if rule["direction"] == 1 else (alpha == 0.0)
        views.append({
            "known": known,
            "win": seq > 0,
            "early_live": early_live,
            "r_dead": float(rates[0]),
            "r_live": float(rates[1]),
        })
    return views


def _ll_from_u(view, u, mask):
    p_same = np.where(view["early_live"], view["r_live"], view["r_dead"])
    p_swap = np.where(view["early_live"], view["r_dead"], view["r_live"])
    p = np.clip((1.0 - u) * p_same + u * p_swap, 1e-6, 1.0 - 1e-6)
    use = view["known"] & mask
    win = view["win"]
    return float(np.sum(np.where(win & use, np.log(p), 0.0) + np.where(~win & use, np.log(1.0 - p), 0.0)))


def drift_schedule(group):
    """Shared linear ramp. Rates are frozen on the first 35% of ticks; the rest scores the ramp."""
    cutoff = group[0]["cutoff"]
    views = _drift_views(group)
    if not views:
        return {
            "anchor": RAMP_START_BACK,
            "length": RAMP_START_BACK - RAMP_END_BACK,
            "u": 1.0,
            "late_ll": {},
            "n_machines": 0,
            "ll_per_machine_vs_no_flip": 0.0,
        }
    score = np.arange(cutoff) >= int(0.35 * cutoff)
    anchors = np.arange(70, 145, 2)
    lengths = np.arange(30, 95, 2)
    ticks = np.arange(cutoff)
    best_ll, best_anchor, best_length = -1e300, RAMP_START_BACK, RAMP_START_BACK - RAMP_END_BACK
    for anchor in anchors:
        # u[length, tick] for this anchor. Small enough to broadcast per machine.
        u = np.clip((ticks - (cutoff - anchor)) / lengths[:, None], 0.0, 1.0)
        ll = np.zeros(len(lengths))
        for view in views:
            p_same = np.where(view["early_live"], view["r_live"], view["r_dead"])
            p_swap = np.where(view["early_live"], view["r_dead"], view["r_live"])
            p = np.clip((1.0 - u) * p_same + u * p_swap, 1e-6, 1.0 - 1e-6)
            use = view["known"] & score
            term = np.where(view["win"], np.log(p), np.log(1.0 - p))
            ll += np.sum(term[:, use], axis=1)
        i = int(np.argmax(ll))
        if ll[i] > best_ll:
            best_ll, best_anchor, best_length = float(ll[i]), int(anchor), int(lengths[i])
    none_ll = sum(_ll_from_u(v, 0.0, score) for v in views)
    fixed_u = np.clip((ticks - (cutoff - RAMP_START_BACK)) / (RAMP_START_BACK - RAMP_END_BACK), 0.0, 1.0)
    fixed_ll = sum(_ll_from_u(v, fixed_u, score) for v in views)
    step_u = (ticks >= cutoff - 40).astype(float)
    done_ll = sum(_ll_from_u(v, step_u, score) for v in views)
    late_mask = ticks >= cutoff - 20
    late = {"n_ticks": int(sum(np.sum(v["known"] & late_mask) for v in views))}
    for name, u in (("u1", 1.0), ("u09", 0.9), ("u07", 0.7), ("u05", 0.5), ("u0", 0.0)):
        late[name] = sum(_ll_from_u(v, u, late_mask) for v in views)
    return {
        "anchor": best_anchor,
        "length": best_length,
        "u": u_at_cutoff(best_anchor, best_length),
        "ll": best_ll,
        "fixed_96_62_ll": fixed_ll,
        "no_flip_ll": none_ll,
        "step_finished_by_40_ll": done_ll,
        "late_ll": late,
        "n_machines": len(views),
        "ll_per_machine_vs_no_flip": (best_ll - none_ll) / len(views),
    }


def shared_flip(group):
    """One flip tick for the episode. Saturated two-rate likelihood, prefix sums."""
    cutoff = group[0]["cutoff"]
    lo = int(0.45 * cutoff)
    hi = int(0.85 * cutoff)
    candidates = np.arange(lo, hi)
    total = np.zeros(len(candidates))
    used = 0
    per_machine = []
    for m in group:
        rule = m["rule"]
        if rule["gap"] < 1.0:
            continue
        used += 1
        ll = _flip_profile(m["seq"], m["alpha"][rule["pair"]], rule["direction"], cutoff, candidates)
        total += ll
        per_machine.append(ll)
    if used == 0:
        c = int(0.66 * cutoff)
        return {"c": c, "ticks_before": cutoff - c, "fraction": c / cutoff, "gain": 0.0, "sharp_sd": None, "n_machines": 0}
    best_i = int(np.argmax(total))
    c = int(candidates[best_i])
    # Curvature: points within 1 nat of the peak, as a tick sd.
    peak = total[best_i]
    close = candidates[total >= peak - 1.0]
    no_flip = 0.0
    for m in group:
        rule = m["rule"]
        if rule["gap"] < 1.0:
            continue
        n, k = _cell_counts(m["seq"], m["alpha"][rule["pair"]], rule["direction"], np.ones(cutoff, dtype=bool), None)
        no_flip += _saturated_ll(k, n)
    return {
        "c": c,
        "ticks_before": cutoff - c,
        "fraction": c / cutoff,
        "gain_vs_no_flip": float(peak - no_flip),
        "sharp_width": int(close.max() - close.min()) if len(close) else 0,
        "n_machines": used,
        "ll": float(peak),
    }


def _saturated_ll(k, n):
    ll = 0.0
    for kk, nn in zip(k, n):
        if nn <= 0:
            continue
        p = min(max(kk / nn, 1e-6), 1.0 - 1e-6)
        ll += kk * math.log(p) + (nn - kk) * math.log(1.0 - p)
    return float(ll)


def _flip_profile(seq, alpha_row, direction, cutoff, candidates):
    outcome = seq[:cutoff]
    observed = outcome >= 0
    known = alpha_row[:cutoff] >= 0
    agree = alpha_row[:cutoff] == 1.0
    early_live = agree if direction == 1 else ~agree
    use = observed & known
    win = outcome > 0
    el = use & early_live
    ed = use & ~early_live
    pre_n_el = np.cumsum(el)
    pre_k_el = np.cumsum(el & win)
    pre_n_ed = np.cumsum(ed)
    pre_k_ed = np.cumsum(ed & win)
    total_n_el = float(pre_n_el[-1])
    total_k_el = float(pre_k_el[-1])
    total_n_ed = float(pre_n_ed[-1])
    total_k_ed = float(pre_k_ed[-1])
    out = np.zeros(len(candidates))
    for i, c in enumerate(candidates):
        n_el_b = float(pre_n_el[c - 1]) if c else 0.0
        k_el_b = float(pre_k_el[c - 1]) if c else 0.0
        n_ed_b = float(pre_n_ed[c - 1]) if c else 0.0
        k_ed_b = float(pre_k_ed[c - 1]) if c else 0.0
        # Live level is early-live before c and early-dead after c.
        k_live = k_el_b + (total_k_ed - k_ed_b)
        n_live = n_el_b + (total_n_ed - n_ed_b)
        k_dead = k_ed_b + (total_k_el - k_el_b)
        n_dead = n_ed_b + (total_n_el - n_el_b)
        out[i] = _saturated_ll(np.array([k_dead, k_live]), np.array([n_dead, n_live]))
    return out


def implied_u(p_same, p_swap, p_true):
    denom = p_same - p_swap
    if abs(denom) < 0.05:
        return None
    return float((p_same - p_true) / denom)


def early_level_probs(machine, prior):
    """Posterior mean of each level using only the pre-change window, MAP pair and direction."""
    cutoff = machine["cutoff"]
    early = np.arange(cutoff) < int(0.40 * cutoff)
    rule = machine["rule"]
    n, k = _cell_counts(machine["seq"], machine["alpha"][rule["pair"]], rule["direction"], early, None)
    out = {}
    for c_i, cell in enumerate(CELLS):
        mu, sd = mu_sd(prior, machine["type"], cell, machine["signal"])
        _, ep, _, _ = level_stats(k[c_i], n[c_i], mu, sd)
        out[cell] = ep
        out[cell + "_n"] = float(n[c_i])
    agree = machine["alpha"][rule["pair"], cutoff]
    if agree < 0:
        return None
    early_live = (agree == 1.0) if rule["direction"] == 1 else (agree == 0.0)
    out["p_same"] = out["live"] if early_live else out["dead"]
    out["p_swap"] = out["dead"] if early_live else out["live"]
    return out


def ols_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    dof = max(len(y) - 2, 1)
    sigma = float(np.sqrt(np.sum(resid ** 2) / dof))
    xtx = np.linalg.inv(x.T @ x)
    se = sigma * np.sqrt(np.diag(xtx))
    return {"intercept": float(beta[0]), "slope": float(beta[1]), "se": float(se[1]), "sigma": sigma, "n": int(len(y))}


def quantiles(values, ps=(0.1, 0.25, 0.5, 0.75, 0.9)):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(values) == 0:
        return {"n": 0}
    return {"n": int(len(values)), "mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            **{f"p{int(100 * p)}": float(np.quantile(values, p)) for p in ps}}


def residual_shape(residuals):
    r = np.asarray(residuals, dtype=float)
    if len(r) < 5:
        return {"n": int(len(r))}
    centered = r - r.mean()
    sd = float(r.std(ddof=1))
    skew = float(np.mean(centered ** 3) / sd ** 3) if sd > 0 else 0.0
    kurt = float(np.mean(centered ** 4) / sd ** 4 - 3.0) if sd > 0 else 0.0
    return {"n": int(len(r)), "mean": float(r.mean()), "sd": sd, "skew": skew, "excess_kurtosis": kurt,
            "min": float(r.min()), "max": float(r.max())}


def score_predictions(p_by_id, sizes, targets, groups, reference):
    preds = {qid: dist_from_p(p, sizes[qid]) for qid, p in p_by_id.items()}
    scored = contest_score(preds, targets, groups, reference, max_points=60)
    return {
        "points": scored["technical_points"],
        "R": scored["mean_loss_ratio"],
        "ratios": {c: scored["by_condition"][c]["loss_ratio"] for c in CONDITIONS},
    }, preds


def bootstrap_points(preds, targets, groups, reference, repeats=2000, seed=314159):
    """Episode bootstrap of official points. Matches contest_score at the full sample."""
    by_q = {t.query_id: t for t in targets}
    episodes = defaultdict(lambda: defaultdict(list))
    for t in targets:
        episodes[groups[t.query_id]][t.episode_id].append(t.query_id)
    ep_err = {}
    ref_err = {}
    for cond, eps in episodes.items():
        for eid, qids in eps.items():
            ep_err[(cond, eid)] = float(np.mean([np.sum((preds[q] - by_q[q].probabilities) ** 2) for q in qids]))
            ref_err[(cond, eid)] = float(np.mean([np.sum((np.asarray(reference[q]) - by_q[q].probabilities) ** 2) for q in qids]))
    rng = np.random.default_rng(seed)
    points = np.empty(repeats)
    conds = list(CONDITIONS)
    pools = {c: np.array(list(episodes[c])) for c in conds}
    for i in range(repeats):
        ratios = []
        for c in conds:
            draw = rng.choice(pools[c], size=len(pools[c]), replace=True)
            num = np.mean([ep_err[(c, e)] for e in draw])
            den = np.mean([ref_err[(c, e)] for e in draw])
            ratios.append(num / max(den, 1e-9))
        points[i] = 60.0 / (1.0 + float(np.mean(ratios)))
    points.sort()
    lo = float(np.quantile(points, 0.025))
    hi = float(np.quantile(points, 0.975))
    return {"low": lo, "high": hi, "repeats": repeats}


def rmse_table(p_hat, machines):
    err = defaultdict(list)
    for m in machines:
        e = (p_hat[m["qid"]] - m["p_win"]) ** 2
        err[("condition", m["condition"])].append(e)
        err[("type", m["type"])].append(e)
        err[("both", m["condition"] + "|" + m["type"])].append(e)
    return {f"{a}:{b}": float(np.sqrt(np.mean(v))) for (a, b), v in sorted(err.items())}


def prepare(public, split, with_targets):
    queries, history = load_inputs(public / split)
    cond_path = public / split / "conditions.csv"
    cond = {r["query_id"]: r["condition"] for r in read_rows(cond_path, ("query_id", "condition"))} if cond_path.exists() else {}
    targets = {t.query_id: np.asarray(t.probabilities, dtype=float) for t in read_targets(public / split / "targets.csv")} if with_targets else {}
    seqs = build_sequences(queries, history)
    fold_of = {}
    if cond:
        by_c = defaultdict(set)
        for q in queries:
            by_c[cond[q.query_id]].add(q.episode_id)
        for episodes in by_c.values():
            for i, eid in enumerate(sorted(episodes)):
                fold_of[eid] = "A" if i % 2 == 0 else "B"
    machines = []
    for q in queries:
        seq = seqs[(q.episode_id, q.machine_id)]
        alpha = pair_alpha(seq, q.cutoff)
        regime = cond.get(q.query_id, "stable")
        rule = map_pair(seq, alpha, q.cutoff, window_for(regime, q.cutoff))
        row = {
            "qid": q.query_id,
            "episode": q.episode_id,
            "type": q.machine_type,
            "signal": int(q.signal),
            "cutoff": int(q.cutoff),
            "condition": regime,
            "fold": fold_of.get(q.episode_id, "A"),
            "seq": seq,
            "alpha": alpha,
            "rule": rule,
        }
        if q.query_id in targets:
            row["target"] = targets[q.query_id]
            row["p_win"] = float(1.0 - targets[q.query_id][0])
            row["logit"] = float(np.log(np.clip(row["p_win"], 1e-8, 1 - 1e-8) / (1 - np.clip(row["p_win"], 1e-8, 1 - 1e-8))))
            row["sizes"] = true_sizes(targets[q.query_id])
        machines.append(row)
    return machines, queries


def predict_fold(machines, prior, mode, shifts=None, schedules=None, flips=None):
    shifts = shifts or {}
    schedules = schedules or {}
    flips = flips or {}
    out = {}
    diag = {}
    for m in machines:
        shift = shifts.get(m["qid"], 0.0)
        sched = schedules.get(m["episode"])
        flip = flips.get(m["episode"])
        ramp = (RAMP_START_BACK, RAMP_END_BACK)
        u_cut = 1.0
        flip_at = None
        if sched and mode == "blend":
            # start_back = anchor, end_back = anchor - length. The pure tail is t >= cutoff - end_back.
            # u(cutoff) = clip(anchor / length, 0, 1). A negative end_back means the ramp is still running.
            ramp = (sched["anchor"], sched["anchor"] - sched["length"])
            u_cut = sched["u"]
        if flip and m["condition"] == "shared_change":
            flip_at = flip["c"]
        pred = predict_machine(
            m["seq"], m["alpha"], m["cutoff"], prior, m["signal"], m["type"], m["condition"],
            mode, flip_at=flip_at, ramp=ramp if mode == "blend" else (RAMP_START_BACK, RAMP_END_BACK),
            u_cut=u_cut if mode == "blend" else 1.0, shift=shift,
        )
        out[m["qid"]] = pred["p"]
        diag[m["qid"]] = pred
    return out, diag


def cross_predict(machines, priors_by_fold, mode, shifts_by_fold=None, schedules=None, flips=None):
    """Fit-fold prior predicts the other fold. `priors_by_fold['A']` was fit on A and scores B."""
    out = {}
    diag = {}
    shifts_by_fold = shifts_by_fold or {"A": {}, "B": {}}
    for held in ("A", "B"):
        source = "B" if held == "A" else "A"
        subset = [m for m in machines if m["fold"] == held]
        # The prior named by the fit fold is applied to the other fold.
        pred, info = predict_fold(subset, priors_by_fold[source], mode, shifts_by_fold.get(source, {}), schedules, flips)
        out.update(pred)
        diag.update(info)
    return out, diag


def run(public, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    val, _ = prepare(public, "validation", True)
    train, _ = prepare(public, "train", False)
    targets = read_targets(public / "validation" / "targets.csv")
    groups = {m["qid"]: m["condition"] for m in val}
    reference = read_submission(public / "validation" / "reference.csv", {t.query_id for t in targets})
    sizes = {m["qid"]: m["sizes"] for m in val}
    by_id = {m["qid"]: m for m in val}

    # Regime label: drift and shared have already flipped by the cutoff. Stable and short have not.
    flipped_regime = {"stable": False, "short_history": False, "drift": True, "shared_change": True}
    id_stats = {}
    for cond in CONDITIONS:
        subset = [m for m in val if m["condition"] == cond]
        id_stats[cond] = {
            "n": len(subset),
            "median_pair_gap_nats": float(np.median([m["rule"]["gap"] for m in subset])),
            "frac_gap_under_3": float(np.mean([m["rule"]["gap"] < 3 for m in subset])),
            "frac_uncertain": float(np.mean([m["rule"]["uncertain"] for m in subset])),
            "missing_cutoff_cell": int(sum(
                cutoff_cell(m["alpha"][m["rule"]["pair"]], m["cutoff"], m["rule"]["direction"], False) is None for m in subset
            )),
            "pair_counts": {str(PAIRS[i]): int(sum(m["rule"]["pair"] == i for m in subset)) for i in range(3)},
        }

    priors = {}
    shared_priors = {}
    nosig_priors = {}
    fit_meta = {}
    for fold in ("A", "B"):
        pool = [m for m in val if m["fold"] == fold]
        rows, meta = label_rows(pool, flipped_regime, ("stable", "short_history"))
        all_rows, all_meta = label_rows(pool, flipped_regime, CONDITIONS)
        priors[fold] = fit_prior(rows)
        shared_priors[fold] = fit_shared_slope(rows)
        nosig_priors[fold] = fit_no_signal(rows)
        fit_meta[fold] = {"stable_short": meta, "all_conditions": all_meta, "all_condition_prior": _dump_prior(fit_prior(all_rows))}
    train_prior, train_corr = train_level_prior(train)

    # Continuity: held-out logit residual around the stable/short prior, MAP cell.
    resid_buckets = defaultdict(list)
    mid_band = defaultdict(int)
    mid_n = defaultdict(int)
    for m in val:
        source = "B" if m["fold"] == "A" else "A"
        cell = cutoff_cell(
            m["alpha"][m["rule"]["pair"]], m["cutoff"], m["rule"]["direction"], flipped_regime[m["condition"]]
        )
        if cell is None or m["rule"]["uncertain"]:
            continue
        mu, sd = mu_sd(priors[source], m["type"], cell, m["signal"])
        resid_buckets[(m["condition"], m["type"], cell)].append(m["logit"] - mu)
        resid_buckets[("all", m["type"], cell)].append(m["logit"] - mu)
        other = "live" if cell == "dead" else "dead"
        mu_other, _ = mu_sd(priors[source], m["type"], other, m["signal"])
        mid = 0.5 * (mu + mu_other)
        # Closer to the midpoint than to the assigned cell: a half-finished blend, or a bad label.
        mid_n[m["condition"]] += 1
        if abs(m["logit"] - mid) < abs(m["logit"] - mu):
            mid_band[m["condition"]] += 1
    continuity = {
        "residual_logit": {f"{a}|{b}|{c}": residual_shape(v) for (a, b, c), v in sorted(resid_buckets.items())},
        "fraction_closer_to_midpoint_than_cell": {c: mid_band[c] / mid_n[c] for c in CONDITIONS if mid_n[c]},
    }

    print("fitting drift schedules and shared flips", flush=True)
    # Drift schedules and shared flips from histories only.
    by_episode = defaultdict(list)
    for m in val:
        by_episode[m["episode"]].append(m)
    schedules = {}
    flips = {}
    for eid, group in by_episode.items():
        cond = group[0]["condition"]
        if cond == "drift":
            schedules[eid] = drift_schedule(group)
        elif cond == "shared_change":
            flips[eid] = shared_flip(group)
    # Same measurements on train, for the train-only schedule the filter should use.
    train_eps = defaultdict(list)
    for m in train:
        train_eps[m["episode"]].append(m)
    train_drift = [drift_schedule(g) for g in train_eps.values() if g[0]["condition"] == "drift"]
    train_flip = [shared_flip(g) for g in train_eps.values() if g[0]["condition"] == "shared_change"]

    # Implied u at the cutoff from pre-change level estimates vs the true probability.
    # Prior is the other fold's stable/short prior, so the target is not in the prior.
    u_rows = defaultdict(list)
    for m in val:
        source = "B" if m["fold"] == "A" else "A"
        levels = early_level_probs(m, priors[source])
        if levels is None:
            continue
        u = implied_u(levels["p_same"], levels["p_swap"], m["p_win"])
        u_rows[m["condition"]].append(u)

    # Cutoff slope and episode shock, cross-fit.
    # Episode offsets come from the train prior, so they do not use validation targets.
    train_pred, train_diag = predict_fold(val, train_prior, "shrink", flips=flips)
    offsets = {}
    for m in val:
        info = train_diag[m["qid"]]
        cell = info["map_cell"]
        if cell not in CELLS:
            offsets[m["qid"]] = 0.0
            continue
        mu, _ = mu_sd(train_prior, m["type"], cell, m["signal"])
        offsets[m["qid"]] = info["map_theta"] - mu
    peer = {}
    for m in val:
        others = [offsets[o["qid"]] for o in by_episode[m["episode"]] if o["qid"] != m["qid"]]
        peer[m["qid"]] = float(np.mean(others)) if others else 0.0
    cutoff_fit = {}
    episode_fit = {}
    shifts = {"A": {}, "B": {}}
    episode_shifts = {"A": {}, "B": {}}
    for fit_fold, test_fold in (("A", "B"), ("B", "A")):
        fit_rows = []
        ep_x, ep_y = [], []
        for m in val:
            if m["fold"] != fit_fold or m["rule"]["uncertain"]:
                continue
            cell = cutoff_cell(m["alpha"][m["rule"]["pair"]], m["cutoff"], m["rule"]["direction"], flipped_regime[m["condition"]])
            if cell is None or m["condition"] not in ("stable", "short_history"):
                continue
            mu, _ = mu_sd(priors[fit_fold], m["type"], cell, m["signal"])
            fit_rows.append(((m["cutoff"] - 224.0) / 16.0, m["logit"] - mu))
            ep_x.append(peer[m["qid"]])
            ep_y.append(m["logit"] - mu)
        cutoff_fit[fit_fold] = ols_slope([r[0] for r in fit_rows], [r[1] for r in fit_rows]) if len(fit_rows) > 5 else None
        episode_fit[fit_fold] = ols_slope(ep_x, ep_y) if len(ep_x) > 5 else None
        slope_c = cutoff_fit[fit_fold]["slope"] if cutoff_fit[fit_fold] else 0.0
        slope_e = episode_fit[fit_fold]["slope"] if episode_fit[fit_fold] else 0.0
        for m in val:
            if m["fold"] != test_fold:
                continue
            shifts[fit_fold][m["qid"]] = slope_c * (m["cutoff"] - 224.0) / 16.0
            episode_shifts[fit_fold][m["qid"]] = slope_e * peer[m["qid"]]

    # Within-episode residual correlation on stable held-out targets.
    icc = {}
    for cond in CONDITIONS:
        num = den = n_pairs = 0.0
        for eid, group in by_episode.items():
            members = []
            for m in group:
                if m["condition"] != cond or m["rule"]["uncertain"]:
                    continue
                cell = cutoff_cell(m["alpha"][m["rule"]["pair"]], m["cutoff"], m["rule"]["direction"], flipped_regime[m["condition"]])
                if cell is None:
                    continue
                source = "B" if m["fold"] == "A" else "A"
                mu, _ = mu_sd(priors[source], m["type"], cell, m["signal"])
                members.append(m["logit"] - mu)
            if len(members) < 3:
                continue
            members = np.asarray(members)
            num += float(np.sum(members[:, None] * members[None, :]) - np.sum(members ** 2))
            den += float((len(members) - 1) * np.sum(members ** 2))
            n_pairs += len(members) * (len(members) - 1)
        icc[cond] = {"pairwise_residual_corr": float(num / den) if den else None, "n_ordered_pairs": int(n_pairs)}

    specs = [
        ("shrink_separate", priors, "shrink"),
        ("shrink_shared_slope", shared_priors, "shrink"),
        ("shrink_no_signal", nosig_priors, "shrink"),
        ("point_separate", priors, "point"),
        ("point_no_signal", nosig_priors, "point"),
        ("shrink_unflipped", priors, "unflipped"),
        ("shrink_blend_u", priors, "blend"),
        ("shrink_train_prior", {fold: train_prior for fold in ("A", "B")}, "shrink"),
        ("shrink_plus_cutoff", priors, "shrink"),
        ("shrink_plus_episode", priors, "shrink"),
    ]
    results = {}
    predictions = {}
    win_rmse = {}
    print("scoring generators", flush=True)
    for name, prior_map, mode in specs:
        extra = None
        if name == "shrink_plus_cutoff":
            extra = shifts
        elif name == "shrink_plus_episode":
            extra = episode_shifts
        pred, diag = cross_predict(
            val, prior_map, mode,
            shifts_by_fold=extra,
            schedules=schedules,
            flips=flips,
        )
        summary, packed = score_predictions(pred, sizes, targets, groups, reference)
        summary["win_rmse"] = rmse_table(pred, val)
        if name in ("shrink_separate", "point_separate", "shrink_unflipped", "shrink_train_prior"):
            summary["ci95"] = bootstrap_points(packed, targets, groups, reference, repeats=2000)
        results[name] = summary
        predictions[name] = pred
        win_rmse[name] = summary["win_rmse"]
        print(name, round(summary["points"], 4), {k: round(v, 4) for k, v in summary["ratios"].items()}, flush=True)

    oracle = {m["qid"]: m["p_win"] for m in val}
    oracle_summary, _ = score_predictions(oracle, sizes, targets, groups, reference)
    anchor, _ = score_predictions(
        {qid: 1.0 - float(reference[qid][0]) for qid in reference},
        {qid: true_sizes(reference[qid]) for qid in reference},
        targets, groups, reference,
    )
    # The reference row is the pooled starter. Scoring it against itself is the 30-point anchor.
    ref_as_pred = {qid: np.asarray(reference[qid], dtype=float) for qid in reference}
    anchor_official = contest_score(ref_as_pred, targets, groups, reference, max_points=60)

    v13_path = Path("/Users/saisreed/Desktop/Coding/TamuDatathon26-PSAZZ/worktrees/_tm/v13/mainFiles/runs/v13/crossfit_validation_predictions.csv")
    v13_points = None
    if v13_path.exists():
        v13 = read_submission(v13_path, {t.query_id for t in targets})
        v13_points = contest_score(v13, targets, groups, reference, max_points=60)["technical_points"]

    # How far is the history posterior from the truth, by slice, for the headline.
    # Once the cutoff cell is updated from its own counts, the other cell's residual
    # should not still predict the probability error if a bivariate shock is redundant.
    other_x, other_y = [], []
    headline_for_resid = predictions["shrink_separate"]
    for m in val:
        if m["condition"] == "shared_change":
            continue
        cell = cutoff_cell(
            m["alpha"][m["rule"]["pair"]], m["cutoff"], m["rule"]["direction"], flipped_regime[m["condition"]]
        )
        if cell is None:
            continue
        source = "B" if m["fold"] == "A" else "A"
        cutoff = m["cutoff"]
        if m["condition"] == "drift":
            ticks = np.arange(cutoff)
            use = (ticks < cutoff - RAMP_START_BACK) | (ticks >= cutoff - RAMP_END_BACK)
            flip_at = cutoff - RAMP_END_BACK
        else:
            use = np.ones(cutoff, dtype=bool)
            flip_at = None
        n, k = _cell_counts(m["seq"], m["alpha"][m["rule"]["pair"]], m["rule"]["direction"], use, flip_at)
        oi = 1 if cell == "dead" else 0
        if n[oi] < 10 or k[oi] <= 0 or k[oi] >= n[oi]:
            continue
        other = "live" if oi == 1 else "dead"
        mu, _ = mu_sd(priors[source], m["type"], other, m["signal"])
        mle = math.log((k[oi] + 0.5) / (n[oi] - k[oi] + 0.5))
        other_x.append(mle - mu)
        other_y.append(m["p_win"] - headline_for_resid[m["qid"]])
    other_x = np.asarray(other_x)
    other_y = np.asarray(other_y)
    other_cell_check = {
        "n": int(len(other_x)),
        "corr": float(np.corrcoef(other_x, other_y)[0, 1]) if len(other_x) > 5 else None,
        "rmse": float(np.sqrt(np.mean(other_y ** 2))) if len(other_y) else None,
    }
    headline = predictions["shrink_separate"]
    gap = {}
    for cond in CONDITIONS:
        for typ in TYPES:
            members = [m for m in val if m["condition"] == cond and m["type"] == typ]
            if not members:
                continue
            err = np.array([headline[m["qid"]] - m["p_win"] for m in members])
            gap[f"{cond}|{typ}"] = {"n": len(members), "rmse": float(np.sqrt(np.mean(err ** 2))), "mean_bias": float(err.mean())}

    # Jackpot live / dead centers the filter has been missing.
    centers = {fold: _dump_prior(priors[fold]) for fold in ("A", "B")}
    pooled_rows, _ = label_rows(val, flipped_regime, CONDITIONS)
    pooled_prior = fit_prior(pooled_rows)
    ramp_ends = [sch["anchor"] - sch["length"] for sch in schedules.values() if sch.get("n_machines")]

    report = {
        "identification": id_stats,
        "prior_fit_counts": fit_meta,
        "prior_crossfit": centers,
        "prior_train_debiased": _dump_prior(train_prior),
        "prior_pooled_targets_descriptive": _dump_prior(pooled_prior),
        "drift_ramp_end_ticks_before_cutoff": quantiles(ramp_ends),
        "train_level_correlation": train_corr,
        "continuity": continuity,
        "drift_validation": _summ_schedules(list(schedules.values())),
        "drift_train": _summ_schedules(train_drift),
        "drift_episodes": [
            {"episode": eid, **{k: v for k, v in sch.items() if k != "late_ll"}, "late_ll": sch.get("late_ll")}
            for eid, sch in schedules.items()
        ],
        "shared_validation": _summ_flips(list(flips.values())),
        "shared_train": _summ_flips(train_flip),
        "implied_u_from_targets": {c: quantiles(vs) for c, vs in u_rows.items()},
        "cutoff_regression": cutoff_fit,
        "episode_regression": episode_fit,
        "episode_residual_corr": icc,
        "other_cell_residual_vs_win_error": other_cell_check,
        "ceilings": results,
        "oracle_true_p_and_true_sizes": oracle_summary,
        "pooled_anchor_points": anchor_official["technical_points"],
        "v13_crossfit_points": v13_points,
        "headline_error_by_condition_type": gap,
        "notes": {
            "headline": "shrink_separate",
            "prior": "Fit on the other fold's stable and short_history targets only, so drift and shared targets test transfer.",
            "sizes": "Each scored vector uses that query's true conditional size distribution.",
            "condition_known": "Ceilings use the true regime to choose flipped vs unflipped. shrink_unflipped is the cost of ignoring the change.",
        },
    }
    path = out_dir / "winform.json"
    path.write_text(json.dumps(report, indent=2, default=_json) + "\n")
    print("wrote", path, flush=True)
    return report


def _dump_prior(prior):
    return {f"{typ}|{cell}": prior[(typ, cell)] for typ in TYPES for cell in CELLS if (typ, cell) in prior}


def _summ_schedules(rows):
    if not rows:
        return {}
    def col(key):
        return quantiles([r[key] for r in rows if key in r])
    late_u1 = [r["late_ll"]["u1"] - r["late_ll"]["u09"] for r in rows if r.get("late_ll") and r["late_ll"].get("n_ticks")]
    return {
        "n": len(rows),
        "anchor": col("anchor"),
        "length": col("length"),
        "u_at_cutoff": col("u"),
        "frac_u_at_least_0.99": float(np.mean([r["u"] >= 0.99 for r in rows])),
        "mean_ll_per_machine_vs_no_flip": float(np.mean([r.get("ll_per_machine_vs_no_flip", 0.0) for r in rows])),
        "late20_u1_minus_u09_total": float(np.sum(late_u1)) if late_u1 else None,
    }


def _summ_flips(rows):
    if not rows:
        return {}
    return {
        "n": len(rows),
        "fraction": quantiles([r["fraction"] for r in rows]),
        "ticks_before_cutoff": quantiles([r["ticks_before"] for r in rows]),
        "min_ticks_before": int(min(r["ticks_before"] for r in rows)),
        "mean_gain_vs_no_flip": float(np.mean([r.get("gain_vs_no_flip", r.get("gain", 0.0)) for r in rows])),
        "median_sharp_width_ticks": float(np.median([r.get("sharp_width", 0) for r in rows])),
    }


def _json(value):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "public")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "winform")
    args = parser.parse_args()
    run(args.data, args.out)


if __name__ == "__main__":
    main()
