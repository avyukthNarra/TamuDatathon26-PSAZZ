"""Changepoint / regime weight builders for W2. Every function returns per-ROW weights (len q.ticks) for the
pair-ridge fit, using only ticks < cutoff."""
import numpy as np
import _mem as m

PAYS = (0, 1, 2, 5, 20)
_C = {}


def classes(g, K):
    """class index per tick (-1 = unobserved). K=2: loss/win, K=5: payout class."""
    out = np.full(len(g), -1)
    ok = ~np.isnan(g)
    if K == 2:
        out[ok] = (g[ok] > 0).astype(int)
    else:
        for k, p in enumerate(PAYS):
            out[ok & (g == p)] = k
    return out


def cum(cl, K):
    c = np.zeros((K, len(cl) + 1))
    for k in range(K):
        c[k, 1:] = np.cumsum(cl == k)
    return c


def bocpd_post(grids, c, h, alpha, K, rho=1.0):
    """grids: list of tick arrays (first = self). Posterior over regime start s in [0, c-1] at the cutoff.
    Adams-MacKay with Dirichlet-multinomial predictive; observations of several machines at the same tick are
    multiplied (log-lik scaled by rho for peers)."""
    cls = [classes(g[:c], K) for g in grids]
    cc = [cum(x, K) for x in cls]
    post = np.array([1.0])
    for t in range(c):
        lp = np.zeros(t + 1)
        for j, (x, cj) in enumerate(zip(cls, cc)):
            k = x[t]
            if k < 0:
                continue
            cnt = cj[:, t:t + 1] - cj[:, :t + 1]          # K x (t+1): counts in [s,t)
            pred = (alpha + cnt[k]) / (K * alpha + cnt.sum(0))
            lp += (1.0 if j == 0 else rho) * np.log(pred)
        pred = np.exp(lp)
        grow = post * pred
        new = np.empty(t + 2)
        new[:t + 1] = grow * (1 - h)
        new[t + 1] = grow.sum() * h
        post = new / new.sum()
    post = post[:c]
    return post / post.sum()


def surv_weights(q, post, floor):
    """weight of row tick u = P(regime start <= u)."""
    cdf = np.cumsum(post)
    w = cdf[np.minimum(q.ticks, len(cdf) - 1)]
    return np.maximum(w, floor) if floor > 0 else w


def start_weights(q, s, floor):
    return np.where(q.ticks >= s, 1.0, floor)


def adwin_start(q, delta, minseg):
    """ADWIN (Bifet-Gavalda) on the win-flag stream of observed ticks; returns tick where the final window starts."""
    obs_t = np.flatnonzero(~np.isnan(q.win))
    x = q.win[obs_t]
    lo = 0
    for n in range(1, len(x) + 1):
        while True:
            w = x[lo:n]
            L = len(w)
            if L < 2 * minseg:
                break
            cs = np.cumsum(w)
            k = np.arange(minseg, L - minseg + 1)
            n0, n1 = k.astype(float), (L - k).astype(float)
            mu0 = cs[k - 1] / n0
            mu1 = (cs[-1] - cs[k - 1]) / n1
            mh = 1.0 / (1.0 / n0 + 1.0 / n1)
            eps = np.sqrt(np.log(4.0 * L / delta) / (2.0 * mh))
            hit = np.flatnonzero(np.abs(mu0 - mu1) >= eps)
            if len(hit) == 0:
                break
            lo += int(k[hit[0]])        # drop the oldest segment up to the first violating cut
    return int(obs_t[lo]) if len(obs_t) else 0


def scan_stats(q, minseg):
    """Backward CUSUM scan and rescaled range on the win-flag stream.
    returns (z_scan, k_best_in_obs_from_end, n, range_stat)."""
    x = q.win[~np.isnan(q.win)]
    n = len(x)
    mu = x.mean()
    sd = max(np.sqrt(mu * (1 - mu)), 1e-3)
    d = x - mu
    rev = np.cumsum(d[::-1])                     # sum over last k obs
    k = np.arange(1, n + 1)
    if n > 2 * minseg:
        sel = (k >= minseg) & (k <= n - minseg)
        z = np.abs(rev) / (sd * np.sqrt(k * (1 - k / n)))
        zz = np.where(sel, z, 0)
        kb = int(np.argmax(zz)) + 1
        zs = float(zz.max())
    else:
        zs, kb = 0.0, 0
    cs = np.concatenate([[0], np.cumsum(d)])
    rng = (cs.max() - cs.min()) / (sd * np.sqrt(max(n, 1)))
    return zs, kb, n, float(rng)


def scan_start(q, kb):
    obs_t = np.flatnonzero(~np.isnan(q.win))
    return int(obs_t[len(obs_t) - kb]) if kb > 0 else 0


def hmm_gamma(q, K, stay, iters=10):
    """Bernoulli-emission K-state HMM on the win-flag stream (ticks < cutoff, missing = no emission), fixed
    deterministic init and `iters` Baum-Welch iterations. Returns (gamma[T,K] smoothed, pi_next[K])."""
    c = q.c
    x = q.win[:c]
    ok = ~np.isnan(x)
    xv = np.where(ok, x, 0.0)
    pbar = float(np.nanmean(x))
    off = (np.arange(K) - (K - 1) / 2.0) * 0.15
    b = np.clip(pbar + off, 0.03, 0.97)                       # P(win | state)
    A = np.full((K, K), (1 - stay) / (K - 1)); np.fill_diagonal(A, stay)
    pi0 = np.ones(K) / K
    for it in range(iters + 1):
        em = np.where(ok[:, None], np.where(xv[:, None] > 0, b[None, :], 1 - b[None, :]), 1.0)   # T x K
        al = np.empty((c, K)); sc = np.empty(c)
        a = pi0 * em[0]; sc[0] = a.sum(); al[0] = a / sc[0]
        for t in range(1, c):
            a = (al[t - 1] @ A) * em[t]; sc[t] = a.sum(); al[t] = a / sc[t]
        be = np.empty((c, K)); be[-1] = 1.0
        for t in range(c - 2, -1, -1):
            be[t] = (A @ (em[t + 1] * be[t + 1])) / sc[t + 1]
        gam = al * be
        gam /= gam.sum(1, keepdims=True)
        if it == iters:
            break
        xi = np.zeros((K, K))
        for t in range(c - 1):
            xi += (al[t][:, None] * A * (em[t + 1] * be[t + 1])[None, :]) / sc[t + 1]
        A = xi + 1e-3; A /= A.sum(1, keepdims=True)
        wgt = gam * ok[:, None]
        b = np.clip((wgt * xv[:, None]).sum(0) / np.maximum(wgt.sum(0), 1e-9), 0.03, 0.97)
        pi0 = gam[0]
    return gam, al[-1] @ A
