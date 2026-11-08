import numpy as np
from scipy.optimize import minimize


def softmax(S):
    S = S - S.max(1, keepdims=True); E = np.exp(S); return E / E.sum(1, keepdims=True)


def fit_gate(Pm, Z, q, ww, lam):
    """Pm (M,n,5) member predictions; Z (n,k) standardised observable features (k may be 0).
    weights_m(i) = softmax_m(b_m + Z_i . g_m); P = sum_m weights_m P_m. Loss = sum ww |P-q|^2 + lam |g|^2
    (ww normalised to sum 1; intercepts unpenalised). Returns coefficient matrix (M,k+1)."""
    M, n, _ = Pm.shape
    X = np.column_stack([np.ones(n), Z])
    k1 = X.shape[1]
    ww = ww / ww.sum()

    def f(x):
        B = x.reshape(M, k1)
        W = softmax(X @ B.T)                       # (n,M)
        P = np.einsum("nm,mnc->nc", W, Pm)
        r = P - q
        L = (ww * (r ** 2).sum(1)).sum() + lam * (B[:, 1:] ** 2).sum()
        dP = 2 * ww[:, None] * r                   # (n,5)
        dot = np.einsum("mnc,nc->nm", Pm, dP)      # (n,M)
        dS = W * (dot - (W * dot).sum(1, keepdims=True))
        G = dS.T @ X
        G[:, 1:] += 2 * lam * B[:, 1:]
        return L, G.ravel()

    r = minimize(f, np.zeros(M * k1), jac=True, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-13, "gtol": 1e-9})
    return r.x.reshape(M, k1)


def pred_gate(B, Pm, Z):
    X = np.column_stack([np.ones(Pm.shape[1]), Z])
    return np.einsum("nm,mnc->nc", softmax(X @ B.T), Pm)
