"""Sharper toy: continuous score highly correlated with selection propensity.
V = m + 0.01*N(0,1),  m=min(T,1), T~Exp(1), C~Exp(1.5) indep;  p(zeta)=e^{-1.5 m}, w=1/p.
High V <-> high m <-> low retention  =>  unweighted quantile biased LOW => undercoverage.
Oracle-weighted (test atom = test's own w) should hold >= 0.90 and be near-tight.
"""

import numpy as np

rng = np.random.default_rng(7)
lam, tstar, alpha = 1.5, 1.0, 0.1


def draw(nsub, trials):
    T = rng.exponential(1.0, size=(trials, nsub))
    C = rng.exponential(1.0 / lam, size=(trials, nsub))
    m = np.minimum(T, tstar)
    delta = m <= C
    w = np.exp(lam * m)
    V = m + 0.01 * rng.standard_normal((trials, nsub))
    return V, w, delta


def wq(V, w, delta, w_test):
    trials = V.shape[0]
    q = np.full(trials, np.inf)
    for t in range(trials):
        idx = delta[t]
        if not idx.any():
            continue
        v, ww = V[t, idx], w[t, idx]
        o = np.argsort(v)
        v, ww = v[o], ww[o]
        cs = np.cumsum(ww) / (ww.sum() + w_test[t])
        h = np.nonzero(cs >= 1 - alpha)[0]
        if h.size:
            q[t] = v[h[0]]
    return q


for n_cal in (20, 100):
    trials = 200_000
    V, w, d = draw(n_cal, trials)
    Vt, wt, _ = draw(1, trials)
    Vt, wt = Vt[:, 0], wt[:, 0]
    cov_w = (Vt <= wq(V, w, d, wt)).mean()
    cov_u = (Vt <= wq(V, np.ones_like(w), d, np.ones(trials))).mean()
    se = np.sqrt(0.9 * 0.1 / trials)
    print(
        f"n={n_cal:4d}: oracle-weighted={cov_w:.4f}  unweighted-on-I={cov_u:.4f}"
        f"  (target 0.900, se≈{se:.4f})"
    )
