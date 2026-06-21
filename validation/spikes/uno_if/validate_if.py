"""Validate the closed-form influence-function CI for crforest's Uno/Wolbers CR
concordance against bootstrap. Staged: (1) pair term vs fixed-weight bootstrap,
(2) full IF (pairs + censoring-G) vs full bootstrap.

Naive O(n^2). Continuous times => Branch B (tied case-times) empty; only the
asymmetric case-vs-comparator (A) and case-vs-competing (C) families are active.
Grounded in metrics.py branches A/C and survC1 unoCW/kmcens.
"""

import sys

import numpy as np

from comprisk.metrics import (
    _km_censor_fit,
    compute_uno_weights,
    concordance_index_uno_cr,
)

CAUSE = 1


def gen(n, seed, signal=0.0):
    rng = np.random.default_rng(seed)
    p = rng.normal(size=n)  # continuous risk scores, no ties
    # informative DGP when signal>0: high p => earlier cause-1 time => C>0.5
    rate1 = np.exp(signal * p)
    t1 = rng.exponential(1.0 / rate1) + rng.random(n) * 1e-3
    tc = rng.exponential(1.0, n)
    to = rng.exponential(1.5, n)
    t = np.minimum.reduce([t1, tc, to])
    e = np.where(t == t1, 1, np.where(t == to, 2, 0)).astype(int)
    return t, e, p


# ---------------------------------------------------------------- estimator mirror
def naive_ND(t, e, p, w):
    """Reproduce metrics.py branches A + C (continuous times). Return N, D and
    per-subject Ndot/Ddot (Hajek projection: subject counted in both pair slots)."""
    n = len(t)
    w1 = np.sqrt(w)
    N = D = 0.0
    Ndot = np.zeros(n)
    Ddot = np.zeros(n)
    keep = w > 0  # library drops weight-0 rows entirely (keep_mask)
    case = np.flatnonzero((e == CAUSE) & keep)
    comp = np.flatnonzero((e > 0) & (e != CAUSE) & keep)
    for i in case:
        # Branch A: j with t_j > t_i (any kept type)
        jA = np.flatnonzero((t > t[i]) & keep)
        if jA.size:
            hh = (p[jA] < p[i]).astype(float)  # no p-ties
            nc = 2.0 * w[i] * hh
            dc = 2.0 * w[i] * np.ones(jA.size)
            N += nc.sum()
            D += dc.sum()
            Ndot[i] += nc.sum()
            Ddot[i] += dc.sum()
            np.add.at(Ndot, jA, nc)
            np.add.at(Ddot, jA, dc)
        # Branch C: competing j with t_j <= t_i
        jC = comp[t[comp] <= t[i]]
        if jC.size:
            hh = (p[jC] < p[i]).astype(float)
            ww = 2.0 * w1[i] * w1[jC]
            nc = ww * hh
            dc = ww
            N += nc.sum()
            D += dc.sum()
            Ndot[i] += nc.sum()
            Ddot[i] += dc.sum()
            np.add.at(Ndot, jC, nc)
            np.add.at(Ddot, jC, dc)
    return N, D, Ndot, Ddot


# ---------------------------------------------------------------- censoring-G IF
def cens_influence(t, e):
    """a_k(t_grid): IID influence of -log G-hat under crforest events-first KM.
    a_k(s) increment = [dN^other_k(s) - Y^r_k(s) dLambda^c(s)] / (r(s)/n),
    r(s)=#{X>=s} - #{X=s,cause}.  Returns (knots, A) with A[k, m]=a_k(knots[m])."""
    n = len(t)
    knots = np.unique(t)
    K = knots.size
    # counts at each knot
    nrisk = np.array([(t >= s).sum() for s in knots], float)
    d_one = np.array([((t == s) & (e == CAUSE)).sum() for s in knots], float)
    d_oth = np.array([((t == s) & (e != CAUSE)).sum() for s in knots], float)
    r = nrisk - d_one
    dLam = np.zeros_like(r)
    ok = (d_oth > 0) & (r > 0)
    dLam[ok] = d_oth[ok] / r[ok]
    pi_r = r / n  # at-risk fraction in the events-first r-set
    incr = np.zeros((n, K))
    for m, s in enumerate(knots):
        if pi_r[m] <= 0:
            continue
        dN_oth = ((t == s) & (e != CAUSE)).astype(float)  # k censored/comp at s
        Yr = (t >= s).astype(float) - ((t == s) & (e == CAUSE))  # in r-set at s
        incr[:, m] = (dN_oth - Yr * dLam[m]) / pi_r[m]
    A = np.cumsum(incr, axis=1)  # a_k(knots[m])
    return knots, A


def a_at(knots, A, times):
    """a_k evaluated at each query time (step function, right-cont at knots)."""
    idx = np.searchsorted(knots, times, side="right") - 1
    idx = np.clip(idx, 0, knots.shape[1 - 1] if False else len(knots) - 1)
    return A[:, idx]  # (n_k, n_query)


def phi_G(t, e, p, w, C, knots, A):
    """Censoring-G influence phi^{(G)}_k for all k. delta_k w_i = 2 w_i a_k(T_i);
    delta_k w1_i = w1_i a_k(T_i)."""
    n = len(t)
    w1 = np.sqrt(w)
    keep = w > 0
    case = np.flatnonzero((e == CAUSE) & keep)
    comp = np.flatnonzero((e > 0) & (e != CAUSE) & keep)
    phi = np.zeros(n)
    # Branch A: weight w_i (case only). centered partner mass S_i = sum_j 2(h-C).
    for i in case:
        jA = np.flatnonzero((t > t[i]) & keep)
        if jA.size:
            S = (2.0 * ((p[jA] < p[i]).astype(float) - C)).sum()
            # contribution to every k: (1/D)* S * delta_k w_i = (1/D)*S*2 w_i a_k(T_i)
            phi += S * 2.0 * w[i] * A[:, np.searchsorted(knots, t[i], side="right") - 1]
    # Branch C: weight w1_i w1_j, two endpoints
    for i in case:
        jC = comp[t[comp] <= t[i]]
        if jC.size:
            cen = 2.0 * ((p[jC] < p[i]).astype(float) - C)  # per-pair centered
            ai = A[:, np.searchsorted(knots, t[i], side="right") - 1]  # (n,)
            mC = np.searchsorted(knots, t[jC], side="right") - 1
            aj = A[:, mC]  # (n, |jC|)
            ww = w1[i] * w1[jC]  # (|jC|,)
            # sum over pairs: cen*ww*(a_k(Ti)+a_k(Tj))
            phi += (cen * ww).sum() * ai
            phi += (aj * (cen * ww)[None, :]).sum(axis=1)
    return phi  # NOT yet divided by D


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 700
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    signal = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    t, e, p = gen(n, seed=7, signal=signal)
    tau = float(np.quantile(t, 0.70))  # Uno/survC1 truncation: stabilise 1/G^2 tail

    def wts(tt, ee):
        ww = compute_uno_weights(tt, ee, gmin="none")
        ww[tt >= tau] = 0.0
        return ww

    w = wts(t, e)

    C_lib = concordance_index_uno_cr(e, t, p, cause=CAUSE, weights=w)
    N, D, Ndot, Ddot = naive_ND(t, e, p, w)
    C_mine = N / D
    print(
        f"[self-check] C_lib={C_lib:.6f}  C_naive={C_mine:.6f}  diff={abs(C_lib - C_mine):.2e}",
        flush=True,
    )

    # ---- phi^{(U)}: pair influence, G fixed. V-statistic IF carries a factor n:
    # phi^U_k = n*(Ndot_k - C*Ddot_k)/D   (ratio-of-V-stats Hajek projection).
    phiU = n * (Ndot - C_mine * Ddot) / D
    seU = np.sqrt((phiU**2).sum()) / n

    # ---- fixed-weight bootstrap (weights frozen at full-sample w)
    rng = np.random.default_rng(123)
    boot_fix = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        boot_fix[b] = concordance_index_uno_cr(e[idx], t[idx], p[idx], cause=CAUSE, weights=w[idx])
    se_fix = np.nanstd(boot_fix, ddof=1)
    print(
        f"[stage1 pairs ] closed seU={seU:.5f}  fix-wt boot={se_fix:.5f}  ratio={seU / se_fix:.3f}",
        flush=True,
    )

    # ---- phi^{(G)}: censoring-G influence
    knots, Aarr = cens_influence(t, e)
    phiG = phi_G(t, e, p, w, C_mine, knots, Aarr) / D
    phi = phiU + phiG
    se_full = np.sqrt((phi**2).sum()) / n

    # ---- a_k calibration check: Var(log Ghat(t*)) closed vs bootstrap
    from comprisk.metrics import _ghat_minus

    t_star = float(np.median(t[e == CAUSE]))
    m_star = np.searchsorted(knots, t_star, side="right") - 1
    var_logG_closed = (Aarr[:, m_star] ** 2).sum() / n**2

    # ---- full bootstrap (re-estimate G each resample) + record Ghat(t*-)
    boot_full = np.empty(B)
    logG_star = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        wb = wts(t[idx], e[idx])
        boot_full[b] = concordance_index_uno_cr(e[idx], t[idx], p[idx], cause=CAUSE, weights=wb)
        tu, G = _km_censor_fit(t[idx], e[idx])
        logG_star[b] = np.log(_ghat_minus(tu, G, np.array([t_star]))[0])
    se_boot = np.nanstd(boot_full, ddof=1)
    var_logG_boot = np.nanvar(logG_star, ddof=1)
    print(
        f"[a_k check    ] Var(logG(t*)) closed={var_logG_closed:.3e}  "
        f"boot={var_logG_boot:.3e}  ratio={var_logG_closed / var_logG_boot:.3f}",
        flush=True,
    )
    print(
        f"[stage2 full  ] closed se={se_full:.5f}  full boot={se_boot:.5f}  "
        f"ratio={se_full / se_boot:.3f}",
        flush=True,
    )
    print(
        f"[components    ] |phiU|={seU:.5f}  |phiG-add|={np.sqrt((phiG**2).sum()) / n:.5f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
