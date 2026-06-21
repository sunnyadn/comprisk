"""Gate 2: validate the closed-form IF with TIED times (Branch B active).

Discrete times => tied event times, incl. tied case-times (Branch B) and
same-time censored comparators in Branch A. Full naive IF over branches A+B+C
with tie handling, validated: self-check (naive C == library C) + IF se vs
bootstrap. Reuses the tie-correct censoring influence a_k from validate_if.
"""

import sys

import numpy as np
from validate_if import cens_influence

from comprisk.metrics import compute_uno_weights, concordance_index_uno_cr

CAUSE = 1


def gen_ties(n, seed, signal=1.2, grid=0.1):
    rng = np.random.default_rng(seed)
    p = rng.normal(size=n)
    t1 = rng.exponential(1.0 / np.exp(signal * p))
    tc = rng.exponential(5.0, n)  # light censoring, no truncation needed
    to = rng.exponential(2.0, n)  # competing
    t = np.minimum.reduce([t1, tc, to])
    e = np.where(t == t1, 1, np.where(t == to, 2, 0)).astype(int)
    t = np.round(t / grid) * grid + grid  # snap to grid => ties; +grid avoids 0
    return t, e, p


def _ak(knots, A, time):
    return A[:, np.searchsorted(knots, time, side="right") - 1]


def naive_full(t, e, p, w):
    """N, D, Ndot, Ddot over branches A+B+C with tie handling (mirrors metrics.py)."""
    n = len(t)
    w1 = np.sqrt(w)
    keep = w > 0
    case = np.flatnonzero((e == CAUSE) & keep)
    comp = np.flatnonzero((e > 0) & (e != CAUSE) & keep)
    N = D = 0.0
    Ndot = np.zeros(n)
    Ddot = np.zeros(n)

    def add(idx_i, idx_j, nc, dc):
        nonlocal N, D
        N += nc.sum()
        D += dc.sum()
        Ndot[idx_i] += nc.sum()
        Ddot[idx_i] += dc.sum()
        np.add.at(Ndot, idx_j, nc)
        np.add.at(Ddot, idx_j, dc)

    for i in case:
        # Branch A: t_j>t_i (any kept) OR t_j==t_i & censored
        jA = np.flatnonzero(((t > t[i]) | ((t == t[i]) & (e == 0))) & keep)
        if jA.size:
            hh = (p[jA] < p[i]).astype(float) + 0.5 * (p[jA] == p[i])
            add(i, jA, 2.0 * w[i] * hh, 2.0 * w[i] * np.ones(jA.size))
        # Branch C: competing j with t_j <= t_i
        jC = comp[t[comp] <= t[i]]
        if jC.size:
            hh = (p[jC] < p[i]).astype(float) + 0.5 * (p[jC] == p[i])
            ww = 2.0 * w1[i] * w1[jC]
            add(i, jC, ww * hh, ww)

    # Branch B: tied case-times, ordered pairs within each equal-time group
    ct = t[case]
    order = np.argsort(ct, kind="stable")
    sc = case[order]
    sct = ct[order]
    g0 = 0
    while g0 < sc.size:
        g1 = g0
        while g1 + 1 < sc.size and sct[g1 + 1] == sct[g0]:
            g1 += 1
        grp = sc[g0 : g1 + 1]
        d = grp.size
        if d >= 2:
            for a in range(d):
                i = grp[a]
                for b in range(d):
                    if a == b:
                        continue
                    ip = grp[b]
                    hB = 0.5 + 0.5 * (p[i] == p[ip])
                    nc = w[i] * hB
                    dc = w[i]
                    N += nc
                    D += dc
                    Ndot[i] += nc
                    Ddot[i] += dc
                    Ndot[ip] += nc
                    Ddot[ip] += dc
        g0 = g1 + 1
    return N, D, Ndot, Ddot


def phi_G_full(t, e, p, w, C, knots, A):
    n = len(t)
    w1 = np.sqrt(w)
    keep = w > 0
    case = np.flatnonzero((e == CAUSE) & keep)
    comp = np.flatnonzero((e > 0) & (e != CAUSE) & keep)
    phi = np.zeros(n)
    for i in case:
        jA = np.flatnonzero(((t > t[i]) | ((t == t[i]) & (e == 0))) & keep)
        if jA.size:
            hh = (p[jA] < p[i]).astype(float) + 0.5 * (p[jA] == p[i])
            S = (2.0 * (hh - C)).sum()
            phi += S * 2.0 * w[i] * _ak(knots, A, t[i])
        jC = comp[t[comp] <= t[i]]
        if jC.size:
            hh = (p[jC] < p[i]).astype(float) + 0.5 * (p[jC] == p[i])
            cen = 2.0 * (hh - C)
            ww = w1[i] * w1[jC]
            phi += (cen * ww).sum() * _ak(knots, A, t[i])
            aj = A[:, np.searchsorted(knots, t[jC], side="right") - 1]
            phi += (aj * (cen * ww)[None, :]).sum(axis=1)
    # Branch B: dc=w_i depends on G(T_i); per ordered pair (i,i'): 2 w_i a_k(T_i)(hB-C)
    ct = t[case]
    order = np.argsort(ct, kind="stable")
    sc = case[order]
    sct = ct[order]
    g0 = 0
    while g0 < sc.size:
        g1 = g0
        while g1 + 1 < sc.size and sct[g1 + 1] == sct[g0]:
            g1 += 1
        grp = sc[g0 : g1 + 1]
        d = grp.size
        if d >= 2:
            for a in range(d):
                i = grp[a]
                SB = 0.0
                for b in range(d):
                    if a == b:
                        continue
                    SB += (0.5 + 0.5 * (p[i] == p[grp[b]])) - C
                phi += SB * 2.0 * w[i] * _ak(knots, A, t[i])
        g0 = g1 + 1
    return phi


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    grid = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    t, e, p = gen_ties(n, seed=11, grid=grid)
    n_evt = (e == CAUSE).sum()
    n_tied = n_evt - np.unique(t[e == CAUSE]).size  # cases sharing a time with another
    w = compute_uno_weights(t, e, gmin="none")

    C_lib = concordance_index_uno_cr(e, t, p, cause=CAUSE, weights=w)
    N, D, Ndot, Ddot = naive_full(t, e, p, w)
    C_mine = N / D
    print(f"[gate2 ties] n={n} cause1={n_evt} tied-case-times(excess)={n_tied}", flush=True)
    print(
        f"  self-check: C_lib={C_lib:.6f} C_naive={C_mine:.6f} diff={abs(C_lib - C_mine):.2e}",
        flush=True,
    )

    phiU = n * (Ndot - C_mine * Ddot) / D
    knots, A = cens_influence(t, e)
    phiG = phi_G_full(t, e, p, w, C_mine, knots, A) / D
    phi = phiU + phiG
    se = np.sqrt((phi**2).sum()) / n

    rng = np.random.default_rng(99)
    boot = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        wb = compute_uno_weights(t[idx], e[idx], gmin="none")
        boot[b] = concordance_index_uno_cr(e[idx], t[idx], p[idx], cause=CAUSE, weights=wb)
    se_boot = np.nanstd(boot, ddof=1)
    print(f"  IF se={se:.5f}  bootstrap se={se_boot:.5f}  ratio={se / se_boot:.3f}", flush=True)


if __name__ == "__main__":
    main()
