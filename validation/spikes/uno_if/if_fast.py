"""Gate 1 capstone: self-contained O(n log n) closed-form IF CI for crforest's
Uno/Wolbers CR concordance. phi^U branches A+C via Fenwick; phi^G via analytic
time-grid reorganization; point estimate C from the same Fenwick sums (no O(n^2)).
Continuous times (Branch B empty -- validated separately in gate2_ties.py).

  python if_fast.py check [n]   -> bit-equivalence vs O(n^2) reference
  python if_fast.py time        -> wall-clock vs naive-IF and bootstrap
"""

import sys
import time as _time

import numpy as np
from validate_if import cens_influence, gen, naive_ND, phi_G

from comprisk.metrics import compute_uno_weights, concordance_index_uno_cr

CAUSE = 1


class WFen:
    __slots__ = ("m", "t")

    def __init__(self, m):
        self.m = m
        self.t = np.zeros(m + 2)

    def add(self, i, v):
        i += 1
        while i <= self.m:
            self.t[i] += v
            i += i & (-i)

    def pref(self, i):
        i += 1
        s = 0.0
        while i > 0:
            s += self.t[i]
            i -= i & (-i)
        return s


def fast_if(t, e, p, w):
    """Return C, se, phi (length n). Pure O(n log n + n*K_knots)."""
    n = len(t)
    w1 = np.sqrt(w)
    keep = w > 0
    is_case = (e == CAUSE) & keep
    is_comp = (e > 0) & (e != CAUSE) & keep
    prank = np.empty(n, dtype=np.int64)
    prank[np.argsort(p, kind="stable")] = np.arange(n)
    m = n
    torder = np.argsort(t, kind="stable")

    Ndot = np.zeros(n)
    Ddot = np.zeros(n)
    lessA = np.zeros(n)
    totA = np.zeros(n)
    lessC = np.zeros(n)
    totC = np.zeros(n)
    gtC = np.zeros(n)
    totCc = np.zeros(n)

    # Branch A, case role (comparator t_j>t_i): count seen comparators by p-rank
    cnt = WFen(m)
    seen = 0
    for idx in range(n - 1, -1, -1):
        s = torder[idx]
        if not keep[s]:
            continue
        if is_case[s]:
            less = cnt.pref(prank[s] - 1)
            lessA[s] = less
            totA[s] = seen
            Ndot[s] += 2 * w[s] * less
            Ddot[s] += 2 * w[s] * seen
        cnt.add(prank[s], 1.0)
        seen += 1
    # Branch A, comparator role
    wA = WFen(m)
    for idx in range(n):
        s = torder[idx]
        if not keep[s]:
            continue
        tot_w = wA.pref(m - 1)
        gt_w = tot_w - wA.pref(prank[s])
        Ndot[s] += gt_w
        Ddot[s] += tot_w
        if is_case[s]:
            wA.add(prank[s], 2 * w[s])
    # Branch C, case role (competing t_j<=t_i)
    fC = WFen(m)
    tw = 0.0
    for idx in range(n):
        s = torder[idx]
        if not keep[s]:
            continue
        if is_case[s]:
            lw = fC.pref(prank[s] - 1)
            lessC[s] = lw
            totC[s] = tw
            Ndot[s] += 2 * w1[s] * lw
            Ddot[s] += 2 * w1[s] * tw
        if is_comp[s]:
            fC.add(prank[s], w1[s])
            tw += w1[s]
    # Branch C, competing role
    gCf = WFen(m)
    twc = 0.0
    for idx in range(n - 1, -1, -1):
        s = torder[idx]
        if not keep[s]:
            continue
        if is_comp[s]:
            gw = twc - gCf.pref(prank[s])
            gtC[s] = gw
            totCc[s] = twc
            Ndot[s] += 2 * w1[s] * gw
            Ddot[s] += 2 * w1[s] * twc
        if is_case[s]:
            gCf.add(prank[s], w1[s])
            twc += w1[s]

    D = Ddot.sum() / 2.0
    C = Ndot.sum() / Ddot.sum()
    phiU = n * (Ndot - C * Ddot) / D

    # centered per-endpoint coefficients for phi^G
    mA = 4 * w * (lessA - C * totA) * is_case
    mC_case = 2 * w1 * (lessC - C * totC) * is_case
    mC_comp = 2 * w1 * (gtC - C * totCc) * is_comp

    # phi^G time-grid reorganization
    knots = np.unique(t)
    K = knots.size
    nrisk = np.array([(t >= s).sum() for s in knots], float)
    d_one = np.array([((t == s) & (e == CAUSE)).sum() for s in knots], float)
    d_oth = np.array([((t == s) & (e != CAUSE)).sum() for s in knots], float)
    r = nrisk - d_one
    pi_r = r / n
    dLam = np.zeros(K)
    ok = (d_oth > 0) & (r > 0)
    dLam[ok] = d_oth[ok] / r[ok]
    mk = np.searchsorted(knots, t)
    Mbar = np.zeros(K)
    np.add.at(Mbar, mk, mA + mC_case + mC_comp)
    Mcum = np.cumsum(Mbar[::-1])[::-1]
    Gl = np.zeros(K)
    Gl[ok] = dLam[ok] * Mcum[ok] / pi_r[ok]
    Gcum_prev = np.concatenate(([0.0], np.cumsum(Gl)))  # Gcum_prev[m]=Gcum[m-1]
    Gcum = Gcum_prev[1:]
    phiG = np.zeros(n)
    cm = is_case
    om = ~is_case
    phiG[cm] = -Gcum_prev[mk[cm]] / D
    mko = mk[om]
    pir_o = np.where(pi_r[mko] > 0, pi_r[mko], np.inf)
    phiG[om] = (Mcum[mko] / pir_o - Gcum[mko]) / D

    phi = phiU + phiG
    se = np.sqrt((phi**2).sum()) / n
    return C, se, phi


def naive_se(t, e, p, w):
    n = len(t)
    N, D, Ndot, Ddot = naive_ND(t, e, p, w)
    C = N / D
    phiU = n * (Ndot - C * Ddot) / D
    knots, A = cens_influence(t, e)
    phiG = phi_G(t, e, p, w, C, knots, A) / D
    return C, np.sqrt(((phiU + phiG) ** 2).sum()) / n


def check(n):
    t, e, p = gen(n, seed=5, signal=1.0)
    w = compute_uno_weights(t, e, gmin="none")
    C_f, se_f, _ = fast_if(t, e, p, w)
    C_n, se_n = naive_se(t, e, p, w)
    print(
        f"[check n={n}] C fast={C_f:.6f} naive={C_n:.6f} dC={abs(C_f - C_n):.1e} | "
        f"se fast={se_f:.6f} naive={se_n:.6f} dse={abs(se_f - se_n):.1e}"
    )


def time_it():
    B = 1000
    print(f"timing: IF(fast) vs naive-IF vs bootstrap B={B}")
    for n in (1000, 2000, 4000, 8000):
        t, e, p = gen(n, seed=1, signal=1.0)
        w = compute_uno_weights(t, e, gmin="none")
        t0 = _time.perf_counter()
        fast_if(t, e, p, w)
        tf = _time.perf_counter() - t0
        t0 = _time.perf_counter()
        naive_se(t, e, p, w)
        tn = _time.perf_counter() - t0
        # one bootstrap iteration cost x B
        rng = np.random.default_rng(0)
        t0 = _time.perf_counter()
        Bs = 20
        for _ in range(Bs):
            idx = rng.integers(0, n, n)
            wb = compute_uno_weights(t[idx], e[idx], gmin="none")
            concordance_index_uno_cr(e[idx], t[idx], p[idx], cause=CAUSE, weights=wb)
        tb1 = (_time.perf_counter() - t0) / Bs
        tboot = tb1 * B
        print(
            f"  n={n:5d}: IF_fast={tf * 1e3:7.1f}ms  IF_naive={tn * 1e3:8.1f}ms  "
            f"boot(B={B})={tboot:7.2f}s  speedup_vs_boot={tboot / tf:7.0f}x  "
            f"fast/naive={tn / tf:5.1f}x"
        )


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "time":
        time_it()
    else:
        for n in (500, 1500, 4000):
            check(n)


if __name__ == "__main__":
    main()
