"""Gate 4: degenerate single-cause case vs survC1 (Uno's own package) as the
external oracle. Light censoring + NO truncation => 1/G^2 well-behaved (IF valid)
and crforest's keep_mask drops nobody (same comparator pool as survC1).

Single covariate => survC1's Cox-fit Wb term ~ 0 (C is rank-based, monotone-
invariant), so Inf.Cval SE == Wa+Wg == our fixed-prediction IF SE.

Writes a CSV for the R side (gate4_degenerate.R) and prints crforest point, IF SE,
and bootstrap SE.
"""

import numpy as np
from validate_if import cens_influence, naive_ND, phi_G

from comprisk.metrics import compute_uno_weights, concordance_index_uno_cr

CAUSE = 1
OUT = "/tmp/gate4_data.csv"


def gen_degenerate(n, seed):
    rng = np.random.default_rng(seed)
    p = rng.normal(size=n)
    t1 = rng.exponential(1.0 / np.exp(1.2 * p)) + rng.random(n) * 1e-4  # informative
    tc = rng.exponential(6.0, n)  # LIGHT censoring, late => G bounded away from 0
    t = np.minimum(t1, tc)
    e = (t == t1).astype(int)  # 1 = cause-1 event, 0 = censored
    return t, e, p


def if_se(t, e, p, w):
    n = len(t)
    N, D, Ndot, Ddot = naive_ND(t, e, p, w)
    C = N / D
    phiU = n * (Ndot - C * Ddot) / D
    knots, A = cens_influence(t, e)
    phiG = phi_G(t, e, p, w, C, knots, A) / D
    phi = phiU + phiG
    return C, np.sqrt((phi**2).sum()) / n


def main():
    n, B = 600, 3000
    t, e, p = gen_degenerate(n, seed=2026)
    cens_rate = float((e == 0).mean())
    w = compute_uno_weights(t, e, gmin="none")  # no truncation

    C_lib = concordance_index_uno_cr(e, t, p, cause=CAUSE, weights=w)
    C_if, se = if_se(t, e, p, w)

    rng = np.random.default_rng(7)
    boot = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        wb = compute_uno_weights(t[idx], e[idx], gmin="none")
        boot[b] = concordance_index_uno_cr(e[idx], t[idx], p[idx], cause=CAUSE, weights=wb)
    se_boot = np.nanstd(boot, ddof=1)

    import csv

    with open(OUT, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["time", "status", "score"])
        for i in range(n):
            wcsv.writerow([t[i], int(e[i]), p[i]])

    print(f"[gate4 crforest] n={n} cens={cens_rate:.2f}")
    print(f"  C (library)   = {C_lib:.6f}")
    print(f"  C (naive IF)  = {C_if:.6f}   self-diff={abs(C_lib - C_if):.2e}")
    print(f"  IF se         = {se:.6f}")
    print(f"  bootstrap se  = {se_boot:.6f}   (IF/boot={se / se_boot:.3f})")
    print(f"  wrote {OUT} for survC1 comparison")


if __name__ == "__main__":
    main()
