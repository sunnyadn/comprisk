"""Gate 3: coverage probability of the closed-form IF 95% CI for crforest's
Uno/Wolbers CR concordance. Estimand fixed via population tau; truth via large
reference sample. Reuses the bootstrap-validated O(n^2) IF from validate_if.py.
"""

import sys

import numpy as np
from validate_if import CAUSE, cens_influence, gen, naive_ND, phi_G

from comprisk.metrics import compute_uno_weights, concordance_index_uno_cr


def wts(tt, ee, tau):
    w = compute_uno_weights(tt, ee, gmin="none")
    w[tt >= tau] = 0.0
    return w


def if_ci(t, e, p, tau, z=1.959964):
    """Closed-form IF point + 95% CI + se."""
    n = len(t)
    w = wts(t, e, tau)
    N, D, Ndot, Ddot = naive_ND(t, e, p, w)
    C = N / D
    phiU = n * (Ndot - C * Ddot) / D
    knots, A = cens_influence(t, e)
    phiG = phi_G(t, e, p, w, C, knots, A) / D
    phi = phiU + phiG
    se = np.sqrt((phi**2).sum()) / n
    return C, C - z * se, C + z * se, se


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    signal = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    tau_q = 0.70

    # --- fixed estimand: population tau + true truncated-Uno-C on a big sample ---
    tbig, ebig, pbig = gen(200_000, seed=999_999, signal=signal)
    tau = float(np.quantile(tbig, tau_q))
    wbig = wts(tbig, ebig, tau)
    C_true = concordance_index_uno_cr(ebig, tbig, pbig, cause=CAUSE, weights=wbig)
    print(f"[truth] signal={signal} tau={tau:.4f} C_true={C_true:.5f}  (n_ref=200k)", flush=True)

    z = 1.959964
    cover = 0  # symmetric Wald
    cover_l = 0  # logit-transform (variance-stabilised)
    Cs = np.empty(R)
    ses = np.empty(R)
    for r in range(R):
        t, e, p = gen(n, seed=10_000 + r, signal=signal)
        C, lo, hi, se = if_ci(t, e, p, tau)
        Cs[r] = C
        ses[r] = se
        if lo <= C_true <= hi:
            cover += 1
        # logit transform: l=log(C/(1-C)), se_l=se/(C(1-C)), CI back-transformed
        lodds = np.log(C / (1 - C))
        se_l = se / (C * (1 - C))
        lo_l = 1 / (1 + np.exp(-(lodds - z * se_l)))
        hi_l = 1 / (1 + np.exp(-(lodds + z * se_l)))
        if lo_l <= C_true <= hi_l:
            cover_l += 1
        if (r + 1) % 200 == 0:
            print(
                f"  ... {r + 1}/{R} wald={cover / (r + 1):.3f} logit={cover_l / (r + 1):.3f}",
                flush=True,
            )

    cov = cover / R
    cov_l = cover_l / R
    mc = np.sqrt(cov * (1 - cov) / R)
    mc_l = np.sqrt(cov_l * (1 - cov_l) / R)
    print(f"\n[gate3 coverage] n={n} R={R} signal={signal}", flush=True)
    print(f"  nominal 95%   Wald  coverage = {cov:.3f} (+/- {1.96 * mc:.3f} MC)", flush=True)
    print(f"                logit coverage = {cov_l:.3f} (+/- {1.96 * mc_l:.3f} MC)", flush=True)
    print(f"  mean IF se    = {ses.mean():.5f}", flush=True)
    print(
        f"  empirical sd(C) over reps = {Cs.std(ddof=1):.5f}  "
        f"(ratio se/sd = {ses.mean() / Cs.std(ddof=1):.3f})",
        flush=True,
    )
    print(f"  mean Chat = {Cs.mean():.5f}  bias = {Cs.mean() - C_true:+.5f}", flush=True)


if __name__ == "__main__":
    main()
