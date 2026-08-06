"""Experiment 2 (design.md 0.6 #2): the estimated-weight gap shrinks at n^{-1/2}.

Validates Corollary cor:rate: the coverage penalty Delta_w under estimated (KM) Ghat
vs oracle G is O_P(n^{-1/2}), driven by ||Ghat-G||_{inf,t*}. We report, across a pool
size grid, (i) the oracle-vs-KM coverage gap and (ii) the KM sup-error, and fit a
log-log slope of the sup-error against n -- it should land near -1/2 (Gill 1980).

Coverage is ORACLE-weighted throughout, so the gap is a genuine population effect of
mis-weighting the CALIBRATION scores, not an artefact of the coverage estimator.

Run:  PYTHONUNBUFFERED=1 uv run python -m validation.spikes.conformal.experiments.e2_deltaw_rate
"""

from __future__ import annotations

import numpy as np
from validation.spikes.conformal.dgp import cr_dgp
from validation.spikes.conformal.experiments.oracle_g import aggregate

ALPHA = 0.1
REPS = 24
N_GRID = (500, 1000, 2000, 4000, 8000)
KW = dict(censor_rate=0.6, competing_frac=0.4, signal=1.0, _alpha=ALPHA)


def main():
    nominal = 1 - ALPHA
    print(
        f"\nExp 2 -- Delta_w rate (alpha={ALPHA}, nominal={nominal:.2f}, reps={REPS}, censor=0.6)\n"
    )
    print(f"  {'n_pool':<9}{'cov_oracle':>11}{'cov_km':>9}{'gap':>9}{'||Gh-G||':>10}")
    ns, errs, gaps = [], [], []
    for n in N_GRID:
        kw = dict(KW)
        ro = aggregate(cr_dgp, kw, reps=REPS, n_pool=n, weight_mode="oracle", atom_mode="gmin")
        rk = aggregate(cr_dgp, kw, reps=REPS, n_pool=n, weight_mode="km", atom_mode="gmin")
        gap = ro["cov_mean"] - rk["cov_mean"]
        print(
            f"  {n:<9}{ro['cov_mean']:>11.3f}{rk['cov_mean']:>9.3f}"
            f"{gap:>+9.3f}{rk['km_err_mean']:>10.4f}"
        )
        ns.append(n)
        errs.append(rk["km_err_mean"])
        gaps.append(abs(gap))

    slope = np.polyfit(np.log(ns), np.log(errs), 1)[0]
    print(f"\nlog-log slope of ||Ghat-G|| vs n_pool = {slope:+.3f}  (expect ~ -0.5, Gill 1980)")
    print("Gap magnitude should be small and non-increasing in n (Delta_w -> 0).")


if __name__ == "__main__":
    main()
