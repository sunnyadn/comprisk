"""Step 1 validation (Gate 1) for the conformal-CR spike.

Checks, at signal == 0 where the marginal CIFs are closed-form:

  1. IPCW-weighted horizon-label proportions reproduce the true F_1(t*), F_2(t*),
     S(t*) within Monte-Carlo noise -> labels AND weights are jointly correct.
  2. Naive (unweighted, observed-only) proportions are visibly biased by censoring
     -> confirms the IPCW correction is doing real work.
  3. Censored-before-t* subjects are flagged not-observed; their count tracks the
     censoring rate.

Run:  PYTHONUNBUFFERED=1 uv run python -m validation.spikes.conformal.dgp_check
"""

from __future__ import annotations

import sys

import numpy as np
from validation.spikes.conformal.dgp import (
    EVENT_FREE,
    cr_dgp,
    horizon_labels,
    ipcw_weights_at_horizon,
)


def _weighted_proportions(y, w, observed, labels):
    """Sum of weights per label over observed subjects, normalised to 1."""
    wo = w[observed]
    yo = y[observed]
    total = wo.sum()
    return {lab: float(wo[yo == lab].sum() / total) for lab in labels}


def _naive_proportions(y, observed, labels):
    yo = y[observed]
    n = yo.size
    return {lab: float(np.mean(yo == lab)) for lab in labels}


def run_one(*, n, censor_rate, competing_frac, t_star, seed):
    _X, time, event, info = cr_dgp(
        n,
        censor_rate=censor_rate,
        competing_frac=competing_frac,
        signal=0.0,
        t_star=t_star,
        seed=seed,
    )
    y, observed = horizon_labels(time, event, t_star)
    w, observed_w = ipcw_weights_at_horizon(time, event, t_star)
    assert np.array_equal(observed, observed_w)

    labels = [1, 2, EVENT_FREE]
    truth = {1: info["F1_true"], 2: info["F2_true"], EVENT_FREE: info["S_true"]}
    weighted = _weighted_proportions(y, w, observed, labels)
    naive = _naive_proportions(y, observed, labels)

    # MC SE proxy for a proportion at this observed sample size.
    n_obs = int(observed.sum())
    se = np.sqrt(0.25 / max(n_obs, 1))

    name = {1: "F1", 2: "F2", EVENT_FREE: "S "}
    print(f"\n[n={n} censor={censor_rate} comp_frac={competing_frac} t*={t_star} seed={seed}]")
    print(
        f"  realized censor rate : {info['realized_censor_rate']:.3f}  "
        f"(censored-before-t* unobserved: {int((~observed).sum())}/{n})"
    )
    print(f"  MC SE (~prop)        : {se:.4f}")
    print(
        f"  {'label':<6}{'true':>9}{'ipcw':>9}{'naive':>9}{'|ipcw-true|':>13}{'|naive-true|':>14}"
    )
    max_w_err = 0.0
    for lab in labels:
        dw = abs(weighted[lab] - truth[lab])
        dn = abs(naive[lab] - truth[lab])
        max_w_err = max(max_w_err, dw)
        print(
            f"  {name[lab]:<6}{truth[lab]:>9.4f}{weighted[lab]:>9.4f}"
            f"{naive[lab]:>9.4f}{dw:>13.4f}{dn:>14.4f}"
        )
    # PASS if IPCW within ~3 SE on every label.
    tol = 3.0 * se
    ok = max_w_err <= tol
    print(f"  -> IPCW max|err| {max_w_err:.4f}  tol(3*SE) {tol:.4f}  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    np.set_printoptions(suppress=True)
    grid = [
        dict(n=40000, censor_rate=0.2, competing_frac=0.4, t_star=1.0, seed=1),
        dict(n=40000, censor_rate=0.4, competing_frac=0.4, t_star=1.0, seed=2),
        dict(n=40000, censor_rate=0.6, competing_frac=0.3, t_star=1.0, seed=3),
        dict(n=40000, censor_rate=0.4, competing_frac=0.5, t_star=1.0, seed=4),
    ]
    results = [run_one(**cfg) for cfg in grid]
    n_pass = sum(results)
    print(f"\n=== Gate 1: {n_pass}/{len(results)} cells PASS ===")
    if n_pass < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
