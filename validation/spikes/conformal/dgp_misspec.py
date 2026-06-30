"""Phase 2 misspecification DGPs for the conformal-CR robustness sweep.

The conformal layer is distribution-free, so coverage should hold even when the
forest's score model is a poor fit to the data-generating hazards. These generators
stress that: non-constant (Weibull) baselines and non-proportional (time-crossing)
covariate effects. No closed-form marginal CIF is needed -- robustness coverage is
measured on observed horizon labels via IPCW, exactly as on real data.

Same return contract as dgp.cr_dgp: (X, time, event, info), event 0=censored / 1 / 2.
"""

from __future__ import annotations

import numpy as np


def weibull_cr_dgp(
    n,
    *,
    censor_rate=0.3,
    competing_frac=0.4,
    signal=1.0,
    shape1=1.6,
    shape2=0.8,
    t_star=1.0,
    p=5,
    seed=0,
):
    """Weibull cause-specific times (non-constant baseline hazards).

    T_k ~ Weibull(shape_k, scale_k); cause-1 scale shrinks with the first covariate
    (higher x0 -> earlier cause-1 event). shape1>1 (increasing hazard), shape2<1
    (decreasing hazard) -> the two causes have qualitatively different baselines, a
    regime constant-hazard intuition misreads.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))

    # Scales chosen so overall event level near S(t*)~0.5; cause mix ~ competing_frac.
    scale1 = (
        1.3
        * np.exp(-signal * X[:, 0] / max(shape1, 1e-6))
        / max(1 - competing_frac, 1e-6) ** (1 / shape1)
    )
    scale2 = np.full(n, 1.3 / max(competing_frac, 1e-6) ** (1 / shape2))

    u1, u2 = rng.random(n), rng.random(n)
    t1 = scale1 * (-np.log(u1)) ** (1.0 / shape1)
    t2 = scale2 * (-np.log(u2)) ** (1.0 / shape2)
    t_event = np.minimum(t1, t2)
    cause = np.where(t1 <= t2, 1, 2).astype(np.int64)

    time, event = _apply_censoring(t_event, cause, censor_rate, t_star, rng)
    return (
        X,
        time,
        event,
        {"realized_censor_rate": float(np.mean(event == 0)), "t_star": t_star, "kind": "weibull"},
    )


def nonph_cr_dgp(
    n,
    *,
    censor_rate=0.3,
    competing_frac=0.4,
    signal=1.2,
    knot=0.5,
    t_star=1.0,
    p=5,
    seed=0,
):
    """Non-proportional cause-1 hazard: the covariate effect REVERSES at ``knot``.

    Piecewise-exponential cause-1 hazard with rate base*exp(+signal*x0) before knot
    and base*exp(-signal*x0) after -- the hazard-ratio ordering of high/low-x0
    subjects crosses over time, violating proportional hazards. Cause 2 constant.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))

    lam = -np.log(0.5) / t_star
    base1 = (1.0 - competing_frac) * lam
    lam_a = base1 * np.exp(signal * X[:, 0])  # rate before knot
    lam_b = base1 * np.exp(-signal * X[:, 0])  # rate after knot (reversed)
    lam2 = competing_frac * lam

    # Inverse-CDF of the piecewise-exponential cumulative hazard.
    e = rng.exponential(1.0, n)
    H_knot = lam_a * knot
    t1 = np.where(e <= H_knot, e / lam_a, knot + (e - H_knot) / lam_b)
    t2 = rng.exponential(1.0 / max(lam2, 1e-12), n)
    t_event = np.minimum(t1, t2)
    cause = np.where(t1 <= t2, 1, 2).astype(np.int64)

    time, event = _apply_censoring(t_event, cause, censor_rate, t_star, rng)
    return (
        X,
        time,
        event,
        {"realized_censor_rate": float(np.mean(event == 0)), "t_star": t_star, "kind": "nonph"},
    )


def _apply_censoring(t_event, cause, censor_rate, t_star, rng):
    """Independent exponential censoring tuned to ~censor_rate; returns (time, event)."""
    n = t_event.shape[0]
    if censor_rate <= 0:
        t_cens = np.full(n, np.inf)
    else:
        # Match the median event time so realized censoring is near the target.
        med = float(np.median(t_event))
        lam_c = (np.log(2) / max(med, 1e-6)) * censor_rate / max(1 - censor_rate, 1e-6)
        t_cens = rng.exponential(1.0 / lam_c, size=n)
    time = np.minimum(t_event, t_cens)
    event = np.where(t_event <= t_cens, cause, 0).astype(np.int64)
    return time, event
