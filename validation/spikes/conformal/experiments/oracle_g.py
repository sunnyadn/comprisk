"""Shared oracle-G machinery for the Phase 0.6 conformal-CR experiments.

The synthetic DGP (``dgp.cr_dgp``) uses independent *exponential* censoring with
rate ``lam_c = lam * censor_rate / (1 - censor_rate)``, ``lam = -log(s_at_tstar)/t*``.
So the TRUE censoring survival is closed form,

    G(u) = exp(-lam_c * u),   G(min(T,t*)^-) = G(min(T,t*))   (C continuous, no atom),

and the ORACLE IPCW weight is w*_i = 1/G(min(T_i,t*)) = exp(lam_c * min(T_i,t*)).
Having the true G is exactly what the coverage.tex Remark rem:conservative flags as
missing: the shipped OOB coverage is computed with the same estimated (clipped) Ghat
it conformalises on, so it is not an independent check of the theorem. Every number
here evaluates population coverage with the ORACLE weights, breaking that circularity.

This module is the single source of truth for oracle-vs-estimated runs; the e1..e4
scripts import ``conformal_run`` and only vary its knobs.
"""

from __future__ import annotations

import numpy as np
from validation.spikes.conformal.conformal import (
    ipcw_coverage,
    prediction_sets,
    weighted_quantile_threshold,
)
from validation.spikes.conformal.dgp import (
    _ghat_minus,
    censoring_km,
    horizon_labels,
    ipcw_weights_at_horizon,
)
from validation.spikes.conformal.scores import (
    nonconformity,
    split_cif_at_horizon,
)

from comprisk import CompetingRiskForest


def oracle_censoring_rate(*, t_star: float, censor_rate: float, s_at_tstar: float) -> float:
    """The true exponential censoring hazard ``lam_c`` used by ``cr_dgp``.

    Mirrors the DGP body exactly: ``lam = -log(s_at_tstar)/t*``,
    ``lam_c = lam * censor_rate/(1-censor_rate)`` (0 when ``censor_rate == 0``).
    """
    if censor_rate <= 0:
        return 0.0
    lam = -np.log(s_at_tstar) / t_star
    return lam * censor_rate / (1.0 - censor_rate)


def oracle_G(u: np.ndarray, lam_c: float) -> np.ndarray:
    """True censoring survival ``G(u) = exp(-lam_c u)`` (marginal, X-free)."""
    return np.exp(-lam_c * np.asarray(u, dtype=float))


def oracle_ipcw_weights_at_horizon(time, event, t_star, *, lam_c, gmin=0.05):
    """Oracle analogue of ``dgp.ipcw_weights_at_horizon``: w*_i = 1/G(min(T_i,t*))
    with the TRUE ``G`` instead of the KM ``Ghat``. Clipped at ``gmin`` for parity
    with the estimated path (inactive on the positivity event, so it adds no bias).
    """
    time = np.asarray(time, dtype=float)
    _, observed = horizon_labels(time, event, t_star)
    m = np.minimum(time, t_star)
    g = np.maximum(oracle_G(m, lam_c), gmin)
    w = np.zeros(time.shape, dtype=float)
    w[observed] = 1.0 / g[observed]
    return w, observed


def km_sup_error(time, event, t_star, *, lam_c, n_grid=200):
    """||Ghat - G||_{inf, [0,t*]} between the calibration-fold KM and the true G.

    Used by e2 to show the coverage gap tracks this quantity at the n^{-1/2} rate.
    """
    t_unique, G = censoring_km(time, event)
    grid = np.linspace(0.0, t_star, n_grid)
    ghat = _ghat_minus(t_unique, G, grid)
    gtrue = oracle_G(grid, lam_c)
    return float(np.max(np.abs(ghat - gtrue)))


def _mean_weight_atom(weights):
    """The old (unproven) mean-calibration-weight atom, kept only so e3 can price
    the conservatism of the 1/g_min atom against it. NOT floor-valid (rem:atom)."""
    w = np.asarray(weights, dtype=float)
    return float(w.mean()) if w.size else 1.0


def conformal_run(
    dgp_fn,
    dgp_kw,
    *,
    gmin=0.05,
    n_pool=2500,
    n_test=4000,
    ntree=100,
    seed=0,
    weight_mode="km",  # "km" (estimated Ghat) or "oracle" (true G)
    atom_mode="gmin",  # "gmin" (1/g_min, the main-theorem atom), "oracle", or "mean"
    t_star=1.0,
    s_at_tstar=0.5,
    eval_mode="oracle",  # coverage always uses ORACLE weights by default
):
    """One split-conformal replication with switchable calibration weights + atom.

    Returns a dict: coverage (population, oracle-weighted unless overridden),
    set size, the atom mass, and ||Ghat-G|| for diagnostics.
    """
    lam_c = oracle_censoring_rate(
        t_star=t_star, censor_rate=dgp_kw.get("censor_rate", 0.0), s_at_tstar=s_at_tstar
    )

    alpha = _alpha_of(dgp_kw)
    # Strip private sentinels (e.g. _alpha) before they reach the DGP signature.
    dgp_kw = {k: v for k, v in dgp_kw.items() if not k.startswith("_")}
    Xp, tp, ep, _ = dgp_fn(n_pool, seed=seed, t_star=t_star, s_at_tstar=s_at_tstar, **dgp_kw)
    Xt, tt, et, _ = dgp_fn(
        n_test, seed=seed + 100_000, t_star=t_star, s_at_tstar=s_at_tstar, **dgp_kw
    )

    h = n_pool // 2
    forest = CompetingRiskForest(n_estimators=ntree, random_state=seed, n_jobs=-1).fit(
        Xp[:h], tp[:h], ep[:h]
    )

    # Calibration fold.
    cal_t, cal_e, cal_X = tp[h:], ep[h:], Xp[h:]
    yc, obs_c = horizon_labels(cal_t, cal_e, t_star)
    if weight_mode == "oracle":
        wc, _ = oracle_ipcw_weights_at_horizon(cal_t, cal_e, t_star, lam_c=lam_c, gmin=gmin)
    else:
        wc, _ = ipcw_weights_at_horizon(cal_t, cal_e, t_star, gmin=gmin)

    pic, pif = split_cif_at_horizon(forest, cal_X, t_star)
    s = nonconformity(pic, pif, yc)

    # Test atom.
    if atom_mode == "mean":
        atom = _mean_weight_atom(wc[obs_c])
    elif atom_mode == "oracle":
        # In simulation the test outcomes are known, so the oracle atom is the mean
        # true test weight (the sharpest admissible average); used for the e1 ceiling.
        m_te = np.minimum(tt, t_star)
        atom = float(np.mean(1.0 / np.maximum(oracle_G(m_te, lam_c), gmin)))
    else:  # "gmin" -- the realizable main-theorem atom
        atom = 1.0 / gmin
    qhat = weighted_quantile_threshold(
        s[obs_c], wc[obs_c], alpha=alpha, test_weight=atom, g_min=gmin
    )

    # Population coverage: ORACLE-weighted test evaluation (independent of Ghat).
    yt, obs_t = horizon_labels(tt, et, t_star)
    if eval_mode == "oracle":
        wt, _ = oracle_ipcw_weights_at_horizon(tt, et, t_star, lam_c=lam_c, gmin=gmin)
    else:
        wt, _ = ipcw_weights_at_horizon(tt, et, t_star, gmin=gmin)

    pic_t, pif_t = split_cif_at_horizon(forest, Xt, t_star)
    sets = prediction_sets(pic_t, pif_t, qhat)
    cov, size = ipcw_coverage(sets, yt, wt, obs_t)

    return {
        "coverage": cov,
        "size": size,
        "atom": atom,
        "qhat": qhat,
        "km_sup_err": km_sup_error(cal_t, cal_e, t_star, lam_c=lam_c),
        "lam_c": lam_c,
        "sets": sets,
        "yt": yt,
        "wt": wt,
        "obs_t": obs_t,
        "pic_t": pic_t,
        "pif_t": pif_t,
    }


# alpha travels in dgp_kw for the sweep scripts; default 0.1.
def _alpha_of(dgp_kw):
    return dgp_kw.get("_alpha", 0.1)


def aggregate(dgp_fn, dgp_kw, *, reps, base_seed=200, **kw):
    """Mean +/- SE of coverage and set size over ``reps`` replications."""
    covs, sizes, errs = [], [], []
    for r in range(reps):
        out = conformal_run(dgp_fn, dgp_kw, seed=base_seed + r, **kw)
        covs.append(out["coverage"])
        sizes.append(out["size"])
        errs.append(out["km_sup_err"])
    covs = np.asarray(covs)
    return {
        "cov_mean": float(covs.mean()),
        "cov_se": float(covs.std(ddof=1) / np.sqrt(reps)) if reps > 1 else 0.0,
        "size_mean": float(np.mean(sizes)),
        "km_err_mean": float(np.mean(errs)),
    }
