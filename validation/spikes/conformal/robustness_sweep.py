"""Phase 2 (Gate 2) robustness sweep for the conformal-CR spike.

Three stresses, all measured by IPCW marginal coverage (no closed-form truth needed):
  A. Misspecification  -- exponential (baseline) vs Weibull vs non-PH DGPs. The
     conformal layer is distribution-free, so coverage should hold for all three;
     misspecification should show up as larger SETS, not lost coverage.
  B. tau (gmin)        -- sweep the IPCW weight clip; confirm a stable region and the
     tail-blowup failure below it.
  C. small-n / extreme censoring -- where the finite-sample (n+1) correction bites.

Run:  PYTHONUNBUFFERED=1 uv run python -m validation.spikes.conformal.robustness_sweep
"""

from __future__ import annotations

import numpy as np
from validation.spikes.conformal.conformal import (
    ipcw_coverage,
    prediction_sets,
    weighted_quantile_threshold,
)
from validation.spikes.conformal.dgp import (
    cr_dgp,
    horizon_labels,
    ipcw_weights_at_horizon,
)
from validation.spikes.conformal.dgp_misspec import nonph_cr_dgp, weibull_cr_dgp
from validation.spikes.conformal.scores import (
    nonconformity,
    oob_cif_at_horizon,
    split_cif_at_horizon,
)

from comprisk import CompetingRiskForest

_ALPHA = 0.1
T_STAR = 1.0

# Sweep configs (module-level so report.py can reuse them as the single source of
# truth instead of copying the bodies out of main()).
REPS = 12
MISSPEC_DGPS = [
    ("exponential", "cr_dgp", dict(censor_rate=0.4, competing_frac=0.4, signal=1.0)),
    ("weibull", "weibull_cr_dgp", dict(censor_rate=0.4, competing_frac=0.4, signal=1.0)),
    ("non-PH", "nonph_cr_dgp", dict(censor_rate=0.4, competing_frac=0.4, signal=1.2)),
]
GMIN_KW = dict(censor_rate=0.6, competing_frac=0.4, signal=1.0)
GMIN_SWEEP = (0.005, 0.02, 0.05, 0.1, 0.2)
SMALLN_NS = (500, 1000)
SMALLN_CENS = (0.6, 0.75)
SMALLN_NTEST = 4000
# Name -> DGP callable, so MISSPEC_DGPS can stay JSON-ish / importable.
DGP_FNS = {"cr_dgp": cr_dgp, "weibull_cr_dgp": weibull_cr_dgp, "nonph_cr_dgp": nonph_cr_dgp}


def _fit(X, time, event, *, ntree, seed):
    return CompetingRiskForest(n_estimators=ntree, random_state=seed, n_jobs=-1).fit(X, time, event)


def _one_rep(dgp_fn, dgp_kw, path, *, gmin, n_pool, n_test, ntree, seed):
    Xp, tp, ep, _ = dgp_fn(n_pool, seed=seed, t_star=T_STAR, **dgp_kw)
    Xt, tt, et, _ = dgp_fn(n_test, seed=seed + 100_000, t_star=T_STAR, **dgp_kw)
    yt, obs_t = horizon_labels(tt, et, T_STAR)
    wt, _ = ipcw_weights_at_horizon(tt, et, T_STAR, gmin=gmin)

    if path == "oob":
        forest = _fit(Xp, tp, ep, ntree=ntree, seed=seed)
        pic, pif, _ = oob_cif_at_horizon(forest, Xp, T_STAR)
        yc, obs_c = horizon_labels(tp, ep, T_STAR)
        wc, _ = ipcw_weights_at_horizon(tp, ep, T_STAR, gmin=gmin)
    else:
        h = n_pool // 2
        forest = _fit(Xp[:h], tp[:h], ep[:h], ntree=ntree, seed=seed)
        pic, pif = split_cif_at_horizon(forest, Xp[h:], T_STAR)
        yc, obs_c = horizon_labels(tp[h:], ep[h:], T_STAR)
        wc, _ = ipcw_weights_at_horizon(tp[h:], ep[h:], T_STAR, gmin=gmin)

    s = nonconformity(pic, pif, yc)
    qhat = weighted_quantile_threshold(s[obs_c], wc[obs_c], alpha=_ALPHA)
    pic_t, pif_t = split_cif_at_horizon(forest, Xt, T_STAR)
    sets = prediction_sets(pic_t, pif_t, qhat)
    return ipcw_coverage(sets, yt, wt, obs_t)


def _cell(dgp_fn, dgp_kw, path, *, reps, gmin=0.05, n_pool=2500, n_test=2500, ntree=100):
    covs, sizes = [], []
    for r in range(reps):
        c, sz = _one_rep(
            dgp_fn, dgp_kw, path, gmin=gmin, n_pool=n_pool, n_test=n_test, ntree=ntree, seed=200 + r
        )
        covs.append(c)
        sizes.append(sz)
    covs = np.array(covs)
    return covs.mean(), covs.std(ddof=1) / np.sqrt(reps), float(np.mean(sizes))


def _print(tag, path, mean, se, size, nominal=1 - _ALPHA):
    dev = mean - nominal
    ok = abs(dev) <= 3 * se if se > 0 else abs(dev) <= 0.01
    print(
        f"  {tag:<26}{path:<7}cov={mean:.3f}±{se:.3f}  dev={dev:+.3f}  "
        f"size={size:.2f}  {'ok' if ok else 'MISS'}"
    )
    return ok, dev


def main():
    reps = REPS
    print(f"\nalpha={_ALPHA} nominal={1 - _ALPHA:.2f} reps={reps}\n")

    print("A. Misspecification (coverage must hold; misspec -> larger sets, not lost cov):")
    a_ok = True
    for name, fn_name, kw in MISSPEC_DGPS:
        fn = DGP_FNS[fn_name]
        for path in ("oob", "split"):
            m, se, sz = _cell(fn, kw, path, reps=reps)
            ok, _ = _print(name, path, m, se, sz)
            a_ok &= ok

    print("\nB. tau / gmin sensitivity (split path, exponential, censor=0.6):")
    for gmin in GMIN_SWEEP:
        m, se, sz = _cell(cr_dgp, GMIN_KW, "split", reps=reps, gmin=gmin)
        _print(f"gmin={gmin}", "split", m, se, sz)

    print("\nC. small-n x extreme censoring (split path):")
    c_ok = True
    for n_pool in SMALLN_NS:
        for cens in SMALLN_CENS:
            kw = dict(censor_rate=cens, competing_frac=0.4, signal=1.0)
            m, se, sz = _cell(cr_dgp, kw, "split", reps=reps, n_pool=n_pool, n_test=SMALLN_NTEST)
            ok, _ = _print(f"n={n_pool} cens={cens}", "split", m, se, sz)
            c_ok &= ok

    print(
        f"\n=== Gate 2: misspec {'PASS' if a_ok else 'REVIEW'}; "
        f"small-n {'PASS' if c_ok else 'REVIEW'} ==="
    )


if __name__ == "__main__":
    main()
