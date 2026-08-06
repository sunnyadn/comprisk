"""Advantage sweep: naive (unweighted) vs IPCW-weighted split-conformal, as a
function of censoring strength, tracking BOTH coverage and efficiency.

Purpose (2026-07-12): produce the "method actually helps" evidence a reader wants.
The finite-sample coverage contract (>=1-alpha) is what EVERY correct method must
meet; the demonstration is therefore two-pronged:

  1. VALIDITY under censoring: the naive unweighted-on-I conformal SILENTLY
     under-covers, and the gap WIDENS as censoring worsens (the label-complete
     calibration set is a censoring-biased sample, missing high-nonconformity
     subjects). Our IPCW-weighted threshold restores >=1-alpha.
  2. EFFICIENCY: among procedures that hold coverage, smaller sets are better. We
     report weighted mean set size so the reader sees the IPCW method buys validity
     without blowing up the set.

Both methods share EVERYTHING except the calibration weight vector fed to the
weighted quantile: naive = uniform on observed calibration points, ipcw = 1/Ghat.
Coverage is the SAME IPCW population estimate over observed test subjects for both,
so the comparison is apples-to-apples (only the threshold differs).

Run:  PYTHONUNBUFFERED=1 uv run python -m validation.spikes.conformal.advantage_sweep
"""

from __future__ import annotations

import numpy as np
from validation.spikes.conformal.conformal import (
    ipcw_coverage,
    prediction_sets,
    weighted_quantile_threshold,
)
from validation.spikes.conformal.dgp import (
    horizon_labels,
    ipcw_weights_at_horizon,
)
from validation.spikes.conformal.scores import (
    nonconformity,
    split_cif_at_horizon,
)

from comprisk import CompetingRiskForest

T_STAR = 1.0


def cr_dgp_informative(
    n,
    *,
    censor_rate=0.4,
    competing_frac=0.4,
    signal=1.0,
    censor_signal=0.0,
    cause_censor=0.0,
    t_star=1.0,
    s_at_tstar=0.5,
    p=5,
    seed=0,
):
    """Competing-risks DGP with informative censoring, two mechanisms:

    - censor_signal: censor hazard ~ exp(censor_signal * X[:,0]) (COVARIATE-dependent;
      still satisfies C |= (T,eps) | X, so the theory's assumption HOLDS and the
      correct weight is covariate-conditional 1/G(.|X)).
    - cause_censor: latent cause-1 subjects get censor hazard multiplied by
      exp(cause_censor) (OUTCOME/CAUSE-dependent; VIOLATES C |= (T,eps) | X, so the
      IPCW identity itself is misspecified -- tests robustness beyond the theory).
      This biases the label-complete calibration sample in LABEL space (cause-1
      under-represented), the most direct stress on cause-set coverage.

    Both zero -> reduces to dgp.cr_dgp (independent censoring, benign).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    lam = -np.log(s_at_tstar) / t_star
    lam2 = competing_frac * lam
    lam1_base = (1.0 - competing_frac) * lam

    mult = np.exp(signal * X[:, 0])
    rate1 = lam1_base * mult / mult.mean()
    rate2 = np.full(n, lam2)
    t1 = rng.exponential(1.0 / np.maximum(rate1, 1e-12))
    t2 = rng.exponential(1.0 / np.maximum(rate2, 1e-12))
    t_event = np.minimum(t1, t2)
    cause = np.where(t1 <= t2, 1, 2).astype(np.int64)

    if censor_rate <= 0:
        t_cens = np.full(n, np.inf)
    else:
        lam_c0 = lam * censor_rate / (1.0 - censor_rate)
        cmult = np.exp(censor_signal * X[:, 0])
        cmult = cmult / cmult.mean()
        # cause-dependent multiplier: cause-1 subjects censored exp(cause_censor)x more
        ccmult = np.where(cause == 1, np.exp(cause_censor), 1.0)
        lam_c = lam_c0 * cmult * ccmult
        t_cens = rng.exponential(1.0 / np.maximum(lam_c, 1e-12), size=n)

    time = np.minimum(t_event, t_cens)
    event = np.where(t_event <= t_cens, cause, 0).astype(np.int64)
    # true per-subject censoring hazard (constant-in-time exponential) for oracle weights
    lam_c_true = np.zeros(n) if censor_rate <= 0 else lam_c0 * cmult * ccmult
    return X, time, event, {"realized_censor_rate": float(np.mean(event == 0)), "lam_c": lam_c_true}


def _oracle_weights(time, event, lam_c, t_star, gmin=0.05):
    """Oracle covariate-conditional IPCW weight w = 1/G(min(T,t*)^-|X) using the
    KNOWN exponential censoring hazard: G(s|x)=exp(-lam_c(x) s), so w=exp(lam_c*q),
    clipped via gmin. Valid ONLY when C |= (T,eps)|X holds (covariate censoring)."""
    _, observed = horizon_labels(time, event, t_star)
    q = np.minimum(np.asarray(time, dtype=float), t_star)
    g = np.exp(-np.asarray(lam_c, dtype=float) * q)
    g = np.maximum(g, gmin)
    w = np.zeros(time.shape, dtype=float)
    w[observed] = 1.0 / g[observed]
    return w


def _fit(X, time, event, *, n_estimators, seed):
    return CompetingRiskForest(n_estimators=n_estimators, random_state=seed, n_jobs=-1).fit(
        X, time, event
    )


def _cause1_coverage(sets, yt, obs_t):
    """Unweighted coverage restricted to observed test subjects whose true horizon
    label is cause 1 (the under-censored class). Reveals label-space undercoverage
    that marginal coverage can mask."""
    K = sets.shape[1] - 1
    obs = obs_t
    y_obs = yt[obs]
    sets_obs = sets[obs]
    mask1 = y_obs == 1
    if mask1.sum() == 0:
        return np.nan
    covered = sets_obs[mask1][:, 0]  # column 0 = cause 1
    return float(covered.mean())


def _one_rep(
    method,
    *,
    censor_rate,
    competing_frac,
    signal,
    censor_signal,
    cause_censor,
    alpha,
    n_pool,
    n_test,
    n_estimators,
    seed,
):
    """Split-conformal single rep. `method` in {'naive','ipcw'} sets ONLY the
    calibration weight vector; everything else is identical."""
    Xp, tp, ep, ip = cr_dgp_informative(
        n_pool,
        censor_rate=censor_rate,
        competing_frac=competing_frac,
        signal=signal,
        censor_signal=censor_signal,
        cause_censor=cause_censor,
        t_star=T_STAR,
        seed=seed,
    )
    Xt, tt, et, it = cr_dgp_informative(
        n_test,
        censor_rate=censor_rate,
        competing_frac=competing_frac,
        signal=signal,
        censor_signal=censor_signal,
        cause_censor=cause_censor,
        t_star=T_STAR,
        seed=seed + 100_000,
    )
    yt, obs_t = horizon_labels(tt, et, T_STAR)
    _wt, _ = ipcw_weights_at_horizon(tt, et, T_STAR)

    h = n_pool // 2
    forest = _fit(Xp[:h], tp[:h], ep[:h], n_estimators=n_estimators, seed=seed)
    pic, pif = split_cif_at_horizon(forest, Xp[h:], T_STAR)
    yc, obs_c = horizon_labels(tp[h:], ep[h:], T_STAR)
    wc, _ = ipcw_weights_at_horizon(tp[h:], ep[h:], T_STAR)

    s = nonconformity(pic, pif, yc)
    if method == "naive":
        # unweighted-on-I: uniform weight on observed calibration points.
        w_cal = np.ones_like(wc)
    elif method == "oracle":
        # oracle covariate-conditional weights from the KNOWN censoring hazard.
        w_cal = _oracle_weights(tp[h:], ep[h:], ip["lam_c"][h:], T_STAR)
    else:  # ipcw = marginal-KM estimate (what the code ships)
        w_cal = wc
    qhat = weighted_quantile_threshold(s[obs_c], w_cal[obs_c], alpha)

    pic_t, pif_t = split_cif_at_horizon(forest, Xt, T_STAR)
    sets = prediction_sets(pic_t, pif_t, qhat)
    # TRUE population coverage: evaluate with ORACLE test weights (same evaluator for
    # all methods, so only the calibration-weight choice differs).
    wt_oracle = _oracle_weights(tt, et, it["lam_c"], T_STAR)
    cov, size = ipcw_coverage(sets, yt, wt_oracle, obs_t)
    cov1 = _cause1_coverage(sets, yt, obs_t)
    return cov, size, cov1


def run_cell(method, *, reps, **cfg):
    covs, sizes, cov1s = [], [], []
    for r in range(reps):
        c, sz, c1 = _one_rep(method, seed=1000 + r, **cfg)
        covs.append(c)
        sizes.append(sz)
        cov1s.append(c1)
    covs = np.array(covs)
    sizes = np.array(sizes)
    cov1s = np.array(cov1s)
    return (
        covs.mean(),
        covs.std(ddof=1) / np.sqrt(reps),
        sizes.mean(),
        sizes.std(ddof=1) / np.sqrt(reps),
        np.nanmean(cov1s),
    )


def main():
    base = dict(
        n_pool=2500,
        n_test=2500,
        n_estimators=100,
        reps=20,
        censor_rate=0.4,
        competing_frac=0.4,
        signal=1.0,
        censor_signal=0.0,
        alpha=0.1,
    )
    nominal = 1.0 - base["alpha"]
    print("\nHOME-TURF test: COVARIATE censoring (assumption C |= (T,eps)|X HOLDS)")
    print(
        "censor hazard ~ exp(censor_signal * X[:,0]); coverage evaluated with ORACLE test weights (true population coverage)"
    )
    print("3 methods: naive (unweighted) / ipcw (marginal-KM, shipped) / oracle (1/G(.|X), known)")
    print(f"config: {base}, t*={T_STAR}, nominal={nominal:.2f}\n")
    print(f"{'cens_sig':>8} {'method':>7} | {'true coverage':>16} | {'size':>8} | valid?")
    print("-" * 62)

    cs_grid = [0.0, 1.5, 3.0]
    rows = []
    for cs in cs_grid:
        cfg = {k: v for k, v in base.items() if k != "reps"}
        cfg["censor_signal"] = cs
        cfg["cause_censor"] = 0.0
        for method in ("naive", "ipcw", "oracle"):
            cov, cov_se, size, _size_se, _ = run_cell(method, reps=base["reps"], **cfg)
            valid = cov >= nominal - 3 * cov_se
            flag = "OK " if valid else "VIOLATES"
            print(f"{cs:>8.1f} {method:>7} | {cov:.3f} +/- {cov_se:.3f}   | {size:>8.2f} | {flag}")
            rows.append(
                dict(cs=cs, method=method, cov=cov, cov_se=cov_se, size=size, valid=bool(valid))
            )
        print("-" * 62)

    print("\n=== Home-turf summary (true population coverage) ===")
    for cs in cs_grid:
        n = next(r for r in rows if r["cs"] == cs and r["method"] == "naive")
        i = next(r for r in rows if r["cs"] == cs and r["method"] == "ipcw")
        o = next(r for r in rows if r["cs"] == cs and r["method"] == "oracle")
        print(
            f"  censor_signal={cs:.1f}: naive={n['cov']:.3f}({'OK' if n['valid'] else 'FAIL'}) "
            f"ipcw={i['cov']:.3f}({'OK' if i['valid'] else 'FAIL'}) "
            f"oracle={o['cov']:.3f}({'OK' if o['valid'] else 'FAIL'})"
        )
    print(
        "\nTheory predicts: strong covariate censoring -> naive undercovers, "
        "oracle 1/G(.|X) restores nominal (marginal-KM ipcw partial)."
    )


if __name__ == "__main__":
    main()
