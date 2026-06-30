"""Phase 1 (Gate 1) of the paper empirics: conformal coverage on REAL cohorts.

No synthetic ground truth -- the horizon label is observed for every uncensored-
before-t* subject, so marginal coverage is IPCW-estimated over observed test
subjects exactly as in the spike. Repeated random pool/test splits give a coverage
mean +/- rep-SE per cohort x horizon x calibration path.

Run:
  PYTHONUNBUFFERED=1 uv run python -m validation.spikes.conformal.real_coverage
  PYTHONUNBUFFERED=1 uv run python -m validation.spikes.conformal.real_coverage --cohort seer
"""

from __future__ import annotations

import argparse

import numpy as np
from validation.spikes.conformal.conformal import (
    ipcw_coverage,
    prediction_sets,
    weighted_quantile_threshold,
)
from validation.spikes.conformal.dgp import horizon_labels, ipcw_weights_at_horizon
from validation.spikes.conformal.scores import (
    nonconformity,
    oob_cif_at_horizon,
    split_cif_at_horizon,
)

from comprisk import CompetingRiskForest


def _fit(X, time, event, *, n_estimators, seed):
    return CompetingRiskForest(n_estimators=n_estimators, random_state=seed, n_jobs=-1).fit(
        X, time, event
    )


def _one_split(path, X, time, event, t_star, *, test_frac, n_estimators, seed):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_test = round(test_frac * n)
    te, pool = perm[:n_test], perm[n_test:]
    Xt, tt, et = X[te], time[te], event[te]
    yt, obs_t = horizon_labels(tt, et, t_star)
    wt, _ = ipcw_weights_at_horizon(tt, et, t_star)

    if path == "oob":
        Xp, tp, ep = X[pool], time[pool], event[pool]
        forest = _fit(Xp, tp, ep, n_estimators=n_estimators, seed=seed)
        pic, pif, _ = oob_cif_at_horizon(forest, Xp, t_star)
        yc, obs_c = horizon_labels(tp, ep, t_star)
        wc, _ = ipcw_weights_at_horizon(tp, ep, t_star)
    else:  # split
        h = len(pool) // 2
        fit_i, cal_i = pool[:h], pool[h:]
        forest = _fit(X[fit_i], time[fit_i], event[fit_i], n_estimators=n_estimators, seed=seed)
        pic, pif = split_cif_at_horizon(forest, X[cal_i], t_star)
        yc, obs_c = horizon_labels(time[cal_i], event[cal_i], t_star)
        wc, _ = ipcw_weights_at_horizon(time[cal_i], event[cal_i], t_star)

    s = nonconformity(pic, pif, yc)
    qhat = weighted_quantile_threshold(s[obs_c], wc[obs_c], alpha=_ALPHA)
    pic_t, pif_t = split_cif_at_horizon(forest, Xt, t_star)
    sets = prediction_sets(pic_t, pif_t, qhat)
    return ipcw_coverage(sets, yt, wt, obs_t)


_ALPHA = 0.1


def run(cohort, *, n_sub, n_estimators, reps, test_frac):
    if cohort == "chf":
        from validation.spikes.conformal.data.chf import HORIZONS, load_chf

        X, time, event, _feats = load_chf(subsample=n_sub, seed=0)
    elif cohort == "seer":
        from validation.spikes.conformal.data.seer import HORIZONS, load_seer

        X, time, event, _feats = load_seer()
        if n_sub and n_sub < X.shape[0]:
            rng = np.random.default_rng(0)
            idx = rng.choice(X.shape[0], n_sub, replace=False)
            X, time, event = X[idx], time[idx], event[idx]
    else:
        raise ValueError(cohort)

    causes = sorted(c for c in np.unique(event) if c >= 1)
    cens = float(np.mean(event == 0))
    print(f"\ncohort={cohort} n={X.shape[0]} p={X.shape[1]} causes={causes} censored={cens:.3f}")
    print(f"alpha={_ALPHA} nominal={1 - _ALPHA:.2f} reps={reps} ntree={n_estimators}\n")

    print(f"  {'horizon':<9}{'path':<7}{'cov':>8}{'±SE':>8}{'dev':>8}{'size':>8}{'obs_te':>9}")
    g1 = []
    for name, t_star in HORIZONS.items():
        for path in ("oob", "split"):
            covs, sizes = [], []
            for r in range(reps):
                c, sz = _one_split(
                    path,
                    X,
                    time,
                    event,
                    t_star,
                    test_frac=test_frac,
                    n_estimators=n_estimators,
                    seed=100 + r,
                )
                covs.append(c)
                sizes.append(sz)
            covs = np.array(covs)
            mean, se = covs.mean(), covs.std(ddof=1) / np.sqrt(reps)
            dev = mean - (1 - _ALPHA)
            ok = abs(dev) <= 3 * se
            g1.append((name, path, ok, dev))
            print(
                f"  {name:<9}{path:<7}{mean:>8.3f}{se:>8.3f}{dev:>+8.3f}"
                f"{np.mean(sizes):>8.2f}{'':>9}{'ok' if ok else 'MISS'}"
            )

    # Gate 1: >=1 path holds across horizons; report OOB-conservativeness sign.
    for path in ("oob", "split"):
        devs = [d for (_, p, _, d) in g1 if p == path]
        oks = [o for (_, p, o, _) in g1 if p == path]
        print(
            f"\n  {path:<7}: {sum(oks)}/{len(oks)} horizons within 3SE; "
            f"mean dev={np.mean(devs):+.3f} "
            f"({'conservative' if np.mean(devs) > 0 else 'anti-conservative'})"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="chf", choices=["chf", "seer"])
    ap.add_argument("--n-sub", type=int, default=20000)
    ap.add_argument("--ntree", type=int, default=100)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--test-frac", type=float, default=0.3)
    a = ap.parse_args()
    run(a.cohort, n_sub=a.n_sub, n_estimators=a.ntree, reps=a.reps, test_frac=a.test_frac)


if __name__ == "__main__":
    main()
