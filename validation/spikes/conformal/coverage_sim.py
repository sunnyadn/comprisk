"""Step 4 (Gate 3 + the deliverable sweep) for the conformal-CR spike.

Compares two calibration substrates at a matched data budget:

  Path A (oob)   : fit the forest on the whole pool, calibrate on OOB scores of
                   that same pool (free calibration; tests Bostroem's open
                   OOB-validity concern).
  Path B (split) : split the pool 50/50, fit on one half, calibrate on the other
                   (clean exchangeability; the safe fallback).

Both evaluate on the SAME fresh test set with the full ensemble, and coverage is an
IPCW population estimate over observed test subjects.

Gate 3 (degenerate, censor=0) must reproduce nominal coverage for both paths before
trusting the censored sweep.

Run:  PYTHONUNBUFFERED=1 uv run python -m validation.spikes.conformal.coverage_sim
"""

from __future__ import annotations

import sys

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
from validation.spikes.conformal.scores import (
    nonconformity,
    oob_cif_at_horizon,
    split_cif_at_horizon,
)

from comprisk import CompetingRiskForest

T_STAR = 1.0


def _fit(X, time, event, *, n_estimators, seed):
    return CompetingRiskForest(n_estimators=n_estimators, random_state=seed, n_jobs=-1).fit(
        X, time, event
    )


def _one_rep(
    path, *, censor_rate, competing_frac, signal, alpha, n_pool, n_test, n_estimators, seed
):
    Xp, tp, ep, _ = cr_dgp(
        n_pool,
        censor_rate=censor_rate,
        competing_frac=competing_frac,
        signal=signal,
        t_star=T_STAR,
        seed=seed,
    )
    Xt, tt, et, _ = cr_dgp(
        n_test,
        censor_rate=censor_rate,
        competing_frac=competing_frac,
        signal=signal,
        t_star=T_STAR,
        seed=seed + 100_000,
    )
    yt, obs_t = horizon_labels(tt, et, T_STAR)
    wt, _ = ipcw_weights_at_horizon(tt, et, T_STAR)

    if path == "oob":
        forest = _fit(Xp, tp, ep, n_estimators=n_estimators, seed=seed)
        pic, pif, _ = oob_cif_at_horizon(forest, Xp, T_STAR)
        yc, obs_c = horizon_labels(tp, ep, T_STAR)
        wc, _ = ipcw_weights_at_horizon(tp, ep, T_STAR)
    else:  # split
        h = n_pool // 2
        forest = _fit(Xp[:h], tp[:h], ep[:h], n_estimators=n_estimators, seed=seed)
        pic, pif = split_cif_at_horizon(forest, Xp[h:], T_STAR)
        yc, obs_c = horizon_labels(tp[h:], ep[h:], T_STAR)
        wc, _ = ipcw_weights_at_horizon(tp[h:], ep[h:], T_STAR)

    s = nonconformity(pic, pif, yc)
    qhat = weighted_quantile_threshold(s[obs_c], wc[obs_c], alpha)

    pic_t, pif_t = split_cif_at_horizon(forest, Xt, T_STAR)
    sets = prediction_sets(pic_t, pif_t, qhat)
    cov, size = ipcw_coverage(sets, yt, wt, obs_t)
    return cov, size


def run_cell(path, *, reps, **cfg):
    covs, sizes = [], []
    for r in range(reps):
        c, sz = _one_rep(path, seed=1000 + r, **cfg)
        covs.append(c)
        sizes.append(sz)
    covs = np.array(covs)
    return covs.mean(), covs.std(ddof=1) / np.sqrt(reps), np.mean(sizes)


def _row(tag, path, *, reps, alpha, **cfg):
    mean, se, size = run_cell(path, reps=reps, alpha=alpha, **cfg)
    nominal = 1.0 - alpha
    dev = mean - nominal
    ok = abs(dev) <= 3 * se if se > 0 else abs(dev) <= 0.01
    print(
        f"  {tag:<28}{path:<7}cov={mean:.3f}±{se:.3f}  nom={nominal:.2f}  "
        f"dev={dev:+.3f}  size={size:.2f}  {'ok' if ok else 'MISS'}"
    )
    return ok, dev, se


def main():
    base = dict(n_pool=2500, n_test=2500, n_estimators=100, reps=15)
    print(f"\nconfig: {base}, t*={T_STAR}\n")

    # --- Gate 3: degenerate (no censoring) must hit nominal for both paths ---
    print("Gate 3 (degenerate, censor=0):")
    g3 = True
    for path in ("oob", "split"):
        ok, _, _ = _row(
            "censor=0 sig=1 a=.1",
            path,
            censor_rate=0.0,
            competing_frac=0.4,
            signal=1.0,
            alpha=0.1,
            **base,
        )
        g3 &= ok
    print(f"  -> Gate 3 {'PASS' if g3 else 'FAIL'}\n")

    # --- the sweep: censoring x competing fraction x alpha ---
    print("Sweep (censored):")
    grid = [
        dict(censor_rate=0.2, competing_frac=0.4, signal=1.0, alpha=0.1),
        dict(censor_rate=0.4, competing_frac=0.4, signal=1.0, alpha=0.1),
        dict(censor_rate=0.6, competing_frac=0.3, signal=1.0, alpha=0.1),
        dict(censor_rate=0.4, competing_frac=0.5, signal=1.0, alpha=0.2),
    ]
    res = {"oob": [], "split": []}
    for cfg in grid:
        tag = f"c={cfg['censor_rate']} cf={cfg['competing_frac']} a={cfg['alpha']}"
        for path in ("oob", "split"):
            ok, dev, se = _row(tag, path, **cfg, **base)
            res[path].append((ok, dev, se))

    print("\n=== Verdict inputs ===")
    for path in ("oob", "split"):
        oks = [r[0] for r in res[path]]
        worst = max(abs(r[1]) for r in res[path])
        print(
            f"  {path:<7}: {sum(oks)}/{len(oks)} cells within 3SE of nominal; "
            f"worst |dev|={worst:.3f}"
        )

    any_path_passes = any(all(r[0] for r in res[path]) for path in ("oob", "split"))
    print(
        f"\n=== Coverage sweep: {'PASS' if (g3 and any_path_passes) else 'REVIEW'} "
        f"(>=1 path holds across grid: {any_path_passes}) ==="
    )
    if not g3:
        sys.exit(1)


if __name__ == "__main__":
    main()
