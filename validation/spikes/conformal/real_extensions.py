"""Phase 3 CR-specific extensions on the REAL cohort (split-conformal calibration).

The Gate-3 extensions (Mondrian per-cause conditional coverage + APS coherent sets)
were validated on synthetic ``cr_dgp`` in ``extensions_eval``. This harness runs the
SAME evaluation on the real local CR cohort, reusing:

  * real_coverage's split scaffold  -- repeated random pool/test splits, fit on half
    the pool, calibrate on the other half, build sets on the held-out test fold.
  * extensions_eval.eval_extensions  -- the single source of truth for the
    marginal / Mondrian / APS math (no duplicated coverage/quantile/set logic).

Mondrian groups by CANDIDATE label (mondrian.py evaluates per-candidate-class), so
there is no true-label-at-test leakage. Coverage is IPCW-estimated over observed
(uncensored-before-t*) test subjects, exactly as in the spike.

CONFIDENTIALITY: the CHF cohort schema is secret. This module prints/writes only
AGGREGATE coverage / set-size numbers per horizon -- never feature names, the source
path, raw class counts, or outcome coding. The loader's feature names are discarded.

Run:
  PYTHONUNBUFFERED=1 uv run python -m validation.spikes.conformal.real_extensions
  PYTHONUNBUFFERED=1 uv run python -m validation.spikes.conformal.real_extensions --quick
"""

from __future__ import annotations

import argparse

import numpy as np
from validation.spikes.conformal.dgp import horizon_labels, ipcw_weights_at_horizon
from validation.spikes.conformal.extensions_eval import _ALPHA, eval_extensions
from validation.spikes.conformal.scores import split_cif_at_horizon

from comprisk import CompetingRiskForest

NOMINAL = 1.0 - _ALPHA


def _fit(X, time, event, *, n_estimators, seed):
    return CompetingRiskForest(n_estimators=n_estimators, random_state=seed, n_jobs=-1).fit(
        X, time, event
    )


def one_split(X, time, event, t_star, *, test_frac, n_estimators, seed):
    """One real-cohort split -> marginal / Mondrian / APS dict (via eval_extensions).

    Split path only: fit on half the pool, calibrate on the other half, build sets
    on the held-out test fold. Mirrors real_coverage._one_split's scaffold.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_test = round(test_frac * n)
    te, pool = perm[:n_test], perm[n_test:]

    Xt, tt, et = X[te], time[te], event[te]
    yt, obs_t = horizon_labels(tt, et, t_star)
    wt, _ = ipcw_weights_at_horizon(tt, et, t_star)

    h = len(pool) // 2
    fit_i, cal_i = pool[:h], pool[h:]
    forest = _fit(X[fit_i], time[fit_i], event[fit_i], n_estimators=n_estimators, seed=seed)

    pic_c, pif_c = split_cif_at_horizon(forest, X[cal_i], t_star)
    yc, obs_c = horizon_labels(time[cal_i], event[cal_i], t_star)
    wc, _ = ipcw_weights_at_horizon(time[cal_i], event[cal_i], t_star)

    pic_t, pif_t = split_cif_at_horizon(forest, Xt, t_star)

    return eval_extensions(pic_c, pif_c, yc, wc, obs_c, pic_t, pif_t, yt, wt, obs_t, _ALPHA)


def _load(cohort, n_sub):
    """Return (X, time, event, HORIZONS); feature names discarded (confidentiality)."""
    if cohort == "chf":
        from validation.spikes.conformal.data.chf import HORIZONS, load_chf

        X, time, event, _ = load_chf(subsample=n_sub, seed=0)
    elif cohort == "seer":
        from validation.spikes.conformal.data.seer import HORIZONS, load_seer

        X, time, event, _ = load_seer()
        if n_sub and n_sub < X.shape[0]:
            rng = np.random.default_rng(0)
            idx = rng.choice(X.shape[0], n_sub, replace=False)
            X, time, event = X[idx], time[idx], event[idx]
    else:
        raise ValueError(cohort)
    return X, time, event, HORIZONS


def run(cohort, *, n_sub, n_estimators, reps, test_frac):
    X, time, event, horizons = _load(cohort, n_sub)
    causes = sorted(int(c) for c in np.unique(event) if c >= 1)
    cens = float(np.mean(event == 0))
    print(f"\ncohort={cohort} n={X.shape[0]} p={X.shape[1]} causes={causes} censored={cens:.3f}")
    print(f"alpha={_ALPHA} nominal={NOMINAL:.2f} reps={reps} ntree={n_estimators} (split path)\n")

    for hname, t_star in horizons.items():
        res = [
            one_split(
                X,
                time,
                event,
                t_star,
                test_frac=test_frac,
                n_estimators=n_estimators,
                seed=100 + r,
            )
            for r in range(reps)
        ]

        def mean(k, _res=res):
            return float(np.mean([r[k] for r in _res]))

        cov_m, size_m = mean("cov_m"), mean("size_m")
        L = len(res[0]["per_class"])
        names = {**{c: f"cause{c + 1}" for c in range(L - 1)}, L - 1: "free"}

        print(f"[{hname}]")
        print(f"  marginal  cov={cov_m:.3f}  size={size_m:.2f}")
        print("  Mondrian per-cause coverage (each should be ~nominal):")
        mon_ok = True
        for c in range(L):
            pc = float(np.nanmean([r["per_class"][c] for r in res]))
            mon_ok &= abs(pc - NOMINAL) <= 0.03
            print(f"    {names[c]:<8} cov={pc:.3f}")
        size_mon = mean("size_mon")
        cov_a, size_a = mean("cov_a"), mean("size_a")
        print(f"    overall size={size_mon:.2f}  (vs marginal {size_m:.2f})")
        print(f"  APS       cov={cov_a:.3f}  size={size_a:.2f}  (vs marginal size {size_m:.2f})")

        aps_cov_ok = cov_a >= NOMINAL - 0.02
        aps_nontrivial = size_a < L - 0.05
        aps_verdict = (
            "PASS"
            if (aps_cov_ok and aps_nontrivial)
            else ("DEGENERATE (covers but trivial full set)" if aps_cov_ok else "REVIEW")
        )
        print(f"  => Mondrian {'PASS' if mon_ok else 'REVIEW'}; APS {aps_verdict}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", default="chf", choices=["chf", "seer"])
    ap.add_argument("--quick", action="store_true", help="fast smoke profile")
    ap.add_argument("--n-sub", type=int, default=None)
    ap.add_argument("--ntree", type=int, default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--test-frac", type=float, default=0.3)
    a = ap.parse_args()

    if a.quick:
        n_sub, ntree, reps = 1500, 20, 2
    else:  # paper-grade
        n_sub, ntree, reps = 20000, 100, 10
    if a.n_sub is not None:
        n_sub = a.n_sub
    if a.ntree is not None:
        ntree = a.ntree
    if a.reps is not None:
        reps = a.reps

    run(a.cohort, n_sub=n_sub, n_estimators=ntree, reps=reps, test_frac=a.test_frac)


if __name__ == "__main__":
    main()
