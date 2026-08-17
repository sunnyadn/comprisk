"""Experiment 5 (design.md 0.6 #5): SEER competing-risks cohort -- EXTERNAL validity.

STATUS: LIVE -- first run 2026-08-10 on the SEER breast cohort Sunny exported under his
own DUA (238,057 cases, 2010-2015). Still refuses to run without a real cohort, so it can
never fabricate an "external validation"; the data itself never enters any repo.

Expected input: a parquet/CSV at $SEER_CR_PATH (or --path) with columns
    time (float), event or status (int: 0=censored, k>=1=cause), plus covariates.
This is the schema gen_seer_breast.py emits (x0..xK + time + status, time in MONTHS,
the SEER survival-time unit); --t-star is read in the cohort's own time unit.
Recommended: breast-cancer CR (cause 1 = breast-cancer death, cause 2 = other-cause
death, censored = alive at last follow-up), horizon t* = 60 months (5 years) -- matching
the CLR-2023 fixed-window comparator we cite for estimand-preservation.

Pipeline (identical estimator to the synthetic experiments, so numbers are comparable):
    load -> horizon_labels at t* -> split fit CompetingRiskForest -> IPCW KM weights
    -> weighted split-conformal (1/g_min atom) -> marginal + per-cause (Mondrian) coverage.

Because true G is unknown on real data, coverage here is the IPCW-KM estimate (the
honest best available) -- flagged as such, unlike the oracle-G synthetic checks.

Run (after export):
    SEER_CR_PATH=/path/to/seer_breast_cr.parquet \
      uv run python -m validation.spikes.conformal.experiments.e5_seer --reps 5

Reported numbers are averaged over `--reps` random train/calibrate/test splits: a single
split fluctuates by about one sd per cause, too noisy to put in a table on its own.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from validation.spikes.conformal.conformal import (
    _label_columns,
    ipcw_coverage,
    prediction_sets,
    weighted_quantile_threshold,
)
from validation.spikes.conformal.dgp import horizon_labels, ipcw_weights_at_horizon
from validation.spikes.conformal.mondrian import assemble_P, mondrian_thresholds
from validation.spikes.conformal.scores import nonconformity, split_cif_at_horizon

from comprisk import CompetingRiskForest

ALPHA = 0.1


def _load(path):
    """Load a cohort into (X, time, event). Supports parquet/csv via pandas."""
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise SystemExit("pandas required to load the SEER export: pip install pandas") from e
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    # gen_seer_breast.py names the cause column "status"; accept either spelling.
    cause_col = next((c for c in ("event", "status") if c in df.columns), None)
    if "time" not in df.columns or cause_col is None:
        raise SystemExit(
            f"SEER export needs columns 'time' and 'event'/'status'; got {list(df.columns)}"
        )
    time = df["time"].to_numpy(float)
    event = df[cause_col].to_numpy(np.int64)
    X = df.drop(columns=["time", cause_col]).to_numpy(float)
    return X, time, event


def _split_once(X, time, event, t_star, seed):
    """One train/calibrate/test split; returns (marginal cov, mean size, per-label cov, counts)."""
    n = time.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    tr, ca = perm[: n // 3], perm[n // 3 : 2 * n // 3]
    te = perm[2 * n // 3 :]

    forest = CompetingRiskForest(n_estimators=200, random_state=seed, n_jobs=-1).fit(
        X[tr], time[tr], event[tr]
    )
    yc, obs_c = horizon_labels(time[ca], event[ca], t_star)
    wc, _ = ipcw_weights_at_horizon(time[ca], event[ca], t_star)
    pic, pif = split_cif_at_horizon(forest, X[ca], t_star)
    K = pic.shape[1]

    s = nonconformity(pic, pif, yc)
    qhat = weighted_quantile_threshold(s[obs_c], wc[obs_c], alpha=ALPHA)
    qm = mondrian_thresholds(assemble_P(pic, pif), yc, wc, obs_c, ALPHA)

    yt, obs_t = horizon_labels(time[te], event[te], t_star)
    wt, _ = ipcw_weights_at_horizon(time[te], event[te], t_star)
    pic_t, pif_t = split_cif_at_horizon(forest, X[te], t_star)
    sets_marg = prediction_sets(pic_t, pif_t, qhat)
    cov, size = ipcw_coverage(sets_marg, yt, wt, obs_t)

    sets_mond = assemble_P(pic_t, pif_t) >= (1.0 - qm)[None, :]
    cols = _label_columns(yt, K)
    per_cause, counts = [], []
    for c in range(K + 1):
        m = obs_t & (cols == c)
        if m.any():
            wm = wt[m]
            per_cause.append(float((wm * sets_mond[m, c]).sum() / wm.sum()))
        else:
            per_cause.append(np.nan)
        counts.append(int(m.sum()))
    return cov, size, np.array(per_cause), counts


def _label_name(c, K):
    return "event-free" if c == K else f"cause {c + 1}"


def run(path, t_star, *, reps=5, seed0=0):
    X, time, event = _load(path)
    n = time.shape[0]
    out = [_split_once(X, time, event, t_star, seed0 + r) for r in range(reps)]
    cov = np.array([o[0] for o in out])
    size = np.array([o[1] for o in out])
    pc = np.vstack([o[2] for o in out])
    counts = out[0][3]
    K = pc.shape[1] - 1

    def sd(v, fmt):
        # A single split has no spread to report; don't print a bare nan.
        return format(np.std(v, ddof=1), fmt) if reps > 1 else "--"

    print(f"\nSEER external check @ t*={t_star} (n={n}, reps={reps} random splits)")
    print("coverage = IPCW-KM estimate (true G unknown on real data), not oracle\n")
    print(
        f"  marginal coverage = {cov.mean():.3f} +/- {sd(cov, '.3f')}   (nominal {1 - ALPHA:.2f})"
    )
    print(f"  mean set size     = {size.mean():.2f} +/- {sd(size, '.2f')}   (max {K + 1})")
    print("\n  per-cause (Mondrian) coverage:")
    print(f"    {'label':<13}{'coverage':>10}{'sd':>8}{'n (rep 1)':>12}")
    for c in range(K + 1):
        print(
            f"    {_label_name(c, K):<13}{pc[:, c].mean():>10.3f}"
            f"{sd(pc[:, c], '.3f'):>8}{counts[c]:>12,}"
        )
    return cov, size, pc


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=os.environ.get("SEER_CR_PATH"))
    # In the cohort's own time unit: SEER survival time is months, so 60 = 5 years.
    ap.add_argument("--t-star", type=float, default=60.0)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    if not args.path:
        sys.exit(
            "e5_seer needs the real SEER cohort; none was given.\n"
            "Set SEER_CR_PATH=/path/to/cohort.parquet (cols: time, event|status, +covariates) "
            "or pass --path.\n"
            "Rebuild it with: python validation/gen_seer_breast.py --src ~/data/seer/export.csv\n"
            "No synthetic fallback: this file only ever runs on the real DUA cohort."
        )
    if not os.path.exists(args.path):
        sys.exit(f"SEER export not found at {args.path!r}")
    run(args.path, args.t_star, reps=args.reps)


if __name__ == "__main__":
    main()
