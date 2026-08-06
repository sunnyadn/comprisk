"""Experiment 5 (design.md 0.6 #5): SEER competing-risks cohort -- EXTERNAL validity.

STATUS: STUB -- [S] Sunny owns the data export (has the SEER DUA); this file is the
analysis pipeline waiting for it, NOT a synthetic sim. It intentionally refuses to run
until a real cohort is present, so it can never fabricate an "external validation".

Expected input: a parquet/CSV at $SEER_CR_PATH (or --path) with columns
    time (float, months or years), event (int: 0=censored, k>=1=cause), plus covariates.
Recommended: breast-cancer CR (cause 1 = breast-cancer death, cause 2 = other-cause
death, censored = alive at last follow-up), horizon t* = 5 years -- matching the
CLR-2023 fixed-window comparator we cite for estimand-preservation.

Pipeline (identical estimator to the synthetic experiments, so numbers are comparable):
    load -> horizon_labels at t* -> split fit CompetingRiskForest -> IPCW KM weights
    -> weighted split-conformal (1/g_min atom) -> marginal + per-cause (Mondrian) coverage.

Because true G is unknown on real data, coverage here is the IPCW-KM estimate (the
honest best available) -- flagged as such, unlike the oracle-G synthetic checks.

Run (after export):
    SEER_CR_PATH=/path/to/seer_breast_cr.parquet \
      uv run python -m validation.spikes.conformal.experiments.e5_seer
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
    for col in ("time", "event"):
        if col not in df.columns:
            raise SystemExit(f"SEER export missing required column '{col}'; got {list(df.columns)}")
    time = df["time"].to_numpy(float)
    event = df["event"].to_numpy(np.int64)
    X = df.drop(columns=["time", "event"]).to_numpy(float)
    return X, time, event


def run(path, t_star, *, seed=0):
    X, time, event = _load(path)
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

    print(f"\nSEER external check @ t*={t_star} (n={n}; coverage = IPCW-KM estimate, not oracle)")
    print(
        f"  marginal coverage = {cov:.3f}   mean set size = {size:.2f}   (nominal {1 - ALPHA:.2f})"
    )
    sets_mond = assemble_P(pic_t, pif_t) >= (1.0 - qm)[None, :]
    cols = _label_columns(yt, K)
    print("  per-cause (Mondrian) coverage:")
    for c in range(K + 1):
        m = obs_t & (cols == c)
        if m.any():
            wm = wt[m]
            pc = (wm * sets_mond[m, c]).sum() / wm.sum()
            print(f"    label {c}: {pc:.3f}  (n={int(m.sum())})")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=os.environ.get("SEER_CR_PATH"))
    ap.add_argument("--t-star", type=float, default=5.0)
    args = ap.parse_args()
    if not args.path:
        sys.exit(
            "e5_seer is a STUB awaiting Sunny's SEER export.\n"
            "Set SEER_CR_PATH=/path/to/cohort.parquet (cols: time,event,+covariates) or pass --path.\n"
            "No synthetic fallback: this file only ever runs on the real DUA cohort."
        )
    if not os.path.exists(args.path):
        sys.exit(f"SEER export not found at {args.path!r}")
    run(args.path, args.t_star)


if __name__ == "__main__":
    main()
