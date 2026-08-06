"""Experiment 4 (design.md 0.6 #4): marginal coverage DECEIVES; per-cause is the story.

Validates Remark rem:mondrian and the paper's clinical sell. A single MARGINAL
threshold gives pooled coverage ~1-alpha but can badly mis-cover individual causes --
especially a RARE competing cause. Mondrian (class-conditional) thresholds restore
P(Y in S | Y=c) >= 1-alpha per cause, at the cost of set size. We report, per label,
coverage under the marginal threshold vs the Mondrian thresholds.

All weights are ORACLE (true G), so per-cause coverage numbers are unbiased population
quantities. cause 2 is made rare via a small ``competing_frac``.

Run:  PYTHONUNBUFFERED=1 uv run python -m validation.spikes.conformal.experiments.e4_mondrian_percause
"""

from __future__ import annotations

import numpy as np
from validation.spikes.conformal.conformal import (
    _label_columns,
    prediction_sets,
    weighted_quantile_threshold,
)
from validation.spikes.conformal.dgp import cr_dgp, horizon_labels
from validation.spikes.conformal.experiments.oracle_g import (
    oracle_censoring_rate,
    oracle_ipcw_weights_at_horizon,
)
from validation.spikes.conformal.mondrian import assemble_P, mondrian_thresholds
from validation.spikes.conformal.scores import nonconformity, split_cif_at_horizon

from comprisk import CompetingRiskForest

ALPHA = 0.1
REPS = 20
T_STAR = 1.0
S_AT_TSTAR = 0.5
KW = dict(censor_rate=0.5, competing_frac=0.15, signal=1.0)  # cause 2 rare
LABELS = {0: "free", 1: "cause1", 2: "cause2"}


def _per_cause_coverage(sets, y, w, observed, K):
    """Oracle-weighted coverage within each true label class."""
    cols = _label_columns(y, K)
    out = {}
    for c in range(K + 1):
        m = observed & (cols == c)
        if not m.any():
            out[c] = (float("nan"), 0.0)
            continue
        incl = sets[m, c]
        wm = w[m]
        out[c] = (float((wm * incl).sum() / wm.sum()), float(wm.sum()))
    return out


def _one_rep(seed):
    lam_c = oracle_censoring_rate(
        t_star=T_STAR, censor_rate=KW["censor_rate"], s_at_tstar=S_AT_TSTAR
    )
    Xp, tp, ep, _ = cr_dgp(3000, seed=seed, t_star=T_STAR, s_at_tstar=S_AT_TSTAR, **KW)
    Xt, tt, et, _ = cr_dgp(6000, seed=seed + 100_000, t_star=T_STAR, s_at_tstar=S_AT_TSTAR, **KW)

    h = 1500
    forest = CompetingRiskForest(n_estimators=100, random_state=seed, n_jobs=-1).fit(
        Xp[:h], tp[:h], ep[:h]
    )
    cal_t, cal_e, cal_X = tp[h:], ep[h:], Xp[h:]
    yc, obs_c = horizon_labels(cal_t, cal_e, T_STAR)
    wc, _ = oracle_ipcw_weights_at_horizon(cal_t, cal_e, T_STAR, lam_c=lam_c)
    pic, pif = split_cif_at_horizon(forest, cal_X, T_STAR)
    P_cal = assemble_P(pic, pif)
    K = pic.shape[1]

    # Marginal threshold.
    s = nonconformity(pic, pif, yc)
    qhat = weighted_quantile_threshold(s[obs_c], wc[obs_c], alpha=ALPHA)
    # Mondrian thresholds.
    qm = mondrian_thresholds(P_cal, yc, wc, obs_c, ALPHA)

    # Test.
    yt, obs_t = horizon_labels(tt, et, T_STAR)
    wt, _ = oracle_ipcw_weights_at_horizon(tt, et, T_STAR, lam_c=lam_c)
    pic_t, pif_t = split_cif_at_horizon(forest, Xt, T_STAR)
    P_te = assemble_P(pic_t, pif_t)

    sets_marg = prediction_sets(pic_t, pif_t, qhat)
    sets_mond = P_te >= (1.0 - qm)[None, :]

    return (
        _per_cause_coverage(sets_marg, yt, wt, obs_t, K),
        _per_cause_coverage(sets_mond, yt, wt, obs_t, K),
        sets_marg[obs_t].sum(1).mean(),
        sets_mond[obs_t].sum(1).mean(),
    )


def main():
    nominal = 1 - ALPHA
    print(f"\nExp 4 -- Mondrian per-cause (alpha={ALPHA}, nominal={nominal:.2f}, reps={REPS})")
    print("cause 2 is rare (competing_frac=0.15). Oracle-weighted per-cause coverage.\n")
    marg = {c: [] for c in LABELS}
    mond = {c: [] for c in LABELS}
    sz_m, sz_d = [], []
    for r in range(REPS):
        cm, cd, sm, sd = _one_rep(200 + r)
        for c in LABELS:
            marg[c].append(cm[c][0])
            mond[c].append(cd[c][0])
        sz_m.append(sm)
        sz_d.append(sd)

    print(f"  {'label':<9}{'marginal-thr':>14}{'mondrian-thr':>14}")
    for c, name in LABELS.items():
        print(f"  {name:<9}{np.nanmean(marg[c]):>14.3f}{np.nanmean(mond[c]):>14.3f}")
    print(f"\n  mean set size: marginal={np.mean(sz_m):.2f}  mondrian={np.mean(sz_d):.2f}")
    print(
        "\nExpect: marginal-thr pools near nominal but the RARE cause2 drifts off it;"
        "\nmondrian-thr pulls every cause back to >= nominal (larger sets = the price)."
    )


if __name__ == "__main__":
    main()
