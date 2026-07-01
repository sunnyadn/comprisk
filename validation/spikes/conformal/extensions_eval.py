"""Phase 3 (Gate 3) evaluation: Mondrian per-cause coverage + APS coherent sets,
compared to the marginal baseline. Split-conformal calibration, synthetic CR data.

Run:  PYTHONUNBUFFERED=1 uv run python -m validation.spikes.conformal.extensions_eval
"""

from __future__ import annotations

import numpy as np
from validation.spikes.conformal.coherent_sets import aps_scores, aps_sets, aps_threshold
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
from validation.spikes.conformal.mondrian import (
    assemble_P,
    mondrian_sets,
    mondrian_thresholds,
    per_class_coverage,
)
from validation.spikes.conformal.scores import nonconformity, split_cif_at_horizon

from comprisk import CompetingRiskForest

_ALPHA = 0.1
T_STAR = 1.0

# Module-level so report.py reuses the exact same config (single source of truth).
REPS = 12
CFG = dict(censor_rate=0.4, competing_frac=0.4, signal=1.5, n_pool=3000, n_test=3000, ntree=100)


def eval_extensions(pic_c, pif_c, yc, wc, obs_c, pic_t, pif_t, yt, wt, obs_t, alpha):
    """Marginal / Mondrian / APS evaluation from a calibrated + test split.

    Single source of truth for the Gate-3 extension math: given the calibration
    fold scores (``*_c`` + IPCW weights + observed mask) and the test fold scores
    (``*_t``), returns the marginal baseline, Mondrian per-class coverage, and APS
    coherent-set coverage. Shared by the synthetic (`_one_rep`) and real-cohort
    (`real_extensions`) harnesses so both compute identical quantities.

    Inputs use the same layout as the primitives: ``pic_*`` is (n, K) cause CIF,
    ``pif_*`` is (n,) event-free prob, ``y*`` horizon labels, ``w*`` IPCW weights,
    ``obs_*`` observed-before-t* mask.
    """
    Pc, Pt = assemble_P(pic_c, pif_c), assemble_P(pic_t, pif_t)

    # --- marginal baseline ---
    s = nonconformity(pic_c, pif_c, yc)
    qm = weighted_quantile_threshold(s[obs_c], wc[obs_c], alpha)
    sets_m = prediction_sets(pic_t, pif_t, qm)
    cov_m, size_m = ipcw_coverage(sets_m, yt, wt, obs_t)

    # --- Mondrian (per-class) ---
    qmon = mondrian_thresholds(Pc, yc, wc, obs_c, alpha)
    sets_mon = mondrian_sets(Pt, qmon)
    per_class, size_mon = per_class_coverage(sets_mon, yt, wt, obs_t)

    # --- APS coherent ---
    sa = aps_scores(Pc, yc)
    qa = aps_threshold(sa[obs_c], wc[obs_c], alpha)
    sets_a = aps_sets(Pt, qa)
    cov_a, size_a = ipcw_coverage(sets_a, yt, wt, obs_t)

    return dict(
        cov_m=cov_m,
        size_m=size_m,
        per_class=per_class,
        size_mon=size_mon,
        cov_a=cov_a,
        size_a=size_a,
    )


def _one_rep(*, censor_rate, competing_frac, signal, n_pool, n_test, ntree, seed):
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
    h = n_pool // 2
    forest = CompetingRiskForest(n_estimators=ntree, random_state=seed, n_jobs=-1).fit(
        Xp[:h], tp[:h], ep[:h]
    )
    pic_c, pif_c = split_cif_at_horizon(forest, Xp[h:], T_STAR)
    yc, obs_c = horizon_labels(tp[h:], ep[h:], T_STAR)
    wc, _ = ipcw_weights_at_horizon(tp[h:], ep[h:], T_STAR)

    pic_t, pif_t = split_cif_at_horizon(forest, Xt, T_STAR)
    yt, obs_t = horizon_labels(tt, et, T_STAR)
    wt, _ = ipcw_weights_at_horizon(tt, et, T_STAR)

    return eval_extensions(pic_c, pif_c, yc, wc, obs_c, pic_t, pif_t, yt, wt, obs_t, _ALPHA)


def main():
    reps = REPS
    cfg = dict(CFG)
    print(f"\nalpha={_ALPHA} nominal={1 - _ALPHA:.2f} reps={reps} cfg={cfg}\n")

    res = [_one_rep(seed=300 + r, **cfg) for r in range(reps)]

    def mean(k):
        return float(np.mean([r[k] for r in res]))

    print("Marginal baseline:")
    print(f"  coverage={mean('cov_m'):.3f}  size={mean('size_m'):.2f}")

    print("\nMondrian (per-class coverage; each should be ~nominal):")
    L = len(res[0]["per_class"])
    names = {**{c: f"cause{c + 1}" for c in range(L - 1)}, L - 1: "free"}
    for c in range(L):
        pc = float(np.nanmean([r["per_class"][c] for r in res]))
        print(f"  {names[c]:<8} cov={pc:.3f}")
    print(f"  overall size={mean('size_mon'):.2f}  (vs marginal {mean('size_m'):.2f})")

    print("\nAPS coherent sets:")
    print(
        f"  coverage={mean('cov_a'):.3f}  size={mean('size_a'):.2f}  "
        f"(vs marginal size {mean('size_m'):.2f})"
    )

    # Gate 3 checks
    mon_ok = all(
        abs(float(np.nanmean([r["per_class"][c] for r in res])) - (1 - _ALPHA)) <= 0.03
        for c in range(L)
    )
    # APS must cover AND be non-trivial (not the full label set). At K=2 the
    # deterministic APS degenerates to the full set -> useful negative result.
    aps_cov_ok = mean("cov_a") >= (1 - _ALPHA) - 0.02
    aps_nontrivial = mean("size_a") < L - 0.05
    aps_verdict = (
        "PASS"
        if (aps_cov_ok and aps_nontrivial)
        else ("DEGENERATE (covers but trivial full set)" if aps_cov_ok else "REVIEW")
    )
    print(
        f"\n=== Gate 3: Mondrian per-class {'PASS' if mon_ok else 'REVIEW'}; APS {aps_verdict} ==="
    )


if __name__ == "__main__":
    main()
