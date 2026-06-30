"""Step 2 validation (Gate 2) for the conformal-CR spike.

Checks:
  1. pi-hat rows are proper distributions: pi_causes >= 0, pi_free >= -eps,
     sum == 1 (trivial by construction, but guards CIF-sum-exceeds-1 / negatives).
  2. Per-tree-loop + horizon-projection wiring is correct: full_cif_at_horizon
     equals forest.predict_cif(X, times=[t*]) to float tolerance (the OOB path
     reuses this exact wiring).
  3. OOB pi-hat on training and split pi-hat on a disjoint fold agree in
     distribution (label-mean per cause within a few SE) -> OOB scores are not
     grossly mis-distributed vs test scores (first read on Bostroem's concern).

Run:  PYTHONUNBUFFERED=1 uv run python -m validation.spikes.conformal.score_check
"""

from __future__ import annotations

import sys

import numpy as np
from validation.spikes.conformal.dgp import cr_dgp
from validation.spikes.conformal.scores import (
    full_cif_at_horizon,
    oob_cif_at_horizon,
    split_cif_at_horizon,
)

from comprisk import CompetingRiskForest


def _fit(X, time, event, *, n_estimators, seed):
    return CompetingRiskForest(n_estimators=n_estimators, random_state=seed, n_jobs=-1).fit(
        X, time, event
    )


def check_distribution(pi_causes, pi_free, tag, eps=1e-9):
    s = pi_causes.sum(axis=1) + pi_free
    ok_sum = np.allclose(s, 1.0, atol=eps)
    ok_neg = (pi_causes >= -eps).all() and (pi_free >= -eps).all()
    print(
        f"  [{tag}] sum==1: {ok_sum} (max|s-1|={np.abs(s - 1).max():.2e}); "
        f"nonneg: {ok_neg} (min pi_free={pi_free.min():.4f})"
    )
    return ok_sum and ok_neg


def main():
    t_star = 1.0
    X, time, event, _ = cr_dgp(
        4000, censor_rate=0.3, competing_frac=0.4, signal=1.0, t_star=t_star, seed=7
    )

    # --- oracle: per-tree loop == predict_cif at t* ---
    forest = _fit(X, time, event, n_estimators=100, seed=11)
    pic_full, pif_full = full_cif_at_horizon(forest, X, t_star)
    ref = forest.predict_cif(X, times=[t_star])[:, :, 0]
    max_dev = np.abs(pic_full - ref).max()
    ok_oracle = max_dev < 1e-10
    print(
        f"\nGate 2.2 per-tree-loop vs predict_cif: max|dev|={max_dev:.2e} "
        f"{'PASS' if ok_oracle else 'FAIL'}"
    )

    ok_dist_full = check_distribution(pic_full, pif_full, "full")

    # --- split vs OOB distribution agreement ---
    ntr = 3000
    Xtr, ttr, etr = X[:ntr], time[:ntr], event[:ntr]
    Xte = X[ntr:]
    forest2 = _fit(Xtr, ttr, etr, n_estimators=200, seed=13)

    pic_oob, pif_oob, count = oob_cif_at_horizon(forest2, Xtr, t_star)
    pic_spl, pif_spl = split_cif_at_horizon(forest2, Xte, t_star)
    ok_dist_oob = check_distribution(pic_oob, pif_oob, "oob")
    ok_dist_spl = check_distribution(pic_spl, pif_spl, "split")

    print(
        f"\n  OOB count per sample: min={count.min()} median={int(np.median(count))} "
        f"(samples with count==0: {(count == 0).sum()})"
    )

    # Label-mean agreement (population-level: both estimate E[F_k(t*)] / E[S(t*)]).
    cols = ["F1", "F2", "Sfree"]
    oob_means = np.r_[pic_oob.mean(0), pif_oob.mean()]
    spl_means = np.r_[pic_spl.mean(0), pif_spl.mean()]
    se = np.sqrt(0.25 / Xte.shape[0])
    print(f"\n  {'q':<7}{'oob':>9}{'split':>9}{'|diff|':>9}{'~3SE':>9}")
    ok_agree = True
    for c, mo, ms in zip(cols, oob_means, spl_means, strict=True):
        d = abs(mo - ms)
        within = d <= 5 * se  # generous: train/test are different folds
        ok_agree &= within
        print(f"  {c:<7}{mo:>9.4f}{ms:>9.4f}{d:>9.4f}{5 * se:>9.4f}  {'ok' if within else 'DIFF'}")

    ok = ok_oracle and ok_dist_full and ok_dist_oob and ok_dist_spl and ok_agree
    print(f"\n=== Gate 2: {'PASS' if ok else 'FAIL'} ===")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
