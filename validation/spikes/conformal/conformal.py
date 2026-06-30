"""Step 3 of the conformal-CR spike: weighted conformal threshold and prediction
sets over the K+1 horizon labels.

Weighted split conformal (Tibshirani et al. 2019) with IPCW calibration weights:
the observed (uncensored-before-t*) subjects are a censoring-biased sample of the
population, so each carries weight w_i = 1/Ghat(min(T_i,t*)^-). The threshold is the
weighted (1-alpha) quantile of calibration nonconformity scores, with a finite-
sample point mass at +inf carrying the expected test-point weight.

Prediction set for a subject:  S(x) = { label l : pi_l(t*|x) >= 1 - qhat }.

Coverage is itself an IPCW estimate over the POPULATION: test subjects are also
censoring-biased, so empirical coverage = (sum_i w_i 1{y_i in S_i}) / (sum_i w_i)
over observed test subjects -- the same correction Step 1 used to recover the true
marginal label proportions.
"""

from __future__ import annotations

import numpy as np
from validation.spikes.conformal.dgp import EVENT_FREE


def weighted_quantile_threshold(scores, weights, alpha, test_weight=None):
    """Smallest s such that the (finite-sample, test-inflated) weighted CDF of
    calibration scores reaches 1 - alpha.

    A +inf atom with mass ``test_weight`` (default: mean calibration weight)
    accounts for the unknown test-point rank, the weighted analogue of the
    (n+1) split-conformal correction. If the atom alone exceeds alpha the set is
    everything (threshold = +inf).
    """
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if scores.size == 0:
        return np.inf
    if test_weight is None:
        test_weight = weights.mean()

    order = np.argsort(scores, kind="stable")
    s_sorted = scores[order]
    w_sorted = weights[order]
    total = weights.sum() + test_weight
    # Normalised cumulative mass up to and including each calibration score.
    cum = np.cumsum(w_sorted) / total
    # First score whose cumulative mass >= 1 - alpha.
    target = 1.0 - alpha
    idx = np.searchsorted(cum, target, side="left")
    if idx >= s_sorted.size:
        return np.inf
    return float(s_sorted[idx])


def prediction_sets(pi_causes, pi_free, qhat):
    """Boolean (n, K+1) inclusion matrix. Columns 0..K-1 = causes 1..K, col K = free.

    Include label l iff pi_l >= 1 - qhat  (i.e. nonconformity 1 - pi_l <= qhat).
    """
    thr = 1.0 - qhat
    n, K = pi_causes.shape
    sets = np.zeros((n, K + 1), dtype=bool)
    sets[:, :K] = pi_causes >= thr
    sets[:, K] = pi_free >= thr
    return sets


def _label_columns(y, K):
    """Map labels {EVENT_FREE, 1..K} to column indices {K, 0..K-1}."""
    y = np.asarray(y, dtype=np.int64)
    col = np.where(y == EVENT_FREE, K, y - 1)
    return col


def ipcw_coverage(sets, y, w, observed):
    """IPCW population coverage and weighted mean set size over observed subjects."""
    K = sets.shape[1] - 1
    obs = observed
    cols = _label_columns(y[obs], K)
    covered = sets[obs][np.arange(obs.sum()), cols]
    wo = w[obs]
    W = wo.sum()
    coverage = float((wo * covered).sum() / W)
    mean_size = float((wo * sets[obs].sum(axis=1)).sum() / W)
    return coverage, mean_size
