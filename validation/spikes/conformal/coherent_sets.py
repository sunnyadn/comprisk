"""Phase 3 extension: cross-cause-coherent (APS-style) prediction sets.

The marginal construction tests each label independently and ignores that the K+1
label probabilities sum to 1. APS (Romano-Sesia-Candes 2020) exploits the simplex:
order labels by predicted prob descending and take the smallest top-set whose
cumulative prob reaches the calibrated threshold. This is the coherent CR set; we
compare its coverage (must stay >= nominal) and efficiency (set size) to marginal.

P is (n, L), columns [cause 1..K, event-free] (same ordering as conformal/mondrian).
"""

from __future__ import annotations

import numpy as np
from validation.spikes.conformal.conformal import (
    _label_columns,
    weighted_quantile_threshold,
)


def aps_scores(P, y):
    """Deterministic APS nonconformity: cumulative prob mass of all labels at least
    as probable as the true label (i.e. mass needed before the true label is in)."""
    L = P.shape[1]
    cols = _label_columns(y, L - 1)
    true_p = P[np.arange(P.shape[0]), cols]
    # sum of probs of all labels at least as probable as the true label
    return np.where(true_p[:, None] <= P, P, 0.0).sum(axis=1)


def aps_threshold(scores, w, alpha):
    return weighted_quantile_threshold(scores, w, alpha)


def aps_sets(P, qhat):
    """Smallest top-prob prefix per row whose cumulative prob >= qhat."""
    order = np.argsort(-P, axis=1)
    sorted_p = np.take_along_axis(P, order, axis=1)
    csum = np.cumsum(sorted_p, axis=1)
    meets = csum >= qhat
    first = np.where(meets.any(axis=1), meets.argmax(axis=1), P.shape[1] - 1)
    ranks = np.arange(P.shape[1])[None, :]
    include_sorted = ranks <= first[:, None]
    sets = np.zeros_like(P, dtype=bool)
    np.put_along_axis(sets, order, include_sorted, axis=1)
    return sets
