"""Phase 3 extension: Mondrian (class-conditional) conformal for CR horizon labels.

Marginal conformal guarantees P(Y in S(X)) >= 1-alpha pooled over labels. Mondrian
guarantees it PER LABEL: P(l in S(X) | Y=l) >= 1-alpha for each l. No test-label
leakage -- the threshold for candidate label l is computed from CALIBRATION points
whose true label is l, and at test we only evaluate candidate l's predicted prob.

P is the (n, L) label-probability matrix with columns [cause 1..K, event-free],
matching conformal.prediction_sets / _label_columns ordering.
"""

from __future__ import annotations

import numpy as np
from validation.spikes.conformal.conformal import (
    _label_columns,
    weighted_quantile_threshold,
)


def assemble_P(pi_causes, pi_free):
    """(pi_causes (n,K), pi_free (n,)) -> P (n, K+1), cols [cause1..causeK, free]."""
    return np.column_stack([pi_causes, pi_free])


def mondrian_thresholds(P, y, w, observed, alpha):
    """Per-label weighted (1-alpha) quantile of class-conditional nonconformity."""
    L = P.shape[1]
    cols = _label_columns(y, L - 1)
    q = np.empty(L)
    for c in range(L):
        m = observed & (cols == c)
        if not m.any():
            q[c] = np.inf
            continue
        s = 1.0 - P[m, c]
        q[c] = weighted_quantile_threshold(s, w[m], alpha)
    return q


def mondrian_sets(P, q_per_label):
    """Include label c iff P[:, c] >= 1 - q_c."""
    return (1.0 - q_per_label)[None, :] <= P


def per_class_coverage(sets, y, w, observed):
    """IPCW coverage within each true-label class: P(l in S | Y=l).

    Returns (per_class dict {col: cov}, weighted mean set size over observed).
    """
    L = sets.shape[1]
    cols = _label_columns(y, L - 1)
    out = {}
    for c in range(L):
        m = observed & (cols == c)
        if not m.any():
            out[c] = float("nan")
            continue
        wc = w[m]
        out[c] = float((wc * sets[m, c]).sum() / wc.sum())
    wo = w[observed]
    mean_size = float((wo * sets[observed].sum(axis=1)).sum() / wo.sum())
    return out, mean_size
