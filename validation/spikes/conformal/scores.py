"""Step 2 of the conformal-CR spike: the score model pi-hat at horizon t*.

For every subject we need a distribution over the K+1 labels {cause 1..K, EVENT_FREE}:

    pi_k(t*|x) = F_k(t*|x)           (cause-k cumulative incidence)
    pi_free(t*|x) = S(t*|x) = 1 - sum_k F_k(t*|x)

These come straight from the forest's Aalen-Johansen CIF and sum to 1 by
construction (CIF + survival identity), so the score vector is already a proper
distribution -- no normalisation needed.

Two calibration substrates (task design.md):
  - split:  ordinary predict_cif on a disjoint fold (clean exchangeability).
  - OOB:    per-tree CIF averaged over only the trees where a training sample was
            out-of-bag (free calibration; tests Bostroem's open OOB-validity question).
"""

from __future__ import annotations

import numpy as np
from validation.spikes.conformal.dgp import EVENT_FREE

from comprisk._binning import apply_bins
from comprisk._hist_tree import predict_tree_hist
from comprisk._tree import predict_tree


def _predictor_and_input(forest, X):
    """Pick the per-tree CIF predictor and prepare X for ``forest.mode``."""
    X = np.asarray(X, dtype=np.float64)
    if forest.mode == "default":
        return predict_tree_hist, apply_bins(X, forest.bin_edges_)
    return predict_tree, X


def _horizon_take(forest, t_star):
    """Right-continuous step index of t* on the forest time grid (matches
    forest._make_time_projection: searchsorted right, minus 1, 0 before grid)."""
    idx = int(np.searchsorted(forest.unique_times_, t_star, side="right")) - 1
    return max(idx, 0), idx < 0


def _split_to_causes_free(cif_kt):
    """(n, n_causes) CIF at t* -> (pi_causes (n,K), pi_free (n,))."""
    pi_causes = cif_kt
    pi_free = 1.0 - pi_causes.sum(axis=1)
    return pi_causes, pi_free


def full_cif_at_horizon(forest, X, t_star):
    """Ensemble-mean CIF at t* via an explicit per-tree loop (every tree, all X).

    Correctness oracle: must equal ``forest.predict_cif(X, times=[t_star])`` to
    float tolerance, validating the per-tree-loop + horizon-projection wiring that
    the OOB path reuses.
    """
    predict_fn, X_input = _predictor_and_input(forest, X)
    take, before = _horizon_take(forest, t_star)
    n = X_input.shape[0]
    acc = np.zeros((n, forest.n_causes_), dtype=np.float64)
    for tree in forest.trees_:
        cif = predict_fn(tree, X_input)  # (n, n_causes, n_time)
        acc += 0.0 if before else cif[:, :, take]
    acc /= len(forest.trees_)
    return _split_to_causes_free(acc)


def split_cif_at_horizon(forest, X, t_star):
    """Path B score: ordinary ensemble CIF at t* (use on a disjoint fold)."""
    cif = forest.predict_cif(X, times=[t_star])[:, :, 0]  # (n, n_causes)
    return _split_to_causes_free(cif)


def oob_cif_at_horizon(forest, X_train, t_star):
    """Path A score: per-tree CIF at t* averaged over OOB trees only.

    Returns
    -------
    pi_causes : (n_train, K)
    pi_free   : (n_train,)
    count     : (n_train,) number of trees for which each sample was OOB
    """
    if not getattr(forest, "_oob_available_", False):
        raise ValueError("OOB CIF needs an out-of-bag set (samptype='swr' or sampsize<n)")
    predict_fn, X_input = _predictor_and_input(forest, X_train)
    take, before = _horizon_take(forest, t_star)
    n = X_input.shape[0]
    acc = np.zeros((n, forest.n_causes_), dtype=np.float64)
    count = np.zeros(n, dtype=np.int64)
    for t, tree in enumerate(forest.trees_):
        oob = forest.oob_indices_[t]
        if len(oob) == 0:
            continue
        cif = predict_fn(tree, X_input[oob])  # (n_oob, n_causes, n_time)
        acc[oob] += 0.0 if before else cif[:, :, take]
        count[oob] += 1
    safe = np.maximum(count, 1)[:, None]
    pi_causes = acc / safe
    pi_free = 1.0 - pi_causes.sum(axis=1)
    return pi_causes, pi_free, count


def label_prob(pi_causes, pi_free, y):
    """pi-hat of the realised label for each subject. y in {EVENT_FREE, 1..K}."""
    y = np.asarray(y, dtype=np.int64)
    out = np.where(y == EVENT_FREE, pi_free, 0.0)
    is_cause = y >= 1
    rows = np.nonzero(is_cause)[0]
    out[rows] = pi_causes[rows, y[rows] - 1]
    return out


def nonconformity(pi_causes, pi_free, y):
    """Conformity score s_i = 1 - pi-hat(realised label)."""
    return 1.0 - label_prob(pi_causes, pi_free, y)
