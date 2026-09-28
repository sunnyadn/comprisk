"""Tests for the reference-mode tree builder and predictor."""

import numpy as np
import pytest

from comprisk._estimators import aalen_johansen, aalen_johansen_from_counts
from comprisk._tree import build_tree, predict_tree


def _toy_separating_dataset():
    # Feature 0 perfectly separates early- vs late-event samples
    rng = np.random.default_rng(0)
    n = 40
    X = np.zeros((n, 2))
    X[:, 0] = np.concatenate([np.zeros(20), np.ones(20)])
    X[:, 1] = rng.uniform(size=n)
    time = np.concatenate([rng.uniform(0.5, 1.5, 20), rng.uniform(5.0, 6.0, 20)])
    event = np.ones(n, dtype=int)  # single cause
    return X, time, event


def test_build_tree_root_split_feature_is_separating_one():
    X, time, event = _toy_separating_dataset()
    unique_times = np.sort(np.unique(time))
    tree = build_tree(
        X,
        time,
        event,
        n_causes=1,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        unique_times=unique_times,
    )
    assert not tree.is_leaf
    assert tree.feature == 0


def test_zero_split_tree_leaf_equals_dataset_wide_cif():
    # Force a zero-split tree by requiring more samples-per-split than we have
    X = np.zeros((5, 2))
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    event = np.array([1, 0, 2, 1, 0])
    unique_times = np.sort(np.unique(time))
    tree = build_tree(
        X,
        time,
        event,
        n_causes=2,
        max_depth=5,
        min_samples_split=100,
        min_samples_leaf=1,
        unique_times=unique_times,
    )
    assert tree.is_leaf
    expected = aalen_johansen(time, event, unique_times, n_causes=2)
    leaf_cif = aalen_johansen_from_counts(tree.event_counts, tree.at_risk, n_causes=2)
    assert np.allclose(leaf_cif, expected, atol=1e-12)


def test_build_tree_rejects_max_features_without_rng():
    X = np.zeros((10, 3))
    time = np.arange(1.0, 11.0)
    event = np.ones(10, dtype=int)
    with pytest.raises(ValueError, match="max_features requires an rng"):
        build_tree(
            X,
            time,
            event,
            n_causes=1,
            max_depth=3,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features=2,
        )


def test_predict_tree_rejects_non_2d_input():
    X, time, event = _toy_separating_dataset()
    unique_times = np.sort(np.unique(time))
    tree = build_tree(
        X,
        time,
        event,
        n_causes=1,
        max_depth=2,
        min_samples_split=4,
        min_samples_leaf=2,
        unique_times=unique_times,
    )
    with pytest.raises(ValueError, match="X must be 2-D"):
        predict_tree(tree, np.array([0.0, 1.0, 2.0]))


def test_build_tree_accepts_splitrule_logrank():
    import numpy as np

    from comprisk._tree import build_tree

    rng = np.random.default_rng(0)
    n = 40
    X = np.zeros((n, 2))
    X[:, 0] = np.concatenate([np.zeros(20), np.ones(20)])
    X[:, 1] = rng.uniform(size=n)
    time = np.concatenate([rng.uniform(0.5, 1.5, 20), rng.uniform(5.0, 6.0, 20)])
    event = np.concatenate([np.ones(20, dtype=int), 2 * np.ones(20, dtype=int)])

    tree = build_tree(
        X,
        time,
        event,
        n_causes=2,
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=2,
        splitrule="logrank",
        cause=1,
    )
    assert not tree.is_leaf
    assert tree.feature == 0


def test_predict_tree_chf_single_leaf_matches_nelson_aalen_cs():
    """A forced single-leaf tree's CHF equals the root Nelson-Aalen CHF."""
    from comprisk._estimators import nelson_aalen_cs
    from comprisk._tree import build_tree, predict_tree_chf

    # 6 samples, 2 features — force a single leaf by making min_samples_split huge
    X = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.2, 0.8], [0.8, 0.2], [0.6, 0.4]])
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    event = np.array([1, 2, 1, 0, 2, 1])
    unique_times = np.sort(np.unique(time))

    tree = build_tree(
        X,
        time,
        event,
        n_causes=2,
        max_depth=5,
        min_samples_split=100,  # ensures root is a leaf
        min_samples_leaf=1,
        unique_times=unique_times,
    )
    assert tree.is_leaf

    chf_pred = predict_tree_chf(tree, X)
    chf_expected = nelson_aalen_cs(time, event, unique_times, n_causes=2)

    assert chf_pred.shape == (6, 2, len(unique_times))
    # Every sample descends to the single leaf, so every row equals the root CHF.
    for i in range(6):
        np.testing.assert_allclose(chf_pred[i], chf_expected, atol=1e-12)


def test_build_tree_nsplit_zero_matches_pre_p3a5_structure():
    """nsplit=0 must preserve the exhaustive-build tree structure."""
    rng_data = np.random.default_rng(10)
    n, p = 40, 2
    X = rng_data.uniform(0, 10, size=(n, p))
    time = rng_data.uniform(1.0, 10.0, n)
    event = rng_data.integers(0, 3, n)
    event[0] = 1
    event[1] = 2

    tree_pre = build_tree(
        X,
        time,
        event,
        n_causes=2,
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=1,
    )
    tree_ns0 = build_tree(
        X,
        time,
        event,
        n_causes=2,
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=1,
        nsplit=0,
    )

    def preorder(n):
        if n.is_leaf:
            return [("leaf",)]
        return [("split", n.feature, n.threshold), *preorder(n.left), *preorder(n.right)]

    assert preorder(tree_pre) == preorder(tree_ns0)
