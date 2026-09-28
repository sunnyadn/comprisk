"""Tests for the CompetingRiskForest ensemble class."""

import numpy as np
import pytest

from comprisk.forest import CompetingRiskForest


def _make_synthetic_cr(seed=0, n=60):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 3))
    # Time roughly anti-correlated with feature 0 (higher feat 0 -> shorter time)
    time = 10.0 - 5.0 * X[:, 0] + rng.normal(scale=0.5, size=n)
    time = np.clip(time, 0.1, None)
    event = rng.integers(0, 3, size=n)  # {0, 1, 2}
    if not np.any(event == 1):
        event[0] = 1
    if not np.any(event == 2):
        event[1] = 2
    return X, time, event


def test_fit_then_predict_cif_shape():
    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, time, event)
    cif = f.predict_cif(X)
    assert cif.shape == (len(X), 2, len(f.unique_times_))


def test_different_seeds_give_different_forests():
    X, time, event = _make_synthetic_cr()
    c1 = CompetingRiskForest(n_estimators=8, random_state=1).fit(X, time, event)
    c2 = CompetingRiskForest(n_estimators=8, random_state=2).fit(X, time, event)
    assert not np.allclose(c1.predict_cif(X), c2.predict_cif(X))


def test_oob_indices_complement_bootstrap_indices():
    X, time, event = _make_synthetic_cr(n=40)
    n = len(X)
    # samptype="swr" exercises the with-replacement draw this test reconstructs.
    f = CompetingRiskForest(n_estimators=4, random_state=0, samptype="swr").fit(X, time, event)

    # Each OOB set is a valid subset of [0, n) with no duplicates
    for oob in f.oob_indices_:
        assert oob.ndim == 1
        assert np.all((oob >= 0) & (oob < n))
        assert len(np.unique(oob)) == len(oob)

    # OOB must be the exact complement of bootstrap indices. Reconstruct
    # the bootstrap draws with the same controller seed to verify.
    rng = np.random.RandomState(0)
    for i in range(len(f.oob_indices_)):
        expected_bootstrap = rng.choice(n, size=n, replace=True)
        rng.randint(0, 2**31)  # consume the per-tree seed, as fit() does
        expected_oob = np.setdiff1d(np.arange(n), expected_bootstrap)
        assert np.array_equal(f.oob_indices_[i], expected_oob)


def test_predict_risk_default_kind_is_integrated_chf():
    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, time, event)
    chf = f.predict_chf(X)
    risk = f.predict_risk(X, cause=1)
    assert np.allclose(risk, chf[:, 0, :].sum(axis=-1))


def test_predict_risk_kind_cif_last_returns_cif_at_last_time():
    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, time, event)
    cif = f.predict_cif(X)
    risk = f.predict_risk(X, cause=1, kind="cif_last")
    assert np.allclose(risk, cif[:, 0, -1])


def test_predict_risk_invalid_kind_raises():
    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, time, event)
    with pytest.raises(ValueError, match="kind must be"):
        f.predict_risk(X, cause=1, kind="bogus")


def test_score_accepts_cause_parameter():
    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, time, event)
    c1 = f.score(X, time, event, cause=1)
    c2 = f.score(X, time, event, cause=2)
    assert 0.0 <= c1 <= 1.0
    assert 0.0 <= c2 <= 1.0
    # Default still uses cause=1
    assert f.score(X, time, event) == c1


def _expected_step_interp(full, ut, times):
    idx = np.searchsorted(ut, times, side="right") - 1
    out = full[:, :, np.clip(idx, 0, None)].copy()
    before = idx < 0
    if before.any():
        out[:, :, before] = 0.0
    return out


def test_predict_cif_with_times_bit_equal_to_full_then_slice():
    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=8, random_state=0).fit(X, time, event)
    full = f.predict_cif(X)
    ut = f.unique_times_
    times = np.array([0.0, ut[0], ut[len(ut) // 2], ut[-1], ut[-1] + 5.0])
    sliced = f.predict_cif(X, times=times)
    np.testing.assert_array_equal(sliced, _expected_step_interp(full, ut, times))


def test_predict_chf_with_times_bit_equal_to_full_then_slice():
    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=8, random_state=0).fit(X, time, event)
    full = f.predict_chf(X)
    ut = f.unique_times_
    times = np.array([0.0, ut[0], ut[len(ut) // 2], ut[-1], ut[-1] + 5.0])
    sliced = f.predict_chf(X, times=times)
    np.testing.assert_array_equal(sliced, _expected_step_interp(full, ut, times))


def test_predict_rejects_wrong_n_features():
    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=3, random_state=0).fit(X, time, event)
    with pytest.raises(ValueError, match="n_features"):
        f.predict_cif(X[:, :2])


def test_default_mode_fit_sets_bin_edges_and_time_grid():
    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=3, mode="default", random_state=0).fit(X, time, event)
    assert hasattr(f, "bin_edges_")
    assert len(f.bin_edges_) == X.shape[1]
    assert hasattr(f, "time_grid_")
    assert f.time_grid_.ndim == 1
    np.testing.assert_array_equal(f.unique_times_, f.time_grid_)


def test_default_mode_uses_flat_trees():
    from comprisk._tree_flat import FlatTree

    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=3, mode="default", random_state=0).fit(X, time, event)
    assert all(isinstance(t, FlatTree) for t in f.trees_)


def test_reference_mode_still_works():
    from comprisk._tree import RefTreeNode

    X, time, event = _make_synthetic_cr()
    f = CompetingRiskForest(n_estimators=3, mode="reference", random_state=0).fit(X, time, event)
    assert all(isinstance(t, RefTreeNode) for t in f.trees_)


def test_invalid_mode_raises():
    X, time, event = _make_synthetic_cr()
    with pytest.raises(ValueError, match="mode"):
        CompetingRiskForest(mode="invalid").fit(X, time, event)


def test_forest_splitrule_unknown_raises():
    import numpy as np
    import pytest

    from comprisk import CompetingRiskForest

    with pytest.raises(ValueError, match="splitrule"):
        CompetingRiskForest(splitrule="bogus").fit(
            np.zeros((10, 2)), np.arange(10.0), np.zeros(10, dtype=np.int64)
        )


def test_forest_logrank_cause_weights_validates_length():
    import numpy as np
    import pytest

    from comprisk import CompetingRiskForest

    forest = CompetingRiskForest(splitrule="logrank", cause_weights=[1.0], mode="reference")
    with pytest.raises(ValueError, match="cause_weights"):
        forest.fit(
            np.zeros((20, 2)),
            np.arange(1.0, 21.0),
            np.array([1, 2] * 10, dtype=np.int64),  # n_causes=2
        )


def test_forest_logrank_cause_weights_rejected_in_default_mode():
    import numpy as np
    import pytest

    from comprisk import CompetingRiskForest

    forest = CompetingRiskForest(splitrule="logrank", cause_weights=[0.6, 0.4], mode="default")
    with pytest.raises(NotImplementedError, match="cause_weights"):
        forest.fit(
            np.zeros((20, 2)),
            np.arange(1.0, 21.0),
            np.array([1, 2] * 10, dtype=np.int64),
        )


def test_forest_logrank_cause_weights_accepted_in_reference_mode():
    import numpy as np

    from comprisk import CompetingRiskForest

    rng = np.random.default_rng(8)
    n = 80
    X = rng.standard_normal((n, 3))
    time = rng.uniform(0.1, 5.0, size=n)
    event = rng.integers(0, 3, size=n).astype(np.int64)

    forest = CompetingRiskForest(
        n_estimators=3,
        max_depth=3,
        random_state=0,
        splitrule="logrank",
        cause_weights=[0.6, 0.4],
        mode="reference",
        n_jobs=1,
    )
    forest.fit(X, time, event)  # should not raise
    cif = forest.predict_cif(X)
    assert cif.shape == (n, 2, len(forest.unique_times_))
