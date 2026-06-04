"""Tests for samptype (swor/swr) + sampsize per-tree subsampling (SUN-83)."""

from __future__ import annotations

import numpy as np
import pytest

from comprisk import CompetingRiskForest


def _toy(n=200, p=4, seed=0, n_causes=2):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    time = rng.uniform(0.1, 10, n)
    event = rng.randint(0, n_causes + 1, n).astype(np.int64)
    return X, time, event


# --------------------------------------------------------------------------- #
# sampsize resolution                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("samptype", "sampsize", "n", "expected"),
    [
        ("swor", None, 1000, 632),  # rfsrc default: round(0.632 * n)
        ("swr", None, 1000, 1000),  # classic full bootstrap
        ("swor", 0.1, 1000, 100),  # float fraction
        ("swr", 0.25, 1000, 250),
        ("swor", 250, 1000, 250),  # absolute int
        ("swr", 1500, 1000, 1500),  # swr may exceed n
        ("swor", 5000, 1000, 1000),  # swor capped at n
        ("swor", 1.0, 1000, 1000),  # full-data fraction
        ("swor", (lambda nn: nn // 4), 1000, 250),  # callable(n) -> int
        ("swor", (lambda nn: int(nn * 0.632)), 1000, 632),  # rfsrc functional form
    ],
)
def test_resolve_sampsize(samptype, sampsize, n, expected):
    f = CompetingRiskForest(samptype=samptype, sampsize=sampsize)
    assert f._resolve_sampsize(n) == expected


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, 2.0])
def test_resolve_sampsize_float_out_of_range(bad):
    f = CompetingRiskForest(samptype="swor", sampsize=bad)
    with pytest.raises(ValueError, match=r"float sampsize must be in \(0, 1\]"):
        f._resolve_sampsize(100)


def test_resolve_sampsize_zero_int_raises():
    f = CompetingRiskForest(samptype="swr", sampsize=0)
    with pytest.raises(ValueError, match="resolved sampsize must be >= 1"):
        f._resolve_sampsize(100)


def test_resolve_sampsize_bool_rejected():
    # bool is an int subclass; guard against True/False slipping through.
    f = CompetingRiskForest(samptype="swr", sampsize=True)
    with pytest.raises(ValueError, match="got bool"):
        f._resolve_sampsize(100)


def test_invalid_samptype_raises():
    X, time, event = _toy()
    with pytest.raises(ValueError, match="samptype must be 'swor' or 'swr'"):
        CompetingRiskForest(samptype="bagging").fit(X, time, event)


# --------------------------------------------------------------------------- #
# per-tree sampling behaviour                                                 #
# --------------------------------------------------------------------------- #


def _draw(f, n, seed=0):
    f._resolved_sampsize_ = f._resolve_sampsize(n)
    return f._sample_indices(np.random.RandomState(seed), n)


def test_swor_draws_unique_indices_of_resolved_size():
    n = 200
    f = CompetingRiskForest(samptype="swor", sampsize=0.5)
    idx, oob = _draw(f, n)
    assert len(idx) == 100
    assert len(np.unique(idx)) == len(idx)  # no duplicates
    # OOB is the exact complement of the (unique) in-bag set.
    assert len(oob) == n - 100
    assert np.array_equal(np.sort(np.concatenate([idx, oob])), np.arange(n))


def test_swr_allows_duplicate_indices():
    n = 200
    f = CompetingRiskForest(samptype="swr", sampsize=None)  # -> n
    idx, oob = _draw(f, n)
    assert len(idx) == n
    assert len(np.unique(idx)) < len(idx)  # with replacement => duplicates
    # OOB is the set of rows never drawn.
    assert np.array_equal(oob, np.setdiff1d(np.arange(n), idx))


def test_full_data_swor_is_deterministic_arange_no_rng():
    n = 200
    f = CompetingRiskForest(samptype="swor", sampsize=1.0)
    f._resolved_sampsize_ = f._resolve_sampsize(n)
    # Two different RNGs must yield the identical (un-permuted) full index set,
    # proving no RNG is consumed for the full-data swor draw.
    idx_a, oob_a = f._sample_indices(np.random.RandomState(1), n)
    idx_b, oob_b = f._sample_indices(np.random.RandomState(999), n)
    assert np.array_equal(idx_a, np.arange(n))
    assert np.array_equal(idx_a, idx_b)
    assert len(oob_a) == 0 and len(oob_b) == 0


def test_swor_subsample_produces_oob_in_fitted_forest():
    X, time, event = _toy(n=200)
    f = CompetingRiskForest(
        n_estimators=5, samptype="swor", sampsize=0.6, random_state=0, n_jobs=1
    ).fit(X, time, event)
    assert f._resolved_sampsize_ == 120
    for oob in f.oob_indices_:
        assert len(oob) == 200 - 120  # complement of the unique 120-row draw


def test_default_is_rfsrc_swor_0632():
    X, time, event = _toy(n=1000)
    f = CompetingRiskForest(n_estimators=3, random_state=0, n_jobs=1).fit(X, time, event)
    assert f.samptype == "swor"
    assert f.sampsize is None
    assert f._resolved_sampsize_ == 632
    assert f._oob_available_ is True


# --------------------------------------------------------------------------- #
# minimal-depth threshold responds to sampsize (regression)                   #
# --------------------------------------------------------------------------- #


def _signal_cr(n=600, p=10, seed=1):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    lin = X[:, 0] * 1.2 - X[:, 1] * 0.8 + X[:, 2] * 0.6
    time = np.clip(rng.exponential(np.exp(-lin)), 0.05, None)
    event = rng.randint(1, 3, n).astype(np.int64)
    event[rng.rand(n) < 0.3] = 0
    return X, time, event


def test_minimal_depth_threshold_rises_as_sampsize_drops():
    """The Ishwaran threshold is geometry-derived: a smaller per-tree sample
    yields shallower trees and therefore a higher expected-minimal-depth
    threshold. Regression guard for the SUN-83 sampling knob."""
    X, time, event = _signal_cr()

    def threshold_at(frac):
        f = CompetingRiskForest(
            n_estimators=40,
            max_depth=None,
            min_samples_split=6,
            min_samples_leaf=3,
            samptype="swor",
            sampsize=frac,
            random_state=0,
            n_jobs=1,
        ).fit(X, time, event)
        return f.minimal_depth()["threshold"].iloc[0]

    thr_full = threshold_at(1.0)
    thr_mid = threshold_at(0.3)
    thr_small = threshold_at(0.1)

    # Strictly increasing as the sample shrinks (gaps are large; ~2.4 -> ~3.3).
    assert thr_full < thr_mid < thr_small
