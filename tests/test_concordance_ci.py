"""Tests for the influence-function concordance CI (concordance_index_ci /
concordance_index_delta_ci)."""

import numpy as np
import pytest

from comprisk import (
    ConcordanceCI,
    DeltaConcordanceCI,
    compute_uno_weights,
    concordance_index_ci,
    concordance_index_delta_ci,
    concordance_index_uno_cr,
)


def _gen(n, seed, signal=1.2, grid=None):
    rng = np.random.default_rng(seed)
    p = rng.normal(size=n)
    t1 = rng.exponential(1.0 / np.exp(signal * p))
    tc = rng.exponential(5.0, n)
    to = rng.exponential(2.0, n)
    t = np.minimum.reduce([t1, tc, to])
    e = np.where(t == t1, 1, np.where(t == to, 2, 0)).astype(int)
    if grid is not None:
        t = np.round(t / grid) * grid + grid
    return t, e, p


def test_point_estimate_matches_concordance_index_uno_cr():
    t, e, p = _gen(800, seed=1)
    res = concordance_index_ci(e, t, p, cause=1, gmin="auto")
    w = compute_uno_weights(t, e, gmin="auto")
    c_ref = concordance_index_uno_cr(e, t, p, cause=1, weights=w)
    assert res.estimate == pytest.approx(c_ref, abs=1e-12)
    assert isinstance(res, ConcordanceCI)


def test_ci_brackets_estimate_and_in_unit_interval():
    t, e, p = _gen(800, seed=2)
    res = concordance_index_ci(e, t, p, cause=1)
    assert res.ci_low < res.estimate < res.ci_high
    assert 0.0 <= res.ci_low < res.ci_high <= 1.0  # logit transform keeps it in [0,1]
    assert res.se > 0
    assert res.n == 800


def test_if_se_matches_bootstrap():
    """The IF standard error should agree with the nonparametric bootstrap to
    Monte-Carlo error under the default (stabilised) weights."""
    t, e, p = _gen(700, seed=7)
    res = concordance_index_ci(e, t, p, cause=1, gmin="auto")

    rng = np.random.default_rng(123)
    n, B = len(t), 1200
    boot = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        wb = compute_uno_weights(t[idx], e[idx], gmin="auto")
        boot[b] = concordance_index_uno_cr(e[idx], t[idx], p[idx], cause=1, weights=wb)
    se_boot = np.nanstd(boot, ddof=1)
    assert res.se == pytest.approx(se_boot, rel=0.15)


def test_handles_tied_event_times():
    t, e, p = _gen(700, seed=11, grid=0.1)  # discrete -> many tied times
    res = concordance_index_ci(e, t, p, cause=1)
    w = compute_uno_weights(t, e, gmin="auto")
    c_ref = concordance_index_uno_cr(e, t, p, cause=1, weights=w)
    assert res.estimate == pytest.approx(c_ref, abs=1e-9)
    assert np.isfinite(res.se) and res.se > 0


def test_transform_none_is_symmetric_wald():
    t, e, p = _gen(800, seed=3)
    res = concordance_index_ci(e, t, p, cause=1, transform="none")
    assert res.ci_low == pytest.approx(2 * res.estimate - res.ci_high, abs=1e-12)


def test_transform_invalid_raises():
    t, e, p = _gen(200, seed=4)
    with pytest.raises(ValueError, match="transform"):
        concordance_index_ci(e, t, p, cause=1, transform="probit")


def test_confidence_level_widens_interval():
    t, e, p = _gen(800, seed=5)
    w90 = concordance_index_ci(e, t, p, cause=1, confidence_level=0.90)
    w99 = concordance_index_ci(e, t, p, cause=1, confidence_level=0.99)
    assert (w99.ci_high - w99.ci_low) > (w90.ci_high - w90.ci_low)


def test_tau_truncation_changes_result():
    t, e, p = _gen(800, seed=6)
    full = concordance_index_ci(e, t, p, cause=1, gmin="none")
    trunc = concordance_index_ci(e, t, p, cause=1, gmin="none", tau=float(np.quantile(t, 0.7)))
    assert trunc.estimate != full.estimate
    assert trunc.n == full.n  # n reports all subjects; truncation acts via weights


def test_delta_identical_models_is_zero():
    t, e, p = _gen(600, seed=8)
    d = concordance_index_delta_ci(e, t, p, p, cause=1)
    assert isinstance(d, DeltaConcordanceCI)
    assert d.estimate == pytest.approx(0.0, abs=1e-12)
    assert d.se == pytest.approx(0.0, abs=1e-12)
    assert d.pvalue == pytest.approx(1.0)


def test_delta_better_model_is_positive_and_significant():
    t, e, p = _gen(1500, seed=9, signal=1.5)
    noise = np.random.default_rng(0).normal(size=len(p))
    d = concordance_index_delta_ci(e, t, p, noise, cause=1)  # signal vs pure noise
    assert d.estimate > 0
    assert d.ci_low > 0  # CI excludes 0
    assert d.pvalue < 0.001


def test_delta_paired_se_below_unpaired():
    """Paired delta SE should be <= sqrt(se_a^2+se_b^2) (positive covariance)."""
    t, e, p = _gen(1000, seed=10, signal=1.2)
    pb = p + np.random.default_rng(1).normal(scale=0.3, size=len(p))  # correlated
    a = concordance_index_ci(e, t, p, cause=1, transform="none")
    b = concordance_index_ci(e, t, pb, cause=1, transform="none")
    d = concordance_index_delta_ci(e, t, p, pb, cause=1)
    assert d.se < np.hypot(a.se, b.se)


def test_no_events_of_cause_returns_nan():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    e = np.array([0, 2, 0, 2])  # no cause-1 events
    p = np.array([0.1, 0.2, 0.3, 0.4])
    res = concordance_index_ci(e, t, p, cause=1)
    assert np.isnan(res.estimate)
    assert np.isnan(res.se)
