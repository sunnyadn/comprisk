"""sklearn drop-in compatibility surface tests.

Verifies CompetingRiskForest is a true sklearn-compatible estimator:

* ``Surv.from_arrays(event, time)`` builds the structured y the same way
  scikit-survival does, so users can swap libraries without rewiring data.
* ``fit(X, y)`` and ``score(X, y)`` accept the structured y form, equivalent
  to the legacy three-argument ``fit(X, time, event)`` / ``score(X, time, event)``.
* ``predict(X)`` is a sklearn alias for ``predict_risk(X, cause=1)``.
* ``cross_val_score`` and ``clone`` work without a wrapper.
* every public estimator is a ``BaseEstimator`` (get_params / set_params / clone).
* ``feature_names_in_`` follows SLEP007.
* ``check_estimator`` fails only for reasons inherent to survival estimators.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from comprisk import (
    CauseSpecificCox,
    CompetingRiskForest,
    FineGrayRegression,
    PenalizedFineGrayRegression,
    Surv,
)


def _toy_cr(n: int = 200, p: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    time = rng.exponential(1.0, n) + 0.1
    event = rng.choice([0, 1, 2], size=n, p=[0.4, 0.3, 0.3]).astype(np.int8)
    return X, time, event


# ---------------------------------------------------------------------------
# Surv.from_arrays
# ---------------------------------------------------------------------------


def test_surv_from_arrays_returns_structured():
    _, time, event = _toy_cr()
    y = Surv.from_arrays(event=event, time=time)
    assert y.dtype.names == ("event", "time")
    assert y.shape == (len(time),)
    np.testing.assert_array_equal(y["time"], time)
    np.testing.assert_array_equal(y["event"], event)


def test_surv_from_arrays_accepts_lists():
    y = Surv.from_arrays(event=[0, 1, 2, 0], time=[1.0, 2.0, 3.0, 0.5])
    np.testing.assert_array_equal(y["event"], [0, 1, 2, 0])
    np.testing.assert_array_equal(y["time"], [1.0, 2.0, 3.0, 0.5])


def test_surv_from_arrays_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        Surv.from_arrays(event=[0, 1], time=[1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# fit / score with structured y
# ---------------------------------------------------------------------------


def test_fit_with_structured_y_equivalent_to_three_arg():
    X, time, event = _toy_cr()
    y = Surv.from_arrays(event=event, time=time)
    f1 = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, time, event)
    f2 = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, y)
    np.testing.assert_array_equal(f1.predict_cif(X[:3]), f2.predict_cif(X[:3]))


def test_fit_accepts_reverse_field_order():
    """Match sksurv's (event, time) order AND any user-built (time, event)."""
    X, time, event = _toy_cr()
    # Build structured y with time-first field order (codebase internal convention).
    y_time_first = np.zeros(len(time), dtype=[("time", np.float64), ("event", np.int8)])
    y_time_first["time"] = time
    y_time_first["event"] = event
    f = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, y_time_first)
    f_ref = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, time, event)
    np.testing.assert_array_equal(f.predict_cif(X[:3]), f_ref.predict_cif(X[:3]))


def test_fit_rejects_non_structured_y_when_event_omitted():
    X, time, _ = _toy_cr()
    f = CompetingRiskForest(n_estimators=3)
    with pytest.raises(TypeError, match="structured array"):
        f.fit(X, time)  # bare time array, no event -> ambiguous


def test_score_with_structured_y_equivalent():
    X, time, event = _toy_cr()
    y = Surv.from_arrays(event=event, time=time)
    f = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, time, event)
    assert f.score(X, time, event, cause=1) == f.score(X, y, cause=1)


# ---------------------------------------------------------------------------
# predict() alias
# ---------------------------------------------------------------------------


def test_predict_alias_matches_predict_risk_cause1():
    X, time, event = _toy_cr()
    f = CompetingRiskForest(n_estimators=5, random_state=0).fit(X, time, event)
    np.testing.assert_array_equal(f.predict(X[:5]), f.predict_risk(X[:5], cause=1))


# ---------------------------------------------------------------------------
# clone + cross_val_score
# ---------------------------------------------------------------------------


def test_clone_preserves_constructor_params():
    f = CompetingRiskForest(n_estimators=42, max_depth=5, random_state=7)
    g = clone(f)
    assert g.n_estimators == 42
    assert g.max_depth == 5
    assert g.random_state == 7
    with pytest.raises(NotFittedError):
        g.predict_cif(np.zeros((3, 5)))


def test_cross_val_score_with_kfold():
    X, time, event = _toy_cr(n=120, p=4)
    y = Surv.from_arrays(event=event, time=time)
    f = CompetingRiskForest(n_estimators=8, random_state=0, n_jobs=1)
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(f, X, y, cv=cv, n_jobs=1)
    assert scores.shape == (3,)
    assert all(0.0 <= s <= 1.0 for s in scores)


# ---------------------------------------------------------------------------
# __init__ stores without validating (sklearn dev guide: validation belongs
# in fit, so set_params / clone / GridSearchCV never trip on a bad value)
# ---------------------------------------------------------------------------


def test_init_does_not_validate_device():
    f = CompetingRiskForest(device="nonsense")
    assert f.device == "nonsense"
    assert clone(f).device == "nonsense"
    f.set_params(device=-1)
    assert f.device == -1


# The matching "fit rejects it" half lives in
# tests/test_forest_device_dispatch.py, which owns device behaviour.


# ---------------------------------------------------------------------------
# Every public estimator is a BaseEstimator: get_params / set_params / clone,
# so all four can go into Pipeline and GridSearchCV
# ---------------------------------------------------------------------------

ALL_ESTIMATORS = [
    CompetingRiskForest,
    PenalizedFineGrayRegression,
    FineGrayRegression,
    CauseSpecificCox,
]


@pytest.mark.parametrize("cls", ALL_ESTIMATORS, ids=lambda c: c.__name__)
def test_estimator_is_base_estimator(cls):
    assert issubclass(cls, BaseEstimator)
    est = cls()
    params = est.get_params()
    # get_params must report exactly the __init__ signature
    expected = {p for p in inspect.signature(cls.__init__).parameters if p != "self"}
    assert set(params) == expected
    assert isinstance(clone(est), cls)


@pytest.mark.parametrize("cls", ALL_ESTIMATORS, ids=lambda c: c.__name__)
def test_set_params_changes_the_value(cls):
    # Every estimator takes `cause`; set it to something other than the default
    # so a no-op set_params would fail rather than pass vacuously.
    est = cls()
    assert est.get_params()["cause"] == 1
    assert est.set_params(cause=2) is est
    assert est.get_params()["cause"] == 2
    assert est.cause == 2


@pytest.mark.parametrize("cls", [FineGrayRegression, CauseSpecificCox], ids=lambda c: c.__name__)
def test_regressor_in_pipeline_and_grid_search(cls):
    X, time, event = _toy_cr(n=150, p=3)
    y = Surv.from_arrays(event=event, time=time)
    pipe = Pipeline([("sc", StandardScaler()), ("m", cls())]).fit(X, y)
    assert hasattr(pipe[-1], "coef_")
    # GridSearchCV needs clone + set_params; `cause` exists on both estimators.
    gs = GridSearchCV(cls(), {"cause": [1, 2]}, cv=KFold(2), scoring=_coef_norm).fit(X, y)
    assert gs.best_params_["cause"] in (1, 2)


def _coef_norm(estimator, X, y):
    """Scorer standing in for a real metric — these estimators expose no score()."""
    return float(np.linalg.norm(estimator.coef_))


# ---------------------------------------------------------------------------
# feature_names_in_ (SLEP007)
# ---------------------------------------------------------------------------


def _toy_df(n=150, p=4, seed=0):
    X, time, event = _toy_cr(n=n, p=p, seed=seed)
    df = pd.DataFrame(X, columns=[f"feat{i}" for i in range(p)])
    return df, Surv.from_arrays(event=event, time=time)


def _fit_any(cls, X, y):
    """Fit with the per-class kwargs needed to keep the toy problem well-posed."""
    kwargs = {"n_estimators": 4, "random_state": 0} if cls is CompetingRiskForest else {}
    return cls(**kwargs).fit(X, y)


@pytest.mark.parametrize("cls", ALL_ESTIMATORS, ids=lambda c: c.__name__)
def test_feature_names_recorded_from_dataframe(cls):
    df, y = _toy_df()
    assert list(_fit_any(cls, df, y).feature_names_in_) == list(df.columns)


@pytest.mark.parametrize("cls", ALL_ESTIMATORS, ids=lambda c: c.__name__)
def test_feature_names_absent_for_ndarray(cls):
    df, y = _toy_df()
    assert not hasattr(_fit_any(cls, df.to_numpy(), y), "feature_names_in_")


def test_refit_on_ndarray_clears_feature_names():
    df, y = _toy_df()
    f = CompetingRiskForest(n_estimators=4, random_state=0).fit(df, y)
    assert hasattr(f, "feature_names_in_")
    f.fit(df.to_numpy(), y)
    assert not hasattr(f, "feature_names_in_")


def test_feature_names_through_pipeline_match_sklearn_semantics():
    # A transformer emitting ndarray (sklearn's default) legitimately drops the
    # names, so the final estimator sees none -- sklearn's own estimators behave
    # identically. They propagate only under set_output(transform="pandas").
    df, y = _toy_df()
    plain = Pipeline(
        [("sc", StandardScaler()), ("f", CompetingRiskForest(n_estimators=4, random_state=0))]
    ).fit(df, y)
    assert not hasattr(plain[-1], "feature_names_in_")

    named = Pipeline(
        [
            ("sc", StandardScaler().set_output(transform="pandas")),
            ("f", CompetingRiskForest(n_estimators=4, random_state=0)),
        ]
    ).fit(df, y)
    assert list(named[-1].feature_names_in_) == list(df.columns)


def test_non_string_columns_do_not_set_names():
    df, y = _toy_df()
    df.columns = range(df.shape[1])  # integer labels — ambiguous, so no names
    f = CompetingRiskForest(n_estimators=4, random_state=0).fit(df, y)
    assert not hasattr(f, "feature_names_in_")


def test_predict_warns_on_feature_name_mismatch():
    df, y = _toy_df()
    f = CompetingRiskForest(n_estimators=4, random_state=0).fit(df, y)
    renamed = df.rename(columns={"feat0": "renamed"})
    with pytest.warns(UserWarning, match="feature names"):
        f.predict_cif(renamed)
    # SHAP is per-feature attribution, so a silent misalignment there is the
    # most damaging; it must warn like the predict path.
    with pytest.warns(UserWarning, match="feature names"):
        f.shap_values(renamed, times=f.unique_times_[[-1]])
    # matching names stay silent
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        f.predict_cif(df)


def test_importance_uses_dataframe_column_names():
    df, y = _toy_df(n=120, p=3)
    f = CompetingRiskForest(n_estimators=4, random_state=0, samptype="swr").fit(df, y)
    imp = f.compute_importance()
    assert set(imp["feature"]) == set(df.columns)


# ---------------------------------------------------------------------------
# check_estimator: the remaining failures must all be inherent to survival
# estimators, never a contract violation we could fix
# ---------------------------------------------------------------------------

# Checks that cannot pass for a competing-risks estimator. Measured against
# scikit-survival's RandomSurvivalForest on 2026-08-06 (sklearn 1.8.0): it fails
# this same family, so these are ecosystem-wide, not comprisk defects.
_TOLERATED_CHECKS = frozenset(
    {
        # -- structured y --------------------------------------------------
        # check_estimator feeds a plain numeric y; competing-risks data needs
        # both time and event. Guessing which column is which would be worse
        # than failing loudly, so fit raises and every check touching fit dies
        # with it.
        "check_complex_data",
        "check_dict_unchanged",
        "check_dont_overwrite_parameters",
        "check_dtype_object",
        "check_estimators_dtypes",
        "check_estimators_fit_returns_self",
        "check_estimators_overwrite_params",
        "check_estimators_pickle",
        "check_f_contiguous_array_estimator",
        "check_fit1d",
        "check_fit2d_1feature",
        "check_fit2d_1sample",
        "check_fit2d_predict1d",
        "check_fit_check_is_fitted",
        "check_fit_idempotent",
        "check_fit_score_takes_y",
        "check_methods_sample_order_invariance",
        "check_methods_subset_invariance",
        "check_n_features_in",
        "check_n_features_in_after_fitting",
        "check_pipeline_consistency",
        "check_readonly_memmap_input",
        # Same root cause: y is unpacked before X is validated, so these two
        # report missing X validation that in fact exists. Confirmed by direct
        # probe -- fit on a NaN-bearing X with a well-formed y raises
        # "X contains non-finite values", and empty X raises "X must have at
        # least one row".
        "check_estimators_empty_data_messages",
        "check_estimators_nan_inf",
        # -- sparse X not supported (sksurv fails these too) ----------------
        "check_estimator_sparse_array",
        "check_estimator_sparse_matrix",
        "check_estimator_sparse_tag",
        # -- negative X values ---------------------------------------------
        # sksurv fails this one as well; no user-facing benefit to chasing it.
        "check_positive_only_tag_during_fit",
    }
)


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (CompetingRiskForest, {"n_estimators": 3, "random_state": 0}),
        (PenalizedFineGrayRegression, {}),
    ],
    ids=lambda v: v.__name__ if inspect.isclass(v) else "",
)
def test_check_estimator_failures_are_all_expected(cls, kwargs):
    from sklearn.utils.estimator_checks import check_estimator

    results = check_estimator(cls(**kwargs), on_fail=None)
    failed = {r["check_name"] for r in results if r["status"] != "passed"}
    # Subset, not equality: a new sklearn release may add checks we already
    # pass, and pinning the exact set would break on every upgrade.
    unexpected = failed - _TOLERATED_CHECKS
    assert not unexpected, f"{cls.__name__} fails checks it should pass: {sorted(unexpected)}"
    # 13 checks passed when this was written; a floor catches a collapse that a
    # subset assertion alone would not (every check erroring out is "no
    # unexpected failures" too).
    assert len(results) - len(failed) >= 13
