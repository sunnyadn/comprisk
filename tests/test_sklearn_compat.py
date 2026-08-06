"""sklearn drop-in compatibility surface tests.

Verifies CompetingRiskForest is a true sklearn-compatible estimator:

* ``Surv.from_arrays(event, time)`` builds the structured y the same way
  scikit-survival does, so users can swap libraries without rewiring data.
* ``fit(X, y)`` and ``score(X, y)`` accept the structured y form, equivalent
  to the legacy three-argument ``fit(X, time, event)`` / ``score(X, time, event)``.
* ``predict(X)`` is a sklearn alias for ``predict_risk(X, cause=1)``.
* ``cross_val_score`` and ``clone`` work without a wrapper.
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


def test_fit_validates_device():
    X, time, event = _toy_cr(n=60, p=3)
    f = CompetingRiskForest(n_estimators=2, random_state=0, device="nonsense")
    with pytest.raises(ValueError, match="device must be one of"):
        f.fit(X, time, event)


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
def test_set_params_roundtrip(cls):
    est = cls()
    name, value = next(iter(est.get_params().items()))
    assert est.set_params(**{name: value}) is est
    assert est.get_params()[name] == value


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


def test_feature_names_recorded_from_dataframe():
    df, y = _toy_df()
    f = CompetingRiskForest(n_estimators=4, random_state=0).fit(df, y)
    assert list(f.feature_names_in_) == list(df.columns)


def test_feature_names_absent_for_ndarray():
    X, time, event = _toy_cr(n=100, p=3)
    f = CompetingRiskForest(n_estimators=4, random_state=0).fit(X, time, event)
    assert not hasattr(f, "feature_names_in_")


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
    # matching names stay silent
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        f.predict_cif(df)


def test_importance_uses_dataframe_column_names():
    df, y = _toy_df(n=120, p=3)
    f = CompetingRiskForest(n_estimators=4, random_state=0, samptype="swr").fit(df, y)
    imp = f.compute_importance()
    assert set(imp["feature"]) == set(df.columns)
