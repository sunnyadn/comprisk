"""Shared input validation for comprisk estimators."""

from __future__ import annotations

import warnings

import numpy as np


def extract_feature_names(X):
    """Column names of a DataFrame-like ``X``, or None.

    Implements the SLEP007 rule without depending on sklearn's
    ``validate_data`` (public only from 1.6, while we declare
    ``scikit-learn>=1.3``) or its private ``_get_feature_names``.

    Names are returned only when *every* column label is a string, matching
    sklearn: a mix of strings and integers is ambiguous, so no names are
    recorded.
    """
    columns = getattr(X, "columns", None)
    if columns is None:
        return None
    names = list(columns)
    if not names or not all(isinstance(name, str) for name in names):
        return None
    return np.asarray(names, dtype=object)


def record_feature_names(estimator, X):
    """Set ``estimator.feature_names_in_`` from ``X``; call this in ``fit``.

    An unnamed ``X`` clears any attribute left by a previous fit — sklearn's
    contract is that the attribute is *absent*, not None, when there are no
    names, and ``fit`` must not leave stale state behind.
    """
    names = extract_feature_names(X)
    if names is None:
        estimator.__dict__.pop("feature_names_in_", None)
    else:
        estimator.feature_names_in_ = names


def warn_if_feature_names_differ(estimator, X):
    """Warn when predict-time column names disagree with those seen in ``fit``.

    Warns rather than raises, mirroring sklearn, so a Pipeline whose earlier
    step drops names stays usable.
    """
    fitted = getattr(estimator, "feature_names_in_", None)
    if fitted is None:
        return
    names = extract_feature_names(X)
    if names is None or np.array_equal(fitted, names):
        return
    warnings.warn(
        "X has feature names that differ from those seen during fit. "
        f"Fitted on {list(fitted)}; got {list(names)}.",
        UserWarning,
        stacklevel=2,
    )


def validate_predict_X(estimator, X):
    """Check ``X`` against what ``estimator`` was fitted on and return it as float64.

    The predict-side counterpart of :func:`record_feature_names`: recording
    ``feature_names_in_`` is only useful if something reads it back, so the name
    check and the shape checks live together and every X-consuming method calls
    this one function.
    """
    warn_if_feature_names_differ(estimator, X)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got ndim={X.ndim}")
    if X.shape[1] != estimator.n_features_in_:
        raise ValueError(
            f"X has wrong n_features: expected {estimator.n_features_in_}, got {X.shape[1]}"
        )
    return X


def check_inputs(X, time, event):
    """Validate and canonicalize (X, time, event) for a CR forest fit.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix. Must be 2-D and numeric with no NaN/inf.
    time : array-like, shape (n_samples,)
        Observed times. Must be 1-D, finite, non-negative.
    event : array-like, shape (n_samples,)
        Event codes. Must be 1-D integer in {0, 1, ..., n_causes}
        where 0 = censored. Causes must form the contiguous set
        {1, ..., n_causes}.

    Returns
    -------
    X : ndarray, float64, shape (n_samples, n_features)
    time : ndarray, float64, shape (n_samples,)
    event : ndarray, int64, shape (n_samples,)
    n_causes : int
    """
    X = np.asarray(X, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    event_raw = np.asarray(event)

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got ndim={X.ndim}")
    if time.ndim != 1:
        raise ValueError(f"time must be 1-D; got ndim={time.ndim}")
    if event_raw.ndim != 1:
        raise ValueError(f"event must be 1-D; got ndim={event_raw.ndim}")

    n = X.shape[0]
    if n == 0:
        raise ValueError("X must have at least one row")
    if time.shape[0] != n or event_raw.shape[0] != n:
        raise ValueError(
            f"length mismatch: X has {n} rows, "
            f"time has {time.shape[0]}, event has {event_raw.shape[0]}"
        )

    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values (NaN or inf)")
    if not np.all(np.isfinite(time)):
        raise ValueError("time contains non-finite values (NaN or inf)")
    if np.any(time < 0):
        raise ValueError("time values must be non-negative")

    if np.issubdtype(event_raw.dtype, np.floating):
        if not np.all(event_raw == np.floor(event_raw)):
            raise ValueError("event values must be integer-valued")
    elif not np.issubdtype(event_raw.dtype, np.integer):
        raise ValueError(f"event must be integer-typed; got dtype={event_raw.dtype}")
    event = event_raw.astype(np.int64)
    if np.any(event < 0):
        raise ValueError("event codes must be non-negative integers (0=censored)")

    causes_present = {int(c) for c in event[event > 0]}
    if not causes_present:
        raise ValueError("event must contain at least one event (> 0)")
    n_causes = max(causes_present)
    expected = set(range(1, n_causes + 1))
    if causes_present != expected:
        raise ValueError(
            "event codes must be contiguous from 1 to n_causes; "
            f"got causes {sorted(causes_present)}, expected {sorted(expected)}"
        )

    return X, time, event, n_causes
