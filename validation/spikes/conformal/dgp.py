"""Step 1 of the conformal-CR spike: synthetic competing-risks DGP, horizon
labelling, and IPCW weights.

Target reframing (see task design.md): we conformalize the *event-type-at-horizon*
label

    y_i(t*) in {1, ..., K, EVENT_FREE}

which is observable for every subject EXCEPT those censored (event == 0) before t*.
Competing events before t* are fully-observed labels, NOT censoring.

Censoring handling here is the standard Graf/Gerds IPCW for at-horizon quantities:
weight observed subjects by 1 / Ghat(min(T_i, t*)^-), where Ghat is the
Kaplan-Meier of the *censoring* distribution (event == 0 as the censoring "event").

NOTE: we deliberately do NOT reuse metrics._km_censor_fit / compute_uno_weights.
Those use the CR "events-first" tie convention that lumps competing events into the
censoring decrement and weight by 1/Ghat^2 (concordance-pair semantics). For
at-horizon classification the correct object is the plain censoring KM with a single
1/Ghat power. Keeping a clean local implementation avoids that semantic mismatch.
"""

from __future__ import annotations

import numpy as np

# Sentinel label for "event-free at t*". Real causes are 1..K.
EVENT_FREE = 0


def cr_dgp(
    n: int,
    *,
    censor_rate: float = 0.3,
    competing_frac: float = 0.4,
    signal: float = 0.0,
    t_star: float = 1.0,
    s_at_tstar: float = 0.5,
    p: int = 5,
    seed: int = 0,
):
    """Two-cause competing-risks data with constant cause-specific hazards.

    Cause-1 hazard carries the covariate signal (first feature); cause 2 is the
    competing risk. Censoring is independent exponential tuned to ``censor_rate``.

    With ``signal == 0`` the marginal (population) CIFs are closed-form, which the
    companion check script uses as ground truth:

        lambda = lambda1 + lambda2,  S(t)   = exp(-lambda t)
                                     F_k(t) = (lambda_k / lambda) (1 - exp(-lambda t))

    ``s_at_tstar`` sets the overall event hazard so that S(t_star) == s_at_tstar at
    signal 0 (keeps a meaningful spread of labels at the horizon).

    Returns
    -------
    X : (n, p) float64
    time : (n,) float64   observed time = min(event time, censoring time)
    event : (n,) int64    0 = censored, k>=1 = cause-k event
    info : dict           base rates + true marginal F_1, F_2, S at t_star (signal 0)
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))

    # Overall event hazard so that exp(-lambda * t_star) == s_at_tstar.
    lam = -np.log(s_at_tstar) / t_star
    lam2 = competing_frac * lam  # competing (cause 2) baseline
    lam1_base = (1.0 - competing_frac) * lam  # primary (cause 1) baseline

    # Cause-1 hazard modulated by the first covariate; renormalise so the mean
    # cause-1 hazard stays at lam1_base (keeps censor_rate / label balance stable
    # as signal grows).
    eta1 = signal * X[:, 0]
    mult = np.exp(eta1)
    rate1 = lam1_base * mult / mult.mean()
    rate2 = np.full(n, lam2)

    t1 = rng.exponential(1.0 / np.maximum(rate1, 1e-12))
    t2 = rng.exponential(1.0 / np.maximum(rate2, 1e-12))
    t_event = np.minimum(t1, t2)
    cause = np.where(t1 <= t2, 1, 2).astype(np.int64)

    # Independent censoring: P(C < T) ~= censor_rate at signal 0.
    if censor_rate <= 0:
        t_cens = np.full(n, np.inf)
    else:
        lam_c = lam * censor_rate / (1.0 - censor_rate)
        t_cens = rng.exponential(1.0 / lam_c, size=n)

    time = np.minimum(t_event, t_cens)
    event = np.where(t_event <= t_cens, cause, 0).astype(np.int64)

    s_true = float(np.exp(-lam * t_star))
    info = {
        "lambda": lam,
        "lambda1": lam1_base,
        "lambda2": lam2,
        "t_star": t_star,
        "F1_true": (lam1_base / lam) * (1.0 - s_true),
        "F2_true": (lam2 / lam) * (1.0 - s_true),
        "S_true": s_true,
        "realized_censor_rate": float(np.mean(event == 0)),
    }
    return X, time, event, info


def horizon_labels(time: np.ndarray, event: np.ndarray, t_star: float):
    """Label each subject at horizon ``t_star``.

    Returns
    -------
    y : (n,) int64        EVENT_FREE (0) or cause k>=1; meaningless where not observed
    observed : (n,) bool  False iff censored (event==0) strictly before t_star
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=np.int64)

    event_before = (time <= t_star) & (event >= 1)
    free_at_tstar = time > t_star  # event-free at t* regardless of later fate
    censored_before = (time <= t_star) & (event == 0)

    y = np.full(time.shape, EVENT_FREE, dtype=np.int64)
    y[event_before] = event[event_before]
    # free_at_tstar already EVENT_FREE; censored_before flagged not-observed below.
    observed = ~censored_before
    return y, observed


def censoring_km(time: np.ndarray, event: np.ndarray):
    """Kaplan-Meier of the censoring distribution (event == 0 as the 'event').

    Standard KM: at each unique time, decrement by censorings / at-risk. Returns
    sorted unique times and the post-update survivor Ghat at each.
    """
    time = np.asarray(time, dtype=float)
    is_cens = (np.asarray(event) == 0).astype(np.int64)
    n = time.size
    if n == 0:
        return np.empty(0), np.empty(0)

    order = np.argsort(time, kind="stable")
    t_sorted = time[order]
    c_sorted = is_cens[order]

    t_unique, first_idx = np.unique(t_sorted, return_index=True)
    bounds = np.append(first_idx, n)

    G = np.empty(t_unique.shape[0], dtype=float)
    surv = 1.0
    for k in range(t_unique.shape[0]):
        s, e = int(bounds[k]), int(bounds[k + 1])
        n_risk = n - s
        d_cens = int(c_sorted[s:e].sum())
        if d_cens > 0 and n_risk > 0:
            surv *= 1.0 - d_cens / n_risk
        G[k] = surv
    return t_unique, G


def _ghat_minus(t_unique: np.ndarray, G: np.ndarray, query: np.ndarray):
    """Left-limit Ghat(q^-): 1.0 at/below the first knot, else G at the largest
    knot strictly less than q."""
    query = np.asarray(query, dtype=float)
    if t_unique.size == 0:
        return np.ones(query.shape)
    idx = np.searchsorted(t_unique, query, side="left")
    out = np.ones(query.shape)
    has_pred = idx > 0
    out[has_pred] = G[idx[has_pred] - 1]
    return out


def ipcw_weights_at_horizon(
    time: np.ndarray,
    event: np.ndarray,
    t_star: float,
    *,
    gmin: float = 0.05,
):
    """Graf/Gerds IPCW weights for at-horizon quantities.

    w_i = 1 / Ghat( min(T_i, t*)^- )   for observed subjects, 0 for censored-before-t*.
    Ghat is clipped below at ``gmin`` (tau / weight stabilisation; mirrors the IF-CI
    spike finding that 1/Ghat right-tail blow-up breaks validity).
    """
    time = np.asarray(time, dtype=float)
    _, observed = horizon_labels(time, event, t_star)

    t_unique, G = censoring_km(time, event)
    q = np.minimum(time, t_star)
    g = _ghat_minus(t_unique, G, q)
    g = np.maximum(g, gmin)

    w = np.zeros(time.shape, dtype=float)
    w[observed] = 1.0 / g[observed]
    return w, observed
