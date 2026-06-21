"""Closed-form influence-function variance for the Uno/Wolbers cause-specific
IPCW concordance index, and confidence intervals derived from it.

The estimator is the one in :func:`comprisk.metrics.concordance_index_uno_cr`
(Wolbers et al., 2014, *Biostatistics* 15(3):526-539, eq. 3.4; Uno et al., 2011,
*Statistics in Medicine* 30:1105-1116). Its influence function is asymptotically
linear (Wolbers Lemma 3.1, Web Appendix D); the operationalisation matches the
perturbation scheme of Uno's ``survC1`` package (``unoCW``: a pair term ``phi^U``
plus a censoring-survivor term ``phi^G``).

For a held-out evaluation the predicted risks are conditioned on (treated as
fixed), so the per-subject influence is ``phi_k = phi^U_k + phi^G_k`` and
``Var(C-hat) = sum_k phi_k^2 / n^2``. For two models on the same subjects the
paired covariance is analytic via ``phi^A_k - phi^B_k``.

The first-order variance is valid only where the censoring survivor is bounded
away from zero: otherwise the ``1/G^2`` weights blow up in the right tail and the
linearisation underestimates the variance. Tail stabilisation -- the default
``gmin="auto"`` ESS clip of :func:`compute_uno_weights`, or an explicit ``tau`` --
is therefore required (mirroring the truncation Uno/survC1 mandate).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from comprisk.metrics import _EPS_T, compute_uno_weights


def concordance_influence(event, time, estimate, weights, *, cause):
    """Point estimate and per-subject influence function of the Uno/Wolbers
    cause-specific IPCW concordance.

    Returns ``(C, phi)`` where ``phi`` has length ``n`` and is scaled so that
    ``Var(C) = sum(phi**2) / n**2``. ``C`` is ``NaN`` when undefined.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event)
    p = np.asarray(estimate, dtype=float)
    w = np.asarray(weights, dtype=float)
    n = time.size
    if n == 0:
        return float("nan"), np.zeros(0)

    w1 = np.sqrt(np.maximum(w, 0.0))
    keep = w > 0
    is_cause = event == cause
    is_cens = event == 0
    case = np.flatnonzero(is_cause & keep)
    comp = np.flatnonzero((event > 0) & (event != cause) & keep)

    Ndot = np.zeros(n)
    Ddot = np.zeros(n)
    # uncentred per-subject endpoint masses feeding phi^G; centre as gh - C*gd
    gh = np.zeros(n)
    gd = np.zeros(n)
    N = 0.0
    D = 0.0

    # ---- Branch A: case i vs comparator j (t_j>t_i, or t_j==t_i & censored) ----
    for i in case:
        jA = np.flatnonzero(((time > time[i]) | ((time == time[i]) & is_cens)) & keep)
        if not jA.size:
            continue
        hh = (p[jA] < p[i]).astype(float) + 0.5 * (p[jA] == p[i])
        sh = hh.sum()
        npair = jA.size
        N += 2.0 * w[i] * sh
        D += 2.0 * w[i] * npair
        Ndot[i] += 2.0 * w[i] * sh
        Ddot[i] += 2.0 * w[i] * npair
        Ndot[jA] += 2.0 * w[i] * hh  # jA is a unique index set (flatnonzero)
        Ddot[jA] += 2.0 * w[i]
        gh[i] += 4.0 * w[i] * sh
        gd[i] += 4.0 * w[i] * npair

    # ---- Branch C: case i vs competing j (t_j <= t_i), symmetric sqrt weights ----
    if comp.size:
        t_comp = time[comp]
        p_comp = p[comp]
        w1_comp = w1[comp]
        for i in case:
            mask = t_comp <= time[i]
            if not mask.any():
                continue
            jC = comp[mask]
            hh = (p_comp[mask] < p[i]).astype(float) + 0.5 * (p_comp[mask] == p[i])
            wj1 = w1_comp[mask]
            ww = 2.0 * w1[i] * wj1
            nc = ww * hh
            N += nc.sum()
            D += ww.sum()
            Ndot[i] += nc.sum()
            Ddot[i] += ww.sum()
            Ndot[jC] += nc  # jC is a unique index set
            Ddot[jC] += ww
            gh[i] += 2.0 * w1[i] * (wj1 * hh).sum()
            gd[i] += 2.0 * w1[i] * wj1.sum()
            gh[jC] += ww * hh
            gd[jC] += ww

    # ---- Branch B: tied case-times (ordered pairs within an equal-time group) ----
    if case.size >= 2:
        ct = time[case]
        order = np.argsort(ct, kind="stable")
        sc = case[order]
        sct = ct[order]
        g0 = 0
        d_total = sc.size
        while g0 < d_total:
            g1 = g0
            while g1 + 1 < d_total and abs(sct[g1 + 1] - sct[g0]) <= _EPS_T:
                g1 += 1
            grp = sc[g0 : g1 + 1]
            d = grp.size
            if d >= 2:
                pg = p[grp]
                for a in range(d):
                    i = grp[a]
                    # ordered pairs (i, i') with i' != i: nc=w_i*hB, dc=w_i,
                    # hB = 0.5 + 0.5*1{prediction tie}
                    other = np.ones(d, dtype=bool)
                    other[a] = False
                    hb = 0.5 + 0.5 * (np.abs(pg[other] - pg[a]) <= _EPS_T)
                    sum_hb = hb.sum()
                    cnt_other = d - 1
                    N += w[i] * sum_hb
                    D += w[i] * cnt_other
                    Ndot[i] += w[i] * sum_hb
                    Ddot[i] += w[i] * cnt_other
                    Ndot[grp[other]] += w[i] * hb  # partner side
                    Ddot[grp[other]] += w[i]
                    gh[i] += 2.0 * w[i] * sum_hb
                    gd[i] += 2.0 * w[i] * cnt_other
            g0 = g1 + 1

    if D <= 0:
        return float("nan"), np.zeros(n)
    C = N / D
    phiU = n * (Ndot - C * Ddot) / D

    # ---- phi^G: propagate G-hat estimation error via the censoring time grid ----
    gcoef = gh - C * gd
    knots, counts = np.unique(time, return_counts=True)
    K = knots.size
    mk = np.searchsorted(knots, time)  # exact knot index per subject
    # per-knot risk-set and event counts in O(n + K) (no per-knot scan of `time`)
    nrisk = (n - np.cumsum(counts) + counts).astype(float)  # #{time >= knots[k]}
    d_one = np.bincount(mk[is_cause], minlength=K).astype(float)
    d_oth = np.bincount(mk[~is_cause], minlength=K).astype(float)
    r = nrisk - d_one
    pi_r = r / n
    dLam = np.zeros(K)
    ok = (d_oth > 0) & (r > 0)
    dLam[ok] = d_oth[ok] / r[ok]
    Mbar = np.bincount(mk, weights=gcoef, minlength=K)
    Mcum = np.cumsum(Mbar[::-1])[::-1]  # Mcum[l] = sum_{m>=l} Mbar[m]
    Gl = np.zeros(K)
    Gl[ok] = dLam[ok] * Mcum[ok] / pi_r[ok]
    Gcum = np.cumsum(Gl)
    Gcum_prev = np.concatenate(([0.0], Gcum))  # Gcum_prev[m] = Gcum[m-1]

    phiG = np.empty(n)
    # split by ACTUAL event type (not keep): a truncated cause event still informs G
    cm = is_cause
    phiG[cm] = -Gcum_prev[mk[cm]] / D
    om = ~cm
    mko = mk[om]
    pir_o = pi_r[mko]
    # truncated-tail knots can have pi_r==0; their Mcum/pi_r term drops out
    mass = np.where(pir_o > 0, Mcum[mko] / np.where(pir_o > 0, pir_o, 1.0), 0.0)
    phiG[om] = (mass - Gcum[mko]) / D

    return C, phiU + phiG


@dataclass(frozen=True)
class ConcordanceCI:
    """Result of :func:`concordance_index_ci`."""

    estimate: float
    se: float
    ci_low: float
    ci_high: float
    confidence_level: float
    n: int


@dataclass(frozen=True)
class DeltaConcordanceCI:
    """Result of :func:`concordance_index_delta_ci` (model A minus model B)."""

    estimate: float
    se: float
    ci_low: float
    ci_high: float
    pvalue: float
    confidence_level: float
    n: int


def _ipcw_weights(time, event, *, cause, gmin, tau, weight_kwargs):
    w = compute_uno_weights(time, event, gmin=gmin, **weight_kwargs)
    if tau is not None:
        w = w.copy()
        w[np.asarray(time, dtype=float) >= float(tau)] = 0.0
    return w


def concordance_index_ci(
    event,
    time,
    estimate,
    *,
    cause: int = 1,
    confidence_level: float = 0.95,
    gmin: float | str = "auto",
    tau: float | None = None,
    transform: str = "logit",
    **weight_kwargs,
) -> ConcordanceCI:
    """Uno/Wolbers cause-specific IPCW concordance with a closed-form
    influence-function confidence interval (no bootstrap).

    The variance is the deterministic IF estimator (Wolbers 2014 / Uno 2011);
    it agrees with the nonparametric bootstrap to Monte-Carlo error and with
    Uno's ``survC1`` perturbation inference, at ``O(n log n)`` cost.

    Parameters
    ----------
    event, time, estimate : array-like
        Event code (0 = censored, ``cause`` = event of interest, other positive
        = competing), observed time, predicted risk score for ``cause``.
    cause : int, default 1
        Cause of interest.
    confidence_level : float, default 0.95
        Two-sided confidence level.
    gmin : float | {"auto", "none"}, default "auto"
        Tail stabilisation passed to :func:`compute_uno_weights`. ``"auto"``
        (ESS clip) keeps the IF valid; ``"none"`` disables stabilisation and is
        only safe together with ``tau``.
    tau : float, optional
        Truncation time; observations with ``time >= tau`` are dropped from the
        IPCW weighting (Uno's truncation). Stabilises the ``1/G^2`` tail.
    transform : {"logit", "none"}, default "logit"
        ``"logit"`` builds the interval on the log-odds scale (better small-sample
        coverage for a bounded statistic) and back-transforms; ``"none"`` is a
        symmetric Wald interval.

    Returns
    -------
    ConcordanceCI
    """
    if transform not in ("logit", "none"):
        raise ValueError(f"transform must be 'logit' or 'none', got {transform!r}")
    time = np.asarray(time, dtype=float)
    w = _ipcw_weights(time, event, cause=cause, gmin=gmin, tau=tau, weight_kwargs=weight_kwargs)
    C, phi = concordance_influence(event, time, estimate, w, cause=cause)
    n = phi.size
    if not np.isfinite(C):
        return ConcordanceCI(C, float("nan"), float("nan"), float("nan"), confidence_level, n)
    se = float(np.sqrt((phi**2).sum()) / n)
    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    if transform == "logit" and 0.0 < C < 1.0:
        # interval on the log-odds scale, back-transformed (better small-n coverage)
        lodds = np.log(C / (1.0 - C))
        se_l = se / (C * (1.0 - C))
        lo = float(1.0 / (1.0 + np.exp(-(lodds - z * se_l))))
        hi = float(1.0 / (1.0 + np.exp(-(lodds + z * se_l))))
    else:  # "none", or "logit" at the C in {0,1} boundary -> symmetric Wald
        lo = float(C - z * se)
        hi = float(C + z * se)
    return ConcordanceCI(float(C), se, lo, hi, confidence_level, n)


def concordance_index_delta_ci(
    event,
    time,
    estimate_a,
    estimate_b,
    *,
    cause: int = 1,
    confidence_level: float = 0.95,
    gmin: float | str = "auto",
    tau: float | None = None,
    **weight_kwargs,
) -> DeltaConcordanceCI:
    """Paired difference in IPCW concordance, ``C(A) - C(B)``, with a closed-form
    influence-function CI and a two-sided Wald p-value.

    Both models are scored on the same subjects and the same IPCW weights, so the
    paired covariance is captured analytically by ``phi_A - phi_B`` -- no shared
    bootstrap resampling needed. This is the inference ``model_compare`` needs.
    """
    time = np.asarray(time, dtype=float)
    w = _ipcw_weights(time, event, cause=cause, gmin=gmin, tau=tau, weight_kwargs=weight_kwargs)
    Ca, phi_a = concordance_influence(event, time, estimate_a, w, cause=cause)
    Cb, phi_b = concordance_influence(event, time, estimate_b, w, cause=cause)
    n = phi_a.size
    delta = float(Ca - Cb)
    if not np.isfinite(delta):
        return DeltaConcordanceCI(
            delta, float("nan"), float("nan"), float("nan"), float("nan"), confidence_level, n
        )
    se = float(np.sqrt(((phi_a - phi_b) ** 2).sum()) / n)
    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    lo = delta - z * se
    hi = delta + z * se
    pvalue = float(2.0 * norm.sf(abs(delta) / se)) if se > 0 else (1.0 if delta == 0 else 0.0)
    return DeltaConcordanceCI(delta, se, lo, hi, pvalue, confidence_level, n)
