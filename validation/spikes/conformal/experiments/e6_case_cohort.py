"""Experiment 6 (Phase 1 / C3): general selection lemma, case-cohort instance.

Validates general-selection.tex Lemma lem:gwe + Theorem thm:gmain, instance (b):
when the selection probability s(Z) is KNOWN by design (case-cohort / two-phase),
the estimated weight equals the oracle (shat = s), so Delta_w = 0 and the floor is the
pure oracle ceiling >= 1-alpha. This isolates the SELECTION mechanism from any
censoring-estimation noise -- the crispest possible check that outcome-dependent
selection is conformalizable at all.

Design (self-contained; oracle score model to isolate selection from model fit):
  - 3-class problem, true class probabilities pi_true(x) = softmax(W x) KNOWN.
  - Nonconformity score v(x,y) = 1 - pi_true[y].
  - Calibration selection is OUTCOME-DEPENDENT and KNOWN: s_sel(y) oversamples the
    rare class (a case-cohort design), s_sel >= g_min. Delta_i ~ Bern(s_sel(y_i)).
  - Weighted split conformal on the retained points with w = 1/s_sel(y), 1/g_min atom.
  - Test point is UNSELECTED (population); coverage = plain mean 1{y in S(x)}.

Three procedures compared, under TWO selection designs (rare-oversampled = the
canonical case-cohort; common-oversampled = the mirror), to show naive's error swings
sign while the weighted procedure is always right:
  oracle-select : weight retained by 1/s_sel (known) -- hits 1-alpha (the lemma).
  naive         : ignore selection (unit weights on retained) -- MIS-covers by an
                  amount and SIGN that depend on the score-class correlation (legitimate
                  here; unlike the real CR pipeline where naive is already valid, 07-12).
  full-oracle   : conformal on the FULL unselected calibration sample -- the ceiling
                  the weighted procedure is trying to recover.

Run:  PYTHONUNBUFFERED=1 uv run python -m validation.spikes.conformal.experiments.e6_case_cohort
"""

from __future__ import annotations

import numpy as np

ALPHA = 0.1
REPS = 40
K = 3
N_CAL = 4000
N_TEST = 8000
# Two case-cohort selection designs (each outcome-dependent, KNOWN, s >= g_min=0.20).
DESIGNS = {
    "rare-oversampled": np.array([0.20, 0.45, 0.90]),  # canonical: oversample rare class 2
    "common-oversampled": np.array([0.90, 0.45, 0.20]),  # mirror: oversample common class 0
}
# True class-score weights (fixed, define pi_true = softmax(W x)); class 2 rare.
W = np.array([[1.4, -0.3], [0.2, 0.9], [-1.1, -0.7]])
BIAS = np.array([0.6, 0.2, -1.3])


def _pi_true(X):
    logits = X @ W.T + BIAS
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def _draw(n, rng):
    X = rng.normal(size=(n, 2))
    pi = _pi_true(X)
    u = rng.random(n)
    y = (u[:, None] > np.cumsum(pi, axis=1)).sum(axis=1)  # inverse-CDF sample
    y = np.clip(y, 0, K - 1)
    return X, y, pi


def _weighted_quantile(scores, weights, alpha, atom):
    order = np.argsort(scores, kind="stable")
    s, w = scores[order], weights[order]
    cum = np.cumsum(w) / (w.sum() + atom)
    idx = np.searchsorted(cum, 1 - alpha, side="left")
    return np.inf if idx >= s.size else float(s[idx])


def _coverage(pi_test, y_test, qhat):
    sets = pi_test >= (1.0 - qhat)
    covered = sets[np.arange(y_test.size), y_test]
    return float(covered.mean()), float(sets.sum(1).mean())


def _one_rep(seed, s_sel):
    rng = np.random.default_rng(seed)
    _Xc, yc, pic = _draw(N_CAL, rng)
    _Xt, yt, pit = _draw(N_TEST, rng)
    g_min = float(s_sel.min())

    v = 1.0 - pic[np.arange(yc.size), yc]  # calibration scores (oracle pi)
    sel = rng.random(yc.size) < s_sel[yc]  # Bernoulli(s_sel(y)) selection
    w_known = 1.0 / s_sel[yc]

    q_oracle = _weighted_quantile(v[sel], w_known[sel], ALPHA, atom=1.0 / g_min)
    q_naive = _weighted_quantile(v[sel], np.ones(sel.sum()), ALPHA, atom=1.0)
    q_full = _weighted_quantile(v, np.ones(yc.size), ALPHA, atom=1.0)

    return (
        _coverage(pit, yt, q_oracle),
        _coverage(pit, yt, q_naive),
        _coverage(pit, yt, q_full),
    )


def main():
    nominal = 1 - ALPHA
    print(
        f"\nExp 6 -- general selection lemma, case-cohort instance "
        f"(alpha={ALPHA}, nominal={nominal:.2f}, reps={REPS})"
    )
    print("Known outcome-dependent selection (shat=s so Delta_w=0); test is unselected.\n")
    for dname, s_sel in DESIGNS.items():
        acc = {"full-oracle": [], "oracle-select": [], "naive": []}
        sz = {"full-oracle": [], "oracle-select": [], "naive": []}
        for r in range(REPS):
            (co, so), (cn, sn), (cf, sf) = _one_rep(100 + r, s_sel)
            acc["oracle-select"].append(co)
            sz["oracle-select"].append(so)
            acc["naive"].append(cn)
            sz["naive"].append(sn)
            acc["full-oracle"].append(cf)
            sz["full-oracle"].append(sf)
        print(f"  design: {dname}  s_sel={s_sel.tolist()}  g_min={float(s_sel.min())}")
        print(f"    {'procedure':<16}{'cov':>8}{'dev':>8}{'size':>8}")
        for name in ("full-oracle", "oracle-select", "naive"):
            m = float(np.mean(acc[name]))
            print(f"    {name:<16}{m:>8.3f}{m - nominal:>+8.3f}{float(np.mean(sz[name])):>8.2f}")
        print()
    print(
        "Read: oracle-select tracks full-oracle to ~nominal in BOTH designs (the lemma:"
        "\nknown outcome-dependent weights => Delta_w=0 => exact ceiling). naive MIS-covers"
        "\nwith a sign that flips between designs -- the weight is doing real work."
    )


if __name__ == "__main__":
    main()
