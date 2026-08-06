"""Adversarial probe of scaffold §2 Step B (weighted exchangeability under IPCW selection).

Probe 1: assignment probability given multiset + |I|=k  -> should be ∝ w, NOT uniform.
Probe 2: coverage of the oracle-weighted procedure (test atom = test's own oracle w)
         vs the unweighted-on-I procedure (what 'common weight w' would literally imply).
Probe 3: |I| independent of retained values (the random-size cancellation).
"""

import numpy as np

rng = np.random.default_rng(20260711)


# ---------------- Probe 1: discrete assignment ratio ----------------
# zeta in {a,b}, P(a)=P(b)=0.5 ; selection p(a)=0.9, p(b)=0.2 ; w=1/p.
# 2 calibration + 1 test. Condition on |I|=1 and observed multiset {a,b}
# (retained cal value and test value are one 'a' and one 'b').
# Claim: P(test=b | .) = w(b)/(w(a)+w(b)) = (1/0.2)/(1/0.9+1/0.2) = 0.81818...
def probe1(n_trials=4_000_000):
    pa, pb = 0.9, 0.2
    vals = rng.integers(0, 2, size=(n_trials, 3))  # 0=a, 1=b ; cols: cal1, cal2, test
    u = rng.random(size=(n_trials, 2))
    p_sel = np.where(vals[:, :2] == 0, pa, pb)
    delta = u < p_sel
    k = delta.sum(axis=1)
    # |I| = 1: exactly one cal retained
    one = k == 1
    ret_val = np.where(delta[:, 0], vals[:, 0], vals[:, 1])  # the retained cal value
    test_val = vals[:, 2]
    mixed = one & (ret_val != test_val)  # multiset {a,b}
    frac_test_b = (test_val[mixed] == 1).mean()
    wa, wb = 1 / pa, 1 / pb
    pred = wb / (wa + wb)
    print(
        f"[P1] trials kept={mixed.sum()}  P(test=b|multiset,|I|=1) = {frac_test_b:.5f}"
        f"   predicted ∝w: {pred:.5f}   uniform would be: 0.50000"
    )


# ---------------- Probes 2&3: survival toy ----------------
# T~Exp(1), t*=1, eps=1 -> Y=1{T<=1}. C~Exp(lam) indep, lam=1.5.
# G(u)=exp(-lam u); m=min(T,1); Delta=1{C>=m}; w=exp(lam*m)  (outcome-dependent).
# score: V = 1-pi_Y + 0.001*jitter,  pi_1=0.6, pi_0=0.35  ->  V(Y=1)~0.4, V(Y=0)~0.65.
# Selection knocks out Y=0 (m=1, w=e^1.5=4.48, retained w.p. 0.223) far more than Y=1.
lam = 1.5
tstar = 1.0


def draw(nsub, trials):
    T = rng.exponential(1.0, size=(trials, nsub))
    C = rng.exponential(1.0 / lam, size=(trials, nsub))
    m = np.minimum(T, tstar)
    delta = m <= C
    w = np.exp(lam * m)
    Y1 = tstar >= T
    V = np.where(Y1, 0.4, 0.65) + 0.001 * rng.random(size=(trials, nsub))
    return V, w, delta


def weighted_quantile(V, w, delta, w_test, alpha):
    """per-trial: q_hat = inf{q: sum_{i in I} p_i 1{V_i<=q} >= 1-alpha}, atom w_test at +inf."""
    trials, _nsub = V.shape
    q = np.full(trials, np.inf)
    for t in range(trials):
        idx = delta[t]
        if not idx.any():
            continue
        v, ww = V[t, idx], w[t, idx]
        order = np.argsort(v)
        v, ww = v[order], ww[order]
        denom = ww.sum() + w_test[t]
        cs = np.cumsum(ww) / denom
        hit = np.nonzero(cs >= 1 - alpha)[0]
        if hit.size:
            q[t] = v[hit[0]]
    return q


def probe2(n_cal=20, trials=200_000, alpha=0.1):
    V, w, delta = draw(n_cal, trials)
    Vt_all, wt_all, _ = draw(1, trials)
    Vt, wt = Vt_all[:, 0], wt_all[:, 0]
    # (a) oracle-weighted, test atom = test's own oracle weight (the ceiling lemma)
    q_w = weighted_quantile(V, w, delta, wt, alpha)
    cov_w = (Vt <= q_w).mean()
    # (b) 'common weight' literal reading -> uniform assignment -> unweighted on I,
    #     atom mass 1/(|I|+1)  == weighted_quantile with all weights 1
    ones = np.ones_like(w)
    q_u = weighted_quantile(V, ones, delta, np.ones(trials), alpha)
    cov_u = (Vt <= q_u).mean()
    se = np.sqrt(0.9 * 0.1 / trials)
    print(f"[P2] n={n_cal} trials={trials}  target={1 - alpha:.3f}  (MC se≈{se:.4f})")
    print(f"     oracle-weighted coverage  = {cov_w:.4f}   (claim: >= 0.900)")
    print(f"     unweighted-on-I coverage  = {cov_u:.4f}   (literal 'common w' reading)")


def probe3(n_cal=50, trials=400_000):
    """|I| vs retained-value independence: E[V_i | i in I, |I|=k] flat in k."""
    V, _w, delta = draw(n_cal, trials)
    k = delta.sum(axis=1)
    print("[P3] mean retained score by |I| (should be flat if size ⟂ values):")
    for kk in [5, 10, 15, 20, 25, 30]:
        sel = k == kk
        if sel.sum() < 500:
            continue
        mv = V[sel][delta[sel]].mean()
        print(f"     |I|={kk:2d}  trials={sel.sum():6d}  mean retained V = {mv:.5f}")
    # theory: E_Qsel[V] = E[p V]/E[p]
    T = rng.exponential(1.0, size=2_000_000)
    m = np.minimum(T, tstar)
    p = np.exp(-lam * m)
    Vv = np.where(tstar >= T, 0.4, 0.65) + 0.0005
    print(
        f"     theory E_Qsel[V] = {(p * Vv).mean() / p.mean():.5f}   (population E[V]={Vv.mean():.5f})"
    )


probe1()
probe2(n_cal=20)
probe2(n_cal=5, trials=200_000)
probe3()
