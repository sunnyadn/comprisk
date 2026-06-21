"""Gate 1 (step 1): O(n log n) Fenwick computation of the Branch-A pair influence
Ndot/Ddot, validated bit-for-bit against the O(n^2) reference (naive_full).

Branch A only here (the dominant cost). For each case i, comparator set is
{t_j>t_i} plus same-time censored; per pair N+=2 w_i h_ij, D+=2 w_i, attributed
to both i and j. We need, for every subject k:
  Ndot_k = (k as case)  2 w_k * sum_j h_kj
         + (k as comparator) sum_{i case, k comparable} 2 w_i h_ik
  Ddot_k = (k as case)  2 w_k * #comparators
         + (k as comparator) sum_{i case, k comparable} 2 w_i
with h_ab = 1{p_b<p_a} + 0.5*1{p_b=p_a}.

Continuous times here (no ties) to isolate the algorithm; the censored same-time
term and ties are exercised by the naive reference too, so any mismatch shows up.
"""

import sys

import numpy as np
from validate_if import gen

from comprisk.metrics import compute_uno_weights


class Fenwick:
    """1-indexed BIT over compressed ranks; stores summed weights."""

    def __init__(self, m):
        self.m = m
        self.t = np.zeros(m + 1)

    def add(self, i, v):
        i += 1
        while i <= self.m:
            self.t[i] += v
            i += i & (-i)

    def pref(self, i):
        # sum over ranks [0, i]
        i += 1
        s = 0.0
        while i > 0:
            s += self.t[i]
            i -= i & (-i)
        return s


def naive_branchA(t, e, p, w, CAUSE=1):
    n = len(t)
    keep = w > 0
    case = np.flatnonzero((e == CAUSE) & keep)
    Ndot = np.zeros(n)
    Ddot = np.zeros(n)
    for i in case:
        jA = np.flatnonzero(((t > t[i]) | ((t == t[i]) & (e == 0))) & keep)
        if jA.size:
            hh = (p[jA] < p[i]).astype(float) + 0.5 * (p[jA] == p[i])
            nc = 2.0 * w[i] * hh
            dc = 2.0 * w[i] * np.ones(jA.size)
            Ndot[i] += nc.sum()
            Ddot[i] += dc.sum()
            np.add.at(Ndot, jA, nc)
            np.add.at(Ddot, jA, dc)
    return Ndot, Ddot


def fenwick_branchA(t, e, p, w, CAUSE=1):
    """Continuous-time Branch A (no same-time censored term): comparator = t_j>t_i.
    Sweep subjects in DECREASING time; a Fenwick over p-rank holds, among already-
    seen (= later-time) subjects, the count and weight by rank so a case can query
    its comparators, and a comparator accumulates from later-seen cases via a second
    pass. Two Fenwicks: one keyed for case->comparator, one for comparator->case."""
    n = len(t)
    keep = w > 0
    # compress p to ranks (no ties assumed); strict < and == handled via rank
    order_p = np.argsort(p, kind="stable")
    prank = np.empty(n, dtype=np.int64)
    prank[order_p] = np.arange(n)
    m = n

    Ndot = np.zeros(n)
    Ddot = np.zeros(n)

    # sort by time ascending; process from latest to earliest
    torder = np.argsort(t, kind="stable")

    # ---- pass 1: case i queries its later comparators (t_j>t_i) ----
    # BIT over comparators' p-rank: count (for h) and #; weight not needed (case uses w_i)
    cnt = Fenwick(m)  # number of seen comparators by p-rank
    seen = 0
    # we need #{j seen: p_j<p_i} and total seen; h uses strict< +0.5 eq(none)
    for idx in reversed(range(n)):
        s = torder[idx]
        if not keep[s]:
            continue
        if e[s] == CAUSE:
            less = cnt.pref(prank[s] - 1)  # seen comparators with p_j<p_i
            tot = seen
            Ndot[s] += 2.0 * w[s] * less
            Ddot[s] += 2.0 * w[s] * tot
        # s becomes a comparator for earlier cases
        cnt.add(prank[s], 1.0)
        seen += 1

    # ---- pass 2: comparator j accumulates from earlier?-no, later cases i (t_i<t_j) ----
    # j is comparator of case i iff t_i < t_j and p_j<p_i (or eq). Sweep latest->earliest
    # accumulating cases' weights keyed by p-rank; when we hit comparator j, sum over
    # cases already seen (t_i>t_j? no). Careful: comparator j has t_j; cases i with
    # t_i<t_j are EARLIER. So sweep earliest->latest, accumulate case weights, query at j.
    wsum = Fenwick(m)  # sum of 2 w_i over seen cases by p-rank
    for idx in range(n):
        s = torder[idx]
        if not keep[s]:
            continue
        # j=s as comparator: cases i seen so far have t_i<=t_s; need t_i<t_s (strict).
        # With continuous times t_i<t_s strict holds for earlier indices. h: p_s<p_i.
        # sum over cases i with p_i>p_s of 2 w_i  => wsum.pref(top) - wsum.pref(prank_s)
        tot_w = wsum.pref(m - 1)
        ge = wsum.pref(prank[s])  # cases with p_i<=p_s
        gt_w = tot_w - ge  # cases with p_i>p_s  => h=1
        Ndot[s] += gt_w
        Ddot[s] += tot_w
        if e[s] == CAUSE:
            wsum.add(prank[s], 2.0 * w[s])
    return Ndot, Ddot


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    t, e, p = gen(n, seed=3, signal=1.0)  # continuous times, no ties
    w = compute_uno_weights(t, e, gmin="none")
    NA, DA = naive_branchA(t, e, p, w)
    NF, DF = fenwick_branchA(t, e, p, w)
    print(f"[fenwick branchA] n={n}")
    print(f"  max|Ndot diff| = {np.abs(NA - NF).max():.2e}")
    print(f"  max|Ddot diff| = {np.abs(DA - DF).max():.2e}")


if __name__ == "__main__":
    main()
