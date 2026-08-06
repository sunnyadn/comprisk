"""Experiment 3 (design.md 0.6 #3): the 1/g_min atom's conservatism is bounded.

Validates Contract C1 / Remark rem:atom. Both atoms use ORACLE calibration weights so
the ONLY moving part is the test atom:
  - atom=gmin : the realizable, floor-PROVEN 1/g_min atom (main theorem).
  - atom=mean : the tighter mean-calibration-weight atom -- empirically near-nominal
    but with NO finite-sample floor proof (an open problem, not shipped as a guarantee).
We sweep the censoring rate and report coverage + set size for each. The story: the
1/g_min atom stays valid everywhere (coverage >= nominal) at the cost of larger sets,
and that cost grows with censoring (smaller g_min => heavier atom) but stays bounded.

Run:  PYTHONUNBUFFERED=1 uv run python -m validation.spikes.conformal.experiments.e3_atom_tradeoff
"""

from __future__ import annotations

from validation.spikes.conformal.dgp import cr_dgp
from validation.spikes.conformal.experiments.oracle_g import aggregate

ALPHA = 0.1
REPS = 20
CENSOR_SWEEP = (0.2, 0.4, 0.6, 0.75)
KW = dict(competing_frac=0.4, signal=1.0, _alpha=ALPHA)


def main():
    nominal = 1 - ALPHA
    print(f"\nExp 3 -- atom trade-off (alpha={ALPHA}, nominal={nominal:.2f}, reps={REPS})")
    print("Oracle calibration weights; only the test atom varies.\n")
    print(f"  {'censor':<9}{'atom':<8}{'cov':>8}{'dev':>8}{'size':>8}{'size_infl':>11}")
    for censor in CENSOR_SWEEP:
        kw = dict(KW, censor_rate=censor)
        r_mean = aggregate(cr_dgp, kw, reps=REPS, weight_mode="oracle", atom_mode="mean")
        r_gmin = aggregate(cr_dgp, kw, reps=REPS, weight_mode="oracle", atom_mode="gmin")
        base = r_mean["size_mean"]
        for tag, r in (("mean", r_mean), ("gmin", r_gmin)):
            infl = r["size_mean"] - base
            print(
                f"  {censor:<9}{tag:<8}{r['cov_mean']:>8.3f}{r['cov_mean'] - nominal:>+8.3f}"
                f"{r['size_mean']:>8.2f}{infl:>+11.2f}"
            )
    print(
        "\nExpect: both atoms >= nominal; gmin's set-size inflation over mean grows"
        "\nwith censoring but stays a bounded, documented cost (C1)."
    )


if __name__ == "__main__":
    main()
