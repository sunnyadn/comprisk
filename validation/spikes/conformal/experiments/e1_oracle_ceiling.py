"""Experiment 1 (design.md 0.6 #1): the oracle-weight CEILING hits 1-alpha sharply.

Validates Lemma lem:oracle / lem:we: with the TRUE censoring weights and the oracle
test atom, finite-sample coverage should sit right at 1-alpha (not merely above it),
confirming the guarantee is real and not vacuous. The realizable 1/g_min atom is then
shown to over-cover (conservative but valid) -- the price C1 quantifies in e3.

Run:  PYTHONUNBUFFERED=1 uv run python -m validation.spikes.conformal.experiments.e1_oracle_ceiling
"""

from __future__ import annotations

from validation.spikes.conformal.dgp import cr_dgp
from validation.spikes.conformal.experiments.oracle_g import aggregate

ALPHA = 0.1
REPS = 20
KW = dict(competing_frac=0.4, signal=1.0, _alpha=ALPHA)


def main():
    nominal = 1 - ALPHA
    print(f"\nExp 1 -- oracle-weight ceiling (alpha={ALPHA}, nominal={nominal:.2f}, reps={REPS})")
    print("Coverage is ORACLE-weighted (independent of Ghat).\n")
    print(f"  {'censor':<9}{'atom':<9}{'cov':>8}{'se':>7}{'dev':>8}{'size':>8}")
    for censor in (0.2, 0.4, 0.6):
        kw = dict(KW, censor_rate=censor)
        for atom in ("oracle", "gmin"):
            r = aggregate(cr_dgp, kw, reps=REPS, weight_mode="oracle", atom_mode=atom)
            dev = r["cov_mean"] - nominal
            print(
                f"  {censor:<9}{atom:<9}{r['cov_mean']:>8.3f}{r['cov_se']:>7.3f}"
                f"{dev:>+8.3f}{r['size_mean']:>8.2f}"
            )
    print(
        "\nExpect: atom=oracle sits ~at nominal (sharp ceiling); atom=gmin sits above"
        "\n(valid but conservative), gap widening as censoring grows (heavier 1/g_min)."
    )


if __name__ == "__main__":
    main()
