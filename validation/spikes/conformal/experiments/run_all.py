"""Run the four synthetic Phase 0.6 experiments in sequence (e5/SEER excluded --
it needs Sunny's DUA export). Prints each experiment's table under a banner.

Run:  PYTHONUNBUFFERED=1 uv run python -m validation.spikes.conformal.experiments.run_all
"""

from __future__ import annotations

from validation.spikes.conformal.experiments import (
    e1_oracle_ceiling,
    e2_deltaw_rate,
    e3_atom_tradeoff,
    e4_mondrian_percause,
    e6_case_cohort,
)

SUITE = [
    ("E1 oracle ceiling", e1_oracle_ceiling.main),
    ("E2 Delta_w rate", e2_deltaw_rate.main),
    ("E3 atom trade-off", e3_atom_tradeoff.main),
    ("E4 Mondrian per-cause", e4_mondrian_percause.main),
    ("E6 general selection / case-cohort", e6_case_cohort.main),
]


def main():
    for name, fn in SUITE:
        print("\n" + "=" * 72)
        print(name)
        print("=" * 72)
        fn()
    print("\n" + "=" * 72)
    print("Synthetic suite complete. e5_seer awaits Sunny's SEER export (see README).")
    print("=" * 72)


if __name__ == "__main__":
    main()
