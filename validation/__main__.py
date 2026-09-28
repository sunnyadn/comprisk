"""CLI for the validation benches.

Subcommands:
  bench-vimp         — time VIMP on a large synthetic workload
"""

from __future__ import annotations

import argparse


def _cmd_bench_vimp(args: argparse.Namespace) -> None:
    from validation.bench_vimp import run

    result = run(
        dataset=args.dataset,
        n=args.n,
        n_repeats=args.n_repeats,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    for k, v in result.items():
        print(f"{k}: {v}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="validation", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_bench = sub.add_parser("bench-vimp", help="Time VIMP on large synthetic workload")
    p_bench.add_argument("--dataset", default="synthetic")
    p_bench.add_argument("--n", type=int, default=100_000)
    p_bench.add_argument("--n-repeats", type=int, default=5)
    p_bench.add_argument("--seed", type=int, default=0)
    p_bench.add_argument("--n-jobs", type=int, default=-1, help="joblib workers (default: -1)")
    p_bench.set_defaults(func=_cmd_bench_vimp)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
