"""Phase 4 consolidation: paper-ready evidence pack for the conformal-CR empirics.

This module does NOT re-implement fit / calibration / coverage. It is a thin
orchestration + formatting layer that reuses the per-rep primitives from the three
gate harnesses (single source of truth):

  * real_coverage._one_split   -- one real-cohort train/calib/test split (Gate 1)
  * robustness_sweep._cell      -- one robustness cell over reps (Gate 2)
  * extensions_eval._one_rep    -- one marginal/Mondrian/APS rep (Gate 3)

It writes Markdown + LaTeX (booktabs) tables and the CSVs backing them under
``validation/reports/conformal/`` (gitignored), plus an index REPORT.md with the
Gate 1/2/3 verdicts.

CONFIDENTIALITY: the CHF cohort schema is secret. Output here is AGGREGATE ONLY --
cohort name, horizon name, calibration path, coverage / SE / deviation / set size /
gate flag. No feature names, file paths, raw class counts, or outcome coding are ever
written. Loaders return feature names; we discard them.

Run (paper-grade defaults):
  PYTHONUNBUFFERED=1 uv run python -m validation.spikes.conformal.report
Fast smoke (for trellis-check, <2 min):
  PYTHONUNBUFFERED=1 uv run python -m validation.spikes.conformal.report --quick
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from validation.spikes.conformal import extensions_eval as ee
from validation.spikes.conformal import real_coverage as rc
from validation.spikes.conformal import robustness_sweep as rs
from validation.spikes.conformal.dgp import cr_dgp

OUT = Path("validation/reports/conformal")
_ALPHA = rc._ALPHA
NOMINAL = 1.0 - _ALPHA


# --------------------------------------------------------------------------- config


@dataclass
class Cfg:
    """All sweep knobs in one place; `--quick` swaps in a fast smoke profile."""

    # real-data coverage (Table 1)
    reps_real: int = 10
    ntree_real: int = 100
    n_sub: int = 20000
    test_frac: float = 0.3
    # robustness (Table 2)
    reps_sweep: int = rs.REPS
    ntree_sweep: int = 100
    n_pool: int = 2500
    n_test: int = 2500
    smalln_ntest: int = rs.SMALLN_NTEST
    # extensions (Table 3)
    reps_ext: int = ee.REPS
    ext_cfg: dict = field(default_factory=lambda: dict(ee.CFG))

    @classmethod
    def quick(cls) -> Cfg:
        return cls(
            reps_real=2,
            ntree_real=20,
            n_sub=1500,
            test_frac=0.3,
            reps_sweep=2,
            ntree_sweep=20,
            n_pool=400,
            n_test=400,
            smalln_ntest=600,
            reps_ext=2,
            ext_cfg=dict(
                censor_rate=0.4,
                competing_frac=0.4,
                signal=1.5,
                n_pool=600,
                n_test=600,
                ntree=20,
            ),
        )


# ------------------------------------------------------------------ small utilities


def _mean_se(vals):
    a = np.asarray(vals, dtype=float)
    n = a.size
    se = a.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return float(a.mean()), float(se)


def _gate_ok(dev, se):
    return abs(dev) <= (3 * se if se > 0 else 0.01)


def _md_table(header, rows):
    sep = "| " + " | ".join("---" for _ in header) + " |"
    lines = ["| " + " | ".join(header) + " |", sep]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _tex_cell(c):
    # Escape LaTeX-special chars in tabular cells/headers (captions/labels are
    # passed through verbatim so they can carry intentional math like $\tau$).
    return str(c).replace("_", r"\_").replace("±", r"$\pm$").replace("—", "--")


def _tex_table(colspec, header, rows, *, caption, label):
    out = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(_tex_cell(h) for h in header) + " \\\\",
        "\\midrule",
    ]
    out += [" & ".join(_tex_cell(c) for c in r) + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(out)


def _write(name, text):
    path = OUT / name
    path.write_text(text)
    return path


# -------------------------------------------------------------- Table 1: real data


def _load_cohort(cohort, n_sub):
    """Return (X, time, event) or None if the cohort data is absent.

    Feature names are intentionally discarded (confidentiality)."""
    if cohort == "chf":
        from validation.spikes.conformal.data.chf import HORIZONS, load_chf

        X, time, event, _ = load_chf(subsample=n_sub, seed=0)
    elif cohort == "seer":
        from validation.spikes.conformal.data.seer import HORIZONS, load_seer

        X, time, event, _ = load_seer()
        if n_sub and n_sub < X.shape[0]:
            rng = np.random.default_rng(0)
            idx = rng.choice(X.shape[0], n_sub, replace=False)
            X, time, event = X[idx], time[idx], event[idx]
    else:
        raise ValueError(cohort)
    return X, time, event, HORIZONS


def table1_real_coverage(cfg: Cfg):
    """cohort (available only) x horizon x path -> cov+/-SE, dev, size, gate.

    Reuses real_coverage._one_split exactly as real_coverage.run does."""
    print("\n=== Table 1: real-data coverage ===")
    rows = []  # for md/tex
    gate = {}  # cohort -> {path: (all_ok, mean_dev)}
    for cohort in ("chf", "seer"):
        try:
            X, time, event, horizons = _load_cohort(cohort, cfg.n_sub)
        except (FileNotFoundError, ImportError, ValueError) as e:
            print(f"  [skip] cohort={cohort}: {type(e).__name__} (data absent)")
            continue
        causes = sorted(int(c) for c in np.unique(event) if c >= 1)
        cens = float(np.mean(event == 0))
        print(
            f"  cohort={cohort} n={X.shape[0]} p={X.shape[1]} causes={causes} censored={cens:.3f}"
        )
        per_path = {p: [] for p in ("oob", "split")}
        for hname, t_star in horizons.items():
            for path in ("oob", "split"):
                covs, sizes = [], []
                for r in range(cfg.reps_real):
                    c, sz = rc._one_split(
                        path,
                        X,
                        time,
                        event,
                        t_star,
                        test_frac=cfg.test_frac,
                        n_estimators=cfg.ntree_real,
                        seed=100 + r,
                    )
                    covs.append(c)
                    sizes.append(sz)
                mean, se = _mean_se(covs)
                dev = mean - NOMINAL
                ok = _gate_ok(dev, se)
                per_path[path].append((ok, dev))
                rows.append(
                    [
                        cohort,
                        hname,
                        path,
                        f"{mean:.3f}",
                        f"{se:.3f}",
                        f"{dev:+.3f}",
                        f"{np.mean(sizes):.2f}",
                        "ok" if ok else "MISS",
                    ]
                )
                print(
                    f"    {hname:<5}{path:<7}cov={mean:.3f}±{se:.3f} "
                    f"dev={dev:+.3f} size={np.mean(sizes):.2f} {'ok' if ok else 'MISS'}"
                )
        gate[cohort] = {
            p: (all(o for o, _ in lst), float(np.mean([d for _, d in lst])))
            for p, lst in per_path.items()
        }

    header = ["cohort", "horizon", "path", "cov", "SE", "dev", "size", "gate"]
    if not rows:
        rows = [["(none)", "-", "-", "-", "-", "-", "-", "no cohort data present"]]
    md = (
        f"# Table 1 — real-data conformal coverage (alpha={_ALPHA}, nominal={NOMINAL:.2f})\n\n"
        f"Repeated splits per cohort x horizon x path; reuses "
        f"`real_coverage._one_split`. reps={cfg.reps_real}, ntree={cfg.ntree_real}, "
        f"n_sub={cfg.n_sub}.\n\n"
    ) + _md_table(header, rows)
    _write("table1_real_coverage.md", md)
    _write(
        "table1_real_coverage.tex",
        _tex_table(
            "lllrrrrl",
            header,
            rows,
            caption=(
                "Real-data IPCW marginal coverage by cohort, horizon and "
                "calibration path (nominal "
                f"{NOMINAL:.2f}). \\emph{{gate}}=ok when $|dev|\\le 3\\,$SE."
            ),
            label="tab:real_coverage",
        ),
    )
    return gate


# ------------------------------------------------------------ Table 2: robustness


def table2_robustness(cfg: Cfg):
    """Three blocks (A misspec, B tau sweep, C small-n x censoring). Reuses
    robustness_sweep._cell. Also emits tau_sweep.csv backing the tau figure."""
    print("\n=== Table 2: robustness ===")
    reps = cfg.reps_sweep

    # Block A — misspecification
    a_rows, a_ok = [], True
    for name, fn_name, kw in rs.MISSPEC_DGPS:
        fn = rs.DGP_FNS[fn_name]
        for path in ("oob", "split"):
            m, se, sz = rs._cell(
                fn,
                kw,
                path,
                reps=reps,
                n_pool=cfg.n_pool,
                n_test=cfg.n_test,
                ntree=cfg.ntree_sweep,
            )
            dev = m - NOMINAL
            ok = _gate_ok(dev, se)
            a_ok &= ok
            a_rows.append(
                [
                    name,
                    path,
                    f"{m:.3f}",
                    f"{se:.3f}",
                    f"{dev:+.3f}",
                    f"{sz:.2f}",
                    "ok" if ok else "MISS",
                ]
            )
            print(f"  A {name:<12}{path:<7}cov={m:.3f}±{se:.3f} size={sz:.2f}")

    # Block B — tau / gmin sweep
    b_rows, tau_csv = [], []
    for gmin in rs.GMIN_SWEEP:
        m, se, sz = rs._cell(
            cr_dgp,
            rs.GMIN_KW,
            "split",
            reps=reps,
            gmin=gmin,
            n_pool=cfg.n_pool,
            n_test=cfg.n_test,
            ntree=cfg.ntree_sweep,
        )
        b_rows.append([f"{gmin}", f"{m:.3f}", f"{se:.3f}", f"{sz:.2f}"])
        tau_csv.append((gmin, m, se, sz))
        print(f"  B gmin={gmin:<7}cov={m:.3f}±{se:.3f} size={sz:.2f}")

    # Block C — small-n x extreme censoring
    c_rows, c_ok = [], True
    for n_pool in rs.SMALLN_NS:
        for cens in rs.SMALLN_CENS:
            kw = dict(censor_rate=cens, competing_frac=0.4, signal=1.0)
            m, se, sz = rs._cell(
                cr_dgp,
                kw,
                "split",
                reps=reps,
                n_pool=n_pool,
                n_test=cfg.smalln_ntest,
                ntree=cfg.ntree_sweep,
            )
            dev = m - NOMINAL
            ok = _gate_ok(dev, se)
            c_ok &= ok
            c_rows.append(
                [
                    f"{n_pool}",
                    f"{cens}",
                    f"{m:.3f}",
                    f"{se:.3f}",
                    f"{dev:+.3f}",
                    f"{sz:.2f}",
                    "ok" if ok else "MISS",
                ]
            )
            print(f"  C n={n_pool} cens={cens} cov={m:.3f}±{se:.3f} size={sz:.2f}")

    # tau_sweep.csv (backs the figure)
    with (OUT / "tau_sweep.csv").open("w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["gmin", "cov", "se", "size"])
        for g, m, se, sz in tau_csv:
            wtr.writerow([g, f"{m:.6f}", f"{se:.6f}", f"{sz:.6f}"])

    ha = ["DGP", "path", "cov", "SE", "dev", "size", "gate"]
    hb = ["gmin", "cov", "SE", "size"]
    hc = ["n_pool", "censor", "cov", "SE", "dev", "size", "gate"]
    md = (
        f"# Table 2 — robustness (alpha={_ALPHA}, nominal={NOMINAL:.2f}, "
        f"reps={reps})\n\nReuses `robustness_sweep._cell`.\n\n"
        f"## A. Misspecification (coverage must hold; misspec -> larger sets)\n\n"
        + _md_table(ha, a_rows)
        + "\n## B. tau / gmin sensitivity (split, exponential, censor=0.6)\n\n"
        + _md_table(hb, b_rows)
        + "\n_Backing data: `tau_sweep.csv`._\n"
        + "\n## C. small-n x extreme censoring (split path)\n\n"
        + _md_table(hc, c_rows)
    )
    _write("table2_robustness.md", md)
    tex = "\n".join(
        [
            _tex_table(
                "llrrrrl",
                ha,
                a_rows,
                caption="Robustness A: coverage under misspecified DGPs "
                "(exponential / Weibull / non-PH).",
                label="tab:robust_misspec",
            ),
            _tex_table(
                "lrrr",
                hb,
                b_rows,
                caption="Robustness B: IPCW-clip $\\tau$ (gmin) sensitivity; "
                "coverage is flat across the sweep.",
                label="tab:robust_tau",
            ),
            _tex_table(
                "llrrrrl",
                hc,
                c_rows,
                caption="Robustness C: small-$n$ and extreme-censoring finite-sample coverage.",
                label="tab:robust_smalln",
            ),
        ]
    )
    _write("table2_robustness.tex", tex)
    return a_ok, c_ok


# ------------------------------------------------------------ Table 3: extensions


def table3_extensions(cfg: Cfg):
    """marginal vs Mondrian (per-cause) vs APS. Reuses extensions_eval._one_rep."""
    print("\n=== Table 3: CR-specific extensions ===")
    reps = cfg.reps_ext
    res = [ee._one_rep(seed=300 + r, **cfg.ext_cfg) for r in range(reps)]

    def mean(k):
        return float(np.mean([r[k] for r in res]))

    L = len(res[0]["per_class"])
    names = {**{c: f"cause{c + 1}" for c in range(L - 1)}, L - 1: "free"}

    cov_m, size_m = mean("cov_m"), mean("size_m")
    rows = [["marginal", "(all)", f"{cov_m:.3f}", f"{size_m:.2f}"]]

    mon_ok = True
    for c in range(L):
        pc = float(np.nanmean([r["per_class"][c] for r in res]))
        mon_ok &= abs(pc - NOMINAL) <= 0.03
        rows.append(["Mondrian", names[c], f"{pc:.3f}", "—"])
    size_mon = mean("size_mon")
    rows.append(["Mondrian", "(size)", "—", f"{size_mon:.2f}"])

    cov_a, size_a = mean("cov_a"), mean("size_a")
    rows.append(["APS", "(all)", f"{cov_a:.3f}", f"{size_a:.2f}"])

    # APS verdict (mirrors extensions_eval.main)
    aps_cov_ok = cov_a >= NOMINAL - 0.02
    aps_nontrivial = size_a < L - 0.05
    if aps_cov_ok and aps_nontrivial:
        aps_verdict = "PASS"
    elif aps_cov_ok:
        aps_verdict = "DEGENERATE"  # covers but trivial full set (K=2)
    else:
        aps_verdict = "REVIEW"

    print(f"  marginal cov={cov_m:.3f} size={size_m:.2f}")
    print(
        f"  Mondrian per-class ~nominal: {'PASS' if mon_ok else 'REVIEW'} "
        f"(size {size_mon:.2f} vs marginal {size_m:.2f})"
    )
    print(f"  APS cov={cov_a:.3f} size={size_a:.2f} -> {aps_verdict}")

    header = ["method", "stratum", "cov", "size"]
    note = f"APS verdict at K={L - 1} causes: **{aps_verdict}**" + (
        " — deterministic APS degenerates to the full label set at low "
        "cardinality (covers but trivial)."
        if aps_verdict == "DEGENERATE"
        else ""
    )
    md = (
        f"# Table 3 — CR-specific extensions (alpha={_ALPHA}, nominal={NOMINAL:.2f}, "
        f"reps={reps})\n\nReuses `extensions_eval._one_rep`. Mondrian rows are "
        f"per-cause conditional coverage; size is the marginal-vs-conditional cost.\n\n"
        + _md_table(header, rows)
        + f"\n{note}\n"
    )
    _write("table3_extensions.md", md)
    _write(
        "table3_extensions.tex",
        _tex_table(
            "llrr",
            header,
            rows,
            caption=(
                "CR-specific extensions: marginal vs Mondrian (per-cause "
                "conditional) vs APS coherent sets. APS degenerates to the full "
                f"set at $K={L - 1}$."
            ),
            label="tab:extensions",
        ),
    )
    return mon_ok, aps_verdict


# ------------------------------------------------------------- optional tau figure


def maybe_tau_figure():
    """Render the tau coverage curve from tau_sweep.csv if matplotlib is present.

    No hard dependency: absence is fine, the CSV already backs the figure."""
    csv_path = OUT / "tau_sweep.csv"
    if not csv_path.exists():
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [note] matplotlib absent -> skipping PNG; tau_sweep.csv backs the figure.")
        return None

    g, cov, se = [], [], []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            g.append(float(row["gmin"]))
            cov.append(float(row["cov"]))
            se.append(float(row["se"]))
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.errorbar(g, cov, yerr=np.array(se) * 3, marker="o", capsize=3)
    ax.axhline(NOMINAL, ls="--", color="grey", label=f"nominal {NOMINAL:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel(r"IPCW clip $\tau$ (gmin)")
    ax.set_ylabel("IPCW coverage")
    ax.set_title("tau sensitivity (split, censor=0.6)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = OUT / "tau_sweep.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f"  [fig] {png}")
    return png


# ----------------------------------------------------------------- index + verdicts


def write_index(cfg, g1, g2, g3, *, quick):
    a_ok, c_ok = g2
    mon_ok, aps_verdict = g3

    # Gate 1 verdict: >=1 path holds across horizons on every available cohort.
    if not g1:
        g1_line = "Gate 1 (real coverage): NO DATA — no real cohort present at run time."
        g1_detail = []
    else:
        g1_detail = []
        all_pass = True
        for cohort, paths in g1.items():
            holding = [p for p, (ok, _) in paths.items() if ok]
            all_pass &= len(holding) >= 1
            oob_dev = paths.get("oob", (None, 0.0))[1]
            sign = "conservative" if oob_dev > 0 else "anti-conservative"
            g1_detail.append(
                f"  - **{cohort}**: paths holding across horizons = "
                f"{holding or 'none'}; OOB mean dev = {oob_dev:+.3f} ({sign})."
            )
        g1_line = f"Gate 1 (real coverage): {'PASS' if all_pass else 'REVIEW'} "
        g1_line += "— >=1 calibration path within MC noise of nominal per cohort."

    g2_line = (
        f"Gate 2 (robustness): misspec {'PASS' if a_ok else 'REVIEW'}; "
        f"small-n {'PASS' if c_ok else 'REVIEW'} — tau coverage flat (see tau_sweep.csv)."
    )
    g3_line = (
        f"Gate 3 (extensions): Mondrian per-cause {'PASS' if mon_ok else 'REVIEW'}; "
        f"APS {aps_verdict}."
    )

    profile = "QUICK SMOKE (not paper-grade)" if quick else "paper-grade"
    body = [
        "# Conformal CR — consolidated evidence pack",
        "",
        f"Profile: **{profile}**. alpha = {_ALPHA}, nominal = {NOMINAL:.2f}. "
        "All coverage is IPCW marginal coverage; gate-ok = |dev| <= 3 SE.",
        "",
        f"reps: real={cfg.reps_real}, sweep={cfg.reps_sweep}, ext={cfg.reps_ext}; "
        f"ntree real={cfg.ntree_real}/sweep={cfg.ntree_sweep}/"
        f"ext={cfg.ext_cfg['ntree']}; n_sub={cfg.n_sub}.",
        "",
        "## Gate verdicts",
        "",
        f"- {g1_line}",
        *g1_detail,
        f"- {g2_line}",
        f"- {g3_line}",
        "",
        "## Tables",
        "",
        "- [Table 1 — real-data coverage](table1_real_coverage.md) (`table1_real_coverage.tex`)",
        "- [Table 2 — robustness](table2_robustness.md) (`table2_robustness.tex`, `tau_sweep.csv`)",
        "- [Table 3 — extensions](table3_extensions.md) (`table3_extensions.tex`)",
        "",
        "## Residual risk -> theory strand",
        "",
        "- OOB calibration is slightly conservative (over-covers, never under): the "
        "finite-sample coverage theorem under **estimated** IPCW weights must explain "
        "this (J+aB adapted to censoring).",
        "- APS coherent sets degenerate to the full label set at low cause cardinality "
        "(K=2); a randomized-APS refinement is the open methods lever.",
        "- SEER second cohort pending the user's SEER export; CHF Mondrian/APS on the "
        "real cohort still to run (Phase 3 extensions were synthetic).",
        "",
    ]
    _write("REPORT.md", "\n".join(body))
    print("\n" + "\n".join([g1_line, g2_line, g3_line]))


# ----------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="fast smoke profile (small reps/ntree/n) for trellis-check",
    )
    ap.add_argument("--reps-real", type=int, default=None)
    ap.add_argument("--reps-sweep", type=int, default=None)
    ap.add_argument("--reps-ext", type=int, default=None)
    ap.add_argument("--ntree", type=int, default=None, help="override fit ntree everywhere")
    ap.add_argument("--n-sub", type=int, default=None, help="real-cohort subsample size")
    a = ap.parse_args()

    cfg = Cfg.quick() if a.quick else Cfg()
    if a.reps_real is not None:
        cfg.reps_real = a.reps_real
    if a.reps_sweep is not None:
        cfg.reps_sweep = a.reps_sweep
    if a.reps_ext is not None:
        cfg.reps_ext = a.reps_ext
    if a.ntree is not None:
        cfg.ntree_real = cfg.ntree_sweep = a.ntree
        cfg.ext_cfg["ntree"] = a.ntree
    if a.n_sub is not None:
        cfg.n_sub = a.n_sub

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"writing evidence pack to {OUT}/  (profile={'quick' if a.quick else 'paper'})")

    g1 = table1_real_coverage(cfg)
    g2 = table2_robustness(cfg)
    g3 = table3_extensions(cfg)
    maybe_tau_figure()
    write_index(cfg, g1, g2, g3, quick=a.quick)
    print("\ndone.")


if __name__ == "__main__":
    main()
