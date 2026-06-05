# comprisk

[![PyPI version](https://img.shields.io/pypi/v/comprisk.svg)](https://pypi.org/project/comprisk/)
[![CI](https://github.com/sunnyadn/comprisk/actions/workflows/ci.yml/badge.svg)](https://github.com/sunnyadn/comprisk/actions/workflows/ci.yml)
[![docs](https://img.shields.io/badge/docs-sunnyadn.github.io%2Fcomprisk-blue)](https://sunnyadn.github.io/comprisk/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19876282-blue)](https://doi.org/10.5281/zenodo.19876282)

**comprisk** — a Python toolkit for competing risks. Ships a scalable,
scikit-learn-compatible competing-risks random survival forest plus the
three canonical regression / non-parametric methods clinical researchers
actually need: Fine-Gray subdistribution-hazard regression, a stand-alone
Aalen-Johansen cumulative-incidence estimator with cmprsk-parity
variance, and cause-specific Cox PH (see [Roadmap](#roadmap)). Designed
to remove the Python → R workflow split that applied researchers
currently endure for competing-risks survival analysis.

> **Status: alpha.** API and internals may change before v1.0.
> **Renamed from `crforest` in 0.3.1** — `pip install comprisk`,
> `from comprisk import CompetingRiskForest`.

## Highlights

- **The four canonical CR methods, native Python** — Fine-Gray (+ penalized),
  cause-specific Cox, Aalen-Johansen CIF, and Gray's test, each validated to
  floating-point tolerances against `cmprsk` / `crrp` / `survival` (parity
  table [below](#regression-and-non-parametric-models)).
- **The only native-Python competing-risks RSF** — cause-specific & composite
  CR log-rank splitting, Aalen-Johansen CIF, Nelson-Aalen CHF, Wolbers + Uno
  IPCW concordance, OOB Breiman VIMP, Ishwaran minimal depth, exact TreeSHAP.
- **CR-aware model evaluation** — `score_cr` (IPCW time-dependent AUC/Brier,
  integrated iAUC/IBS with bootstrap CIs) and `calibration_cr` replace the
  CR-mode `riskRegression::Score()` / `plotCalibration()` blocks in one call.
- **10–22× faster than [randomForestSRC](https://cran.r-project.org/package=randomForestSRC)**
  on real EHR data and **16.6–544× faster than [scikit-survival](https://scikit-survival.readthedocs.io/)**
  (n = 5k → 50k), at matched C ≈ 0.85 and without disabling CIF/CHF
  outputs ([benchmarks](docs/benchmarks.md)).
- **Bit-identical to randomForestSRC** with `equivalence="rfsrc"` — reproduces
  the per-tree mtry/nsplit RNG stream for paper-grade reproducibility and
  rfSRC-baseline migrations.

## comprisk vs alternatives

|                                          | comprisk                       | randomForestSRC                    | scikit-survival          |
|------------------------------------------|:------------------------------:|:----------------------------------:|:------------------------:|
| Language                                 | Python                         | R                                  | Python                   |
| Native competing risks                   | ✓                              | ✓                                  | ✗ (single-event only)    |
| Aalen–Johansen CIF output                | ✓                              | ✓                                  | n/a                      |
| Cumulative hazard at scale               | ✓                              | ✓                                  | ✗¹                       |
| OOB permutation VIMP                     | ✓                              | ✓                                  | ✗                        |
| Bit-identical reproducibility mode       | ✓ (`equivalence="rfsrc"`)      | —                                  | n/a                      |
| Scales to n = 10⁶                        | ✓ (63 s on i7)                 | memory-bound past n ≈ 500 000 on consumer hardware | ✗¹ / OOM²                |
| Default parallelism                      | ✓ (`n_jobs=-1`)                | OpenMP (build-dependent; macOS Apple clang lacks it) | ✓        |
| GPU preview                              | ✓ (CUDA 12)                    | ✗                                  | ✗                        |

¹ sksurv `RandomSurvivalForest(low_memory=True)` is the only mode that
scales beyond ~10k samples, but it disables `predict_cumulative_hazard_function`
and `predict_survival_function` (raises `NotImplementedError`).
² sksurv `low_memory=False` exposes CHF / survival outputs but stores per-leaf
full CHF arrays; peak RSS reaches 16.8 GB at n = 5k on synthetic, OOMs
(> 21.5 GB) at n = 10k on a 24 GB host.

## Install

```bash
pip install comprisk          # or:  uv add comprisk
pip install "comprisk[gpu]"   # or:  uv add 'comprisk[gpu]'
```

Requires Python ≥ 3.10. Core dependencies: numpy, scipy, pandas, joblib,
numba, scikit-learn. GPU extra adds cupy + CUDA 12 runtime libs (preview;
faster only at low feature count today, full rewrite scheduled for v1.1).

## Quickstart

```python
import numpy as np
from comprisk import CompetingRiskForest

# Toy competing-risks data *with signal*: cause-1 risk rises with the first
# two features, cause 2 competes, and some subjects are censored.
rng = np.random.default_rng(42)
n = 1000
X = rng.normal(size=(n, 6))
lp = X[:, 0] + 0.5 * X[:, 1]
t1 = rng.exponential(np.exp(-lp))          # cause 1 fires sooner when lp is high
t2 = rng.exponential(2.0, size=n)          # cause 2 (competing)
tc = rng.exponential(4.0, size=n)          # censoring
time = np.minimum.reduce([t1, t2, tc])
event = np.where((t1 <= t2) & (t1 <= tc), 1, np.where(t2 <= tc, 2, 0))  # 0 = censored

# Fit. Defaults: n_estimators=100, max_features="sqrt", logrankCR, n_jobs=-1.
forest = CompetingRiskForest(n_estimators=200, random_state=42).fit(X, time, event)

# Aalen-Johansen cumulative incidence over the forest's chosen time grid.
cif = forest.predict_cif(X[:5])                       # (5, n_causes, n_times)

# Out-of-bag cause-specific Wolbers concordance — honest (out-of-sample),
# no held-out split needed. (forest.score(X, ...) would report the optimistic
# in-sample value.)
print("OOB C-index, cause 1:", forest.oob_score(cause=1))
```

### Explainability and feature selection

```python
# OOB permutation importance (Uno IPCW-scored).
vimp = forest.compute_importance(random_state=42)

# Ishwaran minimal-depth variable selection.
selected = forest.minimal_depth().query("selected")["feature"].tolist()

# Exact TreeSHAP attributions (Lundberg 2018, Algorithm 2).
shap, base = forest.shap_values(X[:10])               # (n, p, n_times, n_causes)
```

SHAP additivity, per-cause global importance, and per-subject attribution over
the time grid are explored interactively (with sliders) in
[`examples/shap_explain.py`](examples/shap_explain.py) (marimo).

### Regression and non-parametric models

Beyond the forest, comprisk ships the classical competing-risks toolkit — each
validated to floating-point tolerances against its reference R package:

```python
from comprisk import FineGrayRegression

fg = FineGrayRegression(cause=1, robust_se=True).fit(X, time=time, event=event)
print(fg.coef_, fg.se_)                               # log subdistribution-HRs
```

| Estimator | Estimates | R parity |
|---|---|---|
| `FineGrayRegression` | subdistribution-hazard ratios | `cmprsk::crr()` (β̂ to fp noise) |
| `PenalizedFineGrayRegression` | LASSO / ridge / EN / MCP / SCAD path | `crrp::crrp()` to ~1e-6 |
| `CauseSpecificCox` | cause-specific hazard ratios | `survival::coxph()` to 1e-9 |
| `CumulativeIncidence` | non-parametric Aalen-Johansen CIF | `cmprsk::cuminc()` |
| `gray_test` | K-sample test for equal CIFs | `cmprsk::cuminc()$Tests` to 1e-14 |

Worked code for every row — coefficient tables, CIF-by-group plots, the LASSO
path — is in [`examples/02_regression_models.ipynb`](examples/02_regression_models.ipynb);
data format, prediction shapes, cross-validation, GPU, and rfSRC migration are
in [docs/quickstart.md](docs/quickstart.md).

> **scikit-learn drop-in.** `CompetingRiskForest` is a real sklearn
> estimator (`BaseEstimator`, `clone()`-friendly, picklable).
> `cross_val_score`, `KFold`, `Pipeline` work without a wrapper — pass
> `Surv.from_arrays(event, time)` as the `y` argument, or use the legacy
> 3-arg `fit(X, time, event)` form. Full example in
> [docs/quickstart.md § Cross-validation](docs/quickstart.md#cross-validation).

## Roadmap

comprisk is intentionally CR-focused. For non-CR survival methods
(general Cox PH, AFT, parametric, deep-survival, Kaplan-Meier as a
standalone API), use [lifelines](https://lifelines.readthedocs.io/) or
[scikit-survival](https://scikit-survival.readthedocs.io/).

| Version  | Module                                                | Status               |
|----------|-------------------------------------------------------|----------------------|
| v0.3     | `CompetingRiskForest` (CR-RSF)                        | Shipped              |
| **v0.4** | `FineGrayRegression` (subdistribution hazard)         | Shipped              |
| **v0.4** | `CumulativeIncidence` (stand-alone Aalen-Johansen)    | Shipped              |
| **v0.4** | `gray_test` (Gray's K-sample log-rank)                | Shipped              |
| **v0.4** | `CauseSpecificCox` (CR-aware censoring)               | Shipped              |
| **v0.4** | `score_cr` / `calibration_cr` (CR-aware evaluation)   | Shipped              |
| **v0.5** | `PenalizedFineGrayRegression` (LASSO/ridge/elastic-net/MCP/SCAD) | Shipped    |
| v1.0     | API freeze + JMLR MLOSS submission                    | Planned              |
| v1.1     | Full GPU rewrite                                      | Planned              |

## Benchmarks

Headline numbers — full tables, methodology, and reproducibility scripts
in [docs/benchmarks.md](docs/benchmarks.md).

**vs randomForestSRC, matched-pair on real EHR data:**

| Cohort | n × p | Hardware | comprisk | rfSRC OMP-on | Speedup |
|---|---|---|---|---|---|
| CHF (cardio) | 75k × 58 | Apple M4 / i7-14700K / HPC | 5.6–9.4 s | 84.8–207.3 s | **14–22×** |
| SEER breast (oncology) | 238k × 17 | HPC Xeon Gold 6148 | 7.0 s | 81.6 s | **11.6×** |

Both libraries fit similarly well (C ≈ 0.85); the cross-dataset band tracks
feature count (rfSRC's per-split scan scales with p). ~95× vs rfSRC built
without OpenMP (the default R-on-macOS install).

**vs scikit-survival, paired on i7-14700K** — synthetic 2-cause Weibull,
p = 58, both libraries at their best config:

| n | sksurv `low_memory=True` | comprisk | speedup |
|---|---|---|---|
| 5 000 | 18.2 s | 1.10 s | **16.6×** |
| 50 000 | 2935 s (49 min) | 5.40 s | **544×** |

The gap widens super-linearly (sksurv ≈ n^2.2; comprisk ≈ n^0.7), and comprisk
still returns the Aalen-Johansen CIF + Nelson-Aalen CHF that sksurv
`low_memory=True` cannot.

**Scaling on a consumer desktop:** n = 10⁶ in **63 s** on i7-14700K,
14.5 GB RSS. Reproducible via
[`validation/spikes/lambda/exp5_paper_scale_bench.py`](validation/spikes/lambda/exp5_paper_scale_bench.py).

## API

Full parameter lists in the
[API reference](https://sunnyadn.github.io/comprisk/reference/); usage by task
in [docs/quickstart.md](docs/quickstart.md). Two forest splitrules are
available: `logrankCR` (composite competing-risks log-rank, default) and
`logrank` (cause-specific).

## Examples

Runnable notebooks in [`examples/`](examples) (rendered with output on GitHub —
click to view, or open in Colab to run):

- [`01_forest_quickstart.ipynb`](examples/01_forest_quickstart.ipynb)
  [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sunnyadn/comprisk/blob/main/examples/01_forest_quickstart.ipynb)
  — fit → predict CIF → out-of-bag scoring → VIMP → minimal-depth selection
- [`02_regression_models.ipynb`](examples/02_regression_models.ipynb)
  [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sunnyadn/comprisk/blob/main/examples/02_regression_models.ipynb)
  — Fine-Gray, cause-specific Cox, Aalen-Johansen by group, Gray's test, penalized Fine-Gray
- [`shap_explain.py`](examples/shap_explain.py) — interactive
  [marimo](https://marimo.io) app for TreeSHAP attributions (sliders for forest
  size and subject); `uv run --extra examples marimo edit examples/shap_explain.py`

## Documentation

📖 **[Full documentation site](https://sunnyadn.github.io/comprisk/)** — searchable, with autogenerated API reference.

- [Quickstart](docs/quickstart.md) — common tasks with runnable code
- [API reference](https://sunnyadn.github.io/comprisk/reference/) — full parameter lists from the docstrings
- [Equivalence vs rfSRC](docs/equivalence-vs-rfsrc.md) — cross-library validation methodology
- [References](docs/REFERENCES.md) — algorithmic provenance (Park-Miller, Bays-Durham, Wolbers 2009, Uno 2011, Cole & Hernán 2008, Breiman 2001, Ishwaran 2008/2014, etc.)

## Development

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
uv venv
uv pip install -e ".[dev]"
uv run pre-commit install
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Citation

```bibtex
@software{yang_comprisk_2026,
  author    = {Yang, Sunny and Zhao, Wanqi},
  title     = {{comprisk: a Python toolkit for competing risks}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {0.3.1},
  doi       = {10.5281/zenodo.19876282},
  url       = {https://doi.org/10.5281/zenodo.19876282},
}
```

DOI is concept-level (always resolves to the latest version). GitHub's
"Cite this repository" button generates a version-specific record from
[`CITATION.cff`](CITATION.cff). Algorithmic references in
[`docs/REFERENCES.md`](docs/REFERENCES.md).
