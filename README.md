# comprisk

[![PyPI version](https://img.shields.io/pypi/v/comprisk.svg)](https://pypi.org/project/comprisk/)
[![CI](https://github.com/sunnyadn/comprisk/actions/workflows/ci.yml/badge.svg)](https://github.com/sunnyadn/comprisk/actions/workflows/ci.yml)
[![docs](https://img.shields.io/badge/docs-sunnyadn.github.io%2Fcomprisk-blue)](https://sunnyadn.github.io/comprisk/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19876282-blue)](https://doi.org/10.5281/zenodo.19876282)

A Python toolkit for **competing-risks** survival analysis: a scalable,
scikit-learn-compatible competing-risks random survival forest plus the canonical
regression / non-parametric methods — Fine-Gray, Aalen-Johansen CIF, cause-specific
Cox — so applied researchers can drop the Python → R round-trip.

> **Status: alpha** — API may change before v1.0. Renamed from `crforest` in 0.3.1
> (`pip install comprisk`; `from comprisk import CompetingRiskForest`).

## Highlights

- **Four canonical CR methods, native Python** — Fine-Gray (+ penalized),
  cause-specific Cox, Aalen-Johansen CIF, Gray's test — each validated to
  floating-point tolerance against `cmprsk` / `crrp` / `survival`.
- **The only native-Python CR forest** — composite & cause-specific CR log-rank
  splitting, AJ CIF, Nelson-Aalen CHF, Wolbers + Uno IPCW concordance, OOB
  Breiman VIMP, Ishwaran minimal depth, exact TreeSHAP.
- **CR-aware evaluation** — `score_cr` (IPCW time-dependent AUC/Brier + bootstrap
  CIs) and `calibration_cr`, replacing the CR-mode `riskRegression::Score()` block.
- **Fast** — 10–22× vs randomForestSRC on real EHR, 16.6–544× vs scikit-survival
  (n = 5k → 50k), n = 10⁶ in 63 s — at matched C ≈ 0.85. [Benchmarks →](docs/benchmarks.md)
- **Reproducible** — `equivalence="rfsrc"` reproduces rfSRC's per-tree mtry/nsplit
  RNG stream bit-for-bit. [Methodology →](docs/equivalence-vs-rfsrc.md)

## Install

```bash
pip install comprisk          # or:  uv add comprisk
pip install "comprisk[gpu]"   # CUDA 12 preview (faster only at low p today)
```

Python ≥ 3.10. Core deps: numpy, scipy, pandas, joblib, numba, scikit-learn.

## Quickstart

```python
from comprisk import CompetingRiskForest

# event: 0 = censored, k≥1 = cause-k event. Defaults: 100 trees, logrankCR, n_jobs=-1.
forest = CompetingRiskForest(n_estimators=200, random_state=42).fit(X, time, event)

cif  = forest.predict_cif(X[:5])          # (5, n_causes, n_times) — Aalen-Johansen
print(forest.oob_score(cause=1))          # honest out-of-bag C-index (no holdout split)
shap, base = forest.shap_values(X[:10])   # exact TreeSHAP (n, p, n_times, n_causes)
```

Prediction shapes, scoring, cross-validation, VIMP, minimal depth, GPU, and rfSRC
migration — all with runnable code — are in the
**[quickstart](docs/quickstart.md)**. `CompetingRiskForest` is a real sklearn
estimator (`cross_val_score` / `Pipeline` work without a wrapper).

### Regression & non-parametric models

```python
from comprisk import FineGrayRegression

fg = FineGrayRegression(cause=1, robust_se=True).fit(X, time=time, event=event)
print(fg.coef_, fg.se_)                    # log subdistribution-HRs
```

| Estimator | Estimates | R parity |
|---|---|---|
| `FineGrayRegression` | subdistribution-hazard ratios | `cmprsk::crr()` (β̂ to fp noise) |
| `PenalizedFineGrayRegression` | LASSO / ridge / EN / MCP / SCAD path | `crrp::crrp()` to ~1e-6 |
| `CauseSpecificCox` | cause-specific hazard ratios | `survival::coxph()` to 1e-9 |
| `CumulativeIncidence` | non-parametric Aalen-Johansen CIF | `cmprsk::cuminc()` |
| `gray_test` | K-sample test for equal CIFs | `cmprsk::cuminc()$Tests` to 1e-14 |

Worked code for every row is in
[`examples/02_regression_models.ipynb`](examples/02_regression_models.ipynb).

## comprisk vs alternatives

|                                    | comprisk                  | randomForestSRC | scikit-survival       |
|------------------------------------|:-------------------------:|:---------------:|:---------------------:|
| Language                           | Python                    | R               | Python                |
| Native competing risks             | ✓                         | ✓               | ✗ (single-event)      |
| Aalen–Johansen CIF output          | ✓                         | ✓               | n/a                   |
| Cumulative hazard at scale         | ✓                         | ✓               | ✗ (low-memory only)   |
| OOB permutation VIMP               | ✓                         | ✓               | ✗                     |
| Bit-identical reproducibility mode | ✓ (`equivalence="rfsrc"`) | —               | n/a                   |
| Scales to n = 10⁶                  | ✓ (63 s on i7)            | memory-bound    | ✗ / OOM               |
| GPU preview                        | ✓ (CUDA 12)               | ✗               | ✗                     |

scikit-survival's CHF/survival outputs and scaling caveats are detailed in the
[benchmarks](docs/benchmarks.md#vs-scikit-survival-paired-same-machine).

## Benchmarks

Matched-pair, real EHR data (full tables + methodology in [docs/benchmarks.md](docs/benchmarks.md)):

| Cohort | n × p | comprisk | rfSRC (OMP-on) | Speedup |
|---|---|---|---|---|
| CHF (cardio) | 75k × 58 | 5.6–9.4 s | 84.8–207.3 s | **14–22×** |
| SEER breast | 238k × 17 | 7.0 s | 81.6 s | **11.6×** |

Both fit similarly well (C ≈ 0.85); the band tracks feature count. Also 16.6–544×
vs scikit-survival (n = 5k → 50k) and n = 10⁶ in 63 s on a consumer i7.

## Roadmap

comprisk is intentionally CR-focused — for non-CR survival (general Cox, AFT,
deep-survival), use [lifelines](https://lifelines.readthedocs.io/) or
[scikit-survival](https://scikit-survival.readthedocs.io/).

- **Shipped (v0.3–0.6):** CR forest, Fine-Gray (+ penalized), cause-specific Cox,
  Aalen-Johansen CIF, Gray's test, `score_cr` / `calibration_cr`.
- **v1.0 (planned):** API freeze + JMLR MLOSS submission.
- **v1.1 (planned):** full GPU rewrite.

## Documentation

📖 **[Full documentation site](https://sunnyadn.github.io/comprisk/)** — searchable, autogenerated API reference.

- [Quickstart](docs/quickstart.md) — common tasks with runnable code
- [API reference](https://sunnyadn.github.io/comprisk/reference/) — full parameter lists
- [Benchmarks](docs/benchmarks.md) — full tables, methodology, reproduction scripts
- [Equivalence vs rfSRC](docs/equivalence-vs-rfsrc.md) — cross-library validation
- [References](docs/REFERENCES.md) — algorithmic provenance

## Examples

Runnable notebooks in [`examples/`](examples) (rendered on GitHub; open in Colab to run):

- [`01_forest_quickstart.ipynb`](examples/01_forest_quickstart.ipynb) — fit → CIF → OOB scoring → VIMP → minimal-depth selection
- [`02_regression_models.ipynb`](examples/02_regression_models.ipynb) — Fine-Gray, cause-specific Cox, AJ by group, Gray's test, penalized FG
- [`shap_explain.py`](examples/shap_explain.py) — interactive [marimo](https://marimo.io) TreeSHAP app

## Development

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
uv venv && uv pip install -e ".[dev]"
uv run pre-commit install
uv run pytest && uv run ruff check .
```

## License & citation

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). Cite via the DOI below (concept-level,
resolves to latest) or GitHub's "Cite this repository" button ([`CITATION.cff`](CITATION.cff)):

```bibtex
@software{yang_comprisk_2026,
  author    = {Yang, Sunny and Zhao, Wanqi},
  title     = {{comprisk: a Python toolkit for competing risks}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19876282},
  url       = {https://doi.org/10.5281/zenodo.19876282},
}
```
