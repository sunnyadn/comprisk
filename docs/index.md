# comprisk

[![PyPI version](https://img.shields.io/pypi/v/comprisk.svg)](https://pypi.org/project/comprisk/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19876282-blue)](https://doi.org/10.5281/zenodo.19876282)

A Python toolkit for **competing risks**. It ships a scalable,
scikit-learn-compatible competing-risks random survival forest plus the
classical regression / non-parametric methods clinical researchers actually
need — so competing-risks analysis no longer forces a Python → R workflow split.

!!! warning "Status: alpha"
    API and internals may change before v1.0. Renamed from `crforest` in 0.3.1.

## Install

```bash
pip install comprisk
```

## 30-second example

```python
import numpy as np
from comprisk import CompetingRiskForest

rng = np.random.default_rng(42)
n = 1000
X = rng.normal(size=(n, 6))
lp = X[:, 0] + 0.5 * X[:, 1]
t1 = rng.exponential(np.exp(-lp))      # cause 1 (event of interest)
t2 = rng.exponential(2.0, size=n)      # cause 2 (competing)
tc = rng.exponential(4.0, size=n)      # censoring
time = np.minimum.reduce([t1, t2, tc])
event = np.where((t1 <= t2) & (t1 <= tc), 1, np.where(t2 <= tc, 2, 0))  # 0 = censored

forest = CompetingRiskForest(n_estimators=300, random_state=42).fit(X, time, event)
cif = forest.predict_cif(X)                    # (n, n_causes, n_times) Aalen-Johansen CIF
print("OOB C-index, cause 1:", forest.oob_score(cause=1))
```

## What's included

| Tool | Estimates | Validated against |
|---|---|---|
| [`CompetingRiskForest`][comprisk.CompetingRiskForest] | cause-specific CIF, CHF, VIMP, SHAP | randomForestSRC |
| [`FineGrayRegression`][comprisk.FineGrayRegression] | subdistribution-hazard ratios | `cmprsk::crr()` |
| [`PenalizedFineGrayRegression`][comprisk.PenalizedFineGrayRegression] | LASSO/MCP/SCAD Fine-Gray | `crrp::crrp()` |
| [`CauseSpecificCox`][comprisk.CauseSpecificCox] | cause-specific hazard ratios | `survival::coxph()` |
| [`CumulativeIncidence`][comprisk.CumulativeIncidence] | non-parametric Aalen-Johansen CIF | `cmprsk::cuminc()` |
| [`gray_test`][comprisk.gray_test] | K-sample test for equal CIFs | `cmprsk::cuminc()$Tests` |

## Where to next

- **[Quickstart](quickstart.md)** — every task with runnable code (data format,
  prediction shapes, cross-validation, VIMP, minimal depth, SHAP, GPU, rfSRC
  migration).
- **[API reference](reference.md)** — full parameter lists from the docstrings.
- **Notebooks** — [`01_forest_quickstart`](https://github.com/sunnyadn/comprisk/blob/main/examples/01_forest_quickstart.ipynb)
  and [`02_regression_models`](https://github.com/sunnyadn/comprisk/blob/main/examples/02_regression_models.ipynb)
  (rendered on GitHub, runnable in Colab).
- **[Benchmarks](benchmarks.md)** · **[Equivalence vs rfSRC](equivalence-vs-rfsrc.md)** · **[References](REFERENCES.md)**
