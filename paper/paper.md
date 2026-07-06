---
title: 'comprisk: A scikit-learn-compatible Python toolkit for competing-risks survival analysis'
tags:
  - Python
  - survival analysis
  - competing risks
  - random forest
  - biostatistics
  - machine learning
authors:
  - name: Sunny Yang
    orcid: 0009-0006-3160-0860
    affiliation: 1
    corresponding: true
  - name: Weiyan Zhao
    affiliation: 2
  - name: Wanqi Zhao
    affiliation: 3
affiliations:
  - name: University of Illinois Urbana-Champaign, USA
    index: 1
  - name: Northumbria University, UK
    index: 2
  - name: University of Calgary, Canada
    index: 3
date: 3 July 2026
bibliography: paper.bib
---

# Summary

Time-to-event data in medicine are frequently subject to *competing risks*:
a patient may experience one of several mutually exclusive terminal events
(for example, death from heart failure versus death from other causes), and
the occurrence of one event precludes the others. Standard survival methods
that treat competing events as censoring produce biased absolute-risk
estimates [@austin2016; @putter2007]. Correct analysis instead targets the
cause-specific cumulative incidence function (CIF) and models cause-specific
or subdistribution hazards.

`comprisk` is a Python package that brings the canonical toolkit of
competing-risks analysis into a single, `scikit-learn`-compatible library.
It provides a scalable competing-risks random survival forest together with
the standard regression and non-parametric estimators — Fine-Gray
subdistribution-hazard regression (including a penalized variant),
cause-specific Cox regression, the Aalen-Johansen CIF estimator, and Gray's
$K$-sample test. It adds competing-risks-aware model evaluation: inverse
probability of censoring weighted (IPCW) time-dependent AUC and Brier score,
cause-specific concordance indices with closed-form confidence intervals, and
calibration curves. Every estimator is validated numerically against the
established R reference implementations, so users obtain results consistent
with the established literature without leaving Python.

# Statement of need

Competing-risks methodology has, until now, been available to applied
researchers almost exclusively through R packages: `cmprsk` [@cmprsk] for
Fine-Gray regression and Gray's test, `survival` [@survival] for
cause-specific Cox models, `crrp` [@crrp] for penalized Fine-Gray,
`riskRegression` [@riskRegression] for IPCW scoring, and
`randomForestSRC` [@ishwaran2008; @ishwaran2014] for competing-risks forests.
Python's survival ecosystem — `lifelines` [@lifelines] and
`scikit-survival` [@sksurv] — is mature for single-event analysis but does
not natively cover competing risks: `scikit-survival` fits single-event
random survival forests only, and neither package emits Aalen-Johansen CIFs
or subdistribution-hazard regression. Analysts working in Python-based
machine-learning pipelines are therefore forced into a Python-to-R round trip,
which fragments reproducible workflows and raises the barrier to correct
competing-risks analysis.

`comprisk` closes this gap. Its central contribution is a native-Python
competing-risks random survival forest with competing-risks log-rank splitting
(both composite and cause-specific), Aalen-Johansen CIF and Nelson-Aalen
cumulative-hazard prediction, out-of-bag Breiman permutation variable
importance [@breiman2001], Ishwaran minimal-depth variable selection, IPCW
cause-specific concordance [@wolbers2009; @wolbers2014; @uno2011], and exact
cause-specific TreeSHAP attributions. The estimator is a genuine
`scikit-learn` estimator, so it composes with `Pipeline`, `cross_val_score`,
and hyperparameter search without a wrapper.

The forest is engineered for the cohort sizes encountered in modern
electronic-health-record research. A histogram-based split kernel with
`uint8`-binned features, just-in-time compiled with `numba`, gives sub-linear
wall-time growth in sample size: on real EHR cohorts it fits 10–22× faster
than `randomForestSRC` at comparable discrimination (both C-index
$\approx 0.85$, each under its own native concordance scorer), and on a
synthetic feasibility benchmark it scales to $n = 10^6$ in roughly one minute
on a consumer CPU, where existing tools become memory-bound. Output is
bit-identical across thread counts for a fixed random seed. An optional `equivalence="rfsrc"` mode
reproduces `randomForestSRC`'s per-tree random-number stream exactly, enabling
cross-library validation.

Alongside the forest, `comprisk` ships the regression and non-parametric
estimators an applied study needs, each validated to floating-point tolerance
against its R counterpart: Fine-Gray regression [@finegray1999] against
`cmprsk::crr`, penalized Fine-Gray (LASSO / ridge / elastic-net / MCP / SCAD)
against `crrp`, cause-specific Cox against `survival::coxph`, the
Aalen-Johansen estimator against `cmprsk::cuminc`, and Gray's test against
`cmprsk::cuminc`. The evaluation module `score_cr` / `calibration_cr` provides
the IPCW time-dependent AUC, Brier score, integrated variants, and calibration
data that correspond to the competing-risks mode of
`riskRegression::Score`, and the `concordance_index_ci` /
`concordance_index_delta_ci` functions supply closed-form (bootstrap-free)
confidence intervals and paired model-comparison tests for the IPCW
concordance based on its influence-function variance [@wolbers2014].

By consolidating these methods behind a consistent, well-tested,
`scikit-learn`-compatible API, `comprisk` lets applied researchers,
particularly in clinical and epidemiological machine learning, perform
correct and scalable competing-risks analysis entirely within the Python
scientific stack. The intended audience is biostatisticians, epidemiologists,
and data scientists building risk-prediction models on competing-risks
outcomes.

# Functionality

The public API exposes:

- **`CompetingRiskForest`** — competing-risks random survival forest with CIF
  / cumulative-hazard prediction, out-of-bag concordance scoring, permutation
  and minimal-depth variable importance, and exact TreeSHAP.
- **`FineGrayRegression`** and **`PenalizedFineGrayRegression`** —
  subdistribution-hazard regression with robust standard errors, and a
  cross-validated regularization path.
- **`CauseSpecificCox`** — cause-specific proportional-hazards regression.
- **`CumulativeIncidence`** — non-parametric Aalen-Johansen CIF estimation.
- **`gray_test`** — Gray's $K$-sample test for equality of CIFs.
- **`score_cr`**, **`calibration_cr`**, **`concordance_index_ci`**, and
  **`concordance_index_delta_ci`** — competing-risks-aware model evaluation.

The package requires Python $\geq 3.10$ and depends only on the core
scientific-Python stack (`numpy`, `scipy`, `pandas`, `joblib`, `numba`,
`scikit-learn`). It is distributed on PyPI (`pip install comprisk`) and
documented with a quickstart, worked notebooks, an autogenerated API
reference, and a benchmark dossier with reproduction scripts. Correctness is
covered by an extensive test suite, including property-based tests and
paired-seed equivalence checks against the R reference implementations.

# Data availability

The correctness test suite and a public synthetic two-cause Weibull benchmark
(shipped in the repository) can be reproduced by any user. The headline
real-cohort speed comparisons use a de-identified heart-failure electronic
health record cohort and the SEER breast-cancer registry; both are
access-restricted and require the respective data-use agreements, so those
specific tables are reproducible only by users with cohort access. Full
reproduction scripts and per-run provenance are provided in the repository's
benchmark dossier.

# Generative AI disclosure

Generative AI coding assistants were used as an assistive tool for drafting and
refactoring code, tests, and documentation. All methodology, design, and
numerical validation against the R reference implementations were carried out
and reviewed by the authors, who take full responsibility for the software's
correctness and the claims made here.

# Acknowledgements

We thank the maintainers of `randomForestSRC`, `cmprsk`, `crrp`,
`survival`, and `riskRegression`, whose implementations served as
validation references during development.

# References
