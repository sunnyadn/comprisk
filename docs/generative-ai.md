# Use of generative AI

comprisk is developed with Claude Code, Anthropic's command-line coding agent,
using Claude Opus and Claude Sonnet models. It works from written
specifications, edits files in this repository, and runs the test suite and the
benchmarks itself. The authors read its diffs and decide what merges.

Roughly 80% of the test suite, 50% of the library source and 90% of the
documentation began as agent drafts, and agent-drafted code appears in every
module. The authors decided the statistical methodology, the choice and
formulation of each estimator, and the validation described below.

Two mechanical checks run on top of that review. The regression and
non-parametric estimators are gated on R-generated fixtures committed to the
test suite, and the forest is validated against `randomForestSRC` through a
maintainer-run harness under `validation/alignment/`. A regression in any of
them fails CI against numbers `cmprsk`, `crrp`, `survival`, `riskRegression`
and `randomForestSRC` produced.

The authors understand this implementation and take responsibility for the
correctness of the software and the accuracy of the claims made about it.
