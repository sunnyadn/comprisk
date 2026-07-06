# Contributing to comprisk

Thanks for your interest in improving `comprisk`. This document explains how to
report issues, propose changes, and get your contribution merged.

## Reporting bugs and requesting features

Please open an issue on the
[GitHub issue tracker](https://github.com/sunnyadn/comprisk/issues). For bug
reports, include:

- the `comprisk` version (`python -c "import comprisk; print(comprisk.__version__)"`),
  your OS, and Python version;
- a minimal, self-contained reproduction (data-generating code or a small
  attached array is ideal);
- the full traceback, and what you expected to happen instead.

For statistical-correctness reports (e.g. a discrepancy against an R reference
implementation such as `cmprsk`, `survival`, `riskRegression`, or
`randomForestSRC`), please state the reference package and version and, where
possible, the R script that produces the reference number.

## Asking questions / getting help

Use [GitHub Discussions](https://github.com/sunnyadn/comprisk/discussions) or
open an issue with the `question` label. There is no private support channel;
please keep questions public so answers help others.

## Development setup

The project uses [`uv`](https://docs.astral.sh/uv/) for environment management.

```bash
git clone https://github.com/sunnyadn/comprisk.git
cd comprisk
uv sync --extra dev
```

## Running the checks

Before opening a pull request, please make sure the following pass locally:

```bash
uv run pytest          # test suite
uv run ruff check .    # lint
uv run ruff format .   # formatting
```

New behavior should come with tests. For any change touching a numerical
estimator, add or extend a parity/property test rather than only an
input/output smoke test.

## Pull requests

1. Fork the repository and create a topic branch off `main`.
2. Make focused commits with clear messages.
3. Ensure the checks above pass and the test suite is green in CI.
4. Open a pull request describing the change and the motivation. Link any
   related issue.

By contributing, you agree that your contributions are licensed under the
project's [Apache-2.0](LICENSE) license.

## Code of conduct

Participation in this project is governed by our
[Code of Conduct](CODE_OF_CONDUCT.md).
