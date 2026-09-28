"""Sanity check that the public API is importable from the package root."""

import comprisk

EXPECTED_PUBLIC = {
    "CauseSpecificCox",
    "CompetingRiskForest",
    "ConcordanceCI",
    "CumulativeIncidence",
    "DeltaConcordanceCI",
    "FineGrayRegression",
    "GrayTestResult",
    "PenalizedFineGrayRegression",
    "ScoreResult",
    "Surv",
    "__version__",
    "calibration_cr",
    "compute_uno_weights",
    "concordance_index_ci",
    "concordance_index_cr",
    "concordance_index_delta_ci",
    "concordance_index_uno_cr",
    "gray_test",
    "load",
    "score_cr",
}


def test_public_symbols_in_all():
    assert set(comprisk.__all__) == EXPECTED_PUBLIC
    assert all(hasattr(comprisk, name) for name in comprisk.__all__)


def test_n_jobs_defaults_to_minus_one():
    forest = comprisk.CompetingRiskForest()
    assert forest.n_jobs == -1
