"""Tests for validation.alignment.compare_splits."""

import numpy as np
from validation.alignment.compare_splits import (
    comprisk_candidate_stats,
    toy_input,
)

from comprisk._splits import find_best_split


def test_comprisk_candidate_stats_covers_all_features():
    data = toy_input(seed=0, n=30, n_features=3, n_causes=2)
    df = comprisk_candidate_stats(data["X"], data["time"], data["event"], data["n_causes"])
    assert set(df["feature"].unique()) == {0, 1, 2}


def test_comprisk_candidate_stats_best_matches_find_best_split():
    data = toy_input(seed=0, n=30, n_features=3, n_causes=2)
    df = comprisk_candidate_stats(data["X"], data["time"], data["event"], data["n_causes"])
    best_row = df.sort_values("stat", ascending=False).iloc[0]
    _feat, _thresh, stat = find_best_split(
        data["X"], data["time"], data["event"], data["n_causes"], min_samples_leaf=1
    )
    # Ties break on lower-feature, lower-threshold in find_best_split; allow equality in stat.
    assert np.isclose(float(best_row["stat"]), stat, atol=1e-12)
