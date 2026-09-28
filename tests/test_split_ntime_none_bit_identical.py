"""Regression guard: ``split_ntime=None`` tree-building output must be stable.

``ANCHOR_SHA256`` is a digest of ``walk_tree()`` over every tree in the forest:
split features, split bin indices and each leaf's CIF table in pre-order. It
moves only when tree building changes, not when the pickle schema or fitted
attributes change. ``test_parallel_equivalence`` checks the same trees within
one run; this is the cross-version guard for default-mode trees.

Through 0.8.1 the anchor was the SHA-256 of the whole pickled forest, which had to
be re-pinned for every schema or storage change (see git history of this file).
This digest was pinned from the same fit that still reproduced that last pickle
anchor (``3532d3ef..``), so it describes the same trees.
"""

from __future__ import annotations

import hashlib

import numpy as np

from comprisk import CompetingRiskForest
from tests._tree_walkers import walk_tree

ANCHOR_SHA256 = "f6da3384e9f22b43f846f10cea1e11fd9800bb81a678f3e252fc7bd9efc5ac82"


def _fit_forest() -> CompetingRiskForest:
    rng = np.random.default_rng(0)
    n, p = 2000, 10
    X = rng.normal(size=(n, p))
    time = rng.exponential(1.0, size=n) + 0.1
    event = rng.integers(0, 3, size=n)
    # Pin cpu: defensive anchor against a future v1.1 auto→cuda flip
    # (GPU produces different trees by design — DFS vs BFS node order).
    return CompetingRiskForest(
        n_estimators=50,
        max_depth=6,
        random_state=0,
        n_jobs=1,
        mode="default",
        split_ntime=None,
        device="cpu",
        # Pin swr (= the pre-SUN-83 bootstrap=True default) so this digest stays
        # anchored to tree-building behavior under split_ntime=None, independent
        # of the SUN-83 sampling-default change (swr full-n -> swor 0.632n).
        samptype="swr",
    ).fit(X, time, event)


def _trees_digest(forest: CompetingRiskForest) -> str:
    walks = [walk_tree(t) for t in forest.trees_]
    return hashlib.sha256(repr(walks).encode()).hexdigest()


def test_split_ntime_none_matches_anchor_digest() -> None:
    digest = _trees_digest(_fit_forest())
    assert digest == ANCHOR_SHA256, (
        f"tree digest drift: got {digest}, expected {ANCHOR_SHA256}. "
        "split_ntime=None tree building must be stable."
    )
