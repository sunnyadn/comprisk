"""Round-trip tests for the v2 compact FlatTree state and save()/load():
the acceptance bar is bit-identical predictions, with array-level asserts
alongside to localize failures."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from comprisk import CompetingRiskForest
from comprisk._tree_flat import FlatTree, pack_leaf_counts, unpack_leaf_counts


def _make_data(n: int = 600, p: int = 5, seed: int = 0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    time = rng.exponential(10.0, n) + 0.01
    event = rng.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3])
    return X, time, event


def _fit_default(seed: int = 0, **kw) -> tuple[CompetingRiskForest, np.ndarray]:
    X, time, event = _make_data(seed=seed)
    params = dict(n_estimators=5, random_state=42, n_jobs=1)
    params.update(kw)
    return CompetingRiskForest(**params).fit(X, time, event), X


def _assert_identical_predictions(m1, m2, X):
    assert np.array_equal(m1.predict_cif(X), m2.predict_cif(X))
    assert np.array_equal(m1.predict_chf(X), m2.predict_chf(X))
    assert np.array_equal(
        m1.predict_risk(X, cause=1, kind="integrated_chf"),
        m2.predict_risk(X, cause=1, kind="integrated_chf"),
    )
    times = np.linspace(0.5, 15.0, 7)
    assert np.array_equal(m1.predict_cif(X, times=times), m2.predict_cif(X, times=times))


@pytest.mark.parametrize("seed", [0, 1])
def test_pack_unpack_leaf_counts_roundtrip(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_leaves, n_causes, n_t = 40, 2, 120
    ec = np.zeros((n_leaves, n_causes, n_t), dtype=np.uint32)
    mask = rng.random(ec.shape) > 0.97
    ec[mask] = rng.integers(1, 300, size=mask.sum(), dtype=np.uint32)
    # non-increasing at-risk rows with plateaus and trailing zeros
    drops = rng.integers(0, 3, size=(n_leaves, n_t)).astype(np.int64)
    ar = np.maximum(60 - np.cumsum(drops, axis=1), 0).astype(np.uint32)

    ec2, ar2 = unpack_leaf_counts(pack_leaf_counts(ec, ar))
    assert np.array_equal(ec2, ec) and ec2.dtype == np.uint32
    assert np.array_equal(ar2, ar) and ar2.dtype == np.uint32


def test_default_mode_pickle_roundtrip_bit_identical() -> None:
    m, X = _fit_default()
    m2 = pickle.loads(pickle.dumps(m))
    _assert_identical_predictions(m, m2, X)
    for t1, t2 in zip(m.trees_, m2.trees_, strict=True):
        assert np.array_equal(t1.leaf_table, t2.leaf_table)
        assert t2.leaf_table.dtype == np.float64
        assert np.array_equal(t1.leaf_event_counts, t2.leaf_event_counts)
        assert t2.leaf_event_counts.dtype == np.uint32
        assert np.array_equal(t1.leaf_at_risk, t2.leaf_at_risk)
        assert t2.leaf_at_risk.dtype == np.uint32
        for name in ("features", "left_children", "right_children", "leaf_idx_of_node"):
            assert np.array_equal(getattr(t1, name), getattr(t2, name))
            assert getattr(t2, name).dtype == np.int64
        assert np.array_equal(t1.split_values, t2.split_values)
        assert t2.split_values.dtype == t1.split_values.dtype
    for o1, o2 in zip(m.oob_indices_, m2.oob_indices_, strict=True):
        assert np.array_equal(o1, o2)


def test_reference_mode_pickle_roundtrip_bit_identical() -> None:
    m, X = _fit_default(mode="reference", n_estimators=3)
    # populate the _flat / _flat_chf caches so the pickle exercises the
    # counts-less FlatTree branch of __getstate__
    m.predict_cif(X[:5])
    m.predict_chf(X[:5])
    m2 = pickle.loads(pickle.dumps(m))
    _assert_identical_predictions(m, m2, X)


def test_rfsrc_equivalence_pickle_roundtrip_bit_identical() -> None:
    m, X = _fit_default(equivalence="rfsrc", n_estimators=3)
    m2 = pickle.loads(pickle.dumps(m))
    _assert_identical_predictions(m, m2, X)


def test_legacy_dense_state_still_loads() -> None:
    """A pre-v2 pickle carries the plain dense __dict__; it must load unchanged."""
    m, X = _fit_default(n_estimators=2)
    t = m.trees_[0]
    legacy_state = dict(t.__dict__)  # what old pickles recorded
    t2 = FlatTree.__new__(FlatTree)
    t2.__setstate__(legacy_state)
    assert np.array_equal(t.leaf_table, t2.leaf_table)
    from comprisk._tree_flat import predict_leaf_indices

    Xb = np.clip(np.floor(np.abs(X[:20]) * 10), 0, 255).astype(np.uint8)
    assert np.array_equal(predict_leaf_indices(t, Xb), predict_leaf_indices(t2, Xb))


def test_shap_caches_not_serialized() -> None:
    m, X = _fit_default(n_estimators=3)
    baseline = len(pickle.dumps(m))
    m.shap_values(X[:10], n_jobs=1)
    assert any(hasattr(t, "_shap_covers") for t in m.trees_)
    after = len(pickle.dumps(m))
    assert after <= baseline * 1.01
    m2 = pickle.loads(pickle.dumps(m))
    _assert_identical_predictions(m, m2, X)
    # caches were dropped, not restored
    assert not any(hasattr(t, "_shap_covers") for t in m2.trees_)


def test_compact_pickle_is_much_smaller_than_dense() -> None:
    m, _ = _fit_default(n_estimators=10)
    compact = len(pickle.dumps(m.trees_))
    dense = len(pickle.dumps([dict(t.__dict__) for t in m.trees_]))
    assert compact < 0.15 * dense


def test_oob_indices_are_int32() -> None:
    m, _ = _fit_default(n_estimators=2)
    assert all(o.dtype == np.int32 for o in m.oob_indices_)
    # OOB machinery still works on int32 indices
    assert np.isfinite(m.oob_score(cause=1))


# --------------------------------------------------------------------------
# save() / comprisk.load() — the pickle-free container format
# --------------------------------------------------------------------------


def test_forest_save_load_bit_identical(tmp_path) -> None:
    import comprisk

    m, X = _fit_default(n_estimators=6)
    p = tmp_path / "forest.crm"
    m.save(p)
    m2 = comprisk.load(p)
    _assert_identical_predictions(m, m2, X)
    assert np.array_equal(m.time_grid_, m2.time_grid_)
    assert all(np.array_equal(a, b) for a, b in zip(m.bin_edges_, m2.bin_edges_, strict=True))
    assert all(np.array_equal(a, b) for a, b in zip(m.oob_indices_, m2.oob_indices_, strict=True))
    assert np.array_equal(m.oob_score(cause=1), m2.oob_score(cause=1))
    assert m2.n_causes_ == m.n_causes_ and m2.get_params() == m.get_params()


def test_forest_save_load_with_feature_names(tmp_path) -> None:
    import comprisk

    pd = pytest.importorskip("pandas")
    X, time, event = _make_data()
    cols = [f"feat_{j}" for j in range(X.shape[1])]
    m = CompetingRiskForest(n_estimators=3, random_state=0, n_jobs=1).fit(
        pd.DataFrame(X, columns=cols), time, event
    )
    p = tmp_path / "forest.crm"
    m.save(p)
    m2 = comprisk.load(p)
    assert list(m2.feature_names_in_) == cols
    assert np.array_equal(
        m.predict_cif(pd.DataFrame(X, columns=cols)),
        m2.predict_cif(pd.DataFrame(X, columns=cols)),
    )


def test_fine_gray_save_load_bit_identical(tmp_path) -> None:
    import comprisk
    from comprisk import FineGrayRegression

    X, time, event = _make_data(n=400)
    m = FineGrayRegression(cause=1, max_iter=20).fit(X, time=time, event=event)
    p = tmp_path / "fg.crm"
    m.save(p)
    m2 = comprisk.load(p)
    assert np.array_equal(m.coef_, m2.coef_)
    assert np.array_equal(m.se_, m2.se_)
    assert np.array_equal(m.predict(X), m2.predict(X))
    assert np.array_equal(m.predict_cumulative_incidence(X), m2.predict_cumulative_incidence(X))
    times = np.linspace(0.5, 12.0, 5)
    assert np.array_equal(
        m.predict_cumulative_incidence(X, times=times),
        m2.predict_cumulative_incidence(X, times=times),
    )
    assert m2.converged_ == m.converged_ and m2.n_iter_ == m.n_iter_


def test_save_uncompressed_and_deterministic_bytes(tmp_path) -> None:
    import comprisk

    m, X = _fit_default(n_estimators=3)
    p1, p2, p3 = (tmp_path / n for n in ("a.crm", "b.crm", "c.crm"))
    m.save(p1)
    m.save(p2)
    assert p1.read_bytes() == p2.read_bytes()
    # load -> save produces the same bytes again (stable checksums)
    comprisk.load(p1).save(p3)
    assert p3.read_bytes() == p1.read_bytes()
    praw = tmp_path / "raw.crm"
    m.save(praw, compress=False)
    assert praw.stat().st_size > p1.stat().st_size
    _assert_identical_predictions(m, comprisk.load(praw), X)


def test_save_rejects_unsupported_configs(tmp_path) -> None:
    import comprisk

    with pytest.raises(ValueError, match="unfitted"):
        CompetingRiskForest(n_estimators=2).save(tmp_path / "x.crm")
    m_ref, _ = _fit_default(mode="reference", n_estimators=2)
    with pytest.raises(NotImplementedError, match="flat-tree"):
        m_ref.save(tmp_path / "x.crm")
    m_rf, _ = _fit_default(equivalence="rfsrc", n_estimators=2)
    with pytest.raises(NotImplementedError, match="flat-tree"):
        m_rf.save(tmp_path / "x.crm")
    with pytest.raises(ValueError, match="not a comprisk model file"):
        import zipfile

        with zipfile.ZipFile(tmp_path / "empty.zip", "w"):
            pass
        comprisk.load(tmp_path / "empty.zip")


def test_load_rejects_newer_format_version(tmp_path) -> None:
    import json
    import zipfile

    import comprisk

    m, _ = _fit_default(n_estimators=2)
    p = tmp_path / "forest.crm"
    m.save(p)
    with zipfile.ZipFile(p) as zf:
        meta = json.loads(zf.read("meta.json"))
        entries = {n: zf.read(n) for n in zf.namelist() if n != "meta.json"}
    meta["format_version"] = 99
    p2 = tmp_path / "future.crm"
    with zipfile.ZipFile(p2, "w") as zf:
        zf.writestr("meta.json", json.dumps(meta))
        for n, blob in entries.items():
            zf.writestr(n, blob)
    with pytest.raises(ValueError, match="format_version"):
        comprisk.load(p2)
