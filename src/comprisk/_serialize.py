"""Pickle-free persistence: ``save()`` writes zip(meta.json + raw ``.npy``
arrays); ``comprisk.load()`` rebuilds via json + ``np.load(allow_pickle=False)``,
so no pickled code executes on load. Full docs: docs/persistence.md."""

from __future__ import annotations

import json
import zipfile

import numpy as np

from comprisk._tree_flat import FlatTree

FORMAT_VERSION = 1
_TREE_STATE_VERSION = 2  # the FlatTree.__getstate__ layout this container encodes
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)  # fixed timestamp -> deterministic bytes

# Fitted attributes, enumerated explicitly per class; omissions are caught by
# the attribute-completeness and consistency tests in tests/test_serialization.py.
_FOREST_SCALAR_ATTRS = (
    "n_causes_",
    "n_features_in_",
    "_resolved_sampsize_",
    "_oob_available_",
    "_rng_mode_eff_",
    "_split_ntime_eff_",
    "_time_grid_max_eff_",
    "_resolved_nsplit_",
    "_effective_device_",
)
_FOREST_OPTIONAL_ARRAYS = {
    "inbag": "inbag_",
    "cause_weights_arr": "_cause_weights_arr",
    "X_train_oob": "_X_train_oob_",
    "y_train_oob": "_y_train_oob_",
}
_TREE_TOPO_FIELDS = (
    "features",
    "split_values",
    "left_children",
    "right_children",
    "leaf_idx_of_node",
)
_TREE_LEAF_FIELDS = ("ec_indptr", "ec_cause", "ec_time", "ec_val", "ar_indptr", "ar_time", "ar_val")
_FG_SCALAR_ATTRS = (
    "n_features_in_",
    "n_iter_",
    "converged_",
    "log_likelihood_",
    "log_likelihood_null_",
)
_FG_ARRAY_ATTRS = ("coef_", "var_", "se_", "score_", "information_")
_FG_STATE_ARRAY_FIELDS = (
    "cengroups",
    "cengroup_event_times",
    "G_at_event_times",
    "baseline_times",
    "baseline_hazard_increments",
)


def _write_array(zf: zipfile.ZipFile, name: str, arr: np.ndarray) -> None:
    info = zipfile.ZipInfo(f"arrays/{name}.npy", date_time=_ZIP_EPOCH)
    info.compress_type = zf.compression
    # zf.open(ZipInfo) ignores the ZipFile-level compresslevel (stdlib default
    # would be level 6, measured ~4.5x slower than 3 for the same ratio here).
    info._compresslevel = zf.compresslevel
    with zf.open(info, "w") as f:
        np.save(f, np.ascontiguousarray(arr))


def _read_array(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with zf.open(f"arrays/{name}.npy") as f:
        return np.load(f, allow_pickle=False)


def _write_meta(zf: zipfile.ZipFile, meta: dict) -> None:
    info = zipfile.ZipInfo("meta.json", date_time=_ZIP_EPOCH)
    info.compress_type = zf.compression
    info._compresslevel = zf.compresslevel
    try:
        payload = json.dumps(meta, indent=1, sort_keys=True)
    except TypeError as exc:
        raise NotImplementedError(
            "save() requires JSON-serializable constructor params; a callable "
            f"(e.g. sampsize=<function>) cannot be persisted: {exc}"
        ) from None
    zf.writestr(info, payload)


def _py(v):
    """numpy scalar -> native python scalar (json.dumps rejects np.int64 etc.)."""
    return v.item() if isinstance(v, np.generic) else v


def _jsonable_params(estimator) -> dict:
    params = estimator.get_params(deep=False)
    return {k: (v.tolist() if isinstance(v, np.ndarray) else _py(v)) for k, v in params.items()}


def _base_meta(estimator, cls: str, scalar_attrs: tuple[str, ...]) -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "comprisk_version": _comprisk_version(),
        "class": cls,
        "params": _jsonable_params(estimator),
        "scalars": {a: _py(getattr(estimator, a)) for a in scalar_attrs},
        "feature_names_in": _feature_names(estimator),
    }


def _split(concat: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
    return np.split(concat, np.cumsum(lengths)[:-1])


# --------------------------------------------------------------------------
# CompetingRiskForest
# --------------------------------------------------------------------------


def _save_forest(forest, zf: zipfile.ZipFile) -> None:
    from sklearn.utils.validation import check_is_fitted

    check_is_fitted(forest, "trees_")
    trees = forest.trees_
    if not all(
        isinstance(t, FlatTree) and t.leaf_event_counts is not None and t.leaf_at_risk is not None
        for t in trees
    ):
        raise NotImplementedError(
            "save() supports the default flat-tree path only (mode='default', "
            "no equivalence='rfsrc'); use pickle for reference-mode or "
            "rfsrc-aligned forests."
        )

    states = [t.__getstate__() for t in trees]
    if any(s["_cst_v"] != _TREE_STATE_VERSION for s in states):
        raise AssertionError(
            "FlatTree state version changed; update _save_forest's layout and "
            "bump FORMAT_VERSION before shipping."
        )

    meta = _base_meta(forest, "CompetingRiskForest", _FOREST_SCALAR_ATTRS)
    meta["n_trees"] = len(trees)
    _write_meta(zf, meta)

    # per-tree topology, concatenated on the node axis
    for field in _TREE_TOPO_FIELDS:
        _write_array(zf, f"tree_{field}", np.concatenate([s[field] for s in states]))
    _write_array(zf, "tree_n_nodes", np.array([len(s["features"]) for s in states], dtype=np.int64))
    _write_array(
        zf, "tree_n_leaves", np.array([s["leaves"]["shape"][0] for s in states], dtype=np.int64)
    )

    # sparse leaf counts, concatenated (indptrs are per-tree, each 0-based)
    for field in _TREE_LEAF_FIELDS:
        _write_array(zf, f"leaf_{field}", np.concatenate([s["leaves"][field] for s in states]))

    # forest-level arrays
    _write_array(zf, "time_grid", forest.time_grid_)
    _write_array(zf, "bin_edges_concat", np.concatenate(list(forest.bin_edges_)))
    _write_array(zf, "bin_edges_len", np.array([len(e) for e in forest.bin_edges_], dtype=np.int64))
    _write_array(zf, "oob_concat", np.concatenate(list(forest.oob_indices_)))
    _write_array(zf, "oob_len", np.array([len(o) for o in forest.oob_indices_], dtype=np.int64))
    for name, attr in _FOREST_OPTIONAL_ARRAYS.items():
        value = getattr(forest, attr)
        if value is not None:
            _write_array(zf, name, value)


def _load_forest(meta: dict, zf: zipfile.ZipFile):
    from comprisk.forest import CompetingRiskForest

    forest = CompetingRiskForest(**meta["params"])
    for attr, value in meta["scalars"].items():
        setattr(forest, attr, value)
    if meta["feature_names_in"] is not None:
        forest.feature_names_in_ = np.asarray(meta["feature_names_in"], dtype=object)

    forest.time_grid_ = _read_array(zf, "time_grid")
    forest.unique_times_ = forest.time_grid_
    n_causes = int(meta["scalars"]["n_causes_"])
    n_time_bins = len(forest.time_grid_)

    n_nodes = _read_array(zf, "tree_n_nodes")
    n_leaves = _read_array(zf, "tree_n_leaves")
    topo = {field: _split(_read_array(zf, f"tree_{field}"), n_nodes) for field in _TREE_TOPO_FIELDS}
    ec_indptr = _split(_read_array(zf, "leaf_ec_indptr"), n_leaves + 1)
    ar_indptr = _split(_read_array(zf, "leaf_ar_indptr"), n_leaves + 1)
    ec_nnz = np.array([ip[-1] for ip in ec_indptr], dtype=np.int64)
    ar_nnz = np.array([ip[-1] for ip in ar_indptr], dtype=np.int64)
    leaves = {"ec_indptr": ec_indptr, "ar_indptr": ar_indptr}
    for field, nnz in (
        ("ec_cause", ec_nnz),
        ("ec_time", ec_nnz),
        ("ec_val", ec_nnz),
        ("ar_time", ar_nnz),
        ("ar_val", ar_nnz),
    ):
        leaves[field] = _split(_read_array(zf, f"leaf_{field}"), nnz)

    trees = []
    for i in range(meta["n_trees"]):
        state = {
            "_cst_v": _TREE_STATE_VERSION,
            **{field: topo[field][i] for field in _TREE_TOPO_FIELDS},
            "leaves": {
                "shape": (int(n_leaves[i]), n_causes, n_time_bins),
                **{field: leaves[field][i] for field in _TREE_LEAF_FIELDS},
            },
        }
        tree = FlatTree.__new__(FlatTree)
        tree.__setstate__(state)
        trees.append(tree)
    forest.trees_ = trees

    forest.bin_edges_ = _split(
        _read_array(zf, "bin_edges_concat"), _read_array(zf, "bin_edges_len")
    )
    forest.oob_indices_ = _split(_read_array(zf, "oob_concat"), _read_array(zf, "oob_len"))
    present = set(zf.namelist())
    for name, attr in _FOREST_OPTIONAL_ARRAYS.items():
        setattr(forest, attr, _read_array(zf, name) if f"arrays/{name}.npy" in present else None)
    return forest


# --------------------------------------------------------------------------
# FineGrayRegression
# --------------------------------------------------------------------------


def _save_fine_gray(model, zf: zipfile.ZipFile) -> None:
    from sklearn.utils.validation import check_is_fitted

    check_is_fitted(model, "coef_")
    _write_meta(zf, _base_meta(model, "FineGrayRegression", _FG_SCALAR_ATTRS))
    for attr in _FG_ARRAY_ATTRS:
        _write_array(zf, attr, np.asarray(getattr(model, attr)))
    for field in _FG_STATE_ARRAY_FIELDS:
        _write_array(zf, f"state_{field}", getattr(model._state, field))


def _load_fine_gray(meta: dict, zf: zipfile.ZipFile):
    from comprisk.fine_gray import FineGrayRegression, _FGState

    model = FineGrayRegression(**meta["params"])
    for attr, value in meta["scalars"].items():
        setattr(model, attr, value)
    if meta["feature_names_in"] is not None:
        model.feature_names_in_ = np.asarray(meta["feature_names_in"], dtype=object)
    for attr in _FG_ARRAY_ATTRS:
        setattr(model, attr, _read_array(zf, attr))
    model._state = _FGState(
        **{field: _read_array(zf, f"state_{field}") for field in _FG_STATE_ARRAY_FIELDS}
    )
    return model


# --------------------------------------------------------------------------
# public entry points
# --------------------------------------------------------------------------

_SAVERS = {
    "CompetingRiskForest": _save_forest,
    "FineGrayRegression": _save_fine_gray,
}
_LOADERS = {
    "CompetingRiskForest": _load_forest,
    "FineGrayRegression": _load_fine_gray,
}


def _comprisk_version() -> str:
    import comprisk

    return comprisk.__version__


def _feature_names(estimator) -> list | None:
    names = getattr(estimator, "feature_names_in_", None)
    return None if names is None else [str(n) for n in names]


def save_estimator(estimator, path, *, compress: bool = True) -> None:
    cls = type(estimator).__name__
    if cls not in _SAVERS:
        raise NotImplementedError(f"save() is not implemented for {cls}")
    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", compression=compression, compresslevel=3) as zf:
        _SAVERS[cls](estimator, zf)


def load(path):
    """Load an estimator saved with ``estimator.save(path)``; executes no
    pickled code (JSON metadata plus ``np.load(allow_pickle=False)`` arrays)."""
    with zipfile.ZipFile(path) as zf:
        try:
            meta = json.loads(zf.read("meta.json"))
        except KeyError:
            raise ValueError(f"{path} is not a comprisk model file (no meta.json)") from None
        version = meta.get("format_version")
        if not isinstance(version, int) or version > FORMAT_VERSION:
            raise ValueError(
                f"{path} has format_version={version!r}; this comprisk supports up to "
                f"{FORMAT_VERSION}. Upgrade comprisk to load it."
            )
        cls = meta.get("class")
        if cls not in _LOADERS:
            raise ValueError(f"{path} contains unsupported class {cls!r}")
        return _LOADERS[cls](meta, zf)
