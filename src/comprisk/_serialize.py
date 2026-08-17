"""Pickle-free persistence: ``save()`` writes zip(meta.json + raw ``.npy``
arrays); ``comprisk.load()`` rebuilds via json + ``np.load(allow_pickle=False)``,
so no pickled code executes on load. Full docs: docs/persistence.md."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from comprisk._tree_flat import FlatTree

FORMAT_VERSION = 1
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)  # fixed timestamp -> deterministic bytes

# Fitted JSON-scalar attributes, enumerated explicitly per class; omissions
# are caught by the round-trip tests in tests/test_serialization.py.
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
    with zf.open(info, "w") as f:
        np.save(f, np.ascontiguousarray(arr))


def _read_array(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with zf.open(f"arrays/{name}.npy") as f:
        return np.load(f, allow_pickle=False)


def _write_meta(zf: zipfile.ZipFile, meta: dict) -> None:
    info = zipfile.ZipInfo("meta.json", date_time=_ZIP_EPOCH)
    info.compress_type = zf.compression
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


def _scalar_attrs(estimator, attrs: tuple[str, ...]) -> dict:
    return {a: _py(getattr(estimator, a)) for a in attrs}


def _concat(arrays: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(arrays) if arrays else np.empty(0)


def _split(concat: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
    return np.split(concat, np.cumsum(lengths)[:-1]) if len(lengths) else []


# --------------------------------------------------------------------------
# CompetingRiskForest
# --------------------------------------------------------------------------


def _save_forest(forest, zf: zipfile.ZipFile) -> None:
    trees = getattr(forest, "trees_", None)
    if trees is None:
        raise ValueError("Cannot save an unfitted estimator; call fit() first.")
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
    leaf_shapes = [s["leaves"]["shape"] for s in states]
    n_causes, n_time_bins = leaf_shapes[0][1], leaf_shapes[0][2]

    meta = {
        "format_version": FORMAT_VERSION,
        "comprisk_version": _comprisk_version(),
        "class": "CompetingRiskForest",
        "params": _jsonable_params(forest),
        "scalars": _scalar_attrs(forest, _FOREST_SCALAR_ATTRS),
        "feature_names_in": _feature_names(forest),
        "n_trees": len(trees),
        "leaf_grid": [int(n_causes), int(n_time_bins)],
        "optional_arrays": {
            "inbag": forest.inbag_ is not None,
            "cause_weights_arr": forest._cause_weights_arr is not None,
            "X_train_oob": forest._X_train_oob_ is not None,
            "y_train_oob": forest._y_train_oob_ is not None,
        },
    }
    _write_meta(zf, meta)

    # per-tree topology, concatenated on the node axis
    for field in (
        "features",
        "split_values",
        "left_children",
        "right_children",
        "is_leaf_flags",
        "leaf_idx_of_node",
    ):
        _write_array(zf, f"tree_{field}", _concat([s[field] for s in states]))
    _write_array(zf, "tree_n_nodes", np.array([len(s["features"]) for s in states], dtype=np.int64))
    _write_array(zf, "tree_n_leaves", np.array([sh[0] for sh in leaf_shapes], dtype=np.int64))

    # sparse leaf counts, concatenated (indptrs are per-tree, each 0-based)
    for field in ("ec_indptr", "ec_cause", "ec_time", "ec_val", "ar_indptr", "ar_time", "ar_val"):
        _write_array(zf, f"leaf_{field}", _concat([s["leaves"][field] for s in states]))

    # forest-level arrays
    _write_array(zf, "time_grid", forest.time_grid_)
    _write_array(zf, "bin_edges_concat", _concat(list(forest.bin_edges_)))
    _write_array(zf, "bin_edges_len", np.array([len(e) for e in forest.bin_edges_], dtype=np.int64))
    _write_array(zf, "oob_concat", _concat(list(forest.oob_indices_)))
    _write_array(zf, "oob_len", np.array([len(o) for o in forest.oob_indices_], dtype=np.int64))
    if forest.inbag_ is not None:
        _write_array(zf, "inbag", forest.inbag_)
    if forest._cause_weights_arr is not None:
        _write_array(zf, "cause_weights_arr", forest._cause_weights_arr)
    if forest._X_train_oob_ is not None:
        _write_array(zf, "X_train_oob", forest._X_train_oob_)
    if forest._y_train_oob_ is not None:
        _write_array(zf, "y_train_oob", forest._y_train_oob_)


def _load_forest(meta: dict, zf: zipfile.ZipFile):
    from comprisk.forest import CompetingRiskForest

    forest = CompetingRiskForest(**meta["params"])
    for attr, value in meta["scalars"].items():
        setattr(forest, attr, value)
    if meta["feature_names_in"] is not None:
        forest.feature_names_in_ = np.asarray(meta["feature_names_in"], dtype=object)

    n_causes, n_time_bins = meta["leaf_grid"]
    n_nodes = _read_array(zf, "tree_n_nodes")
    n_leaves = _read_array(zf, "tree_n_leaves")
    topo = {
        field: _split(_read_array(zf, f"tree_{field}"), n_nodes)
        for field in (
            "features",
            "split_values",
            "left_children",
            "right_children",
            "is_leaf_flags",
            "leaf_idx_of_node",
        )
    }
    ec_indptr = _split(_read_array(zf, "leaf_ec_indptr"), n_leaves + 1)
    ar_indptr = _split(_read_array(zf, "leaf_ar_indptr"), n_leaves + 1)
    ec_nnz = np.array([ip[-1] for ip in ec_indptr], dtype=np.int64)
    ar_nnz = np.array([ip[-1] for ip in ar_indptr], dtype=np.int64)
    ec_cause = _split(_read_array(zf, "leaf_ec_cause"), ec_nnz)
    ec_time = _split(_read_array(zf, "leaf_ec_time"), ec_nnz)
    ec_val = _split(_read_array(zf, "leaf_ec_val"), ec_nnz)
    ar_time = _split(_read_array(zf, "leaf_ar_time"), ar_nnz)
    ar_val = _split(_read_array(zf, "leaf_ar_val"), ar_nnz)

    trees = []
    for i in range(meta["n_trees"]):
        state = {
            "_cst_v": 2,
            **{field: topo[field][i] for field in topo},
            "leaves": {
                "shape": (int(n_leaves[i]), n_causes, n_time_bins),
                "ec_indptr": ec_indptr[i],
                "ec_cause": ec_cause[i],
                "ec_time": ec_time[i],
                "ec_val": ec_val[i],
                "ar_indptr": ar_indptr[i],
                "ar_time": ar_time[i],
                "ar_val": ar_val[i],
            },
        }
        tree = FlatTree.__new__(FlatTree)
        tree.__setstate__(state)
        trees.append(tree)
    forest.trees_ = trees

    forest.time_grid_ = _read_array(zf, "time_grid")
    forest.unique_times_ = forest.time_grid_
    forest.bin_edges_ = _split(
        _read_array(zf, "bin_edges_concat"), _read_array(zf, "bin_edges_len")
    )
    forest.oob_indices_ = _split(_read_array(zf, "oob_concat"), _read_array(zf, "oob_len"))
    opt = meta["optional_arrays"]
    forest.inbag_ = _read_array(zf, "inbag") if opt["inbag"] else None
    forest._cause_weights_arr = (
        _read_array(zf, "cause_weights_arr") if opt["cause_weights_arr"] else None
    )
    forest._X_train_oob_ = _read_array(zf, "X_train_oob") if opt["X_train_oob"] else None
    forest._y_train_oob_ = _read_array(zf, "y_train_oob") if opt["y_train_oob"] else None
    return forest


# --------------------------------------------------------------------------
# FineGrayRegression
# --------------------------------------------------------------------------


def _save_fine_gray(model, zf: zipfile.ZipFile) -> None:
    if not hasattr(model, "coef_"):
        raise ValueError("Cannot save an unfitted estimator; call fit() first.")
    meta = {
        "format_version": FORMAT_VERSION,
        "comprisk_version": _comprisk_version(),
        "class": "FineGrayRegression",
        "params": _jsonable_params(model),
        "scalars": _scalar_attrs(model, _FG_SCALAR_ATTRS),
        "feature_names_in": _feature_names(model),
    }
    _write_meta(zf, meta)
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

    return getattr(comprisk, "__version__", "unknown")


def _feature_names(estimator) -> list | None:
    names = getattr(estimator, "feature_names_in_", None)
    return None if names is None else [str(n) for n in names]


def save_estimator(estimator, path, *, compress: bool = True) -> None:
    cls = type(estimator).__name__
    if cls not in _SAVERS:
        raise NotImplementedError(f"save() is not implemented for {cls}")
    path = Path(path)
    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", compression=compression, compresslevel=3) as zf:
        _SAVERS[cls](estimator, zf)


def load(path):
    """Load an estimator saved with ``estimator.save(path)``; executes no
    pickled code (JSON metadata plus ``np.load(allow_pickle=False)`` arrays)."""
    path = Path(path)
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
