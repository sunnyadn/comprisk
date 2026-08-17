"""Flattened tree representation and nogil descent kernel.

Both reference-mode and histogram-mode trees flatten into the same
parallel-array layout (``FlatTree``) for vectorized prediction. The only
per-mode differences are (a) the split-value dtype (``float64`` threshold
vs ``int64`` bin index) and (b) how a leaf's leaf quantity (CIF or CHF)
is obtained from its node — these are injected via callables into the
flattening pass (``flatten_tree``). Descent is one shared numba kernel
(``_descend_flat_nogil``) that numba specializes by dtype; the final
leaf-table gather is a plain NumPy fancy-index op.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass
class FlatTree:
    features: np.ndarray  # (n_nodes,) int64
    split_values: np.ndarray  # (n_nodes,) — threshold or bin index
    left_children: np.ndarray  # (n_nodes,) int64
    right_children: np.ndarray  # (n_nodes,) int64
    is_leaf_flags: np.ndarray  # (n_nodes,) bool
    leaf_table: np.ndarray  # (n_leaves, n_causes, n_time_bins) float64
    leaf_idx_of_node: np.ndarray  # (n_nodes,) int64; -1 for internal nodes
    # Optional raw counts persisted by the default-mode njit builder so that
    # secondary leaf quantities (CHF, etc.) can be materialised lazily without
    # rebuilding the tree. None on legacy flatten paths that source from
    # HistTreeNode (which retains its own counts and uses _flat_chf caching).
    leaf_event_counts: np.ndarray | None = None  # (n_leaves, n_causes, n_time_bins) uint32
    leaf_at_risk: np.ndarray | None = None  # (n_leaves, n_time_bins) uint32

    # Serialized-state format version. v2 replaces the dense field dump with
    # sparse leaf counts and drops the (recomputable) leaf_table; a state dict
    # without the marker is a pre-v2 pickle and loads unchanged.
    _STATE_VERSION = 2

    def __getstate__(self):
        # Whitelist the declared fields: dynamic caches planted on instances
        # (_shap_covers/_shap_base/_shap_height, _chf_leaf_table) recompute on
        # demand and would otherwise re-inflate the pickle.
        state = {
            "_cst_v": self._STATE_VERSION,
            "features": self.features.astype(np.int32),
            "split_values": _pack_split_values(self.split_values),
            "left_children": self.left_children.astype(np.int32),
            "right_children": self.right_children.astype(np.int32),
            "is_leaf_flags": self.is_leaf_flags,
            "leaf_idx_of_node": self.leaf_idx_of_node.astype(np.int32),
        }
        if self.leaf_event_counts is not None and self.leaf_at_risk is not None:
            # leaf_table is bit-identical recomputable from the counts (it was
            # produced from them by aalen_johansen_from_counts_batched), so it
            # is not serialized at all.
            state["leaves"] = pack_leaf_counts(self.leaf_event_counts, self.leaf_at_risk)
        else:
            # Legacy flatten paths (reference-mode / HistTreeNode caches) carry
            # no raw counts; the leaf table is the only source of truth.
            state["leaf_table"] = self.leaf_table
        return state

    def __setstate__(self, state):
        if "_cst_v" not in state:  # pre-v2 pickle: plain dense __dict__
            self.__dict__.update(state)
            return
        if state["_cst_v"] > self._STATE_VERSION:
            raise ValueError(
                f"FlatTree state version {state['_cst_v']} is newer than this "
                f"comprisk supports ({self._STATE_VERSION}); upgrade comprisk to load it."
            )
        # The descent kernel is numba-specialized on int64 topology / the
        # original split dtype, so decode restores the exact fit-time dtypes.
        self.features = state["features"].astype(np.int64)
        self.split_values = _unpack_split_values(state["split_values"])
        self.left_children = state["left_children"].astype(np.int64)
        self.right_children = state["right_children"].astype(np.int64)
        self.is_leaf_flags = state["is_leaf_flags"]
        self.leaf_idx_of_node = state["leaf_idx_of_node"].astype(np.int64)
        if "leaves" in state:
            from comprisk._estimators import aalen_johansen_from_counts_batched

            ec, ar = unpack_leaf_counts(state["leaves"])
            self.leaf_event_counts = ec
            self.leaf_at_risk = ar
            self.leaf_table = aalen_johansen_from_counts_batched(ec, ar, ec.shape[1])
        else:
            self.leaf_table = state["leaf_table"]
            self.leaf_event_counts = None
            self.leaf_at_risk = None

    @classmethod
    def from_arrays(
        cls,
        *,
        features: np.ndarray,
        split_values: np.ndarray,
        left_children: np.ndarray,
        right_children: np.ndarray,
        is_leaf_flags: np.ndarray,
        leaf_table: np.ndarray,
        leaf_idx_of_node: np.ndarray,
        leaf_event_counts: np.ndarray | None = None,
        leaf_at_risk: np.ndarray | None = None,
    ) -> FlatTree:
        """Construct a FlatTree from already-flat arrays.

        Used by the njit flat-tree builder. The existing ``flatten_tree``
        path constructs FlatTree internally without going through here.
        """
        n_nodes = features.shape[0]
        for name, arr in (
            ("split_values", split_values),
            ("left_children", left_children),
            ("right_children", right_children),
            ("is_leaf_flags", is_leaf_flags),
            ("leaf_idx_of_node", leaf_idx_of_node),
        ):
            if arr.shape[0] != n_nodes:
                raise ValueError(
                    f"{name} length {arr.shape[0]} does not match features length {n_nodes}"
                )
        return cls(
            features=features,
            split_values=split_values,
            left_children=left_children,
            right_children=right_children,
            is_leaf_flags=is_leaf_flags,
            leaf_table=leaf_table,
            leaf_idx_of_node=leaf_idx_of_node,
            leaf_event_counts=leaf_event_counts,
            leaf_at_risk=leaf_at_risk,
        )


def _pack_split_values(sv: np.ndarray) -> np.ndarray:
    """Downcast integer split values (histogram bin indices) to the smallest
    sufficient dtype; float thresholds (reference mode) pass through."""
    if sv.dtype.kind not in "iu" or sv.size == 0:
        return sv
    lo, hi = int(sv.min()), int(sv.max())
    if lo >= 0 and hi <= np.iinfo(np.uint8).max:
        return sv.astype(np.uint8)
    if lo >= 0 and hi <= np.iinfo(np.uint16).max:
        return sv.astype(np.uint16)
    if np.iinfo(np.int32).min <= lo and hi <= np.iinfo(np.int32).max:
        return sv.astype(np.int32)
    return sv


def _unpack_split_values(sv: np.ndarray) -> np.ndarray:
    return sv.astype(np.int64) if sv.dtype.kind in "iu" else sv


def pack_leaf_counts(leaf_event_counts: np.ndarray, leaf_at_risk: np.ndarray) -> dict:
    """Sparse-encode per-leaf counts for serialization.

    Event counts (measured ~1-2% non-zero at default time_grid) go to
    leaf-major COO with a CSR-style indptr; the monotone non-increasing
    at-risk rows go to step-function breakpoints (every row keeps an
    implicit breakpoint at t=0). Sizes scale with in-bag samples instead
    of n_leaves x n_causes x n_time_bins.
    """
    n_leaves, n_causes, n_time_bins = leaf_event_counts.shape
    l_ec, c_ec, t_ec = np.nonzero(leaf_event_counts)
    vals_ec = leaf_event_counts[l_ec, c_ec, t_ec]
    ec_indptr = np.zeros(n_leaves + 1, dtype=np.int64)
    np.cumsum(np.bincount(l_ec, minlength=n_leaves), out=ec_indptr[1:])
    ec_val_dtype = (
        np.uint16
        if (vals_ec.size == 0 or int(vals_ec.max()) <= np.iinfo(np.uint16).max)
        else np.uint32
    )

    change = np.empty((n_leaves, n_time_bins), dtype=bool)
    change[:, 0] = True
    change[:, 1:] = leaf_at_risk[:, 1:] != leaf_at_risk[:, :-1]
    l_ar, t_ar = np.nonzero(change)
    vals_ar = leaf_at_risk[l_ar, t_ar]
    ar_indptr = np.zeros(n_leaves + 1, dtype=np.int64)
    np.cumsum(np.bincount(l_ar, minlength=n_leaves), out=ar_indptr[1:])

    return {
        "shape": (int(n_leaves), int(n_causes), int(n_time_bins)),
        "ec_indptr": ec_indptr,
        "ec_cause": c_ec.astype(np.uint8),
        "ec_time": t_ec.astype(np.uint16),
        "ec_val": vals_ec.astype(ec_val_dtype),
        "ar_indptr": ar_indptr,
        "ar_time": t_ar.astype(np.uint16),
        "ar_val": vals_ar.astype(np.uint32),
    }


def unpack_leaf_counts(packed: dict) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct dense uint32 (event_counts, at_risk) from pack_leaf_counts."""
    n_leaves, n_causes, n_time_bins = packed["shape"]

    ec = np.zeros((n_leaves, n_causes, n_time_bins), dtype=np.uint32)
    ec_counts = np.diff(packed["ec_indptr"])
    l_ec = np.repeat(np.arange(n_leaves, dtype=np.int64), ec_counts)
    ec[l_ec, packed["ec_cause"].astype(np.int64), packed["ec_time"].astype(np.int64)] = packed[
        "ec_val"
    ]

    ar_indptr = packed["ar_indptr"]
    t_ar = packed["ar_time"].astype(np.int64)
    # Forward-fill each leaf's step function: repeat every breakpoint value up
    # to the next breakpoint (or the end of the grid); each leaf has a
    # breakpoint at t=0, so runs tile the (n_leaves, n_time_bins) grid exactly.
    next_t = np.empty(len(t_ar), dtype=np.int64)
    if len(t_ar):
        next_t[:-1] = t_ar[1:]
        next_t[ar_indptr[1:] - 1] = n_time_bins
    ar = np.repeat(packed["ar_val"].astype(np.uint32), next_t - t_ar).reshape(n_leaves, n_time_bins)
    return ec, ar


def flatten_tree(
    tree,
    get_split_value: Callable,
    get_leaf_table: Callable,
    split_dtype,
    cache_attr: str = "_flat",
) -> FlatTree:
    """Pre-order DFS flattening. Caches the result on ``getattr(tree, cache_attr)``.

    ``get_split_value(node)`` returns the split scalar for an internal node
    (threshold for reference, bin index for histogram). ``get_leaf_table(node)``
    returns the ``(n_causes, n_time_bins)`` leaf quantity — CIF or CHF.
    Separate cache slots (``_flat`` and ``_flat_chf``) let a tree carry
    both flat representations independently.
    """
    cached = getattr(tree, cache_attr, None)
    if cached is not None:
        return cached

    features: list[int] = []
    splits: list = []
    lefts: list[int] = []
    rights: list[int] = []
    is_leaf: list[bool] = []
    leaf_nodes: list[tuple[int, np.ndarray]] = []

    def visit(node) -> int:
        idx = len(features)
        features.append(0)
        splits.append(0)
        lefts.append(0)
        rights.append(0)
        is_leaf.append(False)
        if node.is_leaf:
            is_leaf[idx] = True
            leaf_nodes.append((idx, get_leaf_table(node)))
            return idx
        li = visit(node.left)
        ri = visit(node.right)
        features[idx] = node.feature
        splits[idx] = get_split_value(node)
        lefts[idx] = li
        rights[idx] = ri
        return idx

    visit(tree)

    n_nodes = len(features)
    n_leaves = len(leaf_nodes)
    example = leaf_nodes[0][1]
    leaf_table = np.empty((n_leaves, *example.shape), dtype=np.float64)
    leaf_idx_of_node = np.full(n_nodes, -1, dtype=np.int64)
    for k, (node_idx, val) in enumerate(leaf_nodes):
        leaf_table[k] = val
        leaf_idx_of_node[node_idx] = k

    flat = FlatTree(
        features=np.asarray(features, dtype=np.int64),
        split_values=np.asarray(splits, dtype=split_dtype),
        left_children=np.asarray(lefts, dtype=np.int64),
        right_children=np.asarray(rights, dtype=np.int64),
        is_leaf_flags=np.asarray(is_leaf, dtype=bool),
        leaf_table=leaf_table,
        leaf_idx_of_node=leaf_idx_of_node,
    )
    setattr(tree, cache_attr, flat)
    return flat


@njit(cache=True, nogil=True)
def _descend_flat_nogil(
    features: np.ndarray,
    split_values: np.ndarray,
    left_children: np.ndarray,
    right_children: np.ndarray,
    is_leaf_flags: np.ndarray,
    leaf_idx_of_node: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """Per-sample root-to-leaf descent; returns the leaf-space index for each row.

    ``leaf_idx_of_node[node]`` maps the descended node index into the compact
    ``leaf_table`` index space ``[0, n_leaves)``. Numba specializes by dtype
    on first call for each ``(X.dtype, split_values.dtype)`` combination.
    Both reference-mode (float64, float64) and histogram-mode (uint8, int64)
    are exercised by the test suite.
    """
    n_samples = X.shape[0]
    leaf_idx = np.empty(n_samples, dtype=np.int64)
    for i in range(n_samples):
        node = 0
        while not is_leaf_flags[node]:
            feat = features[node]
            node = left_children[node] if X[i, feat] <= split_values[node] else right_children[node]
        leaf_idx[i] = leaf_idx_of_node[node]
    return leaf_idx


def predict_leaf_indices(flat: FlatTree, X: np.ndarray) -> np.ndarray:
    """Return the compact leaf index each row of ``X`` descends into.

    Thin Python-level wrapper that unpacks ``flat`` into the ndarray
    arguments the jitted kernel requires. The ``<=`` comparison inside
    is specialized per dtype by numba, so both reference mode
    (float64 ``X``, float thresholds) and histogram mode (uint8 ``X``,
    bin-index thresholds) are handled by the same source kernel.
    """
    return _descend_flat_nogil(
        flat.features,
        flat.split_values,
        flat.left_children,
        flat.right_children,
        flat.is_leaf_flags,
        flat.leaf_idx_of_node,
        X,
    )


def predict_with_flat(flat: FlatTree, X: np.ndarray) -> np.ndarray:
    """Vectorized predict: descend each row of ``X`` and return the leaf table."""
    return flat.leaf_table[predict_leaf_indices(flat, X)]
