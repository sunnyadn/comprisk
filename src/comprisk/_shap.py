"""TreeSHAP for cause-specific CIF on comprisk tree ensembles.

Implements Lundberg, Erion & Lee (2018), *Consistent Individualized Feature
Attribution for Tree Ensembles* (arXiv:1802.03888), Algorithm 2 (O(L·D²)),
with leaf values generalised from scalars to ``(n_causes, n_times)`` CIF
tensors.

This module is the host orchestration: it flattens all trees into single
concatenated arrays (``_build_concat``), then calls the parallel kernel
``shap_phi_prange`` (in ``_shap_alg2``), which runs the iterative
explicit-stack recursion in ``numba`` ``prange`` over samples and folds each
leaf's ``(n_causes, n_times)`` value into a per-sample ``phi`` accumulator in
place.  Each sample is owned by one thread (no cross-thread reduction), so the
result is bit-identical across thread counts and scales ~linearly with cores.
``times`` projection and ``time_aggregate`` collapse the leaf time axis here,
before the kernel, so the hot loop is oblivious to the time grid.
"""

from __future__ import annotations

import numba
import numpy as np
from sklearn.utils.validation import check_is_fitted

from comprisk._binning import apply_bins
from comprisk._shap_alg2 import _tree_height, max_path_offset, shap_phi_prange
from comprisk._tree_flat import FlatTree, flatten_tree
from comprisk._validation import validate_predict_X

# ---------------------------------------------------------------------------
# Tree representation helpers (cover counts, flattening)
# ---------------------------------------------------------------------------


def _extract_leaf_counts_reftree(node) -> list[int]:
    """Recursively extract leaf sample counts from a RefTreeNode."""
    if node.is_leaf:
        if node.at_risk is not None and len(node.at_risk) > 0:
            return [round(node.at_risk[0])]
        return [0]
    counts: list[int] = []
    if node.left is not None:
        counts.extend(_extract_leaf_counts_reftree(node.left))
    if node.right is not None:
        counts.extend(_extract_leaf_counts_reftree(node.right))
    return counts


def _extract_leaf_counts_histtree(node) -> list[int]:
    """Recursively extract leaf sample counts from a HistTreeNode."""
    if node.is_leaf:
        if node.at_risk_sparse is not None and len(node.at_risk_sparse.values) > 0:
            return [int(node.at_risk_sparse.values[0])]
        return [0]
    counts: list[int] = []
    if node.left is not None:
        counts.extend(_extract_leaf_counts_histtree(node.left))
    if node.right is not None:
        counts.extend(_extract_leaf_counts_histtree(node.right))
    return counts


def _get_flat_and_leaf_counts(tree):
    """Return (FlatTree, leaf_counts) for any tree representation."""
    if isinstance(tree, FlatTree):
        flat = tree
        if flat.leaf_at_risk is not None:
            leaf_counts = flat.leaf_at_risk[:, 0].astype(np.int64)
        else:
            leaf_counts = None
    elif hasattr(tree, "threshold"):
        flat = flatten_tree(
            tree,
            get_split_value=lambda n: n.threshold,
            get_leaf_table=lambda n: _ensure_cif_refnode(n),
            split_dtype=np.float64,
        )
        leaf_counts = np.asarray(_extract_leaf_counts_reftree(tree), dtype=np.int64)
    elif hasattr(tree, "bin_idx"):
        from comprisk._hist_tree import _flatten_tree_hist

        flat = _flatten_tree_hist(tree)
        leaf_counts = np.asarray(_extract_leaf_counts_histtree(tree), dtype=np.int64)
    else:
        raise TypeError(f"Unknown tree type: {type(tree)}")
    return flat, leaf_counts


def _ensure_cif_refnode(node):
    """Lazy materialise CIF on a RefTreeNode (mirrors _tree.py logic)."""
    from comprisk._estimators import aalen_johansen_from_counts

    if node._cif is None:
        n_causes = node.event_counts.shape[0]
        node._cif = aalen_johansen_from_counts(node.event_counts, node.at_risk, n_causes)
    return node._cif


def _compute_node_covers(
    is_leaf_flags: np.ndarray,
    left_children: np.ndarray,
    right_children: np.ndarray,
    leaf_idx_of_node: np.ndarray,
    leaf_counts: np.ndarray,
) -> np.ndarray:
    """Bottom-up compute cover (training sample count) for each node."""
    n_nodes = len(is_leaf_flags)
    covers = np.zeros(n_nodes, dtype=np.int64)

    def visit(j: int) -> None:
        if is_leaf_flags[j]:
            li = leaf_idx_of_node[j]
            if li >= 0:
                covers[j] = leaf_counts[li]
        else:
            visit(left_children[j])
            visit(right_children[j])
            covers[j] = covers[left_children[j]] + covers[right_children[j]]

    visit(0)
    return covers


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_TIME_AGGREGATORS = ("sum", "trapezoid")


def shap_values(
    forest, X, *, times=None, time_aggregate=None, n_jobs=-1
) -> tuple[np.ndarray, np.ndarray]:
    """Compute TreeSHAP values for cause-specific CIF.

    Parameters
    ----------
    forest : CompetingRiskForest
        A fitted competing-risk forest.
    X : array-like, shape (n_samples, n_features)
        Samples to explain.
    times : array-like of float or None, default=None
        Time points at which to evaluate SHAP.  If ``None``, uses
        the model's ``unique_times_`` grid.
    time_aggregate : {None, "sum", "trapezoid"}, default=None
        If set, collapse the time axis to one scalar per cause *before* the
        attribution (see :meth:`CompetingRiskForest.shap_values`).
    n_jobs : int, default=-1
        Number of numba threads for the parallel-over-samples kernel.
        ``-1`` uses all available cores; ``None`` or ``1`` runs single-threaded.
        Independent of the forest's fit-time ``n_jobs``; the result is
        bit-identical across thread counts (each sample is computed by exactly
        one thread, with no cross-thread reduction).

    Returns
    -------
    shap_values : ndarray
        ``(n_samples, n_features, n_times_out, n_causes)`` when
        ``time_aggregate is None``, else ``(n_samples, n_features, n_causes)``.
    base_value : ndarray
        ``(n_times_out, n_causes)`` or ``(n_causes,)`` to match.

        Additivity holds point-wise:

        .. math::

            sum_d shap_{s,d,t,c} + base_{t,c}
            approx predict_cif(X_s)_{c,t}
    """
    check_is_fitted(forest, "trees_")
    if time_aggregate is not None and time_aggregate not in _TIME_AGGREGATORS:
        raise ValueError(
            f"time_aggregate must be None or one of {_TIME_AGGREGATORS}; got {time_aggregate!r}"
        )
    X = validate_predict_X(forest, X)

    n_features = X.shape[1]
    if times is None:
        times_out = forest.unique_times_
        time_projection = None
    else:
        times_out = np.asarray(times, dtype=np.float64)
        time_projection = _make_time_projection(forest.unique_times_, times_out)
    n_times_out = len(times_out)
    n_causes = forest.n_causes_

    X_input = apply_bins(X, forest.bin_edges_) if forest.mode == "default" else X
    X_input = np.ascontiguousarray(X_input, dtype=np.float64)

    arrays = _build_concat(forest, time_projection, time_aggregate, times_out)

    nthreads = _resolve_threads(n_jobs)
    old_nthreads = numba.get_num_threads()
    numba.set_num_threads(nthreads)
    try:
        phi = shap_phi_prange(
            arrays["feat"],
            arrays["split"],
            arrays["left"],
            arrays["right"],
            arrays["is_leaf"],
            arrays["leaf_idx"],
            arrays["cover"],
            arrays["leaf_tbl"],
            arrays["node_off"],
            X_input,
            arrays["cols"],
            n_features,
            arrays["max_offset"],
        )  # (n_samples, n_features, cols)
    finally:
        numba.set_num_threads(old_nthreads)

    n_trees = len(forest.trees_)
    phi /= n_trees
    base = arrays["base"] / n_trees
    if time_aggregate is not None:
        shap = phi.reshape(len(X), n_features, n_causes)  # (n, F, n_causes)
    else:
        shap = phi.reshape(len(X), n_features, n_causes, n_times_out).transpose(0, 1, 3, 2)
    return shap, base


def _resolve_threads(n_jobs) -> int:
    """Map an ``n_jobs`` value to a numba thread count (clamped to the build max)."""
    maxth = numba.config.NUMBA_NUM_THREADS
    if n_jobs is None:
        return 1
    if n_jobs == -1:
        return maxth
    return max(1, min(int(n_jobs), maxth))  # n_jobs=1 falls through to 1


def _reduce_time_axis(arr: np.ndarray, time_aggregate, times_out: np.ndarray) -> np.ndarray:
    """Collapse the last (time) axis of ``arr`` — plain sum or trapezoidal rule."""
    if time_aggregate == "sum":
        return arr.sum(axis=-1)
    return np.trapezoid(arr, x=times_out, axis=-1)


# Kernel-facing dtype for each concatenated array. astype(copy=False) at the end
# is a no-op when a tree already produced the right dtype, so per-tree casts are
# unnecessary — only the final concatenate (which copies regardless) is paid.
_CONCAT_DTYPES = {
    "feat": np.int64,
    "split": np.float64,
    "left": np.int64,
    "right": np.int64,
    "is_leaf": np.int8,
    "leaf_idx": np.int64,
    "cover": np.float64,
    "leaf_tbl": np.float64,
}


def _build_concat(forest, time_projection, time_aggregate, times_out) -> dict:
    """Flatten every tree into single concatenated arrays for the prange kernel.

    ``left``/``right``/``leaf_idx`` are pre-offset to *global* indices (and -1 at
    leaves / internal nodes respectively); ``node_off[t]`` is tree ``t``'s root.
    Leaf values are time-projected / aggregated here (per the current call's
    ``times`` / ``time_aggregate``) so the kernel stays oblivious to the time
    axis.  ``covers``, the un-projected per-tree ``base``, and tree height are
    cached on the flattened tree across calls.  ``base`` is summed over trees in
    the returned shape ``(n_times_out, n_causes)`` (or ``(n_causes,)`` when
    aggregated); ``cols`` is the leaf-table width the kernel writes per feature.
    """
    rows = []  # one dict of per-tree arrays per tree, concatenated at the end
    node_off = []
    n_off = 0
    l_off = 0
    max_off = 0
    base = None
    for tree in forest.trees_:
        flat, leaf_counts = _get_flat_and_leaf_counts(tree)
        if leaf_counts is None:
            raise RuntimeError("Could not determine leaf sample counts for SHAP cover computation.")
        covers = getattr(flat, "_shap_covers", None)
        if covers is None:
            covers = _compute_node_covers(
                flat.is_leaf_flags,
                flat.left_children,
                flat.right_children,
                flat.leaf_idx_of_node,
                leaf_counts,
            )
            flat._shap_covers = covers
        b = getattr(flat, "_shap_base", None)
        if b is None:
            b = _base_value(flat, covers)  # (n_causes, n_times_full)
            flat._shap_base = b

        leaf_table = flat.leaf_table  # (n_leaves, n_causes, n_times_full)
        if time_projection is not None:
            leaf_table = time_projection(leaf_table)
            b = time_projection(b)
        if time_aggregate is not None:
            leaf_table = _reduce_time_axis(leaf_table, time_aggregate, times_out)  # (n_leaves, nc)
            bt = _reduce_time_axis(b, time_aggregate, times_out)  # (n_causes,)
        else:
            bt = b.T  # (n_times_out, n_causes)
        base = bt.copy() if base is None else base + bt

        ilf = flat.is_leaf_flags  # already bool; used as np.where mask and (as int8) by the kernel
        rows.append(
            {
                "feat": flat.features,
                "split": flat.split_values,
                "left": np.where(ilf, -1, flat.left_children + n_off),
                "right": np.where(ilf, -1, flat.right_children + n_off),
                "is_leaf": ilf,
                "leaf_idx": np.where(ilf, flat.leaf_idx_of_node + l_off, -1),
                "cover": covers,
                "leaf_tbl": leaf_table.reshape(leaf_table.shape[0], -1),
            }
        )
        node_off.append(n_off)
        n_off += len(ilf)
        l_off += leaf_table.shape[0]
        h = getattr(flat, "_shap_height", None)
        if h is None:
            h = int(_tree_height(flat.left_children, flat.right_children, flat.is_leaf_flags))
            flat._shap_height = h
        max_off = max(max_off, max_path_offset(h))

    out = {
        k: np.concatenate([r[k] for r in rows]).astype(dt, copy=False)
        for k, dt in _CONCAT_DTYPES.items()
    }
    out["node_off"] = np.array(node_off, dtype=np.int64)
    out["max_offset"] = max_off
    out["base"] = base
    out["cols"] = out["leaf_tbl"].shape[1]
    return out


def _base_value(flat: FlatTree, covers: np.ndarray) -> np.ndarray:
    """Expected leaf value = sum (cover_leaf / cover_root) * leaf_value."""
    root_cover = covers[0]
    if root_cover == 0:
        return np.zeros(flat.leaf_table.shape[1:], dtype=np.float64)
    base = np.zeros(flat.leaf_table.shape[1:], dtype=np.float64)
    for j in range(len(flat.is_leaf_flags)):
        if flat.is_leaf_flags[j]:
            leaf_idx = flat.leaf_idx_of_node[j]
            base += (covers[j] / root_cover) * flat.leaf_table[leaf_idx]
    return base


def _make_time_projection(unique_times: np.ndarray, target_times: np.ndarray):
    """Project ``(..., n_times_full)`` onto ``target_times`` (right-continuous step)."""
    idx = np.searchsorted(unique_times, target_times, side="right") - 1
    take = np.clip(idx, 0, None)
    before = idx < 0
    before_any = bool(before.any())

    def _project(arr):
        out = arr[..., take]
        if before_any:
            out[..., before] = 0.0
        return out

    return _project
