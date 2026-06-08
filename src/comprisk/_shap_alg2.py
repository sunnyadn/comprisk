"""Algorithm 2 TreeSHAP for comprisk — numba-jitted O(L·D²).

The recursion produces only the *structural* TreeSHAP weights — one scalar
per ``(leaf, path-feature)`` — accumulated into an ``(n_features, n_leaves)``
matrix ``W``.  The leaf values (``(n_causes, n_times)`` CIF tensors) never
enter the hot recursion; SHAP is linear in the leaf value, so

    phi = W @ leaf_table.reshape(n_leaves, n_causes * n_times)

recovers the attributions in a single BLAS matmul (see ``_shap.py``).  This
keeps the ``n_causes * n_times`` factor out of the L·D² inner loop.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------------------
# Path operations (EXTEND / UNWIND / unwound_path_sum)
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def _extend_path(
    path_feature,
    path_z,
    path_o,
    path_w,
    unique_depth,
    zero_fraction,
    one_fraction,
    feature_index,
):
    """Extend decision path with a new feature.

    Operates on the slice starting at index 0 with logical depth ``unique_depth``
    before the call.  After the call the valid entries are ``0 .. unique_depth``.
    """
    path_feature[unique_depth] = feature_index
    path_z[unique_depth] = zero_fraction
    path_o[unique_depth] = one_fraction
    path_w[unique_depth] = 1.0 if unique_depth == 0 else 0.0
    for i in range(unique_depth - 1, -1, -1):
        path_w[i + 1] += one_fraction * path_w[i] * (i + 1) / (unique_depth + 1)
        path_w[i] = zero_fraction * path_w[i] * (unique_depth - i) / (unique_depth + 1)


@njit(cache=True, nogil=True)
def _unwound_path_sum(path_z, path_o, path_w, unique_depth, path_index):
    """Total permutation weight if ``path_index`` were unwound."""
    one_fraction = path_o[path_index]
    zero_fraction = path_z[path_index]
    next_one_portion = path_w[unique_depth]
    total = 0.0
    if one_fraction != 0.0:
        for i in range(unique_depth - 1, -1, -1):
            tmp = next_one_portion / ((i + 1) * one_fraction)
            total += tmp
            next_one_portion = path_w[i] - tmp * zero_fraction * (unique_depth - i)
    else:
        for i in range(unique_depth - 1, -1, -1):
            total += path_w[i] / (zero_fraction * (unique_depth - i))
    return total * (unique_depth + 1)


@njit(cache=True, nogil=True)
def _unwind_path(path_feature, path_z, path_o, path_w, unique_depth, path_index):
    """Undo a previous extension; remove ``path_index`` and shift left."""
    one_fraction = path_o[path_index]
    zero_fraction = path_z[path_index]
    next_one_portion = path_w[unique_depth]
    for i in range(unique_depth - 1, -1, -1):
        if one_fraction != 0.0:
            tmp = path_w[i]
            path_w[i] = next_one_portion * (unique_depth + 1) / ((i + 1) * one_fraction)
            next_one_portion = tmp - path_w[i] * zero_fraction * (unique_depth - i) / (
                unique_depth + 1
            )
        else:
            path_w[i] = path_w[i] * (unique_depth + 1) / (zero_fraction * (unique_depth - i))
    for i in range(path_index, unique_depth):
        path_feature[i] = path_feature[i + 1]
        path_z[i] = path_z[i + 1]
        path_o[i] = path_o[i + 1]


# ---------------------------------------------------------------------------
# Recursive Algorithm 2 core  (offset-based, C++-style pointer arithmetic)
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def _tree_shap_recursive(
    x,
    features,
    split_values,
    left_children,
    right_children,
    is_leaf_flags,
    leaf_idx_of_node,
    covers,
    node,
    unique_depth,
    path_feature,
    path_z,
    path_o,
    path_w,
    path_offset,
    parent_z,
    parent_o,
    parent_feat,
    W,
):
    """Algorithm 2 — recursive descent with EXTEND / UNWIND.

    ``path_offset`` points to the *parent* path in the shared arrays.
    This routine first copies the parent prefix into a new slice at
    ``my_offset = path_offset + unique_depth + 1`` (mirroring C++
    ``unique_path = parent_unique_path + unique_depth + 1``), then
    extends it.  Children receive ``my_offset`` as their parent offset.

    At each leaf it writes the structural weight for every path feature
    into ``W[feature, leaf_idx]`` — the leaf value is multiplied in later
    by a single matmul, so this loop carries no ``n_causes * n_times``
    factor.
    """
    my_offset = path_offset + unique_depth + 1

    # Copy parent path into our working slice (C++ std::copy equivalent)
    for i in range(unique_depth + 1):
        path_feature[my_offset + i] = path_feature[path_offset + i]
        path_z[my_offset + i] = path_z[path_offset + i]
        path_o[my_offset + i] = path_o[path_offset + i]
        path_w[my_offset + i] = path_w[path_offset + i]

    _extend_path(
        path_feature[my_offset:],
        path_z[my_offset:],
        path_o[my_offset:],
        path_w[my_offset:],
        unique_depth,
        parent_z,
        parent_o,
        parent_feat,
    )

    if is_leaf_flags[node]:
        leaf_idx = leaf_idx_of_node[node]
        for i in range(1, unique_depth + 1):
            feat = path_feature[my_offset + i]
            if feat < 0:
                continue
            w = _unwound_path_sum(
                path_z[my_offset:],
                path_o[my_offset:],
                path_w[my_offset:],
                unique_depth,
                i,
            )
            W[feat, leaf_idx] += w * (path_o[my_offset + i] - path_z[my_offset + i])
        return

    feat = features[node]
    threshold = split_values[node]
    if x[feat] <= threshold:
        hot = left_children[node]
        cold = right_children[node]
    else:
        hot = right_children[node]
        cold = left_children[node]

    cover_total = covers[node]
    cover_hot = covers[hot]
    cover_cold = covers[cold]

    hot_z_frac = cover_hot / cover_total if cover_total > 0 else 0.0
    cold_z_frac = cover_cold / cover_total if cover_total > 0 else 0.0

    # Check for repeated feature already on the path
    incoming_z = 1.0
    incoming_o = 1.0
    path_index = unique_depth + 1  # not-found sentinel
    for i in range(unique_depth + 1):
        if path_feature[my_offset + i] == feat:
            path_index = i
            break

    if path_index != unique_depth + 1:
        incoming_z = path_z[my_offset + path_index]
        incoming_o = path_o[my_offset + path_index]
        _unwind_path(
            path_feature[my_offset:],
            path_z[my_offset:],
            path_o[my_offset:],
            path_w[my_offset:],
            unique_depth,
            path_index,
        )
        unique_depth -= 1

    child_offset = my_offset
    child_depth = unique_depth + 1

    # Recurse hot child (branch followed by sample x)
    _tree_shap_recursive(
        x,
        features,
        split_values,
        left_children,
        right_children,
        is_leaf_flags,
        leaf_idx_of_node,
        covers,
        hot,
        child_depth,
        path_feature,
        path_z,
        path_o,
        path_w,
        child_offset,
        hot_z_frac * incoming_z,
        incoming_o,
        feat,
        W,
    )

    # Recurse cold child (branch NOT followed by sample x)
    _tree_shap_recursive(
        x,
        features,
        split_values,
        left_children,
        right_children,
        is_leaf_flags,
        leaf_idx_of_node,
        covers,
        cold,
        child_depth,
        path_feature,
        path_z,
        path_o,
        path_w,
        child_offset,
        cold_z_frac * incoming_z,
        0.0,
        feat,
        W,
    )


# ---------------------------------------------------------------------------
# Driver: structural weights for a batch of samples on one (flattened) tree
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def _tree_height(left_children, right_children, is_leaf_flags) -> int:
    """Length of the longest root-to-leaf path (iterative DFS, no recursion)."""
    n_nodes = is_leaf_flags.shape[0]
    depth = np.zeros(n_nodes, dtype=np.int64)
    stack = np.empty(n_nodes, dtype=np.int64)
    stack[0] = 0
    top = 1
    h = 0
    while top > 0:
        top -= 1
        node = stack[top]
        if is_leaf_flags[node]:
            if depth[node] > h:
                h = depth[node]
        else:
            for child in (left_children[node], right_children[node]):
                depth[child] = depth[node] + 1
                stack[top] = child
                top += 1
    return h


def max_path_offset(height: int) -> int:
    """Scratch length for the offset-based path arrays at a given tree height.

    The recursion's per-level offset sequence is triangular in depth
    (0, 1, 3, 6, ...), so the deepest single root-to-leaf path needs this many
    slots.  Sizing by node count instead would over-allocate by ~10^4x on a
    deep, wide tree.  Shared by ``shap_tree_weights`` (recursive reference) and
    ``_build_concat`` (the prange kernel's scratch).
    """
    return (height + 2) * (height + 3) // 2 + 4


def shap_tree_weights(
    features: np.ndarray,
    split_values: np.ndarray,
    left_children: np.ndarray,
    right_children: np.ndarray,
    is_leaf_flags: np.ndarray,
    leaf_idx_of_node: np.ndarray,
    covers: np.ndarray,
    X: np.ndarray,
    n_features: int,
    n_leaves: int,
) -> np.ndarray:
    """Structural TreeSHAP weights for a batch of samples on one tree.

    Returns ``W`` of shape ``(n_samples, n_features, n_leaves)`` such that
    ``W[s] @ leaf_table.reshape(n_leaves, -1)`` is sample ``s``'s SHAP matrix
    (flattened over ``n_causes * n_times``).  The per-sample loop stays in
    Python — the recursion is jitted, but a recursive ``@njit`` function
    called from *within* another ``@njit`` function is a known crasher, so
    the driver itself is not jitted.
    """
    n_samples = X.shape[0]
    W = np.zeros((n_samples, n_features, n_leaves), dtype=np.float64)

    # The recursion's scratch path-arrays grow with the *tree height* (the
    # offset sequence is triangular in recursion depth: 0, 1, 3, 6, ...), not
    # with the node count — sizing them by ``n_nodes`` would over-allocate by
    # ~10^4x on a deep, wide tree and dominate the runtime.
    height = int(_tree_height(left_children, right_children, is_leaf_flags))
    max_offset = max_path_offset(height)
    path_feature = np.full(max_offset, -1, dtype=np.int64)
    path_z = np.zeros(max_offset, dtype=np.float64)
    path_o = np.zeros(max_offset, dtype=np.float64)
    path_w = np.zeros(max_offset, dtype=np.float64)
    path_z[0] = 1.0
    path_o[0] = 1.0
    path_w[0] = 1.0

    for si in range(n_samples):
        W_one = np.zeros((n_features, n_leaves), dtype=np.float64)
        _tree_shap_recursive(
            X[si],
            features,
            split_values,
            left_children,
            right_children,
            is_leaf_flags,
            leaf_idx_of_node,
            covers,
            0,
            0,
            path_feature,
            path_z,
            path_o,
            path_w,
            0,
            1.0,
            1.0,
            -1,
            W_one,
        )
        W[si] = W_one
    return W


# ---------------------------------------------------------------------------
# Production kernel: iterative explicit-stack Algorithm 2, prange over samples
# ---------------------------------------------------------------------------
#
# The recursive ``_tree_shap_recursive`` above is retained as a readable
# reference / test oracle.  The hot path is the function below: the recursion is
# unrolled into an explicit DFS stack (numba cannot ``prange`` over recursive
# calls), the per-sample loop runs in ``prange`` (fully nogil — no Python in the
# hot path, unlike the old ThreadPoolExecutor-over-trees driver whose Python
# sample loop serialised on the GIL), and each leaf's value is folded into the
# per-sample ``phi`` accumulator *in place* — no dense ``(n, F, n_leaves)`` W
# tensor and no BLAS matmul, so the working set stays cache-resident.  The net
# effect, measured on real multi-NUMA nodes, is ~linear scaling to 50+ cores
# (the old driver collapsed to ~40% efficiency by 8 threads).
#
# All trees are concatenated into flat arrays; ``left``/``right``/``leafidx`` are
# pre-offset to global indices and ``node_off[t]`` is tree ``t``'s root.  Each
# ``prange`` iteration owns one sample's ``out[r]`` row, so the result is
# bit-identical regardless of thread count.


@njit(parallel=True, cache=True, nogil=True)
def shap_phi_prange(
    feat,  # (n_nodes_total,) int64  — split feature per node
    split,  # (n_nodes_total,) float64
    left,  # (n_nodes_total,) int64  — global child index, -1 at leaves
    right,  # (n_nodes_total,) int64
    is_leaf,  # (n_nodes_total,) int8
    leaf_idx,  # (n_nodes_total,) int64  — global leaf-table row, -1 internal
    cover,  # (n_nodes_total,) float64 — training cover per node
    leaf_tbl,  # (n_leaves_total, cols) float64 — leaf values (already time-projected/aggregated)
    node_off,  # (n_trees,) int64 — global index of each tree's root
    Xb,  # (n_samples, n_features) float64
    cols,  # int — n_causes * n_times_out  (or n_causes when time-aggregated)
    n_features,  # int
    mo,  # int — max path-offset across all trees (scratch sizing)
):
    """Per-sample SHAP ``phi`` summed over trees. Returns ``(n_samples, n_features, cols)``."""
    n_samples = Xb.shape[0]
    n_trees = node_off.shape[0]
    out = np.zeros((n_samples, n_features, cols))
    for r in prange(n_samples):
        PF = np.full(mo, -1, np.int64)
        PZ = np.zeros(mo)
        PO = np.zeros(mo)
        PW = np.zeros(mo)
        SN = np.zeros(mo, np.int64)
        SD = np.zeros(mo, np.int64)
        SO = np.zeros(mo, np.int64)
        SPZ = np.zeros(mo)
        SPO = np.zeros(mo)
        SPF = np.zeros(mo, np.int64)
        phi = out[r]
        x = Xb[r]
        for tr in range(n_trees):
            PF[0] = -1
            PZ[0] = 1.0
            PO[0] = 1.0
            PW[0] = 1.0
            SN[0] = node_off[tr]
            SD[0] = 0
            SO[0] = 0
            SPZ[0] = 1.0
            SPO[0] = 1.0
            SPF[0] = -1
            sp = 1
            # The EXTEND / unwound-path-sum / UNWIND blocks below are inlined
            # copies of _extend_path / _unwound_path_sum / _unwind_path. They
            # are NOT factored out on purpose: the recursive reference passes
            # offset *slices* into those helpers, but creating slice views every
            # node visit inside a parallel=True prange body degrades numba's
            # parallel/alias analysis. Keep the three blocks in sync with the
            # helpers above if that math ever changes.
            while sp > 0:
                sp -= 1
                node = SN[sp]
                ud = SD[sp]
                poff = SO[sp]
                parz = SPZ[sp]
                paro = SPO[sp]
                parf = SPF[sp]
                mf = poff + ud + 1
                for i in range(ud + 1):
                    PF[mf + i] = PF[poff + i]
                    PZ[mf + i] = PZ[poff + i]
                    PO[mf + i] = PO[poff + i]
                    PW[mf + i] = PW[poff + i]
                # extend path with the parent edge
                PF[mf + ud] = parf
                PZ[mf + ud] = parz
                PO[mf + ud] = paro
                PW[mf + ud] = 1.0 if ud == 0 else 0.0
                for i in range(ud - 1, -1, -1):
                    PW[mf + i + 1] += paro * PW[mf + i] * (i + 1) / (ud + 1)
                    PW[mf + i] = parz * PW[mf + i] * (ud - i) / (ud + 1)
                if is_leaf[node]:
                    li = leaf_idx[node]
                    for i in range(1, ud + 1):
                        fe = PF[mf + i]
                        if fe < 0:
                            continue
                        one = PO[mf + i]
                        zer = PZ[mf + i]
                        nop = PW[mf + ud]
                        tot = 0.0
                        if one != 0.0:
                            for j in range(ud - 1, -1, -1):
                                tmp = nop / ((j + 1) * one)
                                tot += tmp
                                nop = PW[mf + j] - tmp * zer * (ud - j)
                        else:
                            for j in range(ud - 1, -1, -1):
                                tot += PW[mf + j] / (zer * (ud - j))
                        coef = tot * (ud + 1) * (PO[mf + i] - PZ[mf + i])
                        for c in range(cols):
                            phi[fe, c] += coef * leaf_tbl[li, c]
                    continue
                fe = feat[node]
                if x[fe] <= split[node]:
                    hot = left[node]
                    cold = right[node]
                else:
                    hot = right[node]
                    cold = left[node]
                ct = cover[node]
                hz = cover[hot] / ct if ct > 0 else 0.0
                cz = cover[cold] / ct if ct > 0 else 0.0
                inz = 1.0
                ino = 1.0
                pidx = ud + 1
                for i in range(ud + 1):
                    if PF[mf + i] == fe:
                        pidx = i
                        break
                if pidx != ud + 1:
                    inz = PZ[mf + pidx]
                    ino = PO[mf + pidx]
                    one = PO[mf + pidx]
                    zer = PZ[mf + pidx]
                    nop = PW[mf + ud]
                    for j in range(ud - 1, -1, -1):
                        if one != 0.0:
                            tmp = PW[mf + j]
                            PW[mf + j] = nop * (ud + 1) / ((j + 1) * one)
                            nop = tmp - PW[mf + j] * zer * (ud - j) / (ud + 1)
                        else:
                            PW[mf + j] = PW[mf + j] * (ud + 1) / (zer * (ud - j))
                    for j in range(pidx, ud):
                        PF[mf + j] = PF[mf + j + 1]
                        PZ[mf + j] = PZ[mf + j + 1]
                        PO[mf + j] = PO[mf + j + 1]
                    ud -= 1
                cd = ud + 1
                SN[sp] = cold
                SD[sp] = cd
                SO[sp] = mf
                SPZ[sp] = cz * inz
                SPO[sp] = 0.0
                SPF[sp] = fe
                sp += 1
                SN[sp] = hot
                SD[sp] = cd
                SO[sp] = mf
                SPZ[sp] = hz * inz
                SPO[sp] = ino
                SPF[sp] = fe
                sp += 1
    return out
