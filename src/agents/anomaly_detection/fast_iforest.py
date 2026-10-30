"""Vectorised scorer for a fitted sklearn IsolationForest.

sklearn scores each of the 300 trees in a Python-level loop, which dominates
single-flow latency (~20 ms of ~27 ms). This module packs every tree into flat
NumPy arrays and descends all trees for all rows at once (max depth 12 for
max_samples=4096), reproducing ``IsolationForest.score_samples`` exactly:

    depth_t(x) = node_depth[leaf] + c(n_node_samples[leaf]) - 1
    score(x)   = -2 ** (-sum_t depth_t(x) / (n_trees * c(max_samples)))

Equivalence with sklearn is asserted in tests/anomaly_detection/test_fast_iforest.py.
Only the configuration used by Agent 2 is supported (max_features == all
features, i.e. no per-tree feature subsampling); anything else raises.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.ensemble._iforest import _average_path_length


class CompiledIsolationForest:
    def __init__(self, forest: IsolationForest):
        n_features = forest.n_features_in_
        for feats in forest.estimators_features_:
            if len(feats) != n_features or not np.array_equal(np.sort(feats), np.arange(n_features)):
                raise ValueError("feature subsampling is not supported")
        if forest._max_features != n_features:
            raise ValueError("feature subsampling is not supported")

        trees = [est.tree_ for est in forest.estimators_]
        n_trees, width = len(trees), max(t.node_count for t in trees)
        self.left = np.full((n_trees, width), -1, dtype=np.int32)
        self.right = np.full((n_trees, width), -1, dtype=np.int32)
        self.feature = np.zeros((n_trees, width), dtype=np.int32)
        self.threshold = np.zeros((n_trees, width), dtype=np.float64)
        self.leaf_depth = np.zeros((n_trees, width), dtype=np.float64)
        for i, t in enumerate(trees):
            n = t.node_count
            self.left[i, :n] = t.children_left
            self.right[i, :n] = t.children_right
            self.feature[i, :n] = np.maximum(t.feature, 0)  # leaves store -2
            self.threshold[i, :n] = t.threshold
            # identical to sklearn's per-tree precomputation
            self.leaf_depth[i, :n] = (t.compute_node_depths()
                                      + _average_path_length(t.n_node_samples) - 1.0)
        self.max_depth = max(t.max_depth for t in trees)
        self.n_trees = n_trees
        self.n_features = n_features
        self.denominator = n_trees * _average_path_length([forest._max_samples])[0]
        self._rows = np.arange(n_trees)[:, None]

    def score_samples(self, X: np.ndarray, chunk: int = 4096) -> np.ndarray:
        """Same values as ``IsolationForest.score_samples`` (higher = more normal)."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"expected shape (n, {self.n_features}), got {X.shape}")
        out = np.empty(len(X), dtype=np.float64)
        for s in range(0, len(X), chunk):
            out[s:s + chunk] = self._score(X[s:s + chunk])
        return out

    def _score(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        node = np.zeros((self.n_trees, n), dtype=np.int32)
        cols = np.arange(n)[None, :]
        for _ in range(self.max_depth):
            left = self.left[self._rows, node]
            leaf = left == -1
            if leaf.all():
                break
            f = self.feature[self._rows, node]
            go_left = X[cols, f] <= self.threshold[self._rows, node]
            node = np.where(leaf, node, np.where(go_left, left, self.right[self._rows, node]))
        # Accumulate tree by tree, exactly like sklearn. (A numpy reduction may
        # switch to pairwise summation depending on the array shape; the last-bit
        # differences then flip ties in the benign reference ranking.)
        per_tree = self.leaf_depth[self._rows, node]
        depths = np.zeros(n, dtype=np.float64)
        for t in range(self.n_trees):
            depths += per_tree[t]
        if self.denominator == 0:
            return -np.ones(n)
        return -(2.0 ** (-depths / self.denominator))
