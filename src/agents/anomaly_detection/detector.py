"""Benign-only novelty detector for Agent 2.

Architecture (selected in model_selection.py; see docs/anomaly_detection.md):

    raw flow --FlowPreprocessor--> robust z (69 features)
             --correlation pruning--> 41 model features
             +--> kNN mean distance to 100k benign prototypes  -> benign percentile p_knn
             +--> Isolation Forest path-length score             -> benign percentile p_if
    fused  = (p_knn + p_if) / 2
    anomaly_score = benign percentile of ``fused``

Score semantics
---------------
``anomaly_score`` in [0, 1] is the fraction of held-out BENIGN reference flows
(never used for fitting) whose fused score is <= that of the flow. 0.97 means
"more unusual than 97% of normal traffic". It is *not* a probability that the
flow is an attack. Under the benign distribution it is ~Uniform(0, 1), so a
threshold ``t`` implies an expected benign false-positive rate of ~(1 - t) on
traffic resembling the training benign population.
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

from . import config
from .fast_iforest import CompiledIsolationForest
from .preprocessing import FlowPreprocessor, ImputationReport

MODEL_VERSION = "agent2-knn-iforest-1.1"

# Below this many rows the vectorised forest (exactly equal scores) is faster
# than sklearn's per-tree loop; above it sklearn's Cython path is faster.
FAST_IFOREST_MAX_ROWS = 512
# Below this many rows an exact kd-tree query is faster than brute force
# (~0.9 ms vs ~3.8 ms per single flow); above it brute force wins.
KD_TREE_MAX_ROWS = 8


def _percentile(sorted_ref: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Fraction of reference values <= value (empirical CDF)."""
    return np.searchsorted(sorted_ref, values, side="right") / len(sorted_ref)


def correlation_prune(Z: np.ndarray, threshold: float, seed: int) -> list[int]:
    """Greedy, label-free pruning in original feature order: keep a feature only
    if its |r| with every already-kept feature is below ``threshold``."""
    rng = np.random.default_rng(seed)
    n = min(len(Z), config.CORRELATION_SAMPLE_SIZE)
    S = Z[rng.choice(len(Z), n, replace=False)].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.nan_to_num(np.corrcoef(S.T))
    keep: list[int] = []
    for j in range(Z.shape[1]):
        if all(abs(C[j, k]) < threshold for k in keep):
            keep.append(j)
    return keep


@dataclass
class ScoreResult:
    """Vectorised scoring output (one entry per input row)."""

    anomaly_score: np.ndarray
    knn_percentile: np.ndarray
    iforest_percentile: np.ndarray
    raw: np.ndarray            # raw-unit (imputed) feature matrix, all model features
    z: np.ndarray              # standardised log-space matrix, all model features
    knn_neighbors: np.ndarray  # indices into prototypes, shape (n, k)
    knn_distance: np.ndarray   # mean distance to k neighbours
    report: ImputationReport


class AnomalyDetector:
    def __init__(self, seed: int = config.RANDOM_SEED):
        self.seed = seed
        self.threshold = config.DEFAULT_THRESHOLD
        self.operating_points: dict[str, float] = {}
        self.calibration_info: dict = {}
        self.version = MODEL_VERSION

    # ------------------------------------------------------------------ fit
    def fit(self, benign_fit: pd.DataFrame, benign_reference: pd.DataFrame) -> "AnomalyDetector":
        """Fit on BENIGN flows only. ``benign_reference`` must be disjoint from
        ``benign_fit``; it calibrates the score scale and the evidence reference."""
        self.preprocessor = FlowPreprocessor().fit(benign_fit)
        Z = self.preprocessor.transform(benign_fit)
        self.model_idx_ = correlation_prune(Z, config.MODEL_FEATURE_CORRELATION_PRUNE, self.seed)
        Zm = Z[:, self.model_idx_]

        rng = np.random.default_rng(self.seed)
        n_proto = min(len(Zm), config.KNN_N_PROTOTYPES)
        self.prototypes_ = np.ascontiguousarray(Zm[rng.choice(len(Zm), n_proto, replace=False)])
        self.knn_ = NearestNeighbors(n_neighbors=config.KNN_N_NEIGHBORS,
                                     algorithm="brute").fit(self.prototypes_)
        self._build_caches()
        params = dict(config.ISOLATION_FOREST_PARAMS)
        params["max_samples"] = min(params["max_samples"], len(Zm))
        self.iforest_ = IsolationForest(**params).fit(Zm)
        self.iforest_fast_ = CompiledIsolationForest(self.iforest_)

        # Reference distributions (held-out benign) for score normalisation.
        raw_ref = self.preprocessor.to_raw_matrix(benign_reference)
        Zr = self.preprocessor.raw_to_model(raw_ref)[:, self.model_idx_]
        d_knn, _ = self._knn(Zr)
        s_if = self._iforest_score(Zr)
        self.ref_knn_ = np.sort(d_knn)
        self.ref_if_ = np.sort(s_if)
        fused = 0.5 * (_percentile(self.ref_knn_, d_knn) + _percentile(self.ref_if_, s_if))
        self.ref_fused_ = np.sort(fused)

        # Per-feature benign reference (raw units) for evidence: sorted columns.
        self.ref_raw_sorted_ = np.sort(raw_ref.astype(np.float32), axis=0)
        self.n_fit_, self.n_reference_ = len(benign_fit), len(benign_reference)
        return self

    # --------------------------------------------------------------- scoring
    @property
    def features(self) -> list[str]:
        return self.preprocessor.features_

    @property
    def model_features(self) -> list[str]:
        return [self.preprocessor.features_[i] for i in self.model_idx_]

    def _knn(self, Zm: np.ndarray, batch: int = 20_000):
        """Mean Euclidean distance to the k nearest benign prototypes.

        Neighbours are found with an exact search (kd-tree for tiny inputs,
        brute force otherwise); distances are then recomputed directly in
        float64 so both paths give bit-identical results and near-duplicates
        are not affected by the cancellation error of sklearn's
        ||a||^2 - 2ab + ||b||^2 expansion."""
        k = config.KNN_N_NEIGHBORS
        if len(Zm) == 0:
            return np.empty(0), np.empty((0, k), dtype=np.intp)
        dists, idxs = [], []
        for s in range(0, len(Zm), batch):
            q = Zm[s:s + batch]
            index = self.knn_tree_ if len(q) <= KD_TREE_MAX_ROWS else self.knn_
            i = index.kneighbors(q, return_distance=False)
            diff = q[:, None, :].astype(np.float64) - self.prototypes_[i].astype(np.float64)
            d = np.sqrt(np.einsum("nkf,nkf->nk", diff, diff))
            order = np.argsort(d, axis=1, kind="stable")       # nearest first
            idxs.append(np.take_along_axis(i, order, axis=1))
            dists.append(np.take_along_axis(d, order, axis=1).mean(axis=1))
        return np.concatenate(dists), np.concatenate(idxs)

    # Derived, deterministic acceleration structures are rebuilt on load instead
    # of being pickled (smaller bundle, no stale caches).
    _DERIVED = ("knn_tree_", "iforest_fast_", "_evidence_ref_")

    def _build_caches(self) -> None:
        self.knn_tree_ = NearestNeighbors(n_neighbors=config.KNN_N_NEIGHBORS,
                                          algorithm="kd_tree").fit(self.prototypes_)
        if hasattr(self, "iforest_"):
            self.iforest_fast_ = CompiledIsolationForest(self.iforest_)

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k not in self._DERIVED}

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "prototypes_" in state:
            self._build_caches()

    def _iforest_score(self, Zm: np.ndarray) -> np.ndarray:
        """Isolation Forest anomaly score (higher = more anomalous)."""
        if len(Zm) == 0:
            return np.empty(0)
        if len(Zm) <= FAST_IFOREST_MAX_ROWS:
            return -self.iforest_fast_.score_samples(Zm)
        return -self.iforest_.score_samples(Zm)

    def score(self, X) -> ScoreResult:
        report = ImputationReport()
        raw = self.preprocessor.to_raw_matrix(X, report)
        z = self.preprocessor.raw_to_model(raw)
        Zm = z[:, self.model_idx_]
        d_knn, nbrs = self._knn(Zm)
        s_if = self._iforest_score(Zm)
        p_knn = _percentile(self.ref_knn_, d_knn)
        p_if = _percentile(self.ref_if_, s_if)
        score = _percentile(self.ref_fused_, 0.5 * (p_knn + p_if))
        return ScoreResult(score, p_knn, p_if, raw, z, nbrs, d_knn, report)

    def anomaly_score(self, X) -> np.ndarray:
        return self.score(X).anomaly_score

    def predict(self, X, threshold: float | None = None) -> np.ndarray:
        t = self.threshold if threshold is None else threshold
        return self.anomaly_score(X) >= t

    # ------------------------------------------------------------ persistence
    def save(self, path=config.MODEL_PATH) -> None:
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path=config.MODEL_PATH) -> "AnomalyDetector":
        obj = joblib.load(path)
        if not isinstance(obj, AnomalyDetector):
            raise TypeError(f"{path} does not contain an AnomalyDetector")
        return obj
