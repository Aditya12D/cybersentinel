"""Agent 2 public interface (for Agent 3 integration).

    from src.agents.anomaly_detection import detect_anomaly, detect_anomalies

    result  = detect_anomaly({"record_id": "C000003", "Flow Duration": 48.0, ...})
    results = detect_anomalies(dataframe_or_list_of_dicts)   # same order as input

Input
-----
A flow is a mapping / pandas Series / DataFrame row keyed by the CICIDS2017
feature names used in training (surrounding whitespace in names is tolerated).
Extra keys (record_id, IPs, ports, timestamps, label, other agents' outputs)
are ignored by the model; ``record_id`` is echoed back when present.
Missing, non-numeric and +/-inf values are repaired (see preprocessing.py),
listed in ``data_quality`` and never reported as evidence. A flow in which
*none* of the model features is usable raises ``ValueError``; a non-mapping
input raises ``TypeError``.

Output (JSON-serialisable; identical keys for every flow) - SCHEMA_VERSION 1.1
    record_id       str | None
    anomaly_score   float in [0, 1]: benign percentile, NOT an attack probability
    is_anomaly      bool  (anomaly_score >= threshold)
    evidence        list of {feature, observed_value, reference, severity, direction,
                    benign_tail_probability, benign_median, benign_p01, benign_p99,
                    nearest_benign_range, distance_contribution}
    reason          str   deterministic explanation built from the fields above
    rationale       str   alias of ``reason`` (kept for backward compatibility)
    threshold       float
    components      {knn_benign_percentile, isolation_forest_benign_percentile}
    data_quality    {missing_features, invalid_features, non_finite_features,
                     imputed_model_features, reliable}
    agent, schema_version, model_version

Agent 2 never predicts an attack class, never uses Agent 1's output and never
returns ALLOW/BLOCK; it supplies an independent signal for Agent 3.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from . import config
from .detector import AnomalyDetector
from .evidence import build_rationale, compute_evidence
from .preprocessing import _as_frame

SCHEMA_VERSION = "1.1"
AGENT_NAME = "anomaly_detection"
BATCH_SIZE = 20_000
# A result is marked unreliable when more than this share of the model
# features had to be imputed.
MAX_RELIABLE_IMPUTED_SHARE = 0.10

_lock = threading.Lock()
_cached: dict[str, AnomalyDetector] = {}


def load_detector(path=config.MODEL_PATH) -> AnomalyDetector:
    """Load (once per path, thread-safe) and cache the trained detector bundle."""
    key = str(Path(path).resolve())
    with _lock:
        if key not in _cached:
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"Model not found at {path}. Train it with: "
                    "python -m src.agents.anomaly_detection.train")
            _cached[key] = AnomalyDetector.load(path)
        return _cached[key]


class AnomalyDetectionAgent:
    """Stateless wrapper around a fitted detector. Reuse one instance; it is
    safe to call from multiple threads (scoring does not mutate state)."""

    def __init__(self, detector: AnomalyDetector | None = None, model_path=config.MODEL_PATH,
                 threshold: float | None = None):
        self.detector = detector if detector is not None else load_detector(model_path)
        t = float(self.detector.threshold if threshold is None else threshold)
        if not 0.0 <= t <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {t}")
        self.threshold = t

    def detect_anomaly(self, flow: Mapping) -> dict:
        if not isinstance(flow, (Mapping, pd.Series)):
            raise TypeError(f"flow must be a mapping, got {type(flow).__name__}")
        return self.detect_anomalies([flow])[0]

    def detect_anomalies(self, flows: pd.DataFrame | Iterable[Mapping]) -> list[dict]:
        df = _as_frame(flows)
        results: list[dict] = []
        for s in range(0, len(df), BATCH_SIZE):
            results.extend(self._batch(df.iloc[s:s + BATCH_SIZE], offset=s))
        return results

    def _batch(self, df: pd.DataFrame, offset: int = 0) -> list[dict]:
        if len(df) == 0:
            return []
        det = self.detector
        res = det.score(df)
        cols = np.asarray(det.model_idx_)
        imputed_model = res.report.imputed[:, cols]
        unusable = np.where(imputed_model.all(axis=1))[0]
        if len(unusable):
            raise ValueError(
                f"{len(unusable)} flow(s) contain no usable model features (first at position "
                f"{offset + int(unusable[0])}); expected CICIDS2017 feature names such as "
                f"{det.model_features[:3]}")
        flags = res.anomaly_score >= self.threshold
        evidence = compute_evidence(det, res, flags)
        ids = (df[config.ID_COLUMN].tolist() if config.ID_COLUMN in df.columns else [None] * len(df))
        model_feats = np.array(det.model_features)
        n_model = len(cols)
        out = []
        for i in range(len(df)):
            score, pk, pi = (float(res.anomaly_score[i]), float(res.knn_percentile[i]),
                             float(res.iforest_percentile[i]))
            imp = model_feats[imputed_model[i]].tolist()
            reason = build_rationale(score, self.threshold, bool(flags[i]), pk, pi,
                                     evidence[i], n_imputed=len(imp))
            rid = ids[i]
            out.append({
                "record_id": None if rid is None or (isinstance(rid, float) and np.isnan(rid)) else str(rid),
                "anomaly_score": round(score, 6),
                "is_anomaly": bool(flags[i]),
                "evidence": evidence[i],
                "reason": reason,
                "rationale": reason,
                "threshold": round(self.threshold, 6),
                "components": {"knn_benign_percentile": round(pk, 6),
                               "isolation_forest_benign_percentile": round(pi, 6)},
                "data_quality": {
                    "missing_features": res.report.missing[i],
                    "invalid_features": res.report.invalid[i],
                    "non_finite_features": res.report.non_finite[i],
                    "imputed_model_features": imp,
                    "reliable": len(imp) <= MAX_RELIABLE_IMPUTED_SHARE * n_model,
                },
                "agent": AGENT_NAME,
                "schema_version": SCHEMA_VERSION,
                "model_version": det.version,
            })
        return out


_default_agent: AnomalyDetectionAgent | None = None
_agent_lock = threading.Lock()


def _agent() -> AnomalyDetectionAgent:
    global _default_agent
    with _agent_lock:
        if _default_agent is None:
            _default_agent = AnomalyDetectionAgent()
        return _default_agent


def detect_anomaly(flow: Mapping) -> dict:
    """Assess a single flow. See module docstring for the output schema."""
    return _agent().detect_anomaly(flow)


def detect_anomalies(flows) -> list[dict]:
    """Assess a batch of flows (DataFrame or iterable of mappings), preserving order."""
    return _agent().detect_anomalies(flows)
